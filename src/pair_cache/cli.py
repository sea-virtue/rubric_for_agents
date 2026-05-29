from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


DEFAULT_INPUT_ROOT = Path("data/cache_data/agent-reward-bench/trajectories/cleaned")
DEFAULT_OUTPUT_ROOT = Path("data/cache_pair_data")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build pairwise positive/negative cache data from parsed trajectory caches. "
            "Input layout is expected to be domain/model/job_with_model/job.json; "
            "output layout is domain/job/job_with_model.json plus a pair.json manifest."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument(
        "--include-action-sequence",
        action="store_true",
        help="Include runtime_summary.action_sequence in pair response text.",
    )
    parser.add_argument(
        "--include-steps",
        action="store_true",
        help="Include compact parsed steps when runtime_summary is unavailable.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Read and summarize candidates without writing pair cache files.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    rng = random.Random(args.seed)
    candidates = load_candidates(args.input_root)
    grouped = group_candidates(candidates)

    pair_records: List[Dict[str, Any]] = []
    report_records: List[Dict[str, Any]] = []

    for key in sorted(grouped):
        group_candidates_for_task = sorted(grouped[key], key=lambda item: item["relative_source_path"])
        result = build_pair_record(
            key,
            group_candidates_for_task,
            rng=rng,
            include_action_sequence=args.include_action_sequence,
            include_steps=args.include_steps,
        )
        if result.get("pair"):
            pair_records.append(result["pair"])
        else:
            report_records.append(result["report"])
        if args.max_pairs is not None and len(pair_records) >= args.max_pairs:
            break

    summary = build_summary(candidates, pair_records, report_records, args)

    print(f"input_root: {args.input_root}")
    print(f"candidate_files: {len(candidates)}")
    print(f"task_groups: {len(grouped)}")
    print(f"pairs: {len(pair_records)}")
    print(f"reported_skips: {len(report_records)}")
    print(f"output_root: {args.output_root}")

    if args.dry_run:
        print("dry_run: true")
        print_preview(pair_records, report_records)
        return 0

    write_pair_outputs(
        pair_records,
        report_records,
        summary,
        output_root=args.output_root,
    )
    print(f"pair_index: {args.output_root / 'pair_index.json'}")
    print(f"pair_report: {args.output_root / 'pair_report.json'}")
    print(f"pair_summary: {args.output_root / 'pair_summary.json'}")
    return 0


def load_candidates(input_root: Path) -> List[Dict[str, Any]]:
    if not input_root.exists():
        raise FileNotFoundError(input_root)
    paths = [input_root] if input_root.is_file() else sorted(input_root.rglob("*.json"))
    candidates: List[Dict[str, Any]] = []
    for path in paths:
        source_info = parse_source_path(path, input_root)
        if not source_info:
            continue
        payloads = list(iter_json_payloads(path))
        if not payloads:
            continue
        for item_idx, payload in enumerate(payloads):
            record = dict(payload)
            outcome = infer_outcome(record)
            record_id = str(record.get("__record_id__") or f"{path.stem}_{item_idx}")
            candidates.append(
                {
                    **source_info,
                    "__record_id__": record_id,
                    "outcome": outcome,
                    "task_instruction": pick_task_instruction(record),
                    "record": record,
                }
            )
    return candidates


def parse_source_path(path: Path, input_root: Path) -> Dict[str, Any] | None:
    try:
        relative = path.resolve().relative_to(input_root.resolve())
    except ValueError:
        return None
    parts = relative.parts
    if len(parts) < 4:
        return None
    domain = parts[0]
    model_name = parts[1]
    job_with_model = parts[2]
    job_file = parts[-1]
    jobname = Path(job_file).stem
    return {
        "domain": domain,
        "model_name": model_name,
        "job_with_model": job_with_model,
        "jobname": jobname,
        "source_path": str(path),
        "relative_source_path": str(relative).replace("\\", "/"),
    }


def iter_json_payloads(path: Path) -> Iterable[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        for item in data:
            if isinstance(item, Mapping):
                yield item
    elif isinstance(data, Mapping):
        yield data


def group_candidates(candidates: Sequence[Mapping[str, Any]]) -> Dict[tuple[str, str], List[Dict[str, Any]]]:
    grouped: Dict[tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        grouped[(str(candidate["domain"]), str(candidate["jobname"]))].append(dict(candidate))
    return grouped


def build_pair_record(
    key: tuple[str, str],
    candidates: Sequence[Mapping[str, Any]],
    *,
    rng: random.Random,
    include_action_sequence: bool,
    include_steps: bool,
) -> Dict[str, Any]:
    domain, jobname = key
    successes = [item for item in candidates if item.get("outcome") == "success"]
    failures = [item for item in candidates if item.get("outcome") == "failure"]
    outcome_counts = Counter(str(item.get("outcome", "unknown")) for item in candidates)

    if not successes or not failures:
        return {
            "report": {
                "__record_id__": f"{domain}/{jobname}",
                "domain": domain,
                "jobname": jobname,
                "reason": skip_reason(outcome_counts, successes, failures),
                "outcomes": dict(sorted(outcome_counts.items())),
                "candidate_count": len(candidates),
                "candidates": [candidate_summary(item) for item in candidates],
            }
        }

    positive = rng.choice(list(successes))
    negative = rng.choice(list(failures))
    pair_id = f"{domain}/{jobname}"
    pair = {
        "__record_id__": pair_id,
        "pair_id": pair_id,
        "domain": domain,
        "jobname": jobname,
        "query": pick_pair_query(positive, negative),
        "label_rank": [1, 2],
        "responses": [
            render_response_text(
                positive,
                rank=1,
                include_action_sequence=include_action_sequence,
                include_steps=include_steps,
            ),
            render_response_text(
                negative,
                rank=2,
                include_action_sequence=include_action_sequence,
                include_steps=include_steps,
            ),
        ],
        "selected_records": [
            selected_record_summary(positive, pair_role="positive", label_rank=1),
            selected_record_summary(negative, pair_role="negative", label_rank=2),
        ],
        "candidate_count": len(candidates),
        "outcomes": dict(sorted(outcome_counts.items())),
    }
    return {"pair": pair}


def skip_reason(outcome_counts: Mapping[str, int], successes: Sequence[Any], failures: Sequence[Any]) -> str:
    if successes and not failures:
        return "all_success"
    if failures and not successes:
        return "all_failure"
    if not successes and not failures:
        return "no_success_or_failure"
    if not successes:
        return "no_success"
    if not failures:
        return "no_failure"
    return "unknown"


def infer_outcome(record: Mapping[str, Any]) -> str:
    runtime = record.get("runtime_summary")
    for value in (
        record.get("outcome"),
        runtime.get("outcome") if isinstance(runtime, Mapping) else None,
    ):
        normalized = normalize_outcome(value)
        if normalized != "unknown":
            return normalized

    validation = record.get("validation")
    if not isinstance(validation, Mapping) and isinstance(runtime, Mapping):
        validation = runtime.get("validation")
    if isinstance(validation, Mapping):
        reward = validation.get("reward", validation.get("raw_reward"))
        try:
            return "success" if float(reward) > 0 else "failure"
        except (TypeError, ValueError):
            pass
    return "unknown"


def normalize_outcome(value: Any) -> str:
    if isinstance(value, bool):
        return "success" if value else "failure"
    if isinstance(value, (int, float)):
        return "success" if value > 0 else "failure"
    lowered = str(value or "").strip().lower()
    if lowered in {"success", "succeeded", "pass", "passed", "correct"}:
        return "success"
    if lowered in {"failure", "failed", "fail", "incorrect", "wrong", "error"}:
        return "failure"
    return "unknown"


def pick_task_instruction(record: Mapping[str, Any]) -> str:
    for key in ("task_instruction", "task_instruct", "task", "instruction", "goal", "prompt", "question"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    runtime = record.get("runtime_summary")
    if isinstance(runtime, Mapping):
        for key in ("task_instruction", "task_instruct"):
            value = runtime.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def pick_pair_query(positive: Mapping[str, Any], negative: Mapping[str, Any]) -> str:
    for item in (positive, negative):
        value = str(item.get("task_instruction", "")).strip()
        if value:
            return value
    return str(positive.get("jobname", ""))


def render_response_text(
    candidate: Mapping[str, Any],
    *,
    rank: int,
    include_action_sequence: bool,
    include_steps: bool,
) -> str:
    record = candidate.get("record", {})
    if not isinstance(record, Mapping):
        record = {}
    runtime = record.get("runtime_summary")
    lines = [
        f"record_id: {candidate.get('__record_id__', '')}",
        f"model_name: {candidate.get('model_name', '')}",
        f"outcome: {candidate.get('outcome', 'unknown')}",
        f"label_rank: {rank}",
        f"task_instruction: {pick_task_instruction(record)}",
    ]
    if isinstance(runtime, Mapping):
        lines.extend(render_runtime_summary(runtime, include_action_sequence=include_action_sequence))
    else:
        lines.extend(render_fallback_record(record, include_steps=include_steps))
    return "\n".join(line for line in lines if line)


def render_runtime_summary(runtime: Mapping[str, Any], *, include_action_sequence: bool) -> List[str]:
    lines: List[str] = []
    validation = runtime.get("validation")
    if isinstance(validation, Mapping):
        lines.append(f"validation: {json.dumps(compact_validation(validation), ensure_ascii=False)}")

    if include_action_sequence:
        actions = runtime.get("action_sequence", [])
        if isinstance(actions, list) and actions:
            lines.append("action_sequence:")
            for action in actions:
                lines.append(f"- {json.dumps(action, ensure_ascii=False, default=str)}")

    final_state = runtime.get("final_state")
    if isinstance(final_state, Mapping):
        evidence_lines = final_state.get("evidence_lines", [])
        if isinstance(evidence_lines, list) and evidence_lines:
            lines.append("final_state:")
            for evidence in evidence_lines:
                lines.append(f"- {evidence}")

    cards = runtime.get("state_cards", [])
    if isinstance(cards, list) and cards:
        lines.append("state_cards:")
        for card in cards:
            if not isinstance(card, Mapping):
                continue
            role = card.get("rubric_role", card.get("evidence_role", ""))
            header = f"- {card.get('state_id', '')} | stage={card.get('stage', '')} | role={role}"
            lines.append(header)
            evidence_lines = card.get("evidence_lines", [])
            if isinstance(evidence_lines, list):
                for evidence in evidence_lines:
                    lines.append(f"  evidence: {evidence}")

    risk = runtime.get("risk_signals")
    if isinstance(risk, Mapping) and (risk.get("errors") or risk.get("repeated_action_runs")):
        lines.append("risk_signals:")
        lines.append(json.dumps(risk, ensure_ascii=False, default=str))
    return lines


def render_fallback_record(record: Mapping[str, Any], *, include_steps: bool) -> List[str]:
    lines: List[str] = []
    validation = record.get("validation")
    if isinstance(validation, Mapping):
        lines.append(f"validation: {json.dumps(compact_validation(validation), ensure_ascii=False)}")
    if not include_steps:
        return lines
    steps = record.get("steps")
    if not isinstance(steps, list):
        return lines
    lines.append("steps:")
    for step in steps:
        if not isinstance(step, Mapping):
            continue
        lines.append(f"- step {step.get('step_index', '')}")
        action = step.get("action_signature")
        if isinstance(action, Mapping) and action.get("raw"):
            lines.append(f"  action: {action.get('raw')}")
        obs = step.get("obs_snapshot")
        if isinstance(obs, Mapping):
            for cue in obs.get("task_relevant_cues", []) if isinstance(obs.get("task_relevant_cues"), list) else []:
                lines.append(f"  cue: {cue}")
        error = step.get("error_signal")
        if isinstance(error, Mapping) and error.get("has_error"):
            lines.append(f"  error: {error.get('message', '')}")
    return lines


def compact_validation(validation: Mapping[str, Any]) -> Dict[str, Any]:
    keep = ("reward", "raw_reward", "n_steps", "terminated", "truncated", "has_error", "outcome")
    return {key: validation.get(key) for key in keep if validation.get(key) not in (None, "")}


def selected_record_summary(candidate: Mapping[str, Any], *, pair_role: str, label_rank: int) -> Dict[str, Any]:
    return {
        "__record_id__": candidate.get("__record_id__"),
        "pair_role": pair_role,
        "label_rank": label_rank,
        "outcome": candidate.get("outcome"),
        "domain": candidate.get("domain"),
        "model_name": candidate.get("model_name"),
        "job_with_model": candidate.get("job_with_model"),
        "jobname": candidate.get("jobname"),
        "source_path": candidate.get("source_path"),
        "relative_source_path": candidate.get("relative_source_path"),
    }


def candidate_summary(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "__record_id__": candidate.get("__record_id__"),
        "outcome": candidate.get("outcome"),
        "model_name": candidate.get("model_name"),
        "job_with_model": candidate.get("job_with_model"),
        "relative_source_path": candidate.get("relative_source_path"),
    }


def write_pair_outputs(
    pair_records: Sequence[Mapping[str, Any]],
    report_records: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    *,
    output_root: Path,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    for pair in pair_records:
        domain = safe_path_part(str(pair["domain"]))
        jobname = safe_path_part(str(pair["jobname"]))
        pair_dir = output_root / domain / jobname
        pair_dir.mkdir(parents=True, exist_ok=True)
        write_json(pair_dir / "pair.json", pair)

        selected = pair.get("selected_records", [])
        if isinstance(selected, list):
            for item in selected:
                if not isinstance(item, Mapping):
                    continue
                record = load_selected_record(item)
                if not record:
                    continue
                record = attach_pair_metadata(record, pair, item)
                filename = safe_path_part(str(item.get("job_with_model") or item.get("__record_id__"))) + ".json"
                write_json(pair_dir / filename, record)

    write_json(output_root / "pair_index.json", list(pair_records))
    write_json(output_root / "pair_report.json", list(report_records))
    write_json(output_root / "pair_summary.json", summary)


def load_selected_record(item: Mapping[str, Any]) -> Dict[str, Any]:
    source_path = Path(str(item.get("source_path") or ""))
    if not source_path.exists():
        source_path = ROOT / source_path
    if not source_path.exists():
        return {}
    payloads = list(iter_json_payloads(source_path))
    record_id = str(item.get("__record_id__", ""))
    for payload in payloads:
        if str(payload.get("__record_id__", "")) == record_id:
            return dict(payload)
    return dict(payloads[0]) if payloads else {}


def attach_pair_metadata(
    record: Dict[str, Any],
    pair: Mapping[str, Any],
    selected_item: Mapping[str, Any],
) -> Dict[str, Any]:
    output = dict(record)
    output["_pair_cache"] = {
        "pair_id": pair.get("pair_id"),
        "domain": pair.get("domain"),
        "jobname": pair.get("jobname"),
        "pair_role": selected_item.get("pair_role"),
        "label_rank": selected_item.get("label_rank"),
        "source_path": selected_item.get("source_path"),
        "relative_source_path": selected_item.get("relative_source_path"),
    }
    return output


def build_summary(
    candidates: Sequence[Mapping[str, Any]],
    pair_records: Sequence[Mapping[str, Any]],
    report_records: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    outcome_counts = Counter(str(item.get("outcome", "unknown")) for item in candidates)
    skip_counts = Counter(str(item.get("reason", "unknown")) for item in report_records)
    domain_counts = Counter(str(item.get("domain", "")) for item in pair_records)
    return {
        "input_root": str(args.input_root),
        "output_root": str(args.output_root),
        "seed": args.seed,
        "include_action_sequence": bool(args.include_action_sequence),
        "include_steps": bool(args.include_steps),
        "candidate_files": len(candidates),
        "pairs": len(pair_records),
        "reported_skips": len(report_records),
        "candidate_outcomes": dict(sorted(outcome_counts.items())),
        "skip_reasons": dict(sorted(skip_counts.items())),
        "pairs_by_domain": dict(sorted(domain_counts.items())),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(tmp_path, path)


def safe_path_part(value: str) -> str:
    value = str(value or "").strip()
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value)
    value = value.rstrip(". ")
    return value or "unknown"


def print_preview(pair_records: Sequence[Mapping[str, Any]], report_records: Sequence[Mapping[str, Any]]) -> None:
    print("pair_preview:")
    for pair in pair_records[:5]:
        selected = pair.get("selected_records", [])
        print(f"- {pair.get('pair_id')} selected={len(selected) if isinstance(selected, list) else 0}")
    print("report_preview:")
    for item in report_records[:5]:
        print(f"- {item.get('__record_id__')} reason={item.get('reason')} outcomes={item.get('outcomes')}")


if __name__ == "__main__":
    raise SystemExit(main())

