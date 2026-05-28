from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rubric_miner.llm import async_llm_call, build_client, extract_json_array  # noqa: E402


DEFAULT_RUBRICS = Path("data/rubric/cluster_rubrics.json")
DEFAULT_CLUSTERS = Path("data/cluster/task_clusters.json")
DEFAULT_OUTPUT_DIR = Path("data/rubric_eval")

SEVERITY_WEIGHTS = {
    "high": 1.0,
    "medium": 0.6,
    "low": 0.3,
}

LEAKAGE_PATTERNS = {
    "uuid_or_series_id": re.compile(r"#?[0-9a-f]{6,}-[0-9a-f]?", re.I),
    "record_number": re.compile(r"\b(?:RITM|INC|PRB|CHG|EXP)\d{4,}\b"),
    "literal_url_or_path": re.compile(r"https?://|\.do\b|/now/nav|/catalogsearch"),
    "literal_action_id": re.compile(r"(?:click|fill|select_option|upload_file)\('[A-Za-z0-9_]+'"),
    "hyphenated_person_name": re.compile(r"\b[A-Z][a-z]+-[A-Z][a-z]+\s+[A-Z][a-z]+-[A-Z][a-z]+\b"),
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate mined rubrics as an independent stage. The static audit checks rubric "
            "structure and instance leakage. The judge stage scores cached trajectories with "
            "their cluster rubrics, then compares scores against success/failure labels."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--rubrics", type=Path, default=DEFAULT_RUBRICS)
    parser.add_argument("--clusters", type=Path, default=DEFAULT_CLUSTERS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=os.getenv("RUBRIC_EVAL_MODEL", os.getenv("RUBRIC_MODEL", "qwen3-4b-instruct-2507")))
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:18000/v1"))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-clusters", type=int, default=None)
    parser.add_argument("--cluster-ids", default="", help="Comma-separated cluster ids to evaluate.")
    parser.add_argument("--max-records-per-cluster", type=int, default=4)
    parser.add_argument("--max-chars-per-record", type=int, default=6000)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--include-label-fields",
        action="store_true",
        help=(
            "Include outcome/validation fields in judge prompts. Disabled by default to avoid label leakage; "
            "labels are still used locally for metric computation."
        ),
    )
    parser.add_argument("--static-only", action="store_true", help="Only run static rubric audits; skip LLM judging.")
    parser.add_argument("--refresh", action="store_true", help="Recompute existing judge scores.")
    parser.add_argument("--dry-run", action="store_true", help="Print selected clusters and one judge prompt preview.")
    return parser.parse_args(argv)


async def main_async(args: argparse.Namespace) -> int:
    rubric_records = load_json_array(args.rubrics)
    cluster_records = load_json_array(args.clusters)
    record_index = build_record_index(cluster_records)
    selected = select_rubric_records(
        rubric_records,
        selected_cluster_ids=parse_cluster_ids(args.cluster_ids),
        max_clusters=args.max_clusters,
    )
    if not selected:
        raise ValueError("No rubric clusters selected for evaluation.")

    static_records = [audit_rubric_record(record) for record in selected]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "rubric_static_audit.json", static_records)

    if args.dry_run:
        print(f"rubrics: {args.rubrics}")
        print(f"clusters: {args.clusters}")
        print(f"selected_clusters: {len(selected)}")
        print(f"static_flags: {sum(len(item.get('flags', [])) for item in static_records)}")
        if not args.static_only:
            first = selected[0]
            members = sample_cluster_members(first, record_index, max_records=args.max_records_per_cluster)
            if members:
                sample = load_one_cache_record(members[0])
                messages = judge_prompt_messages(
                    first,
                    sample,
                    args.max_chars_per_record,
                    include_label_fields=args.include_label_fields,
                )
                print(f"prompt_preview_cluster: {first.get('cluster_id')}")
                print(messages[1]["content"][:1600])
        return 0

    if args.static_only:
        summary = build_summary(static_records, [], selected)
        write_json(args.output_dir / "rubric_eval_summary.json", summary)
        print(f"static_audit: {args.output_dir / 'rubric_static_audit.json'}")
        print(f"selected_clusters: {len(selected)}")
        return 0

    score_path = args.output_dir / "trajectory_rubric_scores.json"
    raw_path = args.output_dir / "judge_raw_outputs.json"
    score_records = [] if args.refresh else load_json_array(score_path)
    raw_records = [] if args.refresh else load_json_array(raw_path)
    done = {
        str(record.get("__record_id__"))
        for record in score_records
        if record.get("__record_id__") and not record.get("judge_error")
    }

    client = build_client(api_key_env=args.api_key_env, base_url=args.base_url)
    semaphore = asyncio.Semaphore(max(1, args.concurrency))

    jobs = []
    for rubric_record in selected:
        for member in sample_cluster_members(rubric_record, record_index, max_records=args.max_records_per_cluster):
            job_id = score_record_id(rubric_record, member)
            if job_id in done:
                continue
            jobs.append((rubric_record, member))

    async def process_job(rubric_record: Mapping[str, Any], member: Mapping[str, Any]) -> Dict[str, Any]:
        async with semaphore:
            sample = load_one_cache_record(member)
            messages = judge_prompt_messages(
                rubric_record,
                sample,
                args.max_chars_per_record,
                include_label_fields=args.include_label_fields,
            )
            record_id = score_record_id(rubric_record, member)
            try:
                raw_content = await async_llm_call(
                    client,
                    args.model,
                    messages,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
                raw_record = {
                    "__record_id__": record_id,
                    "cluster_id": rubric_record.get("cluster_id"),
                    "record_id": member.get("__record_id__"),
                    "raw_output": raw_content,
                }
                upsert(raw_records, raw_record, key="__record_id__")
                write_json(raw_path, raw_records)

                raw_scores = extract_json_array(raw_content)
                rubric_scores = normalize_judge_scores(raw_scores, rubric_record.get("rubrics", []))
                weighted_score = weighted_rubric_score(rubric_scores, rubric_record.get("rubrics", []))
                outcome = sample_outcome(sample, member)
                validation_reward = sample_validation_reward(sample)
                return {
                    "__record_id__": record_id,
                    "cluster_id": rubric_record.get("cluster_id"),
                    "record_id": member.get("__record_id__"),
                    "outcome": outcome,
                    "label": outcome_label(outcome),
                    "validation_reward": validation_reward,
                    "weighted_score": weighted_score,
                    "rubric_scores": rubric_scores,
                }
            except Exception as exc:
                return {
                    "__record_id__": record_id,
                    "cluster_id": rubric_record.get("cluster_id"),
                    "record_id": member.get("__record_id__"),
                    "judge_error": str(exc),
                }

    for future in asyncio.as_completed([asyncio.create_task(process_job(*job)) for job in jobs]):
        record = await future
        upsert(score_records, record, key="__record_id__")
        write_json(score_path, score_records)
        print(
            "scored:"
            f" {record.get('cluster_id')}/{record.get('record_id')}"
            f" score={record.get('weighted_score', '')}"
            f" error={record.get('judge_error', '')[:160]}"
        )

    summary = build_summary(static_records, score_records, selected)
    config = {
        "rubrics": str(args.rubrics),
        "clusters": str(args.clusters),
        "output_dir": str(args.output_dir),
        "model": args.model,
        "base_url": args.base_url,
        "max_clusters": args.max_clusters,
        "cluster_ids": parse_cluster_ids(args.cluster_ids),
        "max_records_per_cluster": args.max_records_per_cluster,
        "max_chars_per_record": args.max_chars_per_record,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "include_label_fields": args.include_label_fields,
        "static_only": args.static_only,
        "outputs": {
            "static_audit": str(args.output_dir / "rubric_static_audit.json"),
            "trajectory_scores": str(score_path),
            "judge_raw_outputs": str(raw_path),
            "summary": str(args.output_dir / "rubric_eval_summary.json"),
            "config": str(args.output_dir / "rubric_eval_config.json"),
        },
    }
    write_json(args.output_dir / "rubric_eval_summary.json", summary)
    write_json(args.output_dir / "rubric_eval_config.json", config)
    print(f"static_audit: {args.output_dir / 'rubric_static_audit.json'}")
    print(f"trajectory_scores: {score_path}")
    print(f"summary: {args.output_dir / 'rubric_eval_summary.json'}")
    return 0


def load_json_array(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array")
    return [item for item in data if isinstance(item, dict)]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(tmp_path, path)


def upsert(records: List[Dict[str, Any]], record: Dict[str, Any], *, key: str) -> None:
    value = str(record.get(key))
    for idx, existing in enumerate(records):
        if str(existing.get(key)) == value:
            records[idx] = record
            return
    records.append(record)


def parse_cluster_ids(value: str) -> List[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def build_record_index(cluster_records: Sequence[Mapping[str, Any]]) -> Dict[str, Mapping[str, Any]]:
    return {
        str(record.get("__record_id__")): record
        for record in cluster_records
        if record.get("__record_id__") is not None
    }


def select_rubric_records(
    records: Sequence[Mapping[str, Any]],
    *,
    selected_cluster_ids: Sequence[str],
    max_clusters: Optional[int],
) -> List[Dict[str, Any]]:
    selected = set(selected_cluster_ids)
    output = []
    for record in records:
        cluster_id = str(record.get("cluster_id", ""))
        if selected and cluster_id not in selected:
            continue
        output.append(dict(record))
    output.sort(key=lambda item: (-int(item.get("cluster_size", 0)), str(item.get("cluster_id", ""))))
    if max_clusters is not None:
        output = output[: max(0, max_clusters)]
    return output


def audit_rubric_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    rubrics = [item for item in record.get("rubrics", []) if isinstance(item, Mapping)]
    flags: List[Dict[str, Any]] = []
    dimensions = [str(item.get("dimension", "")).strip().lower() for item in rubrics]
    duplicate_dimensions = [name for name, count in Counter(dimensions).items() if name and count > 1]
    if duplicate_dimensions:
        flags.append({"type": "duplicate_dimensions", "values": duplicate_dimensions})
    if len(rubrics) < 3:
        flags.append({"type": "too_few_rubrics", "count": len(rubrics)})
    if len(rubrics) > 6:
        flags.append({"type": "too_many_rubrics", "count": len(rubrics)})
    if int(record.get("cluster_size", 0)) <= 1:
        flags.append({"type": "singleton_cluster", "cluster_size": record.get("cluster_size", 0)})

    leakage_hits = []
    for idx, rubric in enumerate(rubrics):
        text = json.dumps(rubric, ensure_ascii=False)
        for name, pattern in LEAKAGE_PATTERNS.items():
            match = pattern.search(text)
            if match:
                leakage_hits.append({"rubric_index": idx, "type": name, "example": match.group(0)})
    if leakage_hits:
        flags.append({"type": "instance_leakage", "hits": leakage_hits[:20], "count": len(leakage_hits)})

    return {
        "__record_id__": record.get("__record_id__", record.get("cluster_id")),
        "cluster_id": record.get("cluster_id"),
        "cluster_size": record.get("cluster_size", 0),
        "rubric_count": len(rubrics),
        "field_coverage": {
            "criterion": coverage(rubrics, "criterion"),
            "positive_evidence": coverage(rubrics, "positive_evidence"),
            "negative_evidence": coverage(rubrics, "negative_evidence"),
            "verification_guide": coverage(rubrics, "verification_guide"),
        },
        "flags": flags,
    }


def coverage(items: Sequence[Mapping[str, Any]], key: str) -> float:
    if not items:
        return 0.0
    return sum(1 for item in items if item.get(key)) / len(items)


def sample_cluster_members(
    rubric_record: Mapping[str, Any],
    record_index: Mapping[str, Mapping[str, Any]],
    *,
    max_records: int,
) -> List[Mapping[str, Any]]:
    record_ids = [str(value) for value in rubric_record.get("source_record_ids", [])]
    members = [record_index[record_id] for record_id in record_ids if record_id in record_index]
    if max_records <= 0 or len(members) <= max_records:
        return members
    by_outcome: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for member in members:
        by_outcome[str(member.get("outcome", "unknown"))].append(member)
    selected: List[Mapping[str, Any]] = []
    for outcome in ("success", "failure", "unknown"):
        if by_outcome.get(outcome):
            selected.append(by_outcome[outcome][0])
    remaining = [member for member in members if member not in selected]
    head_count = max(0, (max_records - len(selected) + 1) // 2)
    tail_count = max(0, max_records - len(selected) - head_count)
    selected.extend(remaining[:head_count])
    if tail_count:
        selected.extend(remaining[-tail_count:])
    seen = set()
    deduped = []
    for member in selected:
        record_id = str(member.get("__record_id__"))
        if record_id in seen:
            continue
        seen.add(record_id)
        deduped.append(member)
    return deduped[:max_records]


def load_one_cache_record(member: Mapping[str, Any]) -> Dict[str, Any]:
    source_path = Path(str(member.get("source_path") or member.get("relative_source_path") or ""))
    if not source_path.exists():
        source_path = ROOT / source_path
    if not source_path.exists():
        return {
            "__record_id__": str(member.get("__record_id__", "")),
            "load_error": f"missing source_path: {member.get('source_path')}",
            "task_instruction": member.get("task_instruction", ""),
            "outcome": member.get("outcome", "unknown"),
        }
    data = json.loads(source_path.read_text(encoding="utf-8"))
    record_id = str(member.get("__record_id__", ""))
    if isinstance(data, list):
        for item in data:
            if isinstance(item, Mapping) and str(item.get("__record_id__")) == record_id:
                return dict(item)
        for item in data:
            if isinstance(item, Mapping):
                return dict(item)
    if isinstance(data, Mapping):
        return dict(data)
    return {
        "__record_id__": record_id,
        "load_error": f"unsupported cache payload in {source_path}",
        "task_instruction": member.get("task_instruction", ""),
        "outcome": member.get("outcome", "unknown"),
    }


def judge_prompt_messages(
    rubric_record: Mapping[str, Any],
    sample: Mapping[str, Any],
    max_chars_per_record: int,
    *,
    include_label_fields: bool,
) -> List[Dict[str, str]]:
    rubrics = []
    for idx, item in enumerate(rubric_record.get("rubrics", []), start=1):
        if not isinstance(item, Mapping):
            continue
        rubrics.append(
            {
                "rubric_index": idx,
                "dimension": item.get("dimension", ""),
                "criterion": item.get("criterion", ""),
                "positive_evidence": item.get("positive_evidence", []),
                "negative_evidence": item.get("negative_evidence", []),
                "severity": item.get("severity", "medium"),
                "verification_guide": item.get("verification_guide", {}),
            }
        )
    schema = {
        "rubric_index": 1,
        "score": "0.0 to 1.0, where 1 means clearly satisfied and 0 means clearly violated or absent",
        "evidence": "short trajectory evidence used for the score",
        "rationale": "brief reason",
    }
    return [
        {
            "role": "system",
            "content": (
                "You evaluate one agent trajectory against a fixed list of rubrics. "
                "Return only a strict JSON array with one item per rubric, in the same order."
            ),
        },
        {
            "role": "user",
            "content": (
                "Score each rubric against the trajectory evidence. Use 0.0 for absent or violated, "
                "0.5 for partial/ambiguous, and 1.0 for clearly satisfied. Do not judge the rubric's "
                "quality here; judge whether this trajectory satisfies it.\n\n"
                f"JSON item schema:\n{json.dumps(schema, ensure_ascii=False)}\n\n"
                f"cluster_id: {rubric_record.get('cluster_id')}\n"
                f"RUBRICS:\n{json.dumps(rubrics, ensure_ascii=False, indent=2)}\n\n"
                f"TRAJECTORY:\n{trim_text(render_sample(sample, include_label_fields=include_label_fields), max_chars_per_record)}"
            ),
        },
    ]


def render_sample(sample: Mapping[str, Any], *, include_label_fields: bool) -> str:
    runtime = sample.get("runtime_summary")
    if isinstance(runtime, Mapping):
        return render_runtime_summary(sample, runtime, include_label_fields=include_label_fields)
    lines = [
        f"__record_id__: {sample.get('__record_id__')}",
        f"task_instruction: {sample.get('task_instruction', sample.get('task', ''))}",
    ]
    if include_label_fields:
        lines.append(f"outcome: {sample.get('outcome', 'unknown')}")
        validation = sample.get("validation", {})
        if isinstance(validation, Mapping):
            lines.append(f"validation: {json.dumps(validation, ensure_ascii=False)}")
    steps = sample.get("steps", [])
    if isinstance(steps, list):
        lines.append("steps:")
        for step in steps[:16]:
            if isinstance(step, Mapping):
                lines.append(render_step(step))
    if sample.get("load_error"):
        lines.append(f"load_error: {sample.get('load_error')}")
    return "\n".join(line for line in lines if line)


def render_runtime_summary(sample: Mapping[str, Any], runtime: Mapping[str, Any], *, include_label_fields: bool) -> str:
    lines = [
        f"__record_id__: {sample.get('__record_id__', runtime.get('__record_id__'))}",
        f"task_instruction: {runtime.get('task_instruction', sample.get('task_instruction', ''))}",
    ]
    if include_label_fields:
        lines.append(f"outcome: {runtime.get('outcome', sample.get('outcome', 'unknown'))}")
        validation = runtime.get("validation", {})
        if isinstance(validation, Mapping):
            lines.append(f"validation: {json.dumps(validation, ensure_ascii=False)}")
    final_state = runtime.get("final_state", {})
    if isinstance(final_state, Mapping):
        lines.append("final_state:")
        for item in final_state.get("evidence_lines", [])[:10]:
            lines.append(f"- {item}")
    cards = runtime.get("state_cards", [])
    if isinstance(cards, list):
        lines.append("state_cards:")
        for card in cards[:8]:
            if not isinstance(card, Mapping):
                continue
            lines.append(f"- {card.get('state_id', '')} | role={card.get('rubric_role', card.get('evidence_role', ''))}")
            for evidence in list(card.get("evidence_lines", []))[:6]:
                lines.append(f"  evidence: {evidence}")
    risk = runtime.get("risk_signals", {})
    if isinstance(risk, Mapping) and (risk.get("errors") or risk.get("repeated_action_runs")):
        lines.append("risk_signals:")
        lines.append(json.dumps(risk, ensure_ascii=False)[:1600])
    return "\n".join(line for line in lines if line)


def render_step(step: Mapping[str, Any]) -> str:
    parts = [f"- step {step.get('step_index', '')}"]
    action = step.get("action_signature", {})
    if isinstance(action, Mapping) and action.get("raw"):
        parts.append(f"  action: {action.get('raw')}")
    obs = step.get("obs_snapshot", {})
    if isinstance(obs, Mapping):
        for cue in list(obs.get("task_relevant_cues", []))[:4]:
            parts.append(f"  cue: {cue}")
    return "\n".join(parts)


def trim_text(text: str, limit: int) -> str:
    text = re.sub(r"\n{3,}", "\n\n", str(text)).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 40)].rstrip() + "\n...[truncated]..."


def normalize_judge_scores(raw_items: Sequence[Any], rubrics: Sequence[Any]) -> List[Dict[str, Any]]:
    outputs = []
    for idx, rubric in enumerate(rubrics, start=1):
        if not isinstance(rubric, Mapping):
            continue
        raw = raw_items[idx - 1] if idx - 1 < len(raw_items) and isinstance(raw_items[idx - 1], Mapping) else {}
        outputs.append(
            {
                "rubric_index": idx,
                "dimension": rubric.get("dimension", ""),
                "severity": rubric.get("severity", "medium"),
                "score": normalize_score(raw.get("score", raw.get("rating", raw.get("value", 0.0)))),
                "evidence": str(raw.get("evidence", "")).strip(),
                "rationale": str(raw.get("rationale", raw.get("reason", ""))).strip(),
            }
        )
    return outputs


def normalize_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        text = str(value).lower()
        if "satisf" in text or "pass" in text:
            score = 1.0
        elif "partial" in text or "ambiguous" in text:
            score = 0.5
        else:
            score = 0.0
    return max(0.0, min(1.0, score))


def weighted_rubric_score(scores: Sequence[Mapping[str, Any]], rubrics: Sequence[Any]) -> float:
    total_weight = 0.0
    weighted = 0.0
    for score, rubric in zip(scores, rubrics):
        severity = "medium"
        if isinstance(rubric, Mapping):
            severity = str(rubric.get("severity", "medium")).lower()
        weight = SEVERITY_WEIGHTS.get(severity, SEVERITY_WEIGHTS["medium"])
        total_weight += weight
        weighted += weight * float(score.get("score", 0.0))
    return round(weighted / total_weight, 4) if total_weight else 0.0


def sample_outcome(sample: Mapping[str, Any], member: Mapping[str, Any]) -> str:
    runtime = sample.get("runtime_summary")
    if isinstance(runtime, Mapping) and runtime.get("outcome"):
        return str(runtime.get("outcome"))
    if sample.get("outcome"):
        return str(sample.get("outcome"))
    return str(member.get("outcome", "unknown"))


def sample_validation_reward(sample: Mapping[str, Any]) -> Optional[float]:
    runtime = sample.get("runtime_summary")
    validation = runtime.get("validation", {}) if isinstance(runtime, Mapping) else sample.get("validation", {})
    if not isinstance(validation, Mapping):
        return None
    value = validation.get("reward")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def outcome_label(outcome: str) -> Optional[int]:
    lowered = str(outcome).lower()
    if lowered == "success":
        return 1
    if lowered == "failure":
        return 0
    return None


def score_record_id(rubric_record: Mapping[str, Any], member: Mapping[str, Any]) -> str:
    return f"{rubric_record.get('cluster_id')}::{member.get('__record_id__')}"


def build_summary(
    static_records: Sequence[Mapping[str, Any]],
    score_records: Sequence[Mapping[str, Any]],
    selected_rubrics: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    valid_scores = [record for record in score_records if record.get("weighted_score") is not None and not record.get("judge_error")]
    labels = [record.get("label") for record in valid_scores]
    scores = [float(record.get("weighted_score", 0.0)) for record in valid_scores]
    global_auc = auc_from_scores(labels, scores)
    by_cluster: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for record in valid_scores:
        by_cluster[str(record.get("cluster_id"))].append(record)
    cluster_summaries = []
    for cluster_id, records in sorted(by_cluster.items()):
        cluster_labels = [record.get("label") for record in records]
        cluster_scores = [float(record.get("weighted_score", 0.0)) for record in records]
        success_scores = [score for label, score in zip(cluster_labels, cluster_scores) if label == 1]
        failure_scores = [score for label, score in zip(cluster_labels, cluster_scores) if label == 0]
        cluster_summaries.append(
            {
                "cluster_id": cluster_id,
                "evaluated_records": len(records),
                "success_count": len(success_scores),
                "failure_count": len(failure_scores),
                "auc": auc_from_scores(cluster_labels, cluster_scores),
                "success_mean_score": round(sum(success_scores) / len(success_scores), 4) if success_scores else None,
                "failure_mean_score": round(sum(failure_scores) / len(failure_scores), 4) if failure_scores else None,
                "score_gap": round(
                    (sum(success_scores) / len(success_scores)) - (sum(failure_scores) / len(failure_scores)),
                    4,
                )
                if success_scores and failure_scores
                else None,
            }
        )
    return {
        "selected_clusters": len(selected_rubrics),
        "static": {
            "clusters_with_flags": sum(1 for record in static_records if record.get("flags")),
            "total_flags": sum(len(record.get("flags", [])) for record in static_records),
        },
        "judge": {
            "evaluated_records": len(valid_scores),
            "judge_errors": sum(1 for record in score_records if record.get("judge_error")),
            "global_auc": global_auc,
            "success_count": sum(1 for label in labels if label == 1),
            "failure_count": sum(1 for label in labels if label == 0),
        },
        "clusters": cluster_summaries,
    }


def auc_from_scores(labels: Sequence[Any], scores: Sequence[float]) -> Optional[float]:
    positive = [score for label, score in zip(labels, scores) if label == 1]
    negative = [score for label, score in zip(labels, scores) if label == 0]
    if not positive or not negative:
        return None
    wins = 0.0
    total = 0
    for pos_score in positive:
        for neg_score in negative:
            total += 1
            if pos_score > neg_score:
                wins += 1.0
            elif pos_score == neg_score:
                wins += 0.5
    return round(wins / total, 4) if total else None


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        return asyncio.run(main_async(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
