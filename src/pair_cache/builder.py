from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from typing import Any, Dict, List, Mapping, Sequence


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


def build_summary(
    candidates: Sequence[Mapping[str, Any]],
    pair_records: Sequence[Mapping[str, Any]],
    report_records: Sequence[Mapping[str, Any]],
    *,
    input_root: Any,
    output_root: Any,
    seed: int,
    include_action_sequence: bool,
    include_steps: bool,
) -> Dict[str, Any]:
    outcome_counts = Counter(str(item.get("outcome", "unknown")) for item in candidates)
    skip_counts = Counter(str(item.get("reason", "unknown")) for item in report_records)
    domain_counts = Counter(str(item.get("domain", "")) for item in pair_records)
    return {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "seed": seed,
        "include_action_sequence": bool(include_action_sequence),
        "include_steps": bool(include_steps),
        "candidate_files": len(candidates),
        "pairs": len(pair_records),
        "reported_skips": len(report_records),
        "candidate_outcomes": dict(sorted(outcome_counts.items())),
        "skip_reasons": dict(sorted(skip_counts.items())),
        "pairs_by_domain": dict(sorted(domain_counts.items())),
    }


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
