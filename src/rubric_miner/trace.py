from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from typing import Any, Dict, List, Mapping, Sequence

from .compressor import compact_trace
from .schemas import TraceParsed, model_dump


def stable_record_id(item: Mapping[str, Any], idx: int) -> str:
    for key in ("__record_id__", "id", "trace_id", "sample_id", "record_id"):
        value = item.get(key)
        if value is not None:
            return str(value)
    raw = json.dumps(item, ensure_ascii=False, sort_keys=True, default=str)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"record_{idx:06d}_{digest}"


def scalar_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=False, default=str)


def flatten_trace(value: Any, depth: int = 0, max_depth: int = 8) -> str:
    if depth > max_depth:
        return scalar_to_text(value)[:1000]
    if isinstance(value, dict):
        return _flatten_mapping(value, depth, max_depth)
    if isinstance(value, list):
        return "\n".join(
            text
            for text in (flatten_trace(item, depth + 1, max_depth) for item in value)
            if text
        )
    return scalar_to_text(value)


def _flatten_mapping(value: Mapping[str, Any], depth: int, max_depth: int) -> str:
    preferred = (
        "role",
        "content",
        "message",
        "thought",
        "action",
        "observation",
        "tool",
        "tool_input",
        "tool_output",
        "response",
        "answer",
        "error",
    )
    keys = [key for key in preferred if key in value] + [
        key for key in value if key not in preferred and key != "__record_id__"
    ]
    chunks = []
    for key in keys:
        text = flatten_trace(value[key], depth + 1, max_depth)
        if text:
            chunks.append(f"{key}: {text}")
    return "\n".join(chunks)


def pick_first_text(record: Mapping[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def infer_outcome(record: Mapping[str, Any]) -> str:
    for key in ("outcome", "label", "status", "result"):
        value = record.get(key)
        if isinstance(value, str):
            lowered = value.lower()
            if any(token in lowered for token in ("fail", "failed", "error", "incorrect", "wrong")):
                return "failure"
            if any(token in lowered for token in ("success", "succeed", "pass", "passed", "correct")):
                return "success"
    for key in ("success", "passed", "is_success", "correct"):
        value = record.get(key)
        if isinstance(value, bool):
            return "success" if value else "failure"
    score = record.get("score")
    if isinstance(score, (int, float)):
        return "success" if score >= 0.5 else "failure"
    return "unknown"


def trim_text(text: str, limit: int) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 40].rstrip() + "\n...[truncated]..."


def parse_trace_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    record_id = str(record.get("__record_id__") or stable_record_id(record, 0))
    task = pick_first_text(
        record,
        ("task", "question", "prompt", "instruction", "goal", "user_request", "query"),
    )
    trace_source = _trace_source(record)
    outcome = infer_outcome(record)
    compact = compact_trace(task=task, outcome=outcome, trace_source=trace_source)
    trace_text = str(compact.get("text", "")) or flatten_trace(trace_source)
    compact_for_storage = dict(compact)
    compact_for_storage.pop("text", None)
    structured_sequence = segment_trace(trace_source)
    features = extract_trace_features(structured_sequence, trace_text)
    if not task:
        lines = trace_text.splitlines()
        task = trim_text(lines[0] if lines else "", 400)
    metadata = _record_metadata(record)

    parsed = TraceParsed(
        __record_id__=record_id,
        task=task,
        outcome=outcome,
        trace_text=trim_text(trace_text, 20000),
        compact_trace=compact_for_storage,
        structured_sequence=structured_sequence,
        features=features,
        raw={
            "__record_id__": record_id,
            "task": task,
            "outcome": outcome,
            "metadata": metadata,
        },
        metadata=metadata,
    )
    return model_dump(parsed)


def _record_metadata(record: Mapping[str, Any]) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    existing = record.get("metadata", {})
    if isinstance(existing, Mapping):
        metadata.update(dict(existing))
    for key, value in record.items():
        if key not in {
            "__record_id__",
            "trace",
            "trajectory",
            "messages",
            "steps",
            "events",
            "conversation",
            "log",
            "raw_input",
            "metadata",
            "task",
            "question",
            "prompt",
            "instruction",
            "goal",
            "user_request",
            "query",
            "outcome",
            "label",
            "status",
            "result",
            "success",
            "passed",
            "is_success",
            "correct",
            "score",
        }:
            metadata.setdefault(key, value)
    return metadata


def _trace_source(record: Mapping[str, Any]) -> Any:
    for key in ("trace", "trajectory", "messages", "steps", "events", "conversation", "log"):
        if key in record:
            return record[key]
    return record


def segment_trace(trace_source: Any) -> List[Dict[str, Any]]:
    """Extract [Action, Obs, StateChange, ToolCall, ErrorRecovery] events."""

    raw_events = trace_source if isinstance(trace_source, list) else [trace_source]
    events: List[Dict[str, Any]] = []
    for idx, item in enumerate(raw_events):
        if isinstance(item, dict):
            text = flatten_trace(item, max_depth=4)
            event_type = _event_type_from_mapping(item, text)
            tool_name = _pick_tool_name(item)
            role = str(item.get("role", "")).strip()
        else:
            text = scalar_to_text(item)
            event_type = _event_type_from_text(text)
            tool_name = ""
            role = ""
        events.append(
            {
                "index": idx,
                "type": event_type,
                "role": role,
                "tool_name": tool_name,
                "summary": trim_text(text, 260),
            }
        )
    return events


def extract_trace_features(events: Sequence[Mapping[str, Any]], trace_text: str) -> Dict[str, Any]:
    type_counts = Counter(str(event.get("type", "Action")) for event in events)
    tools = [str(event.get("tool_name", "")) for event in events if event.get("tool_name")]
    action_sequence = [str(event.get("type", "Action")) for event in events]
    strategy_signature = " -> ".join(action_sequence[:40])
    return {
        "num_events": len(events),
        "num_actions": int(type_counts.get("Action", 0)),
        "num_observations": int(type_counts.get("Obs", 0)),
        "num_state_changes": int(type_counts.get("StateChange", 0)),
        "num_tool_calls": int(type_counts.get("ToolCall", 0)),
        "num_error_recoveries": int(type_counts.get("ErrorRecovery", 0)),
        "tool_names": sorted(set(tools)),
        "tool_call_count": len(tools),
        "strategy_signature": strategy_signature,
        "has_error_recovery": bool(type_counts.get("ErrorRecovery", 0)),
        "trace_chars": len(trace_text),
    }


def _event_type_from_mapping(item: Mapping[str, Any], text: str) -> str:
    lowered_keys = {str(key).lower() for key in item}
    lowered = text.lower()
    if lowered_keys & {"tool", "tool_call", "tool_calls", "tool_input", "function_call"}:
        return "ToolCall"
    if lowered_keys & {"observation", "tool_output", "result", "output"}:
        return "Obs"
    if lowered_keys & {"state", "state_change", "diff", "patch", "updated", "changes"}:
        return "StateChange"
    if any(token in lowered for token in ("recover", "retry", "fix", "exception", "traceback", "error")):
        return "ErrorRecovery"
    if lowered_keys & {"action", "thought", "plan", "assistant"}:
        return "Action"
    return _event_type_from_text(text)


def _event_type_from_text(text: str) -> str:
    lowered = text.lower()
    if any(token in lowered for token in ("tool:", "tool_call", "function_call")):
        return "ToolCall"
    if any(token in lowered for token in ("observation:", "tool_output", "result:")):
        return "Obs"
    if any(token in lowered for token in ("state change", "patch", "diff", "updated")):
        return "StateChange"
    if any(token in lowered for token in ("recover", "retry", "fix", "exception", "traceback", "error")):
        return "ErrorRecovery"
    return "Action"


def _pick_tool_name(item: Mapping[str, Any]) -> str:
    for key in ("tool", "tool_name", "name", "function"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, dict) and value.get("name"):
            return str(value["name"]).strip()
    return ""
