from __future__ import annotations

import ast
import hashlib
import json
import re
from collections import Counter
from typing import Any, Dict, List, Mapping, Sequence

from .compressor import extract_ui_cues
from .schemas import CompactStep, TraceParsed, model_dump, model_validate


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


def parse_action_signature(action: Any) -> Dict[str, Any]:
    raw = str(action or "").strip()
    if not raw:
        return {"raw": "", "action_type": "", "target_bid": "", "target_text": "", "params": {}}

    signature: Dict[str, Any] = {
        "raw": raw,
        "action_type": "",
        "target_bid": "",
        "target_text": "",
        "params": {},
    }
    try:
        parsed = ast.parse(raw, mode="eval").body
    except SyntaxError:
        signature["action_type"] = raw.split("(", 1)[0].strip()
        return signature

    if not isinstance(parsed, ast.Call) or not isinstance(parsed.func, ast.Name):
        signature["action_type"] = raw.split("(", 1)[0].strip()
        return signature

    signature["action_type"] = parsed.func.id
    positional: List[Any] = []
    for arg in parsed.args:
        positional.append(_safe_literal_eval(arg))
    keyword_params: Dict[str, Any] = {}
    for kw in parsed.keywords:
        if kw.arg:
            keyword_params[kw.arg] = _safe_literal_eval(kw.value)
    signature["params"] = keyword_params
    if positional:
        signature["target_bid"] = _compact_scalar(positional[0])
    if len(positional) > 1:
        signature["target_text"] = _compact_scalar(positional[1])
    for key in ("value", "options", "text", "query", "target", "target_text", "key_comb", "wait_ms", "delta_y"):
        if key in keyword_params and not signature["target_text"]:
            signature["target_text"] = _compact_scalar(keyword_params[key])
    if not signature["target_bid"]:
        for key in ("bid", "selector", "target", "element"):
            if key in keyword_params:
                signature["target_bid"] = _compact_scalar(keyword_params[key])
                break
    return signature


def _safe_literal_eval(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except Exception:
        return ast.unparse(node) if hasattr(ast, "unparse") else ""


def _compact_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=False, default=str)


def _merge_list_values(*values: Any) -> List[str]:
    seen = set()
    merged: List[str] = []
    for value in values:
        items = value if isinstance(value, list) else [value]
        for item in items:
            text = str(item).strip()
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            merged.append(text)
            seen.add(key)
    return merged


def _obs_snapshot_from_event(event: Mapping[str, Any]) -> Dict[str, Any]:
    segments = event.get("observation_segments")
    cues: List[str] = []
    if isinstance(segments, list):
        for segment in segments:
            if not isinstance(segment, Mapping):
                continue
            section = str(segment.get("section", "observation")).strip() or "observation"
            raw_cues = segment.get("cues", [])
            if not isinstance(raw_cues, list):
                continue
            for cue in raw_cues:
                cue_text = str(cue).strip()
                if cue_text:
                    cues.append(f"{section}: {cue_text}")
    if not cues:
        observation_text = str(event.get("observation", event.get("axtree_pruned", event.get("axtree", ""))))
        cues = extract_ui_cues(observation_text, limit=16, max_chars=220)
    page_title = ""
    for cue in cues:
        match = re.search(r"heading '([^']+)'", cue)
        if match:
            page_title = match.group(1)
            break
    focused_element = str(event.get("focused_element", "")).strip()
    return {
        "source_key": str(event.get("observation_source", event.get("source_key", "observation"))),
        "page_title": page_title,
        "focused_element": focused_element,
        "clickable_options": [cue for cue in cues if any(token in cue.lower() for token in ("button", "link", "option", "checkbox", "combobox", "textbox", "searchbox"))],
        "task_relevant_cues": cues[:12],
    }


def _error_signal_from_event(event: Mapping[str, Any]) -> Dict[str, Any]:
    message = str(event.get("error", event.get("last_action_error", ""))).strip()
    if not message:
        return {"has_error": False, "error_type": "", "message": ""}
    lowered = message.lower()
    error_type = "unknown"
    for candidate in ("timeout", "notfound", "elementnotfound", "detached", "permission", "invalid", "blocked"):
        if candidate in lowered:
            error_type = candidate
            break
    return {"has_error": True, "error_type": error_type, "message": trim_text(message, 300)}


def _compact_step_from_event(event: Mapping[str, Any], fallback_index: int) -> Dict[str, Any] | None:
    action = str(event.get("action", "")).strip()
    thought = str(event.get("thought", event.get("reasoning", ""))).strip()
    observation = event.get("observation")
    error = event.get("error", event.get("last_action_error", ""))
    if not action and not thought and observation is None and not error:
        return None
    raw_index = str(event.get("index", event.get("num", fallback_index)))
    match = re.match(r"^(\d+)", raw_index)
    step_index = int(match.group(1)) if match else fallback_index
    return {
        "step_index": step_index,
        "thought_process": thought,
        "action_signature": parse_action_signature(action),
        "obs_snapshot": {},
        "error_signal": _error_signal_from_event(event),
    }


def _attach_observation(step: Dict[str, Any], event: Mapping[str, Any]) -> None:
    snapshot = _obs_snapshot_from_event(event)
    current = step.get("obs_snapshot", {})
    if not isinstance(current, dict):
        current = {}
    merged = dict(current)
    merged["source_key"] = snapshot.get("source_key", merged.get("source_key", ""))
    merged["page_title"] = snapshot.get("page_title", merged.get("page_title", ""))
    merged["focused_element"] = snapshot.get("focused_element", merged.get("focused_element", ""))
    merged["clickable_options"] = _merge_list_values(merged.get("clickable_options", []), snapshot.get("clickable_options", []))
    merged["task_relevant_cues"] = _merge_list_values(merged.get("task_relevant_cues", []), snapshot.get("task_relevant_cues", []))
    step["obs_snapshot"] = merged


def _finalize_step(step: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(step.get("obs_snapshot"), dict):
        step["obs_snapshot"] = {}
    if not isinstance(step.get("error_signal"), dict):
        step["error_signal"] = {"has_error": False, "error_type": "", "message": ""}
    return step


def parsed_trace_to_text(record: Mapping[str, Any]) -> str:
    task = str(record.get("task_instruction", record.get("task", ""))).strip()
    outcome = str(record.get("outcome", "unknown")).strip()
    steps = record.get("steps", [])
    validation = record.get("validation", {})
    lines = [
        f"task_instruction: {task}",
        f"outcome: {outcome}",
    ]
    if isinstance(validation, Mapping):
        reward = validation.get("reward", validation.get("cum_reward", ""))
        if reward not in ("", None):
            lines.append(f"validation.reward: {reward}")
    lines.append("steps:")
    for step in steps if isinstance(steps, list) else []:
        if not isinstance(step, Mapping):
            continue
        lines.append(f"- step {step.get('step_index', '')}")
        thought = str(step.get("thought_process", "")).strip()
        if thought:
            lines.append(f"  thought_process: {thought}")
        action = step.get("action_signature", {})
        if isinstance(action, Mapping):
            action_bits = [f"{action.get('action_type', '')}({action.get('target_bid', '')})".strip()]
            if action.get("target_text"):
                action_bits.append(f"text={action.get('target_text')}")
            lines.append(f"  action_signature: {', '.join(bit for bit in action_bits if bit and bit != '()')}")
        obs = step.get("obs_snapshot", {})
        if isinstance(obs, Mapping):
            title = str(obs.get("page_title", "")).strip()
            focused = str(obs.get("focused_element", "")).strip()
            cues = obs.get("task_relevant_cues", [])
            if title or focused or cues:
                lines.append("  obs_snapshot:")
                if title:
                    lines.append(f"    page_title: {title}")
                if focused:
                    lines.append(f"    focused_element: {focused}")
                for cue in list(cues)[:6] if isinstance(cues, list) else []:
                    lines.append(f"    cue: {cue}")
        error = step.get("error_signal", {})
        if isinstance(error, Mapping) and error.get("has_error"):
            lines.append(f"  error_signal: {error.get('error_type', 'unknown')} | {error.get('message', '')}")
    return "\n".join(lines)


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
    task_instruction = pick_first_text(
        record,
        ("task", "question", "prompt", "instruction", "goal", "user_request", "query"),
    )
    trace_source = _trace_source(record)
    outcome = infer_outcome(record)
    metadata = _record_metadata(record)
    summary = record.get("summary_info", {})
    chat_messages = record.get("chat_messages")
    if not task_instruction:
        task_instruction = trim_text(str(record.get("goal", "")), 400)
    steps = _parse_compact_steps(trace_source)
    if chat_messages and not steps:
        steps = _parse_steps_from_chat_messages(chat_messages)
    if not steps:
        steps = _parse_steps_from_text(record)

    parsed = TraceParsed(
        __record_id__=record_id,
        task_instruction=task_instruction,
        outcome=outcome,
        steps=[model_validate(CompactStep, step) for step in steps],
        validation=_validation_snapshot(summary, outcome, steps),
        metadata=metadata,
        chat_messages=chat_messages,
    )
    return model_dump(parsed)


def _trace_source_from_chat_messages(chat_messages: Any) -> List[Dict[str, Any]]:
    text = _extract_chat_messages_text(chat_messages)
    if not text:
        return []
    blocks = _split_step_blocks(text)
    if not blocks:
        return []
    return [{"content": block, "index": idx} for idx, block in enumerate(blocks)]


def _parse_steps_from_chat_messages(chat_messages: Any) -> List[Dict[str, Any]]:
    text = _extract_chat_messages_text(chat_messages)
    if not text:
        return []
    steps: List[Dict[str, Any]] = []
    for idx, message in enumerate(_iter_chat_messages(chat_messages)):
        if not isinstance(message, Mapping):
            continue
        role = str(message.get("role", "")).strip().lower()
        content = message.get("content")
        if role == "assistant" and isinstance(content, str):
            if "Reasoning:" in content or "Action:" in content or "Step:" in content:
                steps.extend(_parse_steps_from_plaintext(content))
            else:
                steps.append(
                    {
                        "step_index": idx,
                        "thought_process": trim_text(content, 1200),
                        "action_signature": {"raw": "", "action_type": "", "target_bid": "", "target_text": "", "params": {}},
                        "obs_snapshot": {},
                        "error_signal": {"has_error": False, "error_type": "", "message": ""},
                    }
                )
    return steps


def _parse_steps_from_text(record: Mapping[str, Any]) -> List[Dict[str, Any]]:
    text = _extract_chat_messages_text(record.get("chat_messages", {}))
    if not text:
        return []
    return _parse_steps_from_plaintext(text)


def _extract_chat_messages_text(chat_messages: Any) -> str:
    if isinstance(chat_messages, list):
        messages = chat_messages
    elif isinstance(chat_messages, Mapping):
        messages = []
        for key in ("regular", "pruned"):
            value = chat_messages.get(key)
            if isinstance(value, list):
                messages.extend(value)
    else:
        return ""

    chunks: List[str] = []
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        content = message.get("content")
        if isinstance(content, str):
            chunks.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, Mapping) and isinstance(item.get("text"), str):
                    chunks.append(item["text"])
                elif isinstance(item, str):
                    chunks.append(item)
        elif content is not None:
            chunks.append(str(content))
    return "\n\n".join(chunks)


def _iter_chat_messages(chat_messages: Any) -> List[Mapping[str, Any]]:
    if isinstance(chat_messages, list):
        return [message for message in chat_messages if isinstance(message, Mapping)]
    if isinstance(chat_messages, Mapping):
        messages: List[Mapping[str, Any]] = []
        for key in ("regular", "pruned"):
            value = chat_messages.get(key)
            if isinstance(value, list):
                messages.extend(message for message in value if isinstance(message, Mapping))
        return messages
    return []


def _split_step_blocks(text: str) -> List[str]:
    anchor = "The agent performed the following actions:"
    if anchor in text:
        text = text.split(anchor, 1)[1]
    if "The last accessibility tree is:" in text:
        text = text.split("The last accessibility tree is:", 1)[0]
    return [text]


def _parse_steps_from_plaintext(text: str) -> List[Dict[str, Any]]:
    blocks = _split_step_blocks(text)
    steps: List[Dict[str, Any]] = []
    final_tree = ""
    if "The last accessibility tree is:" in text:
        final_tree = text.split("The last accessibility tree is:", 1)[1].strip()
    for block in blocks:
        parsed_steps = _parse_step_block(block)
        steps.extend(parsed_steps)
    if steps and final_tree:
        _attach_final_tree_to_last_step(steps[-1], final_tree)
    return steps


def _parse_step_block(block: str) -> List[Dict[str, Any]]:
    lines = block.splitlines()
    steps: List[Dict[str, Any]] = []
    current: Dict[str, Any] | None = None
    current_mode = ""
    reasoning_lines: List[str] = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("The last accessibility tree is:"):
            break
        if line == "-----":
            continue
        if line.startswith("Step:"):
            if current is not None:
                current["thought_process"] = trim_text(" ".join(reasoning_lines), 1200)
                steps.append(current)
            match = re.search(r"Step:\s*(\d+)", line)
            current = {
                "step_index": int(match.group(1)) if match else len(steps),
                "thought_process": "",
                "action_signature": {"raw": "", "action_type": "", "target_bid": "", "target_text": "", "params": {}},
                "obs_snapshot": {"page_title": "", "focused_element": "", "clickable_options": [], "task_relevant_cues": []},
                "error_signal": {"has_error": False, "error_type": "", "message": ""},
            }
            current_mode = ""
            reasoning_lines = []
            continue
        if current is None:
            continue
        if line.startswith("URL:"):
            url = line.split("URL:", 1)[1].strip()
            obs = current.setdefault("obs_snapshot", {})
            if isinstance(obs, dict):
                cues = obs.get("task_relevant_cues", [])
                if not isinstance(cues, list):
                    cues = []
                cues.append(f"url: {url}")
                obs["task_relevant_cues"] = cues
            continue
        if line.startswith("Action:"):
            action = line.split("Action:", 1)[1].strip()
            current["action_signature"] = parse_action_signature(action)
            current_mode = ""
            continue
        if line.startswith("Reasoning:"):
            reasoning_lines.append(line.split("Reasoning:", 1)[1].strip())
            current_mode = "reasoning"
            continue
        if current_mode == "reasoning":
            reasoning_lines.append(line)
    if current is not None:
        current["thought_process"] = trim_text(" ".join(reasoning_lines), 1200)
        steps.append(current)
    return steps


def _attach_final_tree_to_last_step(step: Dict[str, Any], final_tree: str) -> None:
    snapshot = _obs_snapshot_from_event({"axtree_pruned": final_tree, "focused_element": step.get("action_signature", {}).get("target_bid", "")})
    obs = step.get("obs_snapshot", {})
    if not isinstance(obs, dict):
        obs = {}
    obs["page_title"] = snapshot.get("page_title", obs.get("page_title", ""))
    obs["focused_element"] = snapshot.get("focused_element", obs.get("focused_element", ""))
    obs["clickable_options"] = _merge_list_values(obs.get("clickable_options", []), snapshot.get("clickable_options", []))
    obs["task_relevant_cues"] = _merge_list_values(obs.get("task_relevant_cues", []), snapshot.get("task_relevant_cues", []))
    step["obs_snapshot"] = obs


def _parse_compact_steps(trace_source: Any) -> List[Dict[str, Any]]:
    raw_events = trace_source if isinstance(trace_source, list) else [trace_source]
    steps: List[Dict[str, Any]] = []
    current: Dict[str, Any] | None = None
    pending_observations: List[Mapping[str, Any]] = []
    pending_errors: List[Mapping[str, Any]] = []

    for idx, event in enumerate(raw_events):
        if not isinstance(event, Mapping):
            continue
        if _is_observation_event(event):
            if current is None:
                pending_observations.append(event)
            else:
                _attach_observation(current, event)
            continue
        if _is_error_event(event):
            if current is None:
                pending_errors.append(event)
            else:
                current["error_signal"] = _merge_error_signals(current.get("error_signal", {}), event)
            continue
        if _is_step_event(event):
            if current is not None:
                steps.append(_finalize_step(current))
            current = _compact_step_from_event(event, len(steps))
            if current is None:
                continue
            if pending_observations:
                for observation in pending_observations:
                    _attach_observation(current, observation)
                pending_observations = []
            if pending_errors:
                current["error_signal"] = _merge_error_signals(current.get("error_signal", {}), pending_errors[-1])
                pending_errors = []
            continue
        if current is None:
            continue
    if current is not None:
        steps.append(_finalize_step(current))
    return steps


def _is_step_event(event: Mapping[str, Any]) -> bool:
    return any(key in event for key in ("action", "reasoning", "thought", "url", "focused_element", "screenshot_path", "num")) or (
        "index" in event and any(key in event for key in ("action", "reasoning", "thought"))
    )


def _is_observation_event(event: Mapping[str, Any]) -> bool:
    return any(key in event for key in ("observation", "observation_segments", "axtree_pruned", "axtree"))


def _is_error_event(event: Mapping[str, Any]) -> bool:
    return any(key in event for key in ("error", "last_action_error"))


def _merge_error_signals(left: Mapping[str, Any], right: Mapping[str, Any]) -> Dict[str, Any]:
    left_message = str(left.get("message", "")).strip() if isinstance(left, Mapping) else ""
    right_signal = _error_signal_from_event(right)
    if right_signal.get("has_error"):
        return right_signal
    return dict(left) if left_message else {"has_error": False, "error_type": "", "message": ""}


def _validation_snapshot(summary: Any, outcome: str, steps: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    summary = summary if isinstance(summary, Mapping) else {}
    reward = summary.get("cum_reward", summary.get("cum_raw_reward"))
    raw_reward = summary.get("cum_raw_reward")
    return {
        "reward": reward,
        "raw_reward": raw_reward,
        "n_steps": summary.get("n_steps", len(steps)),
        "terminated": bool(summary.get("terminated", False)),
        "truncated": bool(summary.get("truncated", False)),
        "has_error": bool(summary.get("err_msg") or summary.get("stack_trace")),
        "outcome": outcome,
    }


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
            "response",
            "chat_messages",
            "trajectory_info",
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
                "summary": trim_text(text, 1000),
            }
        )
    return events


def extract_trace_features(events: Sequence[Mapping[str, Any]], trace_text: str) -> Dict[str, Any]:
    type_counts = Counter(str(event.get("type", "Action")) for event in events)
    tools = [str(event.get("tool_name", "")) for event in events if event.get("tool_name")]
    action_sequence = [str(event.get("type", "Action")) for event in events]
    strategy_signature = " -> ".join(action_sequence[:120])
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
