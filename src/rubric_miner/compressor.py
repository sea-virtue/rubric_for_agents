from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, Sequence


UI_LINE_RE = re.compile(
    r"(button|link|textbox|searchbox|combobox|checkbox|alert|heading|menuitem|option|StaticText|LabelText)",
    re.IGNORECASE,
)


def compact_trace(
    *,
    task: str,
    outcome: str,
    trace_source: Any,
    max_steps: int = 80,
    max_thought_chars: int = 280,
    max_action_chars: int = 320,
    max_cues_per_observation: int = 12,
    max_cue_chars: int = 180,
) -> Dict[str, Any]:
    """Compress a raw trajectory into a compact, evidence-preserving form.

    This is deliberately rule-based. It avoids hallucinated summaries while
    preserving the action sequence, errors, URLs, focused elements, and selected
    UI/state cues from observations.
    """

    events = trace_source if isinstance(trace_source, list) else [trace_source]
    timeline: List[Dict[str, Any]] = []
    pending_observations: Dict[str, List[str]] = {}
    error_recoveries: List[Dict[str, Any]] = []

    for idx, event in enumerate(events[: max_steps * 3]):
        if not isinstance(event, Mapping):
            timeline.append({"step": str(idx), "note": _trim(str(event), max_action_chars)})
            continue

        step = str(event.get("index", event.get("num", idx)))
        base_step = step.split(".", 1)[0]
        if "observation" in event:
            cues = extract_observation_cues(event, max_cues_per_observation, max_cue_chars)
            if timeline and str(timeline[-1].get("step")) == base_step:
                timeline[-1]["state_cues"] = _merge_cues(timeline[-1].get("state_cues", []), cues)
            else:
                pending_observations[base_step] = _merge_cues(pending_observations.get(base_step, []), cues)
            continue

        item = {
            "step": base_step,
            "thought": _trim(str(event.get("thought", event.get("reasoning", ""))), max_thought_chars),
            "action": _trim(str(event.get("action", "")), max_action_chars),
            "url": _trim(str(event.get("url", "")), 220),
            "focused_element": _trim(str(event.get("focused_element", "")), 80),
            "state_cues": pending_observations.pop(base_step, []),
        }
        error = str(event.get("error", event.get("last_action_error", ""))).strip()
        if error:
            item["error"] = _trim(error, max_action_chars)
            error_recoveries.append({"step": base_step, "error": item["error"], "action": item["action"]})
        if item["thought"] or item["action"] or item["state_cues"] or item.get("error"):
            timeline.append(item)
        if len(timeline) >= max_steps:
            break

    for step, cues in pending_observations.items():
        if len(timeline) >= max_steps:
            break
        timeline.append({"step": step, "state_cues": cues})

    final_state = _final_state(timeline)
    action_signature = [item.get("action", "") for item in timeline if item.get("action")]
    compact = {
        "task": task,
        "outcome": outcome,
        "step_count": len(timeline),
        "action_signature": action_signature[:40],
        "timeline": timeline,
        "error_recoveries": error_recoveries,
        "final_state": final_state,
        "evidence_policy": {
            "lossy": True,
            "kept": [
                "thought",
                "action",
                "error",
                "url",
                "focused_element",
                "selected_ui_state_cues",
                "observation_head_tail_relevant_cues",
            ],
            "dropped": ["full_dom", "full_axtree", "bounding_boxes", "chat_history", "package_versions"],
        },
    }
    compact["text"] = compact_trace_to_text(compact)
    return compact


def compact_trace_to_text(compact: Mapping[str, Any]) -> str:
    lines = [
        f"task: {compact.get('task', '')}",
        f"outcome: {compact.get('outcome', 'unknown')}",
        f"step_count: {compact.get('step_count', 0)}",
    ]
    if compact.get("final_state"):
        lines.append(f"final_state: {compact['final_state']}")
    if compact.get("error_recoveries"):
        lines.append("error_recoveries:")
        for item in compact["error_recoveries"][:8]:
            lines.append(f"- step {item.get('step')}: {item.get('error')} | action: {item.get('action')}")

    lines.append("timeline:")
    for item in compact.get("timeline", []):
        header = f"- step {item.get('step')}"
        if item.get("url"):
            header += f" | url: {item.get('url')}"
        lines.append(header)
        if item.get("thought"):
            lines.append(f"  thought: {item.get('thought')}")
        if item.get("action"):
            lines.append(f"  action: {item.get('action')}")
        if item.get("focused_element"):
            lines.append(f"  focused: {item.get('focused_element')}")
        if item.get("error"):
            lines.append(f"  error: {item.get('error')}")
        cues = item.get("state_cues") or []
        if cues:
            lines.append("  state_cues:")
            for cue in _balanced_cues(cues, 8):
                lines.append(f"    - {cue}")
    return "\n".join(lines)


def extract_ui_cues(text: str, limit: int, max_chars: int) -> List[str]:
    cues: List[str] = []
    seen = set()
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line or not UI_LINE_RE.search(line):
            continue
        line = _trim(line, max_chars)
        key = line.lower()
        if key in seen:
            continue
        cues.append(line)
        seen.add(key)
        if len(cues) >= limit:
            break
    if not cues and text.strip():
        cues.append(_trim(re.sub(r"\s+", " ", text.strip()), max_chars))
    return cues


def extract_observation_cues(event: Mapping[str, Any], limit: int, max_chars: int) -> List[str]:
    """Extract balanced cues from a dataloader observation event.

    AgentRewardBench observations may include structured head/task_relevant/tail
    segments. Keep a small quota from each segment so the final page tail cannot
    be crowded out by boilerplate navigation at the top of the axtree.
    """

    segments = event.get("observation_segments")
    if isinstance(segments, list):
        cues = _extract_segmented_cues(segments, limit, max_chars)
        if cues:
            return cues
    return extract_ui_cues(str(event.get("observation", "")), limit, max_chars)


def _extract_segmented_cues(segments: Sequence[Any], limit: int, max_chars: int) -> List[str]:
    normalized_segments: List[tuple[str, List[str]]] = []
    for segment in segments:
        if not isinstance(segment, Mapping):
            continue
        section = str(segment.get("section", "observation")).strip() or "observation"
        raw_cues = segment.get("cues", [])
        if not isinstance(raw_cues, list):
            continue
        cues = [_trim(str(cue), max_chars) for cue in raw_cues if str(cue).strip()]
        if cues:
            normalized_segments.append((section, cues))
    if not normalized_segments:
        return []

    per_segment = max(1, limit // len(normalized_segments))
    selected: List[str] = []
    for section, cues in normalized_segments:
        for cue in cues[:per_segment]:
            selected.append(f"{section}: {cue}")
            if len(selected) >= limit:
                return _dedupe_preserve_order(selected)

    if len(selected) < limit:
        for section, cues in normalized_segments:
            for cue in cues[per_segment:]:
                selected.append(f"{section}: {cue}")
                if len(selected) >= limit:
                    return _dedupe_preserve_order(selected)
    return _dedupe_preserve_order(selected[:limit])


def _final_state(timeline: Sequence[Mapping[str, Any]]) -> str:
    for item in reversed(timeline):
        cues = item.get("state_cues") or []
        if cues:
            return " | ".join(map(str, _balanced_cues(cues, 4)))
    for item in reversed(timeline):
        if item.get("action"):
            return f"last_action: {item['action']}"
    return ""


def _merge_cues(left: Iterable[str], right: Iterable[str]) -> List[str]:
    return _dedupe_preserve_order([*map(str, left or []), *map(str, right or [])])


def _balanced_cues(cues: Iterable[Any], limit: int) -> List[str]:
    clean_cues = _dedupe_preserve_order(map(str, cues or []))
    if len(clean_cues) <= limit:
        return clean_cues

    groups: Dict[str, List[str]] = {}
    order: List[str] = []
    for cue in clean_cues:
        group = cue.split(":", 1)[0] if ":" in cue else "other"
        if group not in groups:
            groups[group] = []
            order.append(group)
        groups[group].append(cue)

    per_group = max(1, limit // max(1, len(order)))
    selected: List[str] = []
    for group in order:
        selected.extend(groups[group][:per_group])
        if len(selected) >= limit:
            return selected[:limit]

    if len(selected) < limit:
        for group in order:
            selected.extend(groups[group][per_group:])
            if len(selected) >= limit:
                return _dedupe_preserve_order(selected)[:limit]
    return _dedupe_preserve_order(selected)[:limit]


def _dedupe_preserve_order(values: Iterable[str]) -> List[str]:
    seen = set()
    output = []
    for value in values:
        clean = str(value).strip()
        if not clean:
            continue
        key = clean.lower()
        if key in seen:
            continue
        output.append(clean)
        seen.add(key)
    return output


def _trim(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 20].rstrip() + " ...[truncated]"
