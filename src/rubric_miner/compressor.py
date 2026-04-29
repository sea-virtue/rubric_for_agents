from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Sequence


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
            pending_observations.setdefault(base_step, []).extend(
                extract_ui_cues(str(event.get("observation", "")), max_cues_per_observation, max_cue_chars)
            )
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
            "kept": ["thought", "action", "error", "url", "focused_element", "selected_ui_state_cues"],
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
            for cue in cues[:8]:
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


def _final_state(timeline: Sequence[Mapping[str, Any]]) -> str:
    for item in reversed(timeline):
        cues = item.get("state_cues") or []
        if cues:
            return " | ".join(map(str, cues[:4]))
    for item in reversed(timeline):
        if item.get("action"):
            return f"last_action: {item['action']}"
    return ""


def _trim(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 20].rstrip() + " ...[truncated]"
