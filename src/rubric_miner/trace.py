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
        cues = extract_rubric_ui_cues(observation_text, limit=48, max_chars=260)
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
        "task_relevant_cues": cues[:24],
    }


RUBRIC_UI_LINE_RE = re.compile(
    r"(StaticText|heading|grid|table|region|main|columnheader|rowgroup|gridcell|button|link|textbox|searchbox|combobox|checkbox|alert|option)",
    re.IGNORECASE,
)


def extract_rubric_ui_cues(text: str, *, limit: int, max_chars: int) -> List[str]:
    """Extract UI cues with rubric evidence roles prioritized over tree order."""

    candidates: List[tuple[int, int, str]] = []
    for idx, raw_line in enumerate(str(text or "").splitlines()):
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line or not RUBRIC_UI_LINE_RE.search(line):
            continue
        lowered = line.lower()
        score = 0
        if any(role in lowered for role in ("heading", "grid '", "table '", "region '", "main '")):
            score += 12
        if any(role in lowered for role in ("statictext", "columnheader", "gridcell")):
            score += 8
        if any(role in lowered for role in ("link", "button", "textbox", "searchbox", "combobox", "checkbox")):
            score += 5
        if any(marker in lowered for marker in ("error", "alert", "invalid", "failed", "no data", "showing rows")):
            score += 4
        if re.search(r"'[^']{3,}'", line):
            score += 2
        if any(boilerplate in lowered for boilerplate in ("global skip links", "my servicenow landing page", "show help", "show notifications")):
            score -= 7
        if any(emptyish in lowered for emptyish in ("generic ''", "section ''", "row ''", "gridcell ''")):
            score -= 3
        # Keep a little early navigation context, but let content evidence win.
        if idx < 20:
            score += 1
        candidates.append((score, idx, trim_text(line, max_chars)))

    if not candidates and text.strip():
        return [trim_text(re.sub(r"\s+", " ", text.strip()), max_chars)]

    candidates.sort(key=lambda item: (-item[0], item[1]))
    selected = [line for score, _, line in candidates if score > -4][:limit]
    return _dedupe_text(selected)


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
        "obs_snapshot": {
            "url": str(event.get("url", "")).strip(),
            "screenshot_path": str(event.get("screenshot_path", "")).strip(),
            "focused_element": str(event.get("focused_element", "")).strip(),
            "task_relevant_cues": [f"url: {event.get('url')}"] if event.get("url") else [],
            "clickable_options": [],
        },
        "error_signal": _error_signal_from_event(event),
    }


def _attach_observation(step: Dict[str, Any], event: Mapping[str, Any]) -> None:
    snapshot = _obs_snapshot_from_event(event)
    current = step.get("obs_snapshot", {})
    if not isinstance(current, dict):
        current = {}
    merged = dict(current)
    merged["source_key"] = snapshot.get("source_key", merged.get("source_key", ""))
    merged["url"] = str(event.get("url", merged.get("url", ""))).strip()
    merged["screenshot_path"] = str(event.get("screenshot_path", merged.get("screenshot_path", ""))).strip()
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
    runtime_summary = record.get("runtime_summary")
    if isinstance(runtime_summary, Mapping):
        return runtime_summary_to_text(runtime_summary)

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


def runtime_summary_to_text(summary: Mapping[str, Any]) -> str:
    """Render the rule-based, rubric-ready summary without step reasoning."""

    lines = [
        f"task_instruction: {summary.get('task_instruction', '')}",
        f"outcome: {summary.get('outcome', 'unknown')}",
    ]
    validation = summary.get("validation", {})
    if isinstance(validation, Mapping):
        reward = validation.get("reward")
        if reward not in (None, ""):
            lines.append(f"validation.reward: {reward}")
        if validation.get("terminated") not in (None, ""):
            lines.append(f"validation.terminated: {validation.get('terminated')}")

    goal_terms = summary.get("goal_terms", [])
    if isinstance(goal_terms, list) and goal_terms:
        lines.append("goal_terms: " + ", ".join(map(str, goal_terms[:12])))

    action_sequence = summary.get("action_sequence", [])
    if isinstance(action_sequence, list) and action_sequence:
        lines.append("action_sequence:")
        for action in action_sequence[:24]:
            if isinstance(action, Mapping):
                text = action.get("action")
                if action.get("target_text"):
                    text = f"{text} text={action.get('target_text')}"
                lines.append(f"- step {action.get('step_index')}: {text}")

    final_state = summary.get("final_state", {})
    if isinstance(final_state, Mapping):
        evidence = final_state.get("evidence_lines", [])
        if evidence:
            lines.append("final_state_evidence:")
            for line in evidence[:12]:
                lines.append(f"- {line}")

    cards = summary.get("state_cards", [])
    if isinstance(cards, list) and cards:
        lines.append("state_cards:")
        for card in cards[:8]:
            if not isinstance(card, Mapping):
                continue
            lines.append(
                f"- {card.get('state_id', '')} | stage={card.get('stage', '')} | "
                f"role={card.get('rubric_role', '')}"
            )
            for line in list(card.get("evidence_lines", []))[:6]:
                lines.append(f"  evidence: {line}")
    return "\n".join(lines)


def build_rubric_ready_views(
    *,
    record_id: str,
    task_instruction: str,
    outcome: str,
    steps: Sequence[Mapping[str, Any]],
    validation: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build conservative parser outputs for later rubric extraction.

    The parser does not decide the rubric. It only recalls likely evidence,
    marks candidate state cards, and records enough audit information to trace
    every compact cue back to a step.
    """

    goal_terms = extract_goal_terms(task_instruction)
    action_sequence = [_runtime_action_item(step) for step in steps if isinstance(step, Mapping)]
    action_sequence = [item for item in action_sequence if item]
    state_cards = build_candidate_state_cards(steps, goal_terms, outcome=outcome, validation=validation)
    final_state = _final_state_summary(steps, goal_terms)
    risk_signals = _risk_signals(steps)

    runtime_summary = {
        "schema_version": "rubric_ready_runtime_v2",
        "__record_id__": record_id,
        "task_instruction": task_instruction,
        "outcome": outcome,
        "validation": {
            "reward": validation.get("reward"),
            "raw_reward": validation.get("raw_reward"),
            "n_steps": validation.get("n_steps"),
            "terminated": validation.get("terminated"),
            "truncated": validation.get("truncated"),
            "has_error": validation.get("has_error"),
        },
        "goal_terms": goal_terms,
        "action_sequence": action_sequence[:120],
        "final_state": final_state,
        "state_cards": state_cards,
        "risk_signals": risk_signals,
        "evidence_policy": {
            "parser_role": "universal_trajectory_evidence_recall_not_rubric_judgment",
            "runtime_omits": ["full_reasoning", "full_axtree", "full_dom", "full_chat_messages"],
            "runtime_keeps": [
                "goal_terms",
                "validation_snapshot",
                "action_sequence",
                "final_state_evidence",
                "universal_state_cards",
                "error_and_loop_signals",
                "screenshot_paths",
                "source_step_indices",
            ],
        },
    }
    audit_trace = {
        "schema_version": "rubric_ready_audit_v1",
        "__record_id__": record_id,
        "parser_note": (
            "Audit trace keeps source pointers and lossy-reduction notes. "
            "Use it to inspect or repair runtime summaries; do not treat it as a rubric."
        ),
        "source_metadata": {
            "benchmark": metadata.get("benchmark"),
            "dataset": metadata.get("dataset"),
            "agent": metadata.get("agent"),
            "model": metadata.get("model"),
            "task_id": metadata.get("task_id"),
            "task_family": metadata.get("task_family"),
            "source_path": metadata.get("source_path"),
            "relative_source_path": metadata.get("relative_source_path"),
        },
        "reasoning_policy": {
            "runtime_summary": "does not include full step reasoning",
            "audit_location": "steps[*].thought_process",
            "reason": "agent reasoning can be useful for inspection but is weaker evidence than observed UI state",
        },
        "state_card_sources": [
            {
                "state_id": card.get("state_id"),
                "card_type": card.get("card_type"),
                "evidence_role": card.get("evidence_role"),
                "source_steps": card.get("source_steps", []),
                "source_fields": card.get("source_fields", []),
                "screenshot_paths": card.get("screenshot_paths", []),
            }
            for card in state_cards
        ],
        "dropped_or_downweighted": [
            "package_version",
            "token/cost statistics",
            "full accessibility tree/object by default",
            "full DOM/pruned HTML by default",
            "full chat history unless explicitly requested",
        ],
    }
    return {"runtime_summary": runtime_summary, "audit_trace": audit_trace}


def _legacy_extract_goal_terms(task_instruction: str) -> List[str]:
    text = str(task_instruction or "")
    terms: List[str] = []

    for match in re.finditer(r"[\"“”']([^\"“”']{3,80})[\"“”']", text):
        _append_term(terms, match.group(1))

    for part in re.split(r">|/|→|->", text):
        clean = re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", part).strip()
        if 3 <= len(clean) <= 80 and any(ch.isupper() for ch in clean):
            _append_term(terms, clean)

    for match in re.finditer(r"\b(?:[A-Z][A-Za-z0-9&.-]+)(?:\s+(?:[A-Z][A-Za-z0-9&.-]+)){0,5}\b", text):
        phrase = match.group(0).strip()
        if len(phrase) >= 3 and phrase.lower() not in _GOAL_TERM_STOPWORDS:
            _append_term(terms, phrase)

    for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_.-]{2,}", text):
        lowered = token.lower()
        if lowered in _GOAL_TERM_STOPWORDS:
            continue
        if any(term.lower() == lowered for term in terms):
            continue
        if any(lowered in term.lower().split() for term in terms):
            continue
        _append_term(terms, token)

    return terms[:24]


_GOAL_TERM_STOPWORDS = {
    "the",
    "and",
    "for",
    "to",
    "your",
    "you",
    "need",
    "navigate",
    "module",
    "application",
    "where",
    "contains",
    "create",
    "complete",
    "following",
    "steps",
    "concretely",
}


def _append_term(terms: List[str], value: Any) -> None:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    text = re.sub(r"^[\s:;,.>-]+|[\s:;,.>-]+$", "", text)
    if not text or len(text) < 3:
        return
    key = text.lower()
    if key in _GOAL_TERM_STOPWORDS:
        return
    if key not in {term.lower() for term in terms}:
        terms.append(text)


def _state_card_from_steps(
    state_id: str,
    stage: str,
    rubric_role: str,
    steps: Sequence[Mapping[str, Any]],
    goal_terms: Sequence[str],
    *,
    max_evidence: int,
) -> Dict[str, Any]:
    evidence: List[str] = []
    source_steps: List[Any] = []
    screenshots: List[str] = []
    fields = {"action_signature", "obs_snapshot"}
    matched: List[str] = []
    for step in steps:
        source_steps.append(step.get("step_index"))
        obs = step.get("obs_snapshot", {})
        if isinstance(obs, Mapping):
            screenshot = str(obs.get("screenshot_path", "")).strip()
            if screenshot:
                screenshots.append(screenshot)
        step_evidence = _step_evidence_lines(step, goal_terms, max_lines=max(4, max_evidence))
        evidence.extend(step_evidence)
        matched.extend(_matched_terms("\n".join(step_evidence), goal_terms))
    evidence = _dedupe_text(evidence)[:max_evidence]
    matched = _dedupe_text(matched)
    if not evidence and stage != "entry_state":
        return {}
    return {
        "state_id": state_id,
        "stage": stage,
        "rubric_role": rubric_role,
        "matched_goal_terms": matched,
        "evidence_lines": evidence,
        "source_steps": source_steps,
        "source_fields": sorted(fields),
        "screenshot_paths": _dedupe_text(screenshots),
        "confidence": "candidate",
    }


def _runtime_action_item(step: Mapping[str, Any]) -> Dict[str, Any]:
    action = step.get("action_signature", {})
    if not isinstance(action, Mapping):
        return {}
    raw = str(action.get("raw", "")).strip()
    action_type = str(action.get("action_type", "")).strip()
    if raw.lower() == "none" or action_type.lower() == "none":
        return {}
    if not raw and not action_type:
        return {}
    obs = step.get("obs_snapshot", {})
    return {
        "step_index": step.get("step_index"),
        "action": raw or action_type,
        "action_type": action_type,
        "target_bid": str(action.get("target_bid", "")).strip(),
        "target_text": str(action.get("target_text", "")).strip(),
        "url": str(obs.get("url", "")).strip() if isinstance(obs, Mapping) else "",
    }


def _final_state_summary(steps: Sequence[Mapping[str, Any]], goal_terms: Sequence[str]) -> Dict[str, Any]:
    if not steps:
        return {"evidence_lines": [], "matched_goal_terms": [], "source_step": None, "screenshot_path": ""}
    final = steps[-1]
    obs = final.get("obs_snapshot", {}) if isinstance(final, Mapping) else {}
    evidence = _step_evidence_lines(final, goal_terms, max_lines=20)
    return {
        "source_step": final.get("step_index") if isinstance(final, Mapping) else None,
        "screenshot_path": str(obs.get("screenshot_path", "")).strip() if isinstance(obs, Mapping) else "",
        "url": str(obs.get("url", "")).strip() if isinstance(obs, Mapping) else "",
        "matched_goal_terms": _matched_terms("\n".join(evidence), goal_terms),
        "evidence_lines": evidence,
    }


def _step_evidence_lines(step: Mapping[str, Any], goal_terms: Sequence[str], *, max_lines: int) -> List[str]:
    lines: List[str] = []
    action = step.get("action_signature", {})
    if isinstance(action, Mapping):
        raw = str(action.get("raw", "")).strip()
        target_text = str(action.get("target_text", "")).strip()
        if raw and raw.lower() != "none":
            lines.append(f"action: {raw}")
        if target_text:
            lines.append(f"action_target_text: {target_text}")
    obs = step.get("obs_snapshot", {})
    if isinstance(obs, Mapping):
        url = str(obs.get("url", "")).strip()
        if url:
            lines.append(f"url: {url}")
        page_title = str(obs.get("page_title", "")).strip()
        if page_title:
            lines.append(f"page_title: {page_title}")
        cues = []
        for key in ("task_relevant_cues", "clickable_options"):
            value = obs.get(key, [])
            if isinstance(value, list):
                cues.extend(str(item) for item in value if str(item).strip())
        lines.extend(_rank_evidence_lines(cues, goal_terms, limit=max_lines))
    return _dedupe_text(lines)[:max_lines]


def _rank_evidence_lines(lines: Sequence[str], goal_terms: Sequence[str], *, limit: int) -> List[str]:
    scored: List[tuple[int, int, str]] = []
    role_terms = (
        "heading",
        "statictext",
        "grid",
        "table",
        "region",
        "link",
        "button",
        "searchbox",
        "textbox",
        "combobox",
        "alert",
        "error",
    )
    for idx, line in enumerate(lines):
        clean = re.sub(r"\s+", " ", str(line)).strip()
        if not clean:
            continue
        lowered = clean.lower()
        goal_hits = sum(1 for term in goal_terms if term and term.lower() in lowered)
        role_hits = sum(1 for term in role_terms if term in lowered)
        # Preserve some non-matching structural cues, but prioritize goal-term hits.
        score = goal_hits * 10 + role_hits
        if score or idx < 4:
            scored.append((score, idx, trim_text(clean, 360)))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [line for _, _, line in scored[:limit]]


def _matched_terms(text: str, goal_terms: Sequence[str]) -> List[str]:
    lowered = str(text or "").lower()
    return [term for term in goal_terms if str(term).strip() and str(term).lower() in lowered]


def _step_evidence_text(step: Mapping[str, Any], *, include_reasoning: bool) -> str:
    chunks = _step_evidence_lines(step, [], max_lines=30)
    if include_reasoning:
        chunks.append(str(step.get("thought_process", "")))
    return "\n".join(chunks)


def _step_search_text(step: Mapping[str, Any]) -> str:
    chunks: List[str] = []
    action = step.get("action_signature", {})
    if isinstance(action, Mapping):
        chunks.extend(str(action.get(key, "")) for key in ("raw", "target_text", "target_bid"))
    obs = step.get("obs_snapshot", {})
    if isinstance(obs, Mapping):
        chunks.extend(str(obs.get(key, "")) for key in ("url", "page_title", "focused_element"))
        for key in ("task_relevant_cues", "clickable_options"):
            value = obs.get(key, [])
            if isinstance(value, list):
                chunks.extend(str(item) for item in value)
    return "\n".join(chunks)


def _risk_signals(steps: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    errors = []
    for step in steps:
        if not isinstance(step, Mapping):
            continue
        error = step.get("error_signal", {})
        if isinstance(error, Mapping) and error.get("has_error"):
            errors.append(
                {
                    "step_index": step.get("step_index"),
                    "error_type": error.get("error_type"),
                    "message": error.get("message"),
                }
            )
    return {
        "errors": errors,
        "repeated_action_runs": _repeated_action_runs(steps),
    }


def _repeated_action_runs(steps: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    current_action = ""
    current_start = None
    current_end = None
    count = 0
    for step in steps:
        if not isinstance(step, Mapping):
            continue
        action = step.get("action_signature", {})
        raw = str(action.get("raw", "")).strip() if isinstance(action, Mapping) else ""
        if not raw:
            continue
        step_idx = step.get("step_index")
        if raw == current_action:
            count += 1
            current_end = step_idx
            continue
        if count >= 3:
            runs.append({"action": current_action, "start_step": current_start, "end_step": current_end, "count": count})
        current_action = raw
        current_start = step_idx
        current_end = step_idx
        count = 1
    if count >= 3:
        runs.append({"action": current_action, "start_step": current_start, "end_step": current_end, "count": count})
    return runs


def _dedupe_text(values: Sequence[Any]) -> List[str]:
    seen = set()
    output: List[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(text)
    return output


_UNIVERSAL_GOAL_TERM_STOPWORDS = _GOAL_TERM_STOPWORDS | {
    "a",
    "an",
    "or",
    "of",
    "in",
    "on",
    "at",
    "by",
    "with",
    "without",
    "from",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "has",
    "have",
    "had",
    "do",
    "does",
    "did",
    "not",
    "that",
    "this",
    "these",
    "those",
    "which",
    "what",
    "when",
    "who",
    "whom",
    "whose",
    "why",
    "how",
    "tell",
    "find",
    "input",
    "image",
    "below",
    "local",
    "path",
    "posixpath",
    "url",
    "https",
    "http",
    "me",
    "my",
    "our",
}


def extract_goal_terms(task_instruction: str) -> List[str]:
    """Extract low-noise task terms for evidence ranking only.

    These terms must not be treated as a task parser. They are weak hints used
    to rank otherwise domain-neutral trajectory evidence.
    """

    text = str(task_instruction or "")
    terms: List[str] = []

    for match in re.finditer(r"[\"'“”]([^\"'“”]{3,80})[\"'“”]", text):
        _append_universal_term(terms, match.group(1))

    for part in re.split(r">|/|->|→", text):
        clean = re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", part).strip()
        if (
            3 <= len(clean) <= 80
            and any(ch.isupper() for ch in clean)
            and "\"" not in clean
            and "'" not in clean
            and not _looks_like_full_question(clean)
        ):
            _append_universal_term(terms, clean)

    for match in re.finditer(r"\b(?:[A-Z][A-Za-z0-9&.-]+)(?:\s+(?:[A-Z][A-Za-z0-9&.-]+)){0,5}\b", text):
        _append_universal_term(terms, match.group(0))

    for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_.@-]{2,}", text):
        lowered = token.lower()
        if lowered in _UNIVERSAL_GOAL_TERM_STOPWORDS:
            continue
        if any(term.lower() == lowered for term in terms):
            continue
        if any(lowered in term.lower().split() for term in terms):
            continue
        _append_universal_term(terms, token)

    return terms[:24]


def _append_universal_term(terms: List[str], value: Any) -> None:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    text = re.sub(r"^[\s:;,.>-]+|[\s:;,.>-]+$", "", text)
    if not text or len(text) < 3:
        return
    key = text.lower()
    if key in _UNIVERSAL_GOAL_TERM_STOPWORDS:
        return
    if key.startswith(("http://", "https://", "posixpath(")):
        return
    if _looks_like_full_question(text):
        return
    if key not in {term.lower() for term in terms}:
        terms.append(text)


def _looks_like_full_question(text: str) -> bool:
    clean = str(text or "").strip()
    lowered = clean.lower()
    if len(clean.split()) >= 8 and lowered.startswith(
        ("what ", "which ", "how ", "where ", "when ", "who ", "tell ", "find ")
    ):
        return True
    if len(clean.split()) >= 5 and lowered.startswith(
        ("navigate ", "create ", "order ", "assign ", "filter ", "submit ", "search ")
    ):
        return True
    return clean.endswith("?") and len(clean.split()) >= 6


def build_candidate_state_cards(
    steps: Sequence[Mapping[str, Any]],
    goal_terms: Sequence[str],
    *,
    outcome: str,
    validation: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """Build domain-neutral state cards from BrowserGym-style trajectories."""

    clean_steps = [step for step in steps if isinstance(step, Mapping)]
    cards: List[Dict[str, Any]] = [_universal_task_card(goal_terms, outcome=outcome, validation=validation)]
    if not clean_steps:
        return cards

    first_card = _universal_state_card_from_steps(
        "initial_observation",
        "initial_observation",
        "initial_context",
        clean_steps[:1],
        goal_terms,
        max_evidence=8,
    )
    if first_card:
        first_card["state_summary"] = "Initial observable state and first action context."
        cards.append(first_card)

    transition_steps = _select_key_transition_steps(clean_steps, goal_terms, limit=5)
    for idx, step in enumerate(transition_steps, start=1):
        card = _universal_state_card_from_steps(
            f"action_transition_{idx}",
            "action_transition",
            "action_and_observation_evidence",
            [step],
            goal_terms,
            max_evidence=10,
        )
        if card:
            card["state_summary"] = (
                "Key action transition selected by observable change, target text, "
                "error state, or task-term evidence."
            )
            cards.append(card)

    excluded = {id(step) for step in transition_steps}
    observation_steps = _select_observation_evidence_steps(clean_steps, goal_terms, excluded=excluded, limit=3)
    for idx, step in enumerate(observation_steps, start=1):
        card = _universal_state_card_from_steps(
            f"evidence_observation_{idx}",
            "evidence_observation",
            "supporting_observation_evidence",
            [step],
            goal_terms,
            max_evidence=12,
        )
        if card:
            card["state_summary"] = "Intermediate observation with dense visible evidence for later rubric extraction."
            cards.append(card)

    final_card = _universal_state_card_from_steps(
        "final_observation",
        "final_observation",
        "terminal_state_evidence",
        clean_steps[-1:],
        goal_terms,
        max_evidence=16,
    )
    if final_card:
        final_card["state_summary"] = "Final observable state for later rubric verification."
        final_card["success_prior"] = {
            "outcome": outcome,
            "reward": validation.get("reward"),
            "note": "This is dataset-level supervision, not parser judgment.",
        }
        cards.append(final_card)

    answer_card = _answer_or_output_card(clean_steps, goal_terms)
    if answer_card:
        cards.append(answer_card)

    media_card = _media_reference_card(clean_steps)
    if media_card:
        cards.append(media_card)

    risk_card = _risk_state_card(clean_steps)
    if risk_card:
        cards.append(risk_card)

    return cards[:14]


def _universal_task_card(
    goal_terms: Sequence[str],
    *,
    outcome: str,
    validation: Mapping[str, Any],
) -> Dict[str, Any]:
    evidence = []
    if goal_terms:
        evidence.append("goal_terms: " + ", ".join(map(str, goal_terms[:12])))
    return {
        "state_id": "task_context",
        "card_type": "task_context",
        "stage": "task_state",
        "evidence_role": "task_definition",
        "rubric_role": "task_definition",
        "state_summary": "Task instruction hints and dataset-level outcome supervision.",
        "matched_goal_terms": list(goal_terms[:12]),
        "evidence_lines": evidence,
        "source_steps": [],
        "source_fields": ["task_instruction", "validation"],
        "screenshot_paths": [],
        "facets": {"has_task_terms": bool(goal_terms)},
        "confidence": "candidate",
        "success_prior": {
            "outcome": outcome,
            "reward": validation.get("reward"),
            "note": "This is dataset-level supervision, not parser judgment.",
        },
    }


def _universal_state_card_from_steps(
    state_id: str,
    card_type: str,
    evidence_role: str,
    steps: Sequence[Mapping[str, Any]],
    goal_terms: Sequence[str],
    *,
    max_evidence: int,
) -> Dict[str, Any]:
    evidence: List[str] = []
    source_steps: List[Any] = []
    screenshots: List[str] = []
    matched: List[str] = []
    fields = {"action_signature", "obs_snapshot"}
    for step in steps:
        source_steps.append(step.get("step_index"))
        obs = step.get("obs_snapshot", {})
        if isinstance(obs, Mapping):
            screenshot = str(obs.get("screenshot_path", "")).strip()
            if screenshot:
                screenshots.append(screenshot)
        step_evidence = _step_evidence_lines(step, goal_terms, max_lines=max(4, max_evidence))
        evidence.extend(step_evidence)
        matched.extend(_matched_terms("\n".join(step_evidence), goal_terms))
    evidence = _dedupe_text(evidence)[:max_evidence]
    matched = _dedupe_text(matched)
    if not evidence and card_type != "initial_observation":
        return {}
    return {
        "state_id": state_id,
        "card_type": card_type,
        "stage": card_type,
        "evidence_role": evidence_role,
        "rubric_role": evidence_role,
        "matched_goal_terms": matched,
        "evidence_lines": evidence,
        "source_steps": source_steps,
        "source_fields": sorted(fields),
        "screenshot_paths": _dedupe_text(screenshots),
        "facets": _card_facets(steps),
        "confidence": "candidate",
    }


def _select_key_transition_steps(
    steps: Sequence[Mapping[str, Any]],
    goal_terms: Sequence[str],
    *,
    limit: int,
) -> List[Mapping[str, Any]]:
    scored: List[tuple[int, int, Mapping[str, Any]]] = []
    previous_url = ""
    for idx, step in enumerate(steps[:-1]):
        score = _step_transition_score(step, goal_terms)
        obs = step.get("obs_snapshot", {})
        url = str(obs.get("url", "")).strip() if isinstance(obs, Mapping) else ""
        if url and url != previous_url:
            score += 5
        previous_url = url or previous_url
        if idx < 2:
            score += 1
        if score > 0:
            scored.append((score, idx, step))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [step for _, _, step in scored[:limit]]


def _select_observation_evidence_steps(
    steps: Sequence[Mapping[str, Any]],
    goal_terms: Sequence[str],
    *,
    excluded: set[int],
    limit: int,
) -> List[Mapping[str, Any]]:
    scored: List[tuple[int, int, Mapping[str, Any]]] = []
    for idx, step in enumerate(steps[1:-1], start=1):
        if id(step) in excluded:
            continue
        score = _observation_evidence_score(step, goal_terms)
        if score > 0:
            scored.append((score, idx, step))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [step for _, _, step in scored[:limit]]


def _step_transition_score(step: Mapping[str, Any], goal_terms: Sequence[str]) -> int:
    action = step.get("action_signature", {})
    raw = str(action.get("raw", "")).strip() if isinstance(action, Mapping) else ""
    action_type = str(action.get("action_type", "")).strip().lower() if isinstance(action, Mapping) else ""
    target_text = str(action.get("target_text", "")).strip() if isinstance(action, Mapping) else ""
    score = 0
    if raw and raw.lower() != "none":
        score += 2
    if action_type in {"fill", "select_option", "click", "goto", "press", "send_msg", "send_message"}:
        score += 2
    if target_text:
        score += 4
    score += len(_matched_terms(_step_search_text(step), goal_terms)) * 4
    error = step.get("error_signal", {})
    if isinstance(error, Mapping) and error.get("has_error"):
        score += 8
    return score


def _observation_evidence_score(step: Mapping[str, Any], goal_terms: Sequence[str]) -> int:
    lines = _step_evidence_lines(step, goal_terms, max_lines=20)
    score = min(len(lines), 12)
    joined = "\n".join(lines).lower()
    score += len(_matched_terms(joined, goal_terms)) * 3
    for token in ("table", "grid", "heading", "statictext", "alert", "error", "image", "search result"):
        if token in joined:
            score += 1
    return score


def _answer_or_output_card(steps: Sequence[Mapping[str, Any]], goal_terms: Sequence[str]) -> Dict[str, Any]:
    candidates: List[Mapping[str, Any]] = []
    for step in steps:
        action = step.get("action_signature", {})
        if not isinstance(action, Mapping):
            continue
        action_type = str(action.get("action_type", "")).strip().lower()
        raw = str(action.get("raw", "")).strip()
        target_text = str(action.get("target_text", "")).strip()
        if action_type in {"send_msg", "send_message", "answer", "final_answer"} or raw.lower().startswith(
            ("send_msg", "send_message", "answer", "final_answer")
        ):
            evidence = [f"action: {raw}"] if raw else []
            if target_text:
                evidence.append(f"answer_text: {target_text}")
            return {
                "state_id": "output_or_answer",
                "card_type": "output_or_answer",
                "stage": "output_or_answer",
                "evidence_role": "final_response_evidence",
                "rubric_role": "final_response_evidence",
                "state_summary": "Explicit final response or message action emitted by the agent.",
                "matched_goal_terms": _matched_terms("\n".join(evidence), goal_terms),
                "evidence_lines": evidence,
                "source_steps": [step.get("step_index")],
                "source_fields": ["action_signature"],
                "screenshot_paths": [],
                "facets": {"has_explicit_answer": True},
                "confidence": "candidate",
            }
        if target_text:
            candidates.append(step)
    return {}


def _media_reference_card(steps: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    screenshots: List[str] = []
    image_labels: List[str] = []
    source_steps: List[Any] = []
    for step in steps:
        obs = step.get("obs_snapshot", {})
        if not isinstance(obs, Mapping):
            continue
        screenshot = str(obs.get("screenshot_path", "")).strip()
        cues = []
        for key in ("task_relevant_cues", "clickable_options"):
            value = obs.get(key, [])
            if isinstance(value, list):
                cues.extend(str(item) for item in value if str(item).strip())
        images = [cue for cue in cues if "image" in cue.lower()]
        if screenshot or images:
            source_steps.append(step.get("step_index"))
        if screenshot:
            screenshots.append(screenshot)
        image_labels.extend(images[:4])
    screenshots = _dedupe_text(screenshots)
    image_labels = _dedupe_text(image_labels)
    if not screenshots and not image_labels:
        return {}
    sampled_screenshots = _dedupe_text(screenshots[:2] + screenshots[-2:])
    deduped_source_steps: List[Any] = []
    seen_steps = set()
    for source_step in source_steps:
        key = str(source_step)
        if key in seen_steps:
            continue
        seen_steps.add(key)
        deduped_source_steps.append(source_step)
    return {
        "state_id": "media_references",
        "card_type": "media_reference",
        "stage": "media_reference",
        "evidence_role": "visual_or_media_pointer",
        "rubric_role": "visual_or_media_pointer",
        "state_summary": "Screenshot paths and image labels retained for optional visual inspection.",
        "matched_goal_terms": [],
        "evidence_lines": image_labels[:12],
        "source_steps": deduped_source_steps[:12],
        "source_fields": ["obs_snapshot.screenshot_path", "obs_snapshot.task_relevant_cues"],
        "screenshot_paths": sampled_screenshots,
        "facets": {"has_screenshot": bool(screenshots), "has_image_label": bool(image_labels)},
        "confidence": "candidate",
    }


def _risk_state_card(steps: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    risk_steps = [
        step
        for step in steps
        if isinstance(step.get("error_signal"), Mapping) and step["error_signal"].get("has_error")
    ]
    repeated = _repeated_action_runs(steps)
    if not risk_steps and not repeated:
        return {}
    card = {
        "state_id": "risk_signals",
        "card_type": "risk_or_error",
        "stage": "risk_state",
        "evidence_role": "negative_or_side_effect_evidence",
        "rubric_role": "negative_or_side_effect_evidence",
        "state_summary": "Errors or repeated actions that may matter for failure/side-effect rubrics.",
        "matched_goal_terms": [],
        "evidence_lines": [],
        "source_steps": [],
        "source_fields": ["error_signal", "action_signature"],
        "screenshot_paths": [],
        "facets": {"has_error": bool(risk_steps), "has_repetition": bool(repeated)},
        "confidence": "candidate",
    }
    for step in risk_steps[:8]:
        card["source_steps"].append(step.get("step_index"))
        error = step.get("error_signal", {})
        card["evidence_lines"].append(f"step {step.get('step_index')} error: {error.get('message', '')}")
    for run in repeated[:8]:
        card["evidence_lines"].append(
            f"repeated action {run['action']} from step {run['start_step']} to {run['end_step']} ({run['count']} times)"
        )
    return card


def _card_facets(steps: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    lines: List[str] = []
    for step in steps:
        if isinstance(step, Mapping):
            lines.extend(_step_evidence_lines(step, [], max_lines=30))
    joined = "\n".join(lines).lower()
    return {
        "has_url": "url:" in joined,
        "has_form_control": any(token in joined for token in ("textbox", "combobox", "checkbox", "button")),
        "has_table_or_grid": any(token in joined for token in ("table", "grid", "gridcell", "columnheader")),
        "has_image_reference": "image" in joined,
        "has_error_text": any(token in joined for token in ("error", "failed", "unexpected", "invalid")),
    }


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
    if not isinstance(summary, Mapping) and isinstance(metadata.get("summary_info"), Mapping):
        summary = metadata.get("summary_info", {})
    elif not summary and isinstance(metadata.get("summary_info"), Mapping):
        summary = metadata.get("summary_info", {})
    if outcome == "unknown" and isinstance(summary, Mapping):
        reward = summary.get("cum_reward", summary.get("reward"))
        if isinstance(reward, (int, float)):
            outcome = "success" if reward > 0 else "failure"
    chat_messages = record.get("chat_messages")
    if not task_instruction:
        task_instruction = trim_text(str(record.get("goal", "")), 400)
    steps = _parse_compact_steps(trace_source)
    if chat_messages and not steps:
        steps = _parse_steps_from_chat_messages(chat_messages)
    if not steps:
        steps = _parse_steps_from_text(record)

    validation = _validation_snapshot(summary, outcome, steps)
    rubric_ready = build_rubric_ready_views(
        record_id=record_id,
        task_instruction=task_instruction,
        outcome=outcome,
        steps=steps,
        validation=validation,
        metadata=metadata,
    )

    parsed = TraceParsed(
        __record_id__=record_id,
        task_instruction=task_instruction,
        outcome=outcome,
        steps=[model_validate(CompactStep, step) for step in steps],
        validation=validation,
        metadata=metadata,
        chat_messages=chat_messages,
        runtime_summary=rubric_ready["runtime_summary"],
        audit_trace=rubric_ready["audit_trace"],
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
        if _is_step_event(event):
            if current is not None:
                steps.append(_finalize_step(current))
            current = _compact_step_from_event(event, len(steps))
            if current is None:
                continue
            if _is_observation_event(event):
                _attach_observation(current, event)
            if _is_error_event(event):
                current["error_signal"] = _merge_error_signals(current.get("error_signal", {}), event)
            if pending_observations:
                for observation in pending_observations:
                    _attach_observation(current, observation)
                pending_observations = []
            if pending_errors:
                current["error_signal"] = _merge_error_signals(current.get("error_signal", {}), pending_errors[-1])
                pending_errors = []
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
    metadata.pop("package_version", None)
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
            "package_version",
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
