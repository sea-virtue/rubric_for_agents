from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Mapping, Sequence


def prompt_messages(
    group: Mapping[str, Any],
    *,
    num_categories: int,
    max_rubrics_per_group: int,
    max_chars_per_rubric: int,
) -> List[Dict[str, str]]:
    schema = {
        "categories": [
            {
                "theme": "concise self-contained evaluation theme",
                "tips": [
                    "specific, observable guidance point that can be applied to future trajectories",
                    "another non-redundant guidance point",
                ],
                "verification_guide": {
                    "what_to_extract": ["facts, final states, answers, side effects, or process signals to inspect"],
                    "checks": ["general checks that use task parameters rather than sample-specific values"],
                    "evidence_pattern": "compact success/failure pattern with placeholders",
                },
                "source_pair_ids": ["pair ids whose rubrics contributed to this category"],
            }
        ],
        "reason": "brief reason for how the source rubrics were merged",
    }
    return [
        {
            "role": "system",
            "content": (
                "You are a professional evaluation criteria aggregation expert. "
                "You merge scattered pair-level rubrics into a compact Theme-Tips rubric set. "
                "Return only a strict JSON object. Do not use markdown or prose outside JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Please aggregate the following pair-level evaluation rubrics into {num_categories} "
                "or fewer structured evaluation rubrics, following the OpenJudge Theme-Tips style.\n\n"
                "Task Requirements:\n"
                "- Rubrics must be fully self-contained so a verifier can apply them without reading source examples.\n"
                "- Each theme should assess an independent trajectory-quality dimension and avoid contradiction with other themes.\n"
                "- Preserve the pair rubrics' ability to distinguish successful trajectories from failed ones.\n"
                "- Merge duplicates and near-duplicates; do not simply summarize each pair separately.\n"
                "- Prefer terminal-state and final-answer checks for task completion unless the task family explicitly evaluates process behavior.\n"
                "- Do not hard-code sample-specific names, dates, request IDs, prices, element IDs, URLs, or row values unless they are reusable task-family concepts.\n"
                "- Use placeholders such as <target_record>, <required_field>, <expected_answer>, <date_range>, <threshold>, <quantity>, or <target_url>.\n"
                "- Tips should be concrete observable checks, not generic advice.\n"
                "- Include source_pair_ids only when the source ids clearly contributed to the merged theme.\n\n"
                f"Output JSON schema:\n{json.dumps(schema, ensure_ascii=False, indent=2)}\n\n"
                f"grouping: {group.get('grouping')}\n"
                f"group_id: {group.get('group_id')}\n"
                f"pair_count: {group.get('pair_count')}\n"
                f"source_rubric_count: {group.get('rubric_count')}\n\n"
                "PAIR_LEVEL_RUBRICS:\n"
                f"{render_group_rubrics(group, max_rubrics=max_rubrics_per_group, max_chars=max_chars_per_rubric)}"
            ),
        },
    ]


def render_group_rubrics(group: Mapping[str, Any], *, max_rubrics: int, max_chars: int) -> str:
    chunks = []
    emitted = 0
    for pair in group.get("pairs", []):
        if not isinstance(pair, Mapping):
            continue
        pair_id = str(pair.get("pair_id") or "")
        query = str(pair.get("query") or "").strip()
        rubrics = pair.get("rubrics", [])
        if not isinstance(rubrics, list):
            continue
        lines = [f"pair_id: {pair_id}", f"query: {trim_text(query, 600)}", "rubrics:"]
        pair_emitted = 0
        for rubric in rubrics:
            if max_rubrics > 0 and emitted >= max_rubrics:
                break
            if not isinstance(rubric, Mapping):
                continue
            lines.append(f"- {trim_text(format_rubric(rubric), max_chars)}")
            emitted += 1
            pair_emitted += 1
        if pair_emitted:
            chunks.append("\n".join(lines))
        if max_rubrics > 0 and emitted >= max_rubrics:
            break
    if max_rubrics > 0 and int(group.get("rubric_count", 0)) > emitted:
        chunks.append(f"...[truncated: rendered {emitted} of {group.get('rubric_count')} rubrics]...")
    return "\n\n---\n\n".join(chunks)


def format_rubric(rubric: Mapping[str, Any]) -> str:
    parts = []
    dimension = str(rubric.get("dimension") or "").strip()
    criterion = str(rubric.get("criterion") or "").strip()
    if dimension:
        parts.append(f"dimension={dimension}")
    if criterion:
        parts.append(f"criterion={criterion}")
    positive = list_text(rubric.get("positive_evidence"))
    negative = list_text(rubric.get("negative_evidence"))
    if positive:
        parts.append("positive_patterns=" + "; ".join(positive[:3]))
    if negative:
        parts.append("negative_patterns=" + "; ".join(negative[:3]))
    guide = rubric.get("verification_guide")
    if isinstance(guide, Mapping):
        checks = list_text(guide.get("checks"))
        what = list_text(guide.get("what_to_extract"))
        pattern = str(guide.get("evidence_pattern") or "").strip()
        if what:
            parts.append("extract=" + "; ".join(what[:3]))
        if checks:
            parts.append("checks=" + "; ".join(checks[:3]))
        if pattern:
            parts.append(f"evidence_pattern={pattern}")
    return " | ".join(part for part in parts if part)


def list_text(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value in (None, ""):
        return []
    return [str(value).strip()]


def trim_text(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if limit <= 0 or len(text) <= limit:
        return text
    return text[: max(0, limit - 20)].rstrip() + " ...[truncated]"

