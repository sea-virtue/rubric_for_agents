from __future__ import annotations

import re
from typing import Any, Dict, Mapping, Sequence

from .text import tokenize


SPECIFIC_MARKERS = {
    "tool",
    "error",
    "retry",
    "evidence",
    "verify",
    "state",
    "observation",
    "action",
    "threshold",
    "exact",
    "specific",
}

VAGUE_MARKERS = {"good", "proper", "appropriate", "reasonable", "clear", "better", "有效", "合理", "清晰"}


def calibrate_rubric_set(record: Mapping[str, Any]) -> Dict[str, Any]:
    rubrics = [item for item in record.get("rubrics", []) if isinstance(item, Mapping)]
    item_scores = [calibrate_rubric_item(item) for item in rubrics]
    return {
        "num_rubrics": len(rubrics),
        "specificity": _mean(item["specificity"] for item in item_scores),
        "retrievability": _mean(item["retrievability"] for item in item_scores),
        "task_specificity": _mean(item["task_specificity"] for item in item_scores),
        "item_scores": item_scores,
    }


def calibrate_rubric_item(item: Mapping[str, Any]) -> Dict[str, Any]:
    text = " ".join(
        [
            str(item.get("dimension", "")),
            str(item.get("criterion", "")),
            " ".join(map(str, item.get("positive_evidence", []))),
            " ".join(map(str, item.get("negative_evidence", []))),
            str(item.get("rationale", "")),
        ]
    )
    tokens = tokenize(text)
    marker_hits = len(set(tokens) & SPECIFIC_MARKERS)
    vague_hits = len(set(tokens) & VAGUE_MARKERS)
    evidence_count = len(item.get("positive_evidence", []) or []) + len(item.get("negative_evidence", []) or [])
    has_binary_language = bool(re.search(r"\b(must|should|only if|when|if|without|fails?|success)\b", text, re.I))

    specificity = min(1.0, (marker_hits + evidence_count + int(has_binary_language)) / 6.0)
    retrievability = min(1.0, (evidence_count + len(tokens) / 30.0) / 3.0)
    task_specificity = max(0.0, min(1.0, specificity - 0.12 * vague_hits + min(0.25, len(tokens) / 80.0)))
    return {
        "dimension": str(item.get("dimension", "")),
        "criterion": str(item.get("criterion", "")),
        "specificity": round(specificity, 4),
        "retrievability": round(retrievability, 4),
        "task_specificity": round(task_specificity, 4),
    }


def _mean(values: Sequence[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)
