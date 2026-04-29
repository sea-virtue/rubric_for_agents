from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from ..io import atomic_write_json_array, upsert
from ..logging_utils import console
from ..schemas import DiscriminativeSignal, RubricItem, model_validate


def progress_bar() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    )


def parse_rubric_items(raw_items: Sequence[Any]) -> List[RubricItem]:
    rubrics: List[RubricItem] = []
    for item in raw_items:
        if not isinstance(item, dict):
            raise ValueError("rubric item must be a JSON object")
        normalized = _normalize_rubric_item(item)
        if not normalized.get("criterion"):
            continue
        rubrics.append(model_validate(RubricItem, normalized))
    return rubrics


def parse_signal_items(raw_items: Sequence[Any]) -> List[DiscriminativeSignal]:
    signals: List[DiscriminativeSignal] = []
    for item in raw_items:
        if not isinstance(item, dict):
            raise ValueError("signal item must be a JSON object")
        signals.append(model_validate(DiscriminativeSignal, item))
    return signals


def _normalize_rubric_item(item: Mapping[str, Any]) -> Dict[str, Any]:
    """Accept common rubric field variants from smaller/local models."""

    dimension = _first_text(
        item,
        "dimension",
        "title",
        "category",
        "capability",
        "failure_mode",
        default="trajectory_quality",
    )
    criterion = _first_text(
        item,
        "criterion",
        "description",
        "rubric",
        "rule",
        "requirement",
        "evaluation_criterion",
    )
    positive = _list_field(item, "positive_evidence", "positive", "success_evidence", "what_successful_traces_do")
    negative = _list_field(item, "negative_evidence", "negative", "failure_evidence", "what_failed_traces_do")
    severity = _first_text(item, "severity", "importance", "weight", default="medium").lower()
    if severity not in {"low", "medium", "high"}:
        severity = _severity_from_weight(severity)
    rationale = _first_text(item, "rationale", "reason", "why", default="")
    return {
        "dimension": dimension or "trajectory_quality",
        "criterion": criterion,
        "positive_evidence": positive,
        "negative_evidence": negative,
        "severity": severity,
        "rationale": rationale,
    }


def _first_text(item: Mapping[str, Any], *keys: str, default: str = "") -> str:
    for key in keys:
        value = item.get(key)
        if value is None or value == "":
            continue
        if isinstance(value, (list, dict)):
            return str(value)
        return str(value).strip()
    return default


def _list_field(item: Mapping[str, Any], *keys: str) -> List[str]:
    for key in keys:
        value = item.get(key)
        if value in (None, ""):
            continue
        if isinstance(value, list):
            return [str(part).strip() for part in value if str(part).strip()]
        return [str(value).strip()]
    return []


def _severity_from_weight(value: str) -> str:
    try:
        numeric = float(value)
    except ValueError:
        return "medium"
    if numeric >= 7:
        return "high"
    if numeric <= 3:
        return "low"
    return "medium"


async def run_parallel_stage(
    label: str,
    items: Sequence[Mapping[str, Any]],
    processor: Any,
    stage_records: List[Dict[str, Any]],
    output_path: Path,
) -> List[Dict[str, Any]]:
    with progress_bar() as progress:
        task = progress.add_task(label, total=len(items))
        pending = [asyncio.create_task(processor(item)) for item in items]
        for future in asyncio.as_completed(pending):
            output = await future
            upsert(stage_records, output)
            atomic_write_json_array(output_path, stage_records)
            progress.advance(task)
    return stage_records
