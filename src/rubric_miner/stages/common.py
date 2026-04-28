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
        rubrics.append(model_validate(RubricItem, item))
    return rubrics


def parse_signal_items(raw_items: Sequence[Any]) -> List[DiscriminativeSignal]:
    signals: List[DiscriminativeSignal] = []
    for item in raw_items:
        if not isinstance(item, dict):
            raise ValueError("signal item must be a JSON object")
        signals.append(model_validate(DiscriminativeSignal, item))
    return signals


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
