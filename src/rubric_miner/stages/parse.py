from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from ..io import atomic_write_json_array, good_record_index, load_json_array, read_input_records, upsert
from ..logging_utils import logger
from ..trace import parse_trace_record, stable_record_id
from .common import progress_bar


async def parse_stage(
    input_path: Path,
    output_path: Path,
    *,
    input_format: Optional[str] = None,
    field_map: Optional[Mapping[str, str]] = None,
    csv_group_by: Optional[str] = None,
) -> List[Dict[str, Any]]:
    raw_records = read_input_records(
        input_path,
        input_format=input_format,
        field_map=field_map,
        csv_group_by=csv_group_by,
    )
    stage_records = load_json_array(output_path)
    ok_index = good_record_index(stage_records)

    logger.info("stage_start", extra={"stage": "trace_parse", "total": len(raw_records)})
    with progress_bar() as progress:
        task = progress.add_task("Trace parsing", total=len(raw_records))
        for idx, raw in enumerate(raw_records):
            record_id = str(raw.get("__record_id__") or stable_record_id(raw, idx))
            raw["__record_id__"] = record_id
            if record_id in ok_index:
                progress.advance(task)
                continue
            try:
                output = parse_trace_record(raw)
            except Exception as exc:
                output = {
                    "__record_id__": record_id,
                    "skipped": True,
                    "parse_error": str(exc),
                    "raw": raw,
                }
                logger.warning(
                    "sample_failed",
                    extra={"stage": "trace_parse", "record_id": record_id, "error": str(exc)},
                )
            upsert(stage_records, output)
            atomic_write_json_array(output_path, stage_records)
            progress.advance(task)
    return stage_records
