from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from pydantic import ValidationError

from ..calibration import calibrate_rubric_set
from ..io import atomic_write_json_array
from ..logging_utils import logger
from ..schemas import (
    DiscriminativeSignal,
    ExportCluster,
    RubricItem,
    has_error,
    model_dump,
    model_validate,
)


def export_stage(refined_records: Sequence[Mapping[str, Any]], output_path: Path) -> List[Dict[str, Any]]:
    exported: List[Dict[str, Any]] = []
    for record in refined_records:
        if has_error(record):
            continue
        try:
            export = ExportCluster(
                __record_id__=record["__record_id__"],
                cluster_id=str(record.get("cluster_id", "")),
                generalized_id=str(record.get("generalized_id", record.get("__record_id__", ""))),
                scope=str(record.get("scope", "cluster_specific")),
                cluster_key=str(record.get("cluster_key", "")),
                source_cluster_ids=list(record.get("source_cluster_ids", [])),
                source_record_ids=list(record.get("source_record_ids", [])),
                rubrics=[model_validate(RubricItem, item) for item in record.get("rubrics", [])],
                discriminative_signals=[
                    model_validate(DiscriminativeSignal, item)
                    for item in record.get("discriminative_signals", [])
                ],
                support_summary=dict(record.get("support_summary", {})),
                calibration=calibrate_rubric_set(record),
            )
            exported.append(model_dump(export))
        except (KeyError, TypeError, ValidationError) as exc:
            logger.warning(
                "sample_failed",
                extra={"stage": "export", "record_id": record.get("__record_id__"), "error": str(exc)},
            )
    atomic_write_json_array(output_path, exported)
    logger.info("stage_done", extra={"stage": "export", "total": len(exported), "path": str(output_path)})
    return exported
