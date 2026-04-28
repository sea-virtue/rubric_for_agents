from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..io import good_record_index, load_json_array
from ..llm import llm_json_array
from ..logging_utils import logger
from ..prompts import refine_messages
from ..schemas import RefinedCluster, RubricItem, has_error, model_dump, model_validate
from ..text import semantic_similarity
from .common import parse_signal_items, run_parallel_stage


async def refine_stage(
    merged_records: Sequence[Mapping[str, Any]],
    groups: Mapping[str, Mapping[str, Any]],
    output_path: Path,
    client: Any,
    merge_model: str,
    concurrency: int,
) -> List[Dict[str, Any]]:
    stage_records = load_json_array(output_path)
    ok_index = good_record_index(stage_records)
    semaphore = asyncio.Semaphore(concurrency)
    candidates = [record for record in merged_records if not has_error(record)]

    async def process_record(merged: Mapping[str, Any]) -> Dict[str, Any]:
        record_id = str(merged.get("__record_id__", merged.get("generalized_id", merged.get("cluster_id"))))
        if record_id in ok_index:
            return dict(ok_index[record_id])
        success, failure = pick_contrast_pair(_source_records(merged, groups))
        if not success or not failure:
            return _refined_without_pair(merged, record_id)
        async with semaphore:
            try:
                raw = await llm_json_array(
                    client,
                    merge_model,
                    refine_messages(merged, success, failure),
                    temperature=0.15,
                )
                refined = RefinedCluster(
                    __record_id__=record_id,
                    cluster_id=str(merged.get("cluster_id", "")),
                    generalized_id=str(merged.get("generalized_id", record_id)),
                    scope=str(merged.get("scope", "cluster_specific")),
                    cluster_key=str(merged.get("cluster_key", "")),
                    source_cluster_ids=list(merged.get("source_cluster_ids", [])),
                    source_record_ids=list(merged.get("source_record_ids", [])),
                    rubrics=[model_validate(RubricItem, item) for item in merged.get("rubrics", [])],
                    discriminative_signals=parse_signal_items(raw),
                    support_summary=dict(merged.get("support_summary", {})),
                )
                return model_dump(refined)
            except Exception as exc:
                logger.warning("sample_failed", extra={"stage": "refine", "record_id": record_id, "error": str(exc)})
                return {
                    "__record_id__": record_id,
                    "cluster_id": str(merged.get("cluster_id", "")),
                    "generalized_id": str(merged.get("generalized_id", record_id)),
                    "scope": str(merged.get("scope", "cluster_specific")),
                    "cluster_key": str(merged.get("cluster_key", "")),
                    "source_cluster_ids": list(merged.get("source_cluster_ids", [])),
                    "source_record_ids": list(merged.get("source_record_ids", [])),
                    "skipped": True,
                    "refine_error": str(exc),
                }

    logger.info("stage_start", extra={"stage": "refine", "total": len(candidates)})
    return await run_parallel_stage(
        "Contrastive refinement",
        candidates,
        process_record,
        stage_records,
        output_path,
    )


def pick_contrast_pair(
    records: Sequence[Mapping[str, Any]],
) -> Tuple[Optional[Mapping[str, Any]], Optional[Mapping[str, Any]]]:
    successes = [record for record in records if record.get("outcome") == "success"]
    failures = [record for record in records if record.get("outcome") == "failure"]
    if not successes or not failures:
        return None, None
    best_pair = (successes[0], failures[0])
    best_score = -1.0
    for success in successes:
        for failure in failures:
            score = semantic_similarity(str(success.get("task", "")), str(failure.get("task", "")))
            if score > best_score:
                best_pair = (success, failure)
                best_score = score
    return best_pair


def _source_records(
    merged: Mapping[str, Any],
    groups: Mapping[str, Mapping[str, Any]],
) -> List[Mapping[str, Any]]:
    cluster_ids = list(merged.get("source_cluster_ids", []))
    if not cluster_ids and merged.get("cluster_id"):
        cluster_ids = [str(merged["cluster_id"])]
    records: List[Mapping[str, Any]] = []
    for cluster_id in cluster_ids:
        records.extend(groups.get(str(cluster_id), {}).get("records", []))
    return records


def _refined_without_pair(merged: Mapping[str, Any], record_id: str) -> Dict[str, Any]:
    refined = RefinedCluster(
        __record_id__=record_id,
        cluster_id=str(merged.get("cluster_id", "")),
        generalized_id=str(merged.get("generalized_id", record_id)),
        scope=str(merged.get("scope", "cluster_specific")),
        cluster_key=str(merged.get("cluster_key", "")),
        source_cluster_ids=list(merged.get("source_cluster_ids", [])),
        source_record_ids=list(merged.get("source_record_ids", [])),
        rubrics=[model_validate(RubricItem, item) for item in merged.get("rubrics", [])],
        discriminative_signals=[],
        support_summary={
            **dict(merged.get("support_summary", {})),
            "contrast_warning": "missing success/failure pair",
        },
    )
    return model_dump(refined)
