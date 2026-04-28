from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from ..io import good_record_index, load_json_array
from ..llm import llm_json_array
from ..logging_utils import logger
from ..prompts import merge_messages
from ..schemas import MergedCluster, RubricItem, has_error, model_dump
from ..text import literal_overlap, semantic_similarity
from .common import parse_rubric_items, run_parallel_stage


async def merge_stage(
    mined_records: Sequence[Mapping[str, Any]],
    output_path: Path,
    client: Any,
    merge_model: str,
    concurrency: int,
    min_model_support: int,
) -> List[Dict[str, Any]]:
    stage_records = load_json_array(output_path)
    ok_index = good_record_index(stage_records)
    semaphore = asyncio.Semaphore(concurrency)
    candidates = [record for record in mined_records if not has_error(record)]

    async def process_record(mined: Mapping[str, Any]) -> Dict[str, Any]:
        cluster_id = str(mined["cluster_id"])
        if cluster_id in ok_index:
            return dict(ok_index[cluster_id])
        async with semaphore:
            try:
                raw = await llm_json_array(client, merge_model, merge_messages(mined), temperature=0.1)
                rubrics_by_model = normalize_rubrics_by_model(mined)
                kept, support = filter_consensus(
                    parse_rubric_items(raw),
                    rubrics_by_model,
                    min_model_support,
                )
                merged = MergedCluster(
                    __record_id__=cluster_id,
                    cluster_id=cluster_id,
                    cluster_key=str(mined.get("cluster_key", "")),
                    source_record_ids=list(mined.get("source_record_ids", [])),
                    merge_model=merge_model,
                    rubrics=kept,
                    support_summary={
                        **support,
                        "min_model_support": min_model_support,
                        "mining_models": list(rubrics_by_model),
                    },
                )
                return model_dump(merged)
            except Exception as exc:
                logger.warning("sample_failed", extra={"stage": "merge", "record_id": cluster_id, "error": str(exc)})
                return {
                    "__record_id__": cluster_id,
                    "cluster_id": cluster_id,
                    "cluster_key": str(mined.get("cluster_key", "")),
                    "source_record_ids": list(mined.get("source_record_ids", [])),
                    "skipped": True,
                    "merge_error": str(exc),
                }

    logger.info("stage_start", extra={"stage": "merge", "total": len(candidates)})
    return await run_parallel_stage(
        "Multi-model aggregation",
        candidates,
        process_record,
        stage_records,
        output_path,
    )


def filter_consensus(
    merged: Sequence[RubricItem],
    rubrics_by_model: Mapping[str, Sequence[Mapping[str, Any]]],
    min_model_support: int,
) -> Tuple[List[RubricItem], Dict[str, Any]]:
    kept: List[RubricItem] = []
    details: List[Dict[str, Any]] = []
    seen: set = set()
    for item in merged:
        data = model_dump(item)
        model_support = {}
        for model, rubrics in rubrics_by_model.items():
            ok, semantic, overlap = supported_by(rubrics, data)
            model_support[model] = {
                "semantic": round(semantic, 4),
                "literal_overlap": round(overlap, 4),
                "supported": ok,
            }
        support_count = sum(1 for value in model_support.values() if value["supported"])
        key = re.sub(r"\s+", " ", f"{item.dimension} {item.criterion}".lower()).strip()
        keep = bool(support_count >= min_model_support and key not in seen)
        details.append(
            {
                "criterion": item.criterion,
                "support_count": support_count,
                "model_support": model_support,
                "kept": keep,
            }
        )
        if keep:
            kept.append(item)
            seen.add(key)
    return kept, {"filter": details, "thresholds": {"semantic": 0.8, "literal_overlap": 0.7}}


def normalize_rubrics_by_model(mined: Mapping[str, Any]) -> Dict[str, Sequence[Mapping[str, Any]]]:
    raw = mined.get("rubrics_by_model")
    if isinstance(raw, dict):
        return {
            str(model): rubrics
            for model, rubrics in raw.items()
            if isinstance(rubrics, list)
        }
    legacy = {}
    if isinstance(mined.get("rubrics_a"), list):
        legacy[str(mined.get("model_a", "model_a"))] = mined["rubrics_a"]
    if isinstance(mined.get("rubrics_b"), list):
        legacy[str(mined.get("model_b", "model_b"))] = mined["rubrics_b"]
    return legacy


def supported_by(items: Sequence[Mapping[str, Any]], candidate: Mapping[str, Any]) -> Tuple[bool, float, float]:
    candidate_text = rubric_item_text(candidate)
    best_semantic = 0.0
    best_overlap = 0.0
    for item in items:
        item_text = rubric_item_text(item)
        best_semantic = max(best_semantic, semantic_similarity(candidate_text, item_text))
        best_overlap = max(best_overlap, literal_overlap(candidate_text, item_text))
    return best_semantic > 0.8 or best_overlap > 0.7, best_semantic, best_overlap


def rubric_item_text(item: Mapping[str, Any]) -> str:
    evidence = " ".join(map(str, item.get("positive_evidence", []) + item.get("negative_evidence", [])))
    return " ".join(
        [
            str(item.get("dimension", "")),
            str(item.get("criterion", "")),
            evidence,
            str(item.get("rationale", "")),
        ]
    )
