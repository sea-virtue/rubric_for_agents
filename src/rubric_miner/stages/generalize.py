from __future__ import annotations

import asyncio
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from ..io import good_record_index, load_json_array
from ..llm import llm_json_array
from ..logging_utils import logger
from ..prompts import generalize_messages
from ..schemas import GeneralizedRubricSet, RubricItem, has_error, model_dump
from ..text import literal_overlap, semantic_similarity, top_keywords
from .common import parse_rubric_items, run_parallel_stage


async def generalize_stage(
    merged_records: Sequence[Mapping[str, Any]],
    output_path: Path,
    client: Any,
    merge_model: str,
    concurrency: int,
    bucket_threshold: float,
) -> List[Dict[str, Any]]:
    stage_records = load_json_array(output_path)
    ok_index = good_record_index(stage_records)
    buckets = build_generalization_buckets(merged_records, bucket_threshold)
    semaphore = asyncio.Semaphore(concurrency)

    async def process_bucket(bucket: Mapping[str, Any]) -> Dict[str, Any]:
        record_id = str(bucket["__record_id__"])
        if record_id in ok_index:
            return dict(ok_index[record_id])
        if len(bucket["source_cluster_ids"]) == 1:
            return _single_cluster_output(bucket, merge_model)
        async with semaphore:
            try:
                raw = await llm_json_array(client, merge_model, generalize_messages(bucket), temperature=0.1)
                rubrics = parse_rubric_items(raw)
                kept, support = _filter_cross_cluster_support(rubrics, bucket)
                generalized = GeneralizedRubricSet(
                    __record_id__=record_id,
                    generalized_id=record_id,
                    scope="shared" if len(bucket["source_cluster_ids"]) > 1 else "cluster_specific",
                    cluster_key=str(bucket.get("cluster_key", "")),
                    source_cluster_ids=list(bucket["source_cluster_ids"]),
                    source_record_ids=list(bucket["source_record_ids"]),
                    merge_model=merge_model,
                    rubrics=kept,
                    support_summary=support,
                )
                return model_dump(generalized)
            except Exception as exc:
                logger.warning("sample_failed", extra={"stage": "generalize", "record_id": record_id, "error": str(exc)})
                return {
                    "__record_id__": record_id,
                    "generalized_id": record_id,
                    "scope": str(bucket.get("scope", "shared")),
                    "cluster_key": str(bucket.get("cluster_key", "")),
                    "source_cluster_ids": list(bucket.get("source_cluster_ids", [])),
                    "source_record_ids": list(bucket.get("source_record_ids", [])),
                    "skipped": True,
                    "generalize_error": str(exc),
                }

    logger.info("stage_start", extra={"stage": "generalize", "total": len(buckets)})
    return await run_parallel_stage(
        "Cross-cluster aggregation",
        buckets,
        process_bucket,
        stage_records,
        output_path,
    )


def build_generalization_buckets(
    merged_records: Sequence[Mapping[str, Any]],
    threshold: float,
) -> List[Dict[str, Any]]:
    buckets: List[Dict[str, Any]] = []
    for record in merged_records:
        if has_error(record):
            continue
        for rubric in record.get("rubrics", []):
            if not isinstance(rubric, dict):
                continue
            _place_item(buckets, record, rubric, threshold)
    return [_finalize_bucket(bucket) for bucket in buckets]


def _place_item(
    buckets: List[Dict[str, Any]],
    record: Mapping[str, Any],
    rubric: Mapping[str, Any],
    threshold: float,
) -> None:
    item_text = rubric_item_text(rubric)
    best_idx = -1
    best_score = 0.0
    for idx, bucket in enumerate(buckets):
        score = max(semantic_similarity(item_text, bucket["representative_text"]), literal_overlap(item_text, bucket["representative_text"]))
        if score > best_score:
            best_idx, best_score = idx, score
    if best_idx == -1 or best_score < threshold:
        buckets.append(
            {
                "items": [],
                "representative_text": item_text,
                "cluster_rubrics": [],
                "source_cluster_ids": [],
                "source_record_ids": [],
                "cluster_keys": [],
            }
        )
        best_idx = len(buckets) - 1
    bucket = buckets[best_idx]
    cluster_id = str(record.get("cluster_id", record.get("__record_id__", "")))
    bucket["items"].append(rubric)
    bucket["cluster_rubrics"].append(
        {
            "cluster_id": cluster_id,
            "cluster_key": record.get("cluster_key", ""),
            "rubric": rubric,
        }
    )
    if cluster_id not in bucket["source_cluster_ids"]:
        bucket["source_cluster_ids"].append(cluster_id)
    for source_id in record.get("source_record_ids", []):
        if source_id not in bucket["source_record_ids"]:
            bucket["source_record_ids"].append(source_id)
    if record.get("cluster_key"):
        bucket["cluster_keys"].append(str(record["cluster_key"]))


def _finalize_bucket(bucket: Mapping[str, Any]) -> Dict[str, Any]:
    source_cluster_ids = list(bucket["source_cluster_ids"])
    digest = hashlib.sha1(
        json.dumps(source_cluster_ids + [bucket["representative_text"]], sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    return {
        "__record_id__": f"generalized_{digest}",
        "bucket_id": f"generalized_{digest}",
        "scope": "shared" if len(source_cluster_ids) > 1 else "cluster_specific",
        "cluster_key": top_keywords(bucket.get("cluster_keys", [])),
        "source_cluster_ids": source_cluster_ids,
        "source_record_ids": list(bucket["source_record_ids"]),
        "cluster_rubrics": list(bucket["cluster_rubrics"]),
        "items": list(bucket["items"]),
    }


def _single_cluster_output(bucket: Mapping[str, Any], merge_model: str) -> Dict[str, Any]:
    record_id = str(bucket["__record_id__"])
    rubrics = [RubricItem(**item) if isinstance(item, dict) else item for item in bucket.get("items", [])]
    generalized = GeneralizedRubricSet(
        __record_id__=record_id,
        generalized_id=record_id,
        scope="cluster_specific",
        cluster_key=str(bucket.get("cluster_key", "")),
        source_cluster_ids=list(bucket["source_cluster_ids"]),
        source_record_ids=list(bucket["source_record_ids"]),
        merge_model=merge_model,
        rubrics=rubrics,
        support_summary={"mode": "preserved_cluster_specific", "supporting_clusters": len(bucket["source_cluster_ids"])},
    )
    return model_dump(generalized)


def _filter_cross_cluster_support(
    rubrics: Sequence[RubricItem],
    bucket: Mapping[str, Any],
) -> Tuple[List[RubricItem], Dict[str, Any]]:
    by_cluster: Dict[str, List[Mapping[str, Any]]] = {}
    for entry in bucket.get("cluster_rubrics", []):
        by_cluster.setdefault(str(entry.get("cluster_id", "")), []).append(entry.get("rubric", {}))

    kept: List[RubricItem] = []
    details: List[Dict[str, Any]] = []
    min_support = 2 if len(by_cluster) > 1 else 1
    seen = set()
    for rubric in rubrics:
        data = model_dump(rubric)
        supporting = []
        for cluster_id, items in by_cluster.items():
            if any(_is_supported_by(data, item) for item in items):
                supporting.append(cluster_id)
        key = re.sub(r"\s+", " ", f"{rubric.dimension} {rubric.criterion}".lower()).strip()
        keep = len(supporting) >= min_support and key not in seen
        details.append({"criterion": rubric.criterion, "supporting_clusters": supporting, "kept": keep})
        if keep:
            kept.append(rubric)
            seen.add(key)
    return kept, {
        "mode": "cross_cluster_conservative_merge",
        "min_supporting_clusters": min_support,
        "support": details,
    }


def _is_supported_by(candidate: Mapping[str, Any], source: Mapping[str, Any]) -> bool:
    candidate_text = rubric_item_text(candidate)
    source_text = rubric_item_text(source)
    return semantic_similarity(candidate_text, source_text) > 0.8 or literal_overlap(candidate_text, source_text) > 0.7


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
