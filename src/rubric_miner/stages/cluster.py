from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..io import atomic_write_json_array, good_record_index, load_json_array, upsert
from ..llm import async_embedding_batch_call
from ..logging_utils import logger
from ..schemas import ClusterAssignment, has_error, model_dump
from ..text import cluster_text, cosine_counts, cosine_vectors, sequence_similarity, token_counter, top_keywords
from .common import progress_bar


async def cluster_stage(
    parsed_records: Sequence[Mapping[str, Any]],
    output_path: Path,
    threshold: float,
    client: Any = None,
    embedding_model: Optional[str] = None,
    embedding_instruction: str = "",
    embedding_max_chars: int = 20000,
    embedding_batch_size: int = 16,
    min_cluster_size: int = 2,
    algorithm: str = "dbscan",
    partition_metadata_keys: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    parsed_by_id = {
        str(record.get("__record_id__")): record
        for record in parsed_records
        if record.get("__record_id__") is not None and not has_error(record)
    }
    stage_records = [
        record
        for record in load_json_array(output_path)
        if str(record.get("__record_id__")) in parsed_by_id
    ]
    ok_index = good_record_index(stage_records)
    partition_metadata_keys = list(partition_metadata_keys or [])
    assigned = _existing_assignments(stage_records, parsed_by_id)
    pending_ids = [record_id for record_id in parsed_by_id if record_id not in ok_index]

    logger.info("stage_start", extra={"stage": "cluster", "total": len(parsed_by_id)})
    if embedding_model and client and pending_ids:
        stage_records = await _embedding_cluster_stage(
            parsed_by_id,
            pending_ids,
            stage_records,
            output_path,
            client,
            embedding_model,
            embedding_instruction,
            embedding_max_chars,
            embedding_batch_size,
            threshold,
            min_cluster_size,
            algorithm,
            partition_metadata_keys,
        )
        return stage_records

    with progress_bar() as progress:
        task = progress.add_task("Clustering", total=len(parsed_by_id))
        for record_id, parsed in parsed_by_id.items():
            if record_id in ok_index:
                progress.advance(task)
                continue
            partition_assigned = _assigned_in_partition(assigned, parsed_by_id, parsed, partition_metadata_keys)
            output = _assign_cluster(record_id, parsed, parsed_by_id, partition_assigned, threshold)
            upsert(stage_records, output)
            assigned[str(output.get("cluster_id", ""))].append(record_id)
            atomic_write_json_array(output_path, stage_records)
            progress.advance(task)
    atomic_write_json_array(output_path, stage_records)
    return stage_records


async def _embedding_cluster_stage(
    parsed_by_id: Mapping[str, Mapping[str, Any]],
    pending_ids: Sequence[str],
    stage_records: List[Dict[str, Any]],
    output_path: Path,
    client: Any,
    embedding_model: str,
    embedding_instruction: str,
    embedding_max_chars: int,
    embedding_batch_size: int,
    threshold: float,
    min_cluster_size: int,
    algorithm: str,
    partition_metadata_keys: Sequence[str],
) -> List[Dict[str, Any]]:
    ids = list(parsed_by_id)
    embeddings: Dict[str, List[float]] = {}
    with progress_bar() as progress:
        task = progress.add_task("Embedding traces", total=len(ids))
        batch_size = max(1, embedding_batch_size)
        for start in range(0, len(ids), batch_size):
            batch_ids = ids[start : start + batch_size]
            batch_texts = [
                _embedding_text(parsed_by_id[record_id], embedding_instruction, embedding_max_chars)
                for record_id in batch_ids
            ]
            batch_embeddings = await async_embedding_batch_call(
                client,
                embedding_model,
                batch_texts,
            )
            for record_id, embedding in zip(batch_ids, batch_embeddings):
                embeddings[record_id] = embedding
            progress.advance(task, len(batch_ids))

    label_by_id: Dict[str, int] = {}
    cluster_members_by_id: Dict[str, List[str]] = {}
    cluster_id_by_record: Dict[str, str] = {}

    partitions: Dict[tuple[str, ...], List[str]] = defaultdict(list)
    for record_id in ids:
        partitions[_partition_key(parsed_by_id[record_id], partition_metadata_keys)].append(record_id)

    for partition_key, partition_ids in partitions.items():
        labels = _cluster_labels(partition_ids, parsed_by_id, embeddings, threshold, min_cluster_size, algorithm)
        grouped: Dict[int, List[str]] = defaultdict(list)
        for record_id, label in zip(partition_ids, labels):
            label_by_id[record_id] = label
            grouped[label].append(record_id)

        partition_digest = _partition_digest(partition_key)
        for label, member_ids in grouped.items():
            if label < 0:
                for member_id in member_ids:
                    cluster_id_by_record[member_id] = make_cluster_id(member_id)
                    cluster_members_by_id[member_id] = [member_id]
                continue
            cluster_id = f"cluster_{partition_digest}_{label}"
            for member_id in member_ids:
                cluster_id_by_record[member_id] = cluster_id
                cluster_members_by_id[member_id] = list(member_ids)

    with progress_bar() as progress:
        task = progress.add_task("Assigning embedding clusters", total=len(pending_ids))
        for record_id in pending_ids:
            cluster_members = cluster_members_by_id.get(record_id, [record_id])
            cluster_id = cluster_id_by_record.get(record_id, make_cluster_id(record_id))
            cluster = ClusterAssignment(
                __record_id__=record_id,
                cluster_id=cluster_id,
                cluster_key=top_keywords(cluster_text(parsed_by_id[mid]) for mid in cluster_members),
                similarity=round(_mean_similarity(record_id, cluster_members, parsed_by_id, embeddings), 4),
            )
            output = model_dump(cluster)
            output["cluster_algorithm"] = algorithm
            output["embedding_model"] = embedding_model
            output["embedding_max_chars"] = embedding_max_chars
            output["embedding_batch_size"] = embedding_batch_size
            output["strategy_similarity_weight"] = 0.35
            output["partition_metadata_keys"] = list(partition_metadata_keys)
            output["embedding_cluster_label"] = label_by_id.get(record_id)
            upsert(stage_records, output)
            atomic_write_json_array(output_path, stage_records)
            progress.advance(task)
    return stage_records


def _cluster_labels(
    ids: Sequence[str],
    parsed_by_id: Mapping[str, Mapping[str, Any]],
    embeddings: Mapping[str, Sequence[float]],
    threshold: float,
    min_cluster_size: int,
    algorithm: str,
) -> List[int]:
    distance_matrix = _distance_matrix(ids, parsed_by_id, embeddings)
    if algorithm.lower() == "dbscan":
        try:
            from sklearn.cluster import DBSCAN

            model = DBSCAN(eps=max(0.01, 1.0 - threshold), min_samples=max(1, min_cluster_size), metric="precomputed")
            return [int(label) for label in model.fit_predict(distance_matrix)]
        except Exception as exc:
            logger.warning("cluster_fallback", extra={"reason": str(exc), "fallback": "connected_components"})
    return _connected_component_labels(distance_matrix, max_distance=1.0 - threshold, min_cluster_size=min_cluster_size)


def _distance_matrix(
    ids: Sequence[str],
    parsed_by_id: Mapping[str, Mapping[str, Any]],
    embeddings: Mapping[str, Sequence[float]],
) -> List[List[float]]:
    return [
        [
            0.0 if left == right else 1.0 - _hybrid_similarity(left, right, parsed_by_id, embeddings)
            for right in ids
        ]
        for left in ids
    ]


def _hybrid_similarity(
    left_id: str,
    right_id: str,
    parsed_by_id: Mapping[str, Mapping[str, Any]],
    embeddings: Mapping[str, Sequence[float]],
) -> float:
    left = parsed_by_id[left_id]
    right = parsed_by_id[right_id]
    emb_score = cosine_vectors(embeddings[left_id], embeddings[right_id])
    seq_score = sequence_similarity(_strategy_sequence(left), _strategy_sequence(right))
    tool_score = sequence_similarity(_tool_sequence(left), _tool_sequence(right))
    return max(0.0, min(1.0, 0.65 * emb_score + 0.25 * seq_score + 0.10 * tool_score))


def _connected_component_labels(
    distance_matrix: Sequence[Sequence[float]],
    max_distance: float,
    min_cluster_size: int,
) -> List[int]:
    n = len(distance_matrix)
    labels = [-1] * n
    current = 0
    for start in range(n):
        if labels[start] != -1:
            continue
        stack = [start]
        component = []
        labels[start] = -2
        while stack:
            idx = stack.pop()
            component.append(idx)
            for nxt, distance in enumerate(distance_matrix[idx]):
                if labels[nxt] == -1 and distance <= max_distance:
                    labels[nxt] = -2
                    stack.append(nxt)
        if len(component) >= min_cluster_size:
            for idx in component:
                labels[idx] = current
            current += 1
        else:
            for idx in component:
                labels[idx] = idx + 100000
    return labels


def _mean_similarity(
    record_id: str,
    member_ids: Sequence[str],
    parsed_by_id: Mapping[str, Mapping[str, Any]],
    embeddings: Mapping[str, Sequence[float]],
) -> float:
    scores = [
        _hybrid_similarity(record_id, other_id, parsed_by_id, embeddings)
        for other_id in member_ids
        if other_id != record_id
    ]
    return sum(scores) / len(scores) if scores else 1.0


def _strategy_sequence(record: Mapping[str, Any]) -> List[str]:
    events = record.get("structured_sequence", [])
    if not isinstance(events, list):
        return []
    return [str(event.get("type", "")) for event in events if isinstance(event, dict)]


def _tool_sequence(record: Mapping[str, Any]) -> List[str]:
    features = record.get("features", {})
    if isinstance(features, dict) and isinstance(features.get("tool_names"), list):
        return [str(tool) for tool in features["tool_names"]]
    return []


def _embedding_text(record: Mapping[str, Any], instruction: str, max_chars: int) -> str:
    text = cluster_text(record)[: max(1000, max_chars)]
    if not instruction:
        return text
    return f"Instruct: {instruction}\nQuery: {text}"


def build_groups(
    parsed_records: Sequence[Mapping[str, Any]],
    cluster_records: Sequence[Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    parsed_by_id = {
        str(record.get("__record_id__")): record
        for record in parsed_records
        if record.get("__record_id__") is not None and not has_error(record)
    }
    groups: Dict[str, Dict[str, Any]] = {}
    for cluster in cluster_records:
        if has_error(cluster):
            continue
        record_id = str(cluster.get("__record_id__"))
        if record_id not in parsed_by_id:
            continue
        cluster_id = str(cluster["cluster_id"])
        group = groups.setdefault(
            cluster_id,
            {"cluster_id": cluster_id, "cluster_key": cluster.get("cluster_key", ""), "records": []},
        )
        group["records"].append(parsed_by_id[record_id])
    for group in groups.values():
        group["source_record_ids"] = [str(record["__record_id__"]) for record in group["records"]]
        if not group.get("cluster_key"):
            group["cluster_key"] = top_keywords(cluster_text(record) for record in group["records"])
    return groups


def _existing_assignments(
    stage_records: Sequence[Mapping[str, Any]],
    parsed_by_id: Mapping[str, Mapping[str, Any]],
) -> Dict[str, List[str]]:
    assigned: Dict[str, List[str]] = defaultdict(list)
    for record in stage_records:
        if has_error(record):
            continue
        record_id = str(record.get("__record_id__"))
        cluster_id = str(record.get("cluster_id", ""))
        if record_id in parsed_by_id and cluster_id:
            assigned[cluster_id].append(record_id)
    return assigned


def _assigned_in_partition(
    assigned: Mapping[str, Sequence[str]],
    parsed_by_id: Mapping[str, Mapping[str, Any]],
    parsed: Mapping[str, Any],
    partition_metadata_keys: Sequence[str],
) -> Dict[str, List[str]]:
    if not partition_metadata_keys:
        return {cluster_id: list(member_ids) for cluster_id, member_ids in assigned.items()}
    key = _partition_key(parsed, partition_metadata_keys)
    filtered: Dict[str, List[str]] = {}
    for cluster_id, member_ids in assigned.items():
        kept = [
            member_id
            for member_id in member_ids
            if member_id in parsed_by_id and _partition_key(parsed_by_id[member_id], partition_metadata_keys) == key
        ]
        if kept:
            filtered[cluster_id] = kept
    return filtered


def _partition_key(record: Mapping[str, Any], keys: Sequence[str]) -> tuple[str, ...]:
    metadata = record.get("metadata", {}) if isinstance(record.get("metadata"), Mapping) else {}
    return tuple(str(metadata.get(key, "")) for key in keys)


def _assign_cluster(
    record_id: str,
    parsed: Mapping[str, Any],
    parsed_by_id: Mapping[str, Mapping[str, Any]],
    assigned: Dict[str, List[str]],
    threshold: float,
) -> Dict[str, Any]:
    try:
        text_counter = token_counter(cluster_text(parsed))
        best_cluster, best_score = _best_cluster(text_counter, assigned, parsed_by_id)
        if not best_cluster or best_score < threshold:
            best_cluster = make_cluster_id(record_id)
            assigned[best_cluster] = []
            best_score = 1.0
        cluster = ClusterAssignment(
            __record_id__=record_id,
            cluster_id=best_cluster,
            cluster_key=top_keywords(cluster_text(parsed_by_id[mid]) for mid in assigned[best_cluster]),
            similarity=round(float(best_score), 4),
        )
        return model_dump(cluster)
    except Exception as exc:
        logger.warning("sample_failed", extra={"stage": "cluster", "record_id": record_id, "error": str(exc)})
        return {"__record_id__": record_id, "skipped": True, "cluster_error": str(exc)}


def _best_cluster(
    text_counter: Any,
    assigned: Mapping[str, Sequence[str]],
    parsed_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[str, float]:
    best_cluster = ""
    best_score = 0.0
    for cluster_id, member_ids in assigned.items():
        rep_text = "\n".join(cluster_text(parsed_by_id[mid]) for mid in member_ids[:24])
        score = cosine_counts(text_counter, token_counter(rep_text))
        if score > best_score:
            best_cluster, best_score = cluster_id, score
    return best_cluster, best_score


def make_cluster_id(record_id: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9_.-]+", "_", record_id).strip("_")
    digest = hashlib.sha1(record_id.encode("utf-8")).hexdigest()[:10]
    prefix = clean[:68].strip("_") or "record"
    return f"cluster_{prefix}_{digest}"


def _partition_digest(partition_key: Sequence[str]) -> str:
    raw = "\n".join(map(str, partition_key))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
