from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rubric_miner.llm import async_embedding_batch_call, build_client  # noqa: E402


DEFAULT_INPUT_ROOT = Path("data/cache_data/agent-reward-bench/trajectories/cleaned")
DEFAULT_OUTPUT_DIR = Path("data/cluster")
DEFAULT_THRESHOLDS = "0.16,0.20,0.24,0.28,0.32"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cluster parsed trajectory cache files by task instruction embeddings. "
            "This is a standalone clustering module and does not run rubric mining."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--task-field",
        default="task_instruct",
        help=(
            "Preferred task field. The loader also falls back to task_instruction, "
            "task, instruction, and runtime_summary.task_instruction."
        ),
    )
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("TRACE_EMBEDDING_MODEL", "qwen3-embedding-8b"),
        help="OpenAI-compatible embedding model name.",
    )
    parser.add_argument(
        "--embedding-base-url",
        default=os.getenv("EMBEDDING_BASE_URL", "http://127.0.0.1:8001/v1"),
        help="OpenAI-compatible embedding endpoint base URL.",
    )
    parser.add_argument("--embedding-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--max-task-chars", type=int, default=2048)
    parser.add_argument(
        "--embedding-instruction",
        default="",
        help="Optional instruction prefix for embedding models that expect Instruct/Query format.",
    )
    parser.add_argument(
        "--thresholds",
        default=DEFAULT_THRESHOLDS,
        help="Comma-separated cosine-distance thresholds for the hierarchical clustering ensemble.",
    )
    parser.add_argument(
        "--consensus-threshold",
        type=float,
        default=0.60,
        help="Minimum co-clustering fraction used for the final consensus hierarchical clustering.",
    )
    parser.add_argument("--min-cluster-size", type=int, default=2)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        default=None,
        help="Embedding cache path. Defaults to <output-dir>/task_instruction_embeddings.json.",
    )
    parser.add_argument("--refresh-embeddings", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only read cache files and print a preview; do not call the embedding server.",
    )
    return parser.parse_args(argv)


async def main_async(args: argparse.Namespace) -> int:
    records = load_task_records(
        args.input_root,
        task_field=args.task_field,
        max_task_chars=args.max_task_chars,
        max_records=args.max_records,
    )
    if args.dry_run:
        print(f"input_root: {args.input_root}")
        print(f"records_with_task: {len(records)}")
        for record in records[:5]:
            print(f"- {record['record_id']} | {record['task_text'][:160]}")
        return 0
    if not records:
        raise ValueError(f"No cache records with task text found under {args.input_root}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    embedding_cache = args.embedding_cache or args.output_dir / "task_instruction_embeddings.json"
    embeddings = await load_or_create_embeddings(
        records,
        cache_path=embedding_cache,
        model=args.embedding_model,
        base_url=args.embedding_base_url,
        api_key_env=args.embedding_api_key_env,
        batch_size=args.embedding_batch_size,
        instruction=args.embedding_instruction,
        refresh=args.refresh_embeddings,
    )

    thresholds = parse_thresholds(args.thresholds)
    distance_matrix = cosine_distance_matrix(embeddings)
    labels_by_threshold = {
        f"{threshold:.4f}": agglomerative_labels(distance_matrix, threshold)
        for threshold in thresholds
    }
    consensus = consensus_matrix(list(labels_by_threshold.values()))
    consensus_distance = invert_consensus(consensus)
    final_labels = agglomerative_labels(consensus_distance, 1.0 - args.consensus_threshold)
    final_labels = relabel_small_clusters(final_labels, min_cluster_size=args.min_cluster_size)
    cluster_ids = stable_cluster_ids(records, final_labels)

    result_records = build_result_records(
        records,
        cluster_ids=cluster_ids,
        final_labels=final_labels,
        labels_by_threshold=labels_by_threshold,
        consensus=consensus,
    )
    summary_records = build_cluster_summary(result_records)
    config = {
        "input_root": str(args.input_root),
        "output_dir": str(args.output_dir),
        "task_field": args.task_field,
        "embedding_model": args.embedding_model,
        "embedding_base_url": args.embedding_base_url,
        "embedding_batch_size": args.embedding_batch_size,
        "max_task_chars": args.max_task_chars,
        "embedding_instruction": args.embedding_instruction,
        "thresholds": thresholds,
        "consensus_threshold": args.consensus_threshold,
        "min_cluster_size": args.min_cluster_size,
        "num_records": len(records),
        "num_clusters": len({record["cluster_id"] for record in result_records}),
        "embedding_cache": str(embedding_cache),
        "outputs": {
            "clusters": str(args.output_dir / "task_clusters.json"),
            "summary": str(args.output_dir / "task_cluster_summary.json"),
            "config": str(args.output_dir / "task_cluster_config.json"),
        },
    }

    write_json(args.output_dir / "task_clusters.json", result_records)
    write_json(args.output_dir / "task_cluster_summary.json", summary_records)
    write_json(args.output_dir / "task_cluster_config.json", config)

    print(f"records: {len(records)}")
    print(f"clusters: {config['num_clusters']}")
    print(f"cluster_file: {args.output_dir / 'task_clusters.json'}")
    print(f"summary_file: {args.output_dir / 'task_cluster_summary.json'}")
    return 0


def load_task_records(
    input_root: Path,
    *,
    task_field: str,
    max_task_chars: int,
    max_records: Optional[int],
) -> List[Dict[str, Any]]:
    if not input_root.exists():
        raise FileNotFoundError(input_root)
    paths = [input_root] if input_root.is_file() else sorted(input_root.rglob("*.json"))
    records: List[Dict[str, Any]] = []
    for path in paths:
        for item_idx, payload in enumerate(iter_json_payloads(path)):
            task_text = pick_task_text(payload, task_field)
            if not task_text:
                continue
            record_id = str(payload.get("__record_id__") or f"{path.stem}_{item_idx}")
            metadata = payload.get("metadata", {})
            runtime = payload.get("runtime_summary", {})
            records.append(
                {
                    "record_id": record_id,
                    "source_path": str(path),
                    "relative_source_path": relative_to_root(path),
                    "task_text": trim_task(task_text, max_task_chars),
                    "outcome": payload.get("outcome") or runtime.get("outcome", "unknown")
                    if isinstance(runtime, Mapping)
                    else payload.get("outcome", "unknown"),
                    "metadata": metadata if isinstance(metadata, Mapping) else {},
                }
            )
            if max_records is not None and len(records) >= max_records:
                return records
    return records


def iter_json_payloads(path: Path) -> Iterable[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        for item in data:
            if isinstance(item, Mapping):
                yield item
    elif isinstance(data, Mapping):
        yield data


def pick_task_text(payload: Mapping[str, Any], task_field: str) -> str:
    candidates = [
        task_field,
        "task_instruct",
        "task_instruction",
        "task",
        "instruction",
        "goal",
        "prompt",
        "question",
    ]
    for key in candidates:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    runtime = payload.get("runtime_summary")
    if isinstance(runtime, Mapping):
        value = runtime.get("task_instruction") or runtime.get("task_instruct")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


async def load_or_create_embeddings(
    records: Sequence[Mapping[str, Any]],
    *,
    cache_path: Path,
    model: str,
    base_url: str,
    api_key_env: str,
    batch_size: int,
    instruction: str,
    refresh: bool,
) -> List[List[float]]:
    cached = {} if refresh else load_embedding_cache(cache_path, model=model)
    embeddings_by_key: Dict[str, List[float]] = {}
    pending: List[Mapping[str, Any]] = []
    for record in records:
        key = embedding_cache_key(record)
        cached_item = cached.get(key)
        if cached_item and cached_item.get("task_text") == record.get("task_text"):
            embeddings_by_key[key] = list(cached_item["embedding"])
        else:
            pending.append(record)

    if pending:
        client = build_client(api_key_env=api_key_env, base_url=base_url)
        for start in range(0, len(pending), max(1, batch_size)):
            batch = pending[start : start + max(1, batch_size)]
            texts = [embedding_input(str(record["task_text"]), instruction) for record in batch]
            batch_embeddings = await async_embedding_batch_call(client, model, texts)
            for record, embedding in zip(batch, batch_embeddings):
                embeddings_by_key[embedding_cache_key(record)] = embedding
            print(f"embedded: {min(start + len(batch), len(pending))}/{len(pending)} new")

    embeddings = [embeddings_by_key[embedding_cache_key(record)] for record in records]
    cache_items = [
        {
            "record_id": record["record_id"],
            "relative_source_path": record["relative_source_path"],
            "task_text": record["task_text"],
            "embedding": embeddings[idx],
        }
        for idx, record in enumerate(records)
    ]
    write_json(
        cache_path,
        {
            "embedding_model": model,
            "embedding_base_url": base_url,
            "items": cache_items,
        },
    )
    return embeddings


def load_embedding_cache(path: Path, *, model: str) -> Dict[str, Mapping[str, Any]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, Mapping) or data.get("embedding_model") != model:
        return {}
    items = data.get("items", [])
    if not isinstance(items, list):
        return {}
    index = {}
    for item in items:
        if isinstance(item, Mapping) and isinstance(item.get("embedding"), list):
            index[embedding_cache_key(item)] = item
    return index


def cosine_distance_matrix(embeddings: Sequence[Sequence[float]]) -> List[List[float]]:
    normalized = [normalize_vector(vector) for vector in embeddings]
    matrix: List[List[float]] = []
    for i, left in enumerate(normalized):
        row = []
        for j, right in enumerate(normalized):
            if i == j:
                row.append(0.0)
            else:
                similarity = sum(left[k] * right[k] for k in range(min(len(left), len(right))))
                row.append(max(0.0, min(2.0, 1.0 - similarity)))
        matrix.append(row)
    return matrix


def agglomerative_labels(distance_matrix: Sequence[Sequence[float]], threshold: float) -> List[int]:
    n = len(distance_matrix)
    if n == 0:
        return []
    if n == 1:
        return [0]
    try:
        from sklearn.cluster import AgglomerativeClustering

        try:
            model = AgglomerativeClustering(
                n_clusters=None,
                metric="precomputed",
                linkage="average",
                distance_threshold=threshold,
            )
        except TypeError:  # sklearn < 1.2
            model = AgglomerativeClustering(
                n_clusters=None,
                affinity="precomputed",
                linkage="average",
                distance_threshold=threshold,
            )
        return [int(label) for label in model.fit_predict(distance_matrix)]
    except ImportError as exc:
        if n > 500:
            raise RuntimeError(
                "scikit-learn is required for clustering more than 500 records. "
                "Install it on the Linux server with: pip install scikit-learn"
            ) from exc
        return pure_python_agglomerative(distance_matrix, threshold)


def pure_python_agglomerative(distance_matrix: Sequence[Sequence[float]], threshold: float) -> List[int]:
    clusters = [[idx] for idx in range(len(distance_matrix))]
    while True:
        best_pair = (-1, -1)
        best_distance = float("inf")
        for left_idx in range(len(clusters)):
            for right_idx in range(left_idx + 1, len(clusters)):
                distance = average_cluster_distance(clusters[left_idx], clusters[right_idx], distance_matrix)
                if distance < best_distance:
                    best_distance = distance
                    best_pair = (left_idx, right_idx)
        if best_pair == (-1, -1) or best_distance > threshold:
            break
        left_idx, right_idx = best_pair
        clusters[left_idx].extend(clusters[right_idx])
        del clusters[right_idx]

    labels = [-1] * len(distance_matrix)
    for label, members in enumerate(clusters):
        for member in members:
            labels[member] = label
    return labels


def consensus_matrix(labels_by_run: Sequence[Sequence[int]]) -> List[List[float]]:
    if not labels_by_run:
        return []
    n = len(labels_by_run[0])
    counts = [[0.0 for _ in range(n)] for _ in range(n)]
    for labels in labels_by_run:
        members_by_label: Dict[int, List[int]] = defaultdict(list)
        for idx, label in enumerate(labels):
            members_by_label[int(label)].append(idx)
        for members in members_by_label.values():
            for left in members:
                for right in members:
                    counts[left][right] += 1.0
    denom = float(len(labels_by_run))
    return [[value / denom for value in row] for row in counts]


def invert_consensus(consensus: Sequence[Sequence[float]]) -> List[List[float]]:
    return [
        [0.0 if i == j else 1.0 - float(value) for j, value in enumerate(row)]
        for i, row in enumerate(consensus)
    ]


def relabel_small_clusters(labels: Sequence[int], *, min_cluster_size: int) -> List[int]:
    members_by_label: Dict[int, List[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        members_by_label[int(label)].append(idx)
    next_label = max(members_by_label, default=-1) + 1
    output = list(map(int, labels))
    for members in members_by_label.values():
        if len(members) >= min_cluster_size:
            continue
        for member in members:
            output[member] = next_label
            next_label += 1
    return output


def stable_cluster_ids(records: Sequence[Mapping[str, Any]], labels: Sequence[int]) -> List[str]:
    members_by_label: Dict[int, List[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        members_by_label[int(label)].append(idx)
    ordered = sorted(
        members_by_label.items(),
        key=lambda item: (-len(item[1]), str(records[min(item[1])]["relative_source_path"])),
    )
    label_to_cluster_id = {
        label: f"task_cluster_{rank:04d}"
        for rank, (label, _members) in enumerate(ordered)
    }
    return [label_to_cluster_id[int(label)] for label in labels]


def build_result_records(
    records: Sequence[Mapping[str, Any]],
    *,
    cluster_ids: Sequence[str],
    final_labels: Sequence[int],
    labels_by_threshold: Mapping[str, Sequence[int]],
    consensus: Sequence[Sequence[float]],
) -> List[Dict[str, Any]]:
    members_by_cluster: Dict[str, List[int]] = defaultdict(list)
    for idx, cluster_id in enumerate(cluster_ids):
        members_by_cluster[cluster_id].append(idx)

    results = []
    for idx, record in enumerate(records):
        cluster_id = cluster_ids[idx]
        members = members_by_cluster[cluster_id]
        results.append(
            {
                "__record_id__": record["record_id"],
                "cluster_id": cluster_id,
                "cluster_size": len(members),
                "task_instruction": record["task_text"],
                "source_path": record["source_path"],
                "relative_source_path": record["relative_source_path"],
                "outcome": record.get("outcome", "unknown"),
                "metadata": compact_metadata(record.get("metadata", {})),
                "consensus_strength": round(mean_pair_consensus(idx, members, consensus), 4),
                "hierarchical_labels": {
                    threshold: int(labels[idx])
                    for threshold, labels in labels_by_threshold.items()
                },
                "final_label": int(final_labels[idx]),
            }
        )
    return sorted(results, key=lambda item: (item["cluster_id"], item["relative_source_path"]))


def build_cluster_summary(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    members_by_cluster: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        members_by_cluster[str(record["cluster_id"])].append(record)
    summary = []
    for cluster_id, members in members_by_cluster.items():
        outcomes = defaultdict(int)
        benchmarks = defaultdict(int)
        task_families = defaultdict(int)
        agents = defaultdict(int)
        for member in members:
            outcomes[str(member.get("outcome", "unknown"))] += 1
            metadata = member.get("metadata", {})
            if isinstance(metadata, Mapping):
                benchmarks[str(metadata.get("benchmark", ""))] += 1
                task_families[str(metadata.get("task_family", ""))] += 1
                agents[str(metadata.get("agent", ""))] += 1
        summary.append(
            {
                "cluster_id": cluster_id,
                "cluster_size": len(members),
                "mean_consensus_strength": round(
                    sum(float(member.get("consensus_strength", 0.0)) for member in members) / len(members),
                    4,
                ),
                "outcomes": clean_counter(outcomes),
                "benchmarks": clean_counter(benchmarks),
                "task_families": clean_counter(task_families),
                "agents": clean_counter(agents),
                "sample_tasks": [member["task_instruction"] for member in members[:8]],
                "member_record_ids": [member["__record_id__"] for member in members],
            }
        )
    return sorted(summary, key=lambda item: (-item["cluster_size"], item["cluster_id"]))


def embedding_input(task_text: str, instruction: str) -> str:
    task_text = task_text.strip()
    if not instruction:
        return task_text
    return f"Instruct: {instruction.strip()}\nQuery: {task_text}"


def embedding_cache_key(record: Mapping[str, Any]) -> str:
    return f"{record.get('record_id')}::{record.get('relative_source_path')}"


def average_cluster_distance(
    left: Sequence[int],
    right: Sequence[int],
    distance_matrix: Sequence[Sequence[float]],
) -> float:
    total = 0.0
    count = 0
    for left_idx in left:
        for right_idx in right:
            total += float(distance_matrix[left_idx][right_idx])
            count += 1
    return total / max(1, count)


def normalize_vector(vector: Sequence[float]) -> List[float]:
    norm = math.sqrt(sum(float(value) * float(value) for value in vector))
    if not norm:
        return [0.0 for _ in vector]
    return [float(value) / norm for value in vector]


def mean_pair_consensus(idx: int, members: Sequence[int], consensus: Sequence[Sequence[float]]) -> float:
    others = [member for member in members if member != idx]
    if not others:
        return 1.0
    return sum(float(consensus[idx][member]) for member in others) / len(others)


def parse_thresholds(value: str) -> List[float]:
    thresholds = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not thresholds:
        raise ValueError("--thresholds must contain at least one value")
    for threshold in thresholds:
        if threshold < 0 or threshold > 2:
            raise ValueError("cosine-distance thresholds must be in [0, 2]")
    return sorted(set(thresholds))


def trim_task(text: str, limit: int) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 16)].rstrip() + " ...[truncated]"


def compact_metadata(metadata: Any) -> Dict[str, Any]:
    if not isinstance(metadata, Mapping):
        return {}
    keep = (
        "dataset",
        "benchmark",
        "agent",
        "model",
        "experiment",
        "task_id",
        "task_family",
        "relative_source_path",
    )
    return {key: metadata.get(key) for key in keep if metadata.get(key) not in (None, "")}


def clean_counter(counter: Mapping[str, int]) -> Dict[str, int]:
    return {
        key: int(value)
        for key, value in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
        if key
    }


def relative_to_root(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(tmp_path, path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        return asyncio.run(main_async(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
