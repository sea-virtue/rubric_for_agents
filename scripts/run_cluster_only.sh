#!/usr/bin/env bash
set -euo pipefail

# Run only the trace parsing + clustering stages of rubric_miner.
#
# Examples:
#   bash scripts/run_cluster_only.sh
#   bash scripts/run_cluster_only.sh --force --cluster-threshold 0.48
#   bash scripts/run_cluster_only.sh --config configs/local_qwen3_vllm_full.json --force
#   bash scripts/run_cluster_only.sh --no-embeddings --max-input-records 40
#
# Outputs:
#   parsed_traces.json      canonical parsed traces used by clustering
#   clusters.json           one cluster assignment per parsed trace
#   cluster_summary.json    cluster-level statistics and member previews
#   cluster_report.md       human-readable cluster report
#   cluster_only.log.jsonl  structured log for this isolated run

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export RUBRIC_MINER_REPO_ROOT="${REPO_ROOT}"

python - "$@" <<'PY'
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

repo_root = Path(os.environ.get("RUBRIC_MINER_REPO_ROOT", ".")).resolve()
sys.path.insert(0, str(repo_root / "src"))

from rubric_miner.cli import parse_field_map
from rubric_miner.config import load_config
from rubric_miner.llm import build_client
from rubric_miner.logging_utils import configure_logging, console, logger
from rubric_miner.stages import build_groups, cluster_stage, parse_stage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run only rubric_miner trace parsing and clustering.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--input-format", default=None)
    parser.add_argument("--field-map", default=None)
    parser.add_argument("--csv-group-by", default=None)
    parser.add_argument("--max-input-records", type=int, default=None)
    parser.add_argument("--agent-reward-observation-chars", type=int, default=None)
    parser.add_argument(
        "--agent-reward-observation-policy",
        choices=["last", "last_and_errors", "all", "none"],
        default=None,
    )
    parser.add_argument("--agent-reward-sample-per-bucket", type=int, default=None)
    parser.add_argument("--agent-reward-sample-seed", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--cluster-threshold", type=float, default=None)
    parser.add_argument("--cluster-algorithm", choices=["dbscan", "connected"], default=None)
    parser.add_argument(
        "--cluster-partition-metadata-keys",
        default=None,
        help="Comma-separated metadata keys that cannot cross cluster boundaries.",
    )
    parser.add_argument("--min-cluster-size", type=int, default=None)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--embedding-base-url", default=None)
    parser.add_argument("--embedding-instruction", default=None)
    parser.add_argument("--embedding-max-chars", type=int, default=None)
    parser.add_argument("--embedding-batch-size", type=int, default=None)
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Force the local lexical clustering path even if config/env sets an embedding model.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete this run's parsed/clusters/report files before running so threshold changes take effect.",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--preview-records", type=int, default=5)
    parser.add_argument("--preview-chars", type=int, default=900)
    return parser.parse_args()


def apply_overrides(config: Any, args: argparse.Namespace) -> None:
    for key in (
        "input",
        "input_format",
        "csv_group_by",
        "max_input_records",
        "agent_reward_observation_chars",
        "agent_reward_observation_policy",
        "agent_reward_sample_per_bucket",
        "agent_reward_sample_seed",
        "out_dir",
        "cluster_threshold",
        "cluster_algorithm",
        "min_cluster_size",
        "embedding_model",
        "embedding_base_url",
        "embedding_instruction",
        "embedding_max_chars",
        "embedding_batch_size",
        "verbose",
    ):
        value = getattr(args, key, None)
        if value is not None and value is not False:
            setattr(config, key, value)
    if args.no_embeddings:
        config.embedding_model = ""
    if args.cluster_partition_metadata_keys is not None:
        config.cluster_partition_metadata_keys = [
            item.strip()
            for item in args.cluster_partition_metadata_keys.split(",")
            if item.strip()
        ]
    if args.field_map:
        config.field_map = parse_field_map(args.field_map)
    if not config.input:
        default_input = Path("data/agent-reward-bench/trajectories")
        if default_input.exists():
            config.input = default_input
            config.input_format = config.input_format or "agent_reward_bench"
        else:
            raise ValueError("Missing input. Provide --input or --config.")
    if config.input_format is None and str(config.input).replace("\\", "/").endswith("agent-reward-bench/trajectories"):
        config.input_format = "agent_reward_bench"
    if args.out_dir is None and not args.config:
        config.out_dir = Path("outputs/cluster_only")
    if not 0 <= float(config.cluster_threshold) <= 1:
        raise ValueError("cluster_threshold must be in [0, 1]")
    if int(config.min_cluster_size) < 1:
        raise ValueError("min_cluster_size must be >= 1")


def remove_run_files(out_dir: Path) -> None:
    for name in (
        "parsed_traces.json",
        "clusters.json",
        "cluster_summary.json",
        "cluster_report.md",
        "cluster_only.log.jsonl",
    ):
        path = out_dir / name
        if path.exists():
            path.unlink()


def truncate_text(value: Any, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def short_actions(record: Mapping[str, Any], limit: int = 10) -> list[str]:
    compact = record.get("compact_trace")
    if isinstance(compact, Mapping):
        timeline = compact.get("timeline")
        if isinstance(timeline, list):
            actions = []
            for event in timeline:
                if not isinstance(event, Mapping):
                    continue
                action = str(event.get("action") or "").strip()
                if not action:
                    continue
                actions.append(truncate_text(action, 80))
                if len(actions) >= limit:
                    return actions

    sequence = record.get("structured_sequence")
    if not isinstance(sequence, list):
        return []
    actions: list[str] = []
    for event in sequence:
        if not isinstance(event, Mapping):
            continue
        action = str(event.get("action") or "").strip()
        if not action:
            summary = str(event.get("summary") or "")
            match = re.search(r"(?:^|\n)action:\s*(.+)", summary)
            action = match.group(1).strip() if match else ""
        if not action:
            continue
        actions.append(truncate_text(action, 80))
        if len(actions) >= limit:
            break
    return actions


def write_reports(
    *,
    out_dir: Path,
    parsed: Sequence[Mapping[str, Any]],
    clusters: Sequence[Mapping[str, Any]],
    preview_records: int,
    preview_chars: int,
) -> None:
    parsed_by_id = {
        str(record.get("__record_id__")): record
        for record in parsed
        if record.get("__record_id__") is not None and not record.get("skipped")
    }
    groups = build_groups(parsed, clusters)
    cluster_rows: list[dict[str, Any]] = []
    for cluster_id, group in groups.items():
        records = list(group.get("records", []))
        outcomes = Counter(str(record.get("outcome", "unknown")) for record in records)
        benchmarks = Counter(
            str(record.get("metadata", {}).get("benchmark", "unknown"))
            for record in records
        )
        task_families = Counter(
            str(record.get("metadata", {}).get("task_family", "unknown"))
            for record in records
        )
        agents = Counter(
            str(record.get("metadata", {}).get("agent", "unknown"))
            for record in records
        )
        similarities = [
            float(cluster.get("similarity", 0.0))
            for cluster in clusters
            if str(cluster.get("cluster_id")) == str(cluster_id)
        ]
        examples = []
        for record in records[: max(0, preview_records)]:
            meta = record.get("metadata", {}) if isinstance(record.get("metadata"), Mapping) else {}
            examples.append(
                {
                    "record_id": record.get("__record_id__"),
                    "task_id": meta.get("task_id"),
                    "outcome": record.get("outcome"),
                    "task_family": meta.get("task_family"),
                    "source_path": meta.get("relative_source_path") or meta.get("source_path"),
                    "task_preview": truncate_text(record.get("task"), preview_chars),
                    "actions_preview": short_actions(record),
                }
            )
        cluster_rows.append(
            {
                "cluster_id": cluster_id,
                "cluster_key": group.get("cluster_key", ""),
                "size": len(records),
                "avg_similarity": round(sum(similarities) / len(similarities), 4) if similarities else None,
                "outcomes": dict(outcomes),
                "benchmarks": dict(benchmarks),
                "task_families": dict(task_families),
                "agents": dict(agents),
                "source_record_ids": list(group.get("source_record_ids", [])),
                "examples": examples,
            }
        )

    cluster_rows.sort(key=lambda item: (-int(item["size"]), str(item["cluster_id"])))
    singleton_count = sum(1 for item in cluster_rows if item["size"] == 1)
    summary = {
        "num_parsed_records": len(parsed_by_id),
        "num_cluster_assignments": len([item for item in clusters if not item.get("skipped")]),
        "num_clusters": len(cluster_rows),
        "num_singletons": singleton_count,
        "largest_cluster_size": max((int(item["size"]) for item in cluster_rows), default=0),
        "outcomes": dict(Counter(str(record.get("outcome", "unknown")) for record in parsed_by_id.values())),
        "benchmarks": dict(
            Counter(
                str(record.get("metadata", {}).get("benchmark", "unknown"))
                for record in parsed_by_id.values()
            )
        ),
        "clusters": cluster_rows,
    }
    (out_dir / "cluster_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Cluster Report",
        "",
        f"- parsed records: {summary['num_parsed_records']}",
        f"- cluster assignments: {summary['num_cluster_assignments']}",
        f"- clusters: {summary['num_clusters']}",
        f"- singleton clusters: {summary['num_singletons']}",
        f"- largest cluster size: {summary['largest_cluster_size']}",
        f"- outcomes: `{json.dumps(summary['outcomes'], ensure_ascii=False)}`",
        f"- benchmarks: `{json.dumps(summary['benchmarks'], ensure_ascii=False)}`",
        "",
    ]
    for item in cluster_rows:
        lines.extend(
            [
                f"## {item['cluster_id']}",
                "",
                f"- size: {item['size']}",
                f"- key: `{item.get('cluster_key', '')}`",
                f"- avg_similarity: `{item.get('avg_similarity')}`",
                f"- outcomes: `{json.dumps(item['outcomes'], ensure_ascii=False)}`",
                f"- task_families: `{json.dumps(item['task_families'], ensure_ascii=False)}`",
                "",
            ]
        )
        for example in item["examples"]:
            actions = "; ".join(example.get("actions_preview", []))
            lines.extend(
                [
                    f"### {example.get('record_id')}",
                    "",
                    f"- task_id: `{example.get('task_id')}`",
                    f"- outcome: `{example.get('outcome')}`",
                    f"- task_family: `{example.get('task_family')}`",
                    f"- source: `{example.get('source_path')}`",
                    f"- task: {example.get('task_preview')}",
                    f"- actions: {actions or '(none)'}",
                    "",
                ]
            )
    (out_dir / "cluster_report.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


async def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    apply_overrides(config, args)

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.force:
        remove_run_files(out_dir)

    configure_logging(out_dir / "cluster_only.log.jsonl", bool(config.verbose))
    logger.info(
        "cluster_only_start",
        extra={
            "input": str(config.input),
            "input_format": config.input_format,
            "out_dir": str(out_dir),
            "threshold": config.cluster_threshold,
            "algorithm": config.cluster_algorithm,
            "embedding_model": config.embedding_model,
            "partition_metadata_keys": config.cluster_partition_metadata_keys,
        },
    )

    parsed = await parse_stage(
        Path(config.input),
        out_dir / "parsed_traces.json",
        input_format=config.input_format,
        field_map=config.field_map,
        csv_group_by=config.csv_group_by,
        max_records=config.max_input_records,
        agent_reward_observation_chars=config.agent_reward_observation_chars,
        agent_reward_observation_policy=config.agent_reward_observation_policy,
        agent_reward_sample_per_bucket=config.agent_reward_sample_per_bucket,
        agent_reward_sample_seed=config.agent_reward_sample_seed,
    )
    embedding_client = (
        build_client(api_key_env=config.embedding_api_key_env, base_url=config.embedding_base_url)
        if config.embedding_model
        else None
    )
    clusters = await cluster_stage(
        parsed,
        out_dir / "clusters.json",
        float(config.cluster_threshold),
        client=embedding_client,
        embedding_model=config.embedding_model or None,
        embedding_instruction=config.embedding_instruction,
        embedding_max_chars=int(config.embedding_max_chars),
        embedding_batch_size=int(config.embedding_batch_size),
        min_cluster_size=int(config.min_cluster_size),
        algorithm=str(config.cluster_algorithm),
        partition_metadata_keys=config.cluster_partition_metadata_keys,
    )
    write_reports(
        out_dir=out_dir,
        parsed=parsed,
        clusters=clusters,
        preview_records=args.preview_records,
        preview_chars=args.preview_chars,
    )
    logger.info("cluster_only_done", extra={"out_dir": str(out_dir)})
    console.print(f"[bold green]Cluster-only run complete.[/bold green] Outputs written to {out_dir}")
    console.print(f"  - {out_dir / 'parsed_traces.json'}")
    console.print(f"  - {out_dir / 'clusters.json'}")
    console.print(f"  - {out_dir / 'cluster_summary.json'}")
    console.print(f"  - {out_dir / 'cluster_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
PY
