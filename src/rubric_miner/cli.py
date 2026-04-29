from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Optional, Sequence

from .config import load_config
from .llm import build_client
from .logging_utils import configure_logging, console, logger
from .stages import (
    build_groups,
    cluster_stage,
    export_stage,
    generalize_stage,
    merge_stage,
    mine_stage,
    parse_stage,
    refine_stage,
)


async def run_pipeline(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    apply_cli_overrides(config, args)

    out_dir = config.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(config.log_file or out_dir / "miner.log.jsonl", config.verbose)

    logger.info(
        "pipeline_start",
        extra={
            "input": str(config.input),
            "out_dir": str(out_dir),
            "rubric_models": config.rubric_models,
            "merge_model": config.merge_model,
            "embedding_model": config.embedding_model,
            "base_url": config.base_url,
            "embedding_base_url": config.embedding_base_url,
            "min_model_support": config.min_model_support,
        },
    )

    client = build_client(api_key_env=config.api_key_env, base_url=config.base_url)
    embedding_client = (
        build_client(api_key_env=config.embedding_api_key_env, base_url=config.embedding_base_url)
        if config.embedding_model
        else None
    )
    parsed = await parse_stage(
        config.input,
        out_dir / "parsed_traces.json",
        input_format=config.input_format,
        field_map=config.field_map,
        csv_group_by=config.csv_group_by,
        max_records=config.max_input_records,
        agent_reward_observation_chars=config.agent_reward_observation_chars,
        agent_reward_observation_policy=config.agent_reward_observation_policy,
    )
    clusters = await cluster_stage(
        parsed,
        out_dir / "clusters.json",
        config.cluster_threshold,
        client=embedding_client,
        embedding_model=config.embedding_model or None,
        min_cluster_size=config.min_cluster_size,
        algorithm=config.cluster_algorithm,
    )
    groups = build_groups(parsed, clusters)

    mined = await mine_stage(
        groups,
        out_dir / "mined.json",
        client,
        config.rubric_models,
        config.concurrency,
        config.max_records_per_cluster,
        config.max_chars_per_trace,
    )
    merged = await merge_stage(
        mined,
        out_dir / "merged.json",
        client,
        config.merge_model,
        config.concurrency,
        config.min_model_support,
    )
    generalized = await generalize_stage(
        merged,
        out_dir / "generalized.json",
        client,
        config.merge_model,
        config.concurrency,
        config.generalization_threshold,
    )
    refined = await refine_stage(generalized, groups, out_dir / "refined.json", client, config.merge_model, config.concurrency)
    exported = export_stage(refined, out_dir / "rubrics.json")

    logger.info("pipeline_done", extra={"exported": len(exported), "path": str(out_dir / "rubrics.json")})
    console.print(f"[bold green]Done.[/bold green] Exported {len(exported)} clusters to {out_dir / 'rubrics.json'}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mine coarse-to-fine rubrics from agent trajectories.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=None, help="Per-task JSON/YAML config file")
    parser.add_argument("--input", type=Path, default=None, help="Input .json or .jsonl trace file")
    parser.add_argument(
        "--input-format",
        default=None,
        help="Override input format: json/jsonl/yaml/csv/agent_reward_bench",
    )
    parser.add_argument("--field-map", default=None, help="Comma pairs, e.g. task=question,trace=trajectory")
    parser.add_argument("--csv-group-by", default=None, help="Column used to group multi-row CSV traces")
    parser.add_argument("--max-input-records", type=int, default=None, help="Load only the first N input records")
    parser.add_argument(
        "--agent-reward-observation-chars",
        type=int,
        default=None,
        help="Max chars kept from AgentRewardBench axtree/axtree_pruned per observation",
    )
    parser.add_argument(
        "--agent-reward-observation-policy",
        choices=["last", "last_and_errors", "all", "none"],
        default=None,
        help="Which AgentRewardBench steps keep axtree observations",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible chat base URL")
    parser.add_argument("--embedding-base-url", default=None, help="OpenAI-compatible embedding base URL")
    parser.add_argument("--rubric-models", default=None, help="Comma-separated mining models; overrides config")
    parser.add_argument("--merge-model", default=None, help="Overrides MERGE_MODEL")
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--cluster-threshold", type=float, default=None)
    parser.add_argument("--cluster-algorithm", choices=["dbscan", "connected"], default=None)
    parser.add_argument("--embedding-model", default=None, help="Overrides TRACE_EMBEDDING_MODEL")
    parser.add_argument("--min-cluster-size", type=int, default=None)
    parser.add_argument("--generalization-threshold", type=float, default=None)
    parser.add_argument("--min-model-support", type=int, default=None)
    parser.add_argument("--max-records-per-cluster", type=int, default=None)
    parser.add_argument("--max-chars-per-trace", type=int, default=None)
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        asyncio.run(run_pipeline(args))
    except KeyboardInterrupt:
        logger.warning("pipeline_interrupted")
        return 130
    except Exception as exc:
        logger.exception("pipeline_failed", extra={"error": str(exc)})
        return 1
    return 0


def apply_cli_overrides(config: object, args: argparse.Namespace) -> None:
    for key in (
        "input",
        "input_format",
        "csv_group_by",
        "max_input_records",
        "agent_reward_observation_chars",
        "agent_reward_observation_policy",
        "out_dir",
        "merge_model",
        "concurrency",
        "cluster_threshold",
        "cluster_algorithm",
        "embedding_model",
        "min_cluster_size",
        "generalization_threshold",
        "min_model_support",
        "max_records_per_cluster",
        "max_chars_per_trace",
        "log_file",
        "verbose",
        "base_url",
        "embedding_base_url",
    ):
        value = getattr(args, key, None)
        if value is not None and value is not False:
            setattr(config, key, value)
    if args.rubric_models:
        config.rubric_models = [model.strip() for model in args.rubric_models.split(",") if model.strip()]
    if args.field_map:
        config.field_map = parse_field_map(args.field_map)
    if not config.input:
        raise ValueError("Missing input. Provide --input or set input in the config file.")
    if config.concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    if not 0 <= config.cluster_threshold <= 1:
        raise ValueError("cluster_threshold must be in [0, 1]")
    if not 0 <= config.generalization_threshold <= 1:
        raise ValueError("generalization_threshold must be in [0, 1]")
    if config.min_cluster_size < 1:
        raise ValueError("min_cluster_size must be >= 1")
    if not config.rubric_models:
        raise ValueError("rubric_models must contain at least one model")
    config.min_model_support = max(1, min(config.min_model_support, len(config.rubric_models)))


def parse_field_map(value: str) -> dict[str, str]:
    mapping = {}
    for pair in value.split(","):
        if not pair.strip():
            continue
        if "=" not in pair:
            raise ValueError("--field-map entries must look like canonical=source_column")
        left, right = pair.split("=", 1)
        mapping[left.strip()] = right.strip()
    return mapping
