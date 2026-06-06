from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Optional, Sequence


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rubric_merge.io import build_cluster_groups, build_domain_groups, load_json_array, load_pair_contexts, parse_csv  # noqa: E402
from rubric_merge.paths import DEFAULT_OUTPUT_DIR, DEFAULT_PAIR_RUBRICS, DEFAULT_PAIRS  # noqa: E402
from rubric_merge.prompting import prompt_messages  # noqa: E402
from rubric_merge.runner import merge_group_rubrics  # noqa: E402


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge pair-level rubrics into broader Theme-Tips rubric groups.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pair-rubrics", type=Path, default=DEFAULT_PAIR_RUBRICS)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS, help="Pair cache root, pair_index.json, or one pair.json.")
    parser.add_argument("--clusters", type=Path, default=Path("data/cluster/task_clusters.json"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--grouping", choices=("domain", "cluster"), default="domain")
    parser.add_argument("--group-ids", default="", help="Comma-separated domain or cluster ids to process first/only.")
    parser.add_argument("--min-pairs", type=int, default=1)
    parser.add_argument("--max-groups", type=int, default=None)
    parser.add_argument("--num-categories", type=int, default=8, help="Maximum merged Theme-Tips categories per group.")
    parser.add_argument(
        "--selection-method",
        choices=("mcr", "none"),
        default="mcr",
        help="Select representative source rubrics before LLM categorization.",
    )
    parser.add_argument("--embedding-model", default=os.getenv("RUBRIC_EMBEDDING_MODEL", "qwen3-embedding-8b"))
    parser.add_argument(
        "--embedding-base-url",
        default=os.getenv("OPENAI_EMBEDDING_BASE_URL", os.getenv("EMBEDDING_BASE_URL", "http://127.0.0.1:8001/v1")),
    )
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--max-selected-rubrics", type=int, default=80)
    parser.add_argument("--mcr-batch-size", type=int, default=5)
    parser.add_argument("--mcr-eps", type=float, default=0.1)
    parser.add_argument("--mcr-min-increment-threshold", type=float, default=0.001)
    parser.add_argument("--mcr-patience", type=int, default=3)
    parser.add_argument("--max-rubrics-per-group", type=int, default=180, help="Prompt cap for source pair-level rubric items; <=0 means no cap.")
    parser.add_argument("--max-chars-per-rubric", type=int, default=1200)
    parser.add_argument("--model", default=os.getenv("RUBRIC_MODEL", "gpt-5.4-mini"))
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.gpt.ge/v1/"))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--refresh", action="store_true", help="Recompute groups already present in output.")
    parser.add_argument("--dry-run", action="store_true", help="Load groups and print prompt previews without calling a model.")
    parser.add_argument("--preview-chars", type=int, default=2600)
    return parser.parse_args(argv)


async def main_async(args: argparse.Namespace) -> int:
    pair_rubrics = load_json_array(args.pair_rubrics)
    pair_contexts = load_pair_contexts(args.pairs)
    if args.grouping == "cluster":
        groups = build_cluster_groups(
            load_json_array(args.clusters),
            pair_rubrics,
            pair_contexts,
            selected_group_ids=parse_csv(args.group_ids),
            min_pairs=args.min_pairs,
            max_groups=args.max_groups,
        )
    else:
        groups = build_domain_groups(
            pair_rubrics,
            pair_contexts,
            selected_group_ids=parse_csv(args.group_ids),
            min_pairs=args.min_pairs,
            max_groups=args.max_groups,
        )
    if not groups:
        raise ValueError("No groups selected for rubric merge.")

    if args.dry_run:
        print(f"pair_rubrics: {args.pair_rubrics}")
        print(f"pairs: {args.pairs}")
        if args.grouping == "cluster":
            print(f"clusters: {args.clusters}")
        print(f"grouping: {args.grouping}")
        print(f"selection_method: {args.selection_method} (not executed in dry-run)")
        if args.selection_method == "mcr":
            print(f"embedding_model: {args.embedding_model}")
            print(f"embedding_base_url: {args.embedding_base_url}")
        print(f"selected_groups: {len(groups)}")
        for group in groups[:3]:
            print(
                f"- {group.get('group_id')} pairs={group.get('pair_count')} "
                f"rubrics={group.get('rubric_count')} missing_pair_rubrics={len(group.get('missing_rubric_pair_ids', []))}"
            )
            messages = prompt_messages(
                group,
                num_categories=args.num_categories,
                max_rubrics_per_group=args.max_rubrics_per_group,
                max_chars_per_rubric=args.max_chars_per_rubric,
            )
            print(messages[1]["content"][: max(0, args.preview_chars)])
            print("---")
        return 0

    return await merge_group_rubrics(
        groups,
        output_dir=args.output_dir,
        pair_rubrics_path=args.pair_rubrics,
        pairs_path=args.pairs,
        model=args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        concurrency=args.concurrency,
        grouping=args.grouping,
        group_ids=args.group_ids,
        max_groups=args.max_groups,
        min_pairs=args.min_pairs,
        num_categories=args.num_categories,
        selection_method=args.selection_method,
        embedding_model=args.embedding_model,
        embedding_base_url=args.embedding_base_url,
        embedding_batch_size=args.embedding_batch_size,
        max_selected_rubrics=args.max_selected_rubrics,
        mcr_batch_size=args.mcr_batch_size,
        mcr_eps=args.mcr_eps,
        mcr_min_increment_threshold=args.mcr_min_increment_threshold,
        mcr_patience=args.mcr_patience,
        max_rubrics_per_group=args.max_rubrics_per_group,
        max_chars_per_rubric=args.max_chars_per_rubric,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        refresh=args.refresh,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        return asyncio.run(main_async(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
