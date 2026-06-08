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

from .build import build_bank_v1  # noqa: E402
from .eval import evaluate_bank_v1  # noqa: E402


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and evaluate bank_v1 from unmerged pair rubrics with task-embedding soft applicability.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-bank-v1", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    build.add_argument("--pair-rubrics", type=Path, default=Path("outputs/pair/pair_rubric/pair_rubrics.json"))
    build.add_argument("--pair-embedding-cache", type=Path, default=Path("outputs/pair/cluster_pair_v3_looser/pair_task_embeddings.json"))
    build.add_argument("--output-dir", type=Path, default=Path("outputs/pair/bank_v1"))
    build.add_argument("--output-file", type=Path, default=None)
    build.add_argument("--embedding-model", default=os.getenv("TRACE_EMBEDDING_MODEL", "qwen3-embedding-8b"))
    build.add_argument("--embedding-base-url", default=os.getenv("EMBEDDING_BASE_URL", "http://127.0.0.1:8001/v1"))
    build.add_argument("--api-key-env", default="OPENAI_API_KEY")
    build.add_argument("--embedding-batch-size", type=int, default=16)
    build.add_argument("--refresh-embeddings", action="store_true")
    build.add_argument("--dry-run", action="store_true")

    evaluate = subparsers.add_parser("evaluate-bank-v1", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    evaluate.add_argument("--bank", type=Path, default=Path("outputs/pair/bank_v1/rubric_bank_v1.json"))
    evaluate.add_argument("--pairs", type=Path, default=Path("data/cache_pair_data"))
    evaluate.add_argument("--pair-embedding-cache", type=Path, default=Path("outputs/pair/cluster_pair_v3_looser/pair_task_embeddings.json"))
    evaluate.add_argument("--output-dir", type=Path, default=Path("outputs/pair/bank_v1_eval"))
    evaluate.add_argument("--model", default=os.getenv("RUBRIC_EVAL_MODEL", os.getenv("RUBRIC_MODEL", "qwen3-4b-instruct-2507")))
    evaluate.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:28000/v1"))
    evaluate.add_argument("--embedding-model", default=os.getenv("TRACE_EMBEDDING_MODEL", "qwen3-embedding-8b"))
    evaluate.add_argument("--embedding-base-url", default=os.getenv("EMBEDDING_BASE_URL", "http://127.0.0.1:8001/v1"))
    evaluate.add_argument("--api-key-env", default="OPENAI_API_KEY")
    evaluate.add_argument("--concurrency", type=int, default=1)
    evaluate.add_argument("--max-pairs", type=int, default=None)
    evaluate.add_argument("--pair-ids", default="")
    evaluate.add_argument("--softmax-temperature", type=float, default=0.05)
    evaluate.add_argument("--rubric-batch-size", type=int, default=32)
    evaluate.add_argument("--score-top-k", type=int, default=32, help="Score only top-k rubrics by a_r. Use 0 to score all bank rubrics.")
    evaluate.add_argument("--max-chars-per-response", type=int, default=80000)
    evaluate.add_argument("--no-truncate", action="store_true")
    evaluate.add_argument("--card-order", choices=("priority", "source"), default="priority")
    evaluate.add_argument("--max-tokens", type=int, default=4096)
    evaluate.add_argument("--temperature", type=float, default=0.0)
    evaluate.add_argument("--refresh", action="store_true")
    evaluate.add_argument("--dry-run", action="store_true")
    evaluate.add_argument("--preview-chars", type=int, default=5000)
    return parser.parse_args(argv)


async def main_async(args: argparse.Namespace) -> int:
    if args.command == "build-bank-v1":
        return await build_bank_v1(
            pair_rubrics_path=args.pair_rubrics,
            pair_embedding_cache=args.pair_embedding_cache,
            output_dir=args.output_dir,
            output_file=args.output_file,
            embedding_model=args.embedding_model,
            embedding_base_url=args.embedding_base_url,
            api_key_env=args.api_key_env,
            embedding_batch_size=args.embedding_batch_size,
            refresh_embeddings=args.refresh_embeddings,
            dry_run=args.dry_run,
        )
    if args.command == "evaluate-bank-v1":
        return await evaluate_bank_v1(
            bank_path=args.bank,
            pairs_path=args.pairs,
            pair_embedding_cache=args.pair_embedding_cache,
            output_dir=args.output_dir,
            model=args.model,
            base_url=args.base_url,
            embedding_model=args.embedding_model,
            embedding_base_url=args.embedding_base_url,
            api_key_env=args.api_key_env,
            concurrency=args.concurrency,
            max_pairs=args.max_pairs,
            pair_ids=args.pair_ids,
            softmax_temperature=args.softmax_temperature,
            rubric_batch_size=args.rubric_batch_size,
            score_top_k=args.score_top_k,
            max_chars_per_response=None if args.no_truncate else args.max_chars_per_response,
            card_order=args.card_order,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            refresh=args.refresh,
            dry_run=args.dry_run,
            preview_chars=args.preview_chars,
        )
    raise ValueError(f"unsupported command: {args.command}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        return asyncio.run(main_async(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
