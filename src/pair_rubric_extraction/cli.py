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

from pair_rubric_extraction.io import load_pair_records, parse_csv, select_pair_records  # noqa: E402
from pair_rubric_extraction.paths import DEFAULT_OUTPUT_DIR, DEFAULT_PAIRS  # noqa: E402
from pair_rubric_extraction.prompting import prompt_messages  # noqa: E402
from pair_rubric_extraction.runner import extract_pair_rubrics  # noqa: E402


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract pair-level rubrics from cleaned positive/negative pair cache records.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS, help="Pair cache root, pair_index.json, or one pair.json.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=os.getenv("RUBRIC_MODEL", "qwen3-4b-instruct-2507"))
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:28000/v1"))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--pair-ids", default="", help="Comma-separated pair ids to process first/only.")
    parser.add_argument("--max-chars-per-response", type=int, default=16000)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--refresh", action="store_true", help="Recompute pairs already present in output.")
    parser.add_argument("--dry-run", action="store_true", help="Load pairs and print prompt previews without calling a model.")
    parser.add_argument("--preview-chars", type=int, default=2200, help="Characters of each dry-run prompt to print.")
    return parser.parse_args(argv)


async def main_async(args: argparse.Namespace) -> int:
    pair_records = select_pair_records(
        load_pair_records(args.pairs),
        selected_pair_ids=parse_csv(args.pair_ids),
        max_pairs=args.max_pairs,
    )
    if not pair_records:
        raise ValueError("No pair records selected for pair-level rubric extraction.")

    if args.dry_run:
        print(f"pairs: {args.pairs}")
        print(f"selected_pairs: {len(pair_records)}")
        for pair in pair_records[:3]:
            messages = prompt_messages(pair, max_chars_per_response=args.max_chars_per_response)
            print(f"- {pair.get('pair_id')}")
            print(messages[1]["content"][: max(0, args.preview_chars)])
            print("---")
        return 0

    return await extract_pair_rubrics(
        pair_records,
        output_dir=args.output_dir,
        pairs_path=args.pairs,
        model=args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        concurrency=args.concurrency,
        max_pairs=args.max_pairs,
        pair_ids=args.pair_ids,
        max_chars_per_response=args.max_chars_per_response,
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
