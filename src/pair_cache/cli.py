from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pair_cache.builder import build_pair_record, build_summary, group_candidates  # noqa: E402
from pair_cache.loader import load_candidates  # noqa: E402
from pair_cache.paths import DEFAULT_INPUT_ROOT, DEFAULT_OUTPUT_ROOT  # noqa: E402
from pair_cache.writer import write_pair_outputs  # noqa: E402


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build pairwise positive/negative cache data from parsed trajectory caches. "
            "Input layout is expected to be domain/model/job_with_model/job.json; "
            "output layout is domain/job/job_with_model.json plus a pair.json manifest."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument(
        "--include-action-sequence",
        action="store_true",
        help="Include runtime_summary.action_sequence in pair response text.",
    )
    parser.add_argument(
        "--include-steps",
        action="store_true",
        help="Include compact parsed steps when runtime_summary is unavailable.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Read and summarize candidates without writing pair cache files.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    rng = random.Random(args.seed)
    candidates = load_candidates(args.input_root)
    grouped = group_candidates(candidates)

    pair_records: List[Dict[str, Any]] = []
    report_records: List[Dict[str, Any]] = []

    for key in sorted(grouped):
        group_candidates_for_task = sorted(grouped[key], key=lambda item: item["relative_source_path"])
        result = build_pair_record(
            key,
            group_candidates_for_task,
            rng=rng,
            include_action_sequence=args.include_action_sequence,
            include_steps=args.include_steps,
        )
        if result.get("pair"):
            pair_records.append(result["pair"])
        else:
            report_records.append(result["report"])
        if args.max_pairs is not None and len(pair_records) >= args.max_pairs:
            break

    summary = build_summary(
        candidates,
        pair_records,
        report_records,
        input_root=args.input_root,
        output_root=args.output_root,
        seed=args.seed,
        include_action_sequence=args.include_action_sequence,
        include_steps=args.include_steps,
    )

    print(f"input_root: {args.input_root}")
    print(f"candidate_files: {len(candidates)}")
    print(f"task_groups: {len(grouped)}")
    print(f"pairs: {len(pair_records)}")
    print(f"reported_skips: {len(report_records)}")
    print(f"output_root: {args.output_root}")

    if args.dry_run:
        print("dry_run: true")
        print_preview(pair_records, report_records)
        return 0

    write_pair_outputs(
        pair_records,
        report_records,
        summary,
        output_root=args.output_root,
    )
    print(f"pair_index: {args.output_root / 'pair_index.json'}")
    print(f"pair_report: {args.output_root / 'pair_report.json'}")
    print(f"pair_summary: {args.output_root / 'pair_summary.json'}")
    return 0


def print_preview(pair_records: Sequence[Mapping[str, Any]], report_records: Sequence[Mapping[str, Any]]) -> None:
    print("pair_preview:")
    for pair in pair_records[:5]:
        selected = pair.get("selected_records", [])
        print(f"- {pair.get('pair_id')} selected={len(selected) if isinstance(selected, list) else 0}")
    print("report_preview:")
    for item in report_records[:5]:
        print(f"- {item.get('__record_id__')} reason={item.get('reason')} outcomes={item.get('outcomes')}")


if __name__ == "__main__":
    raise SystemExit(main())
