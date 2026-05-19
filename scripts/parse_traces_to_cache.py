from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rubric_miner.io import read_input_records  # noqa: E402
from rubric_miner.trace import parse_trace_record, stable_record_id  # noqa: E402


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parse raw traces into a rubric-ready cached summary dataset. "
            "This stage is standalone and does not run clustering or rubric extraction."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True, help="Input trace file or AgentRewardBench trajectory root")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/cache_data/rubric_ready_parsed.json"),
        help="Output JSON array path. Ignored when --preserve-tree is set.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/cache_data"),
        help="Output root for --preserve-tree. Input paths under data/ are mirrored below this root.",
    )
    parser.add_argument(
        "--preserve-tree",
        action="store_true",
        help=(
            "Write one parsed file per input JSON while preserving the source directory tree. "
            "For example data/agent-reward-bench/... becomes data/cache_data/agent-reward-bench/..."
        ),
    )
    parser.add_argument(
        "--input-format",
        default=None,
        help="Optional format override: json/jsonl/yaml/csv/agent_reward_bench",
    )
    parser.add_argument("--max-records", type=int, default=None, help="Parse only the first N loaded records")
    parser.add_argument(
        "--include-chat-messages",
        action="store_true",
        help="Keep chat_messages in cached output. They are not needed for runtime_summary by default.",
    )
    parser.add_argument(
        "--agent-reward-observation-chars",
        type=int,
        default=2400,
        help="Max chars kept for each AgentRewardBench observation digest before rubric-ready parsing",
    )
    parser.add_argument(
        "--agent-reward-observation-policy",
        choices=["last", "last_and_errors", "all", "none"],
        default="all",
        help=(
            "Which AgentRewardBench steps keep compressed axtree observations. "
            "Use 'all' for rubric-ready state-card candidate extraction."
        ),
    )
    parser.add_argument(
        "--runtime-only",
        action="store_true",
        help="Write only runtime_summary objects instead of the full parsed+audit records",
    )
    parser.add_argument("--preview-records", type=int, default=3, help="Print a short preview for the first N records")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.preserve_tree:
        return _main_preserve_tree(args)

    records = read_input_records(
        args.input,
        input_format=args.input_format,
        max_records=args.max_records,
        agent_reward_observation_chars=args.agent_reward_observation_chars,
        agent_reward_observation_policy=args.agent_reward_observation_policy,
        parse_include_chat_messages=True,
    )

    parsed_records = []
    for idx, raw in enumerate(records):
        record_id = str(raw.get("__record_id__") or stable_record_id(raw, idx))
        raw["__record_id__"] = record_id
        try:
            parsed = parse_trace_record(raw)
            if not args.include_chat_messages:
                parsed["chat_messages"] = None
            parsed_records.append(parsed.get("runtime_summary", parsed) if args.runtime_only else parsed)
        except Exception as exc:
            parsed_records.append(
                {
                    "__record_id__": record_id,
                    "skipped": True,
                    "parse_error": str(exc),
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(parsed_records, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"loaded_records: {len(records)}")
    print(f"written_records: {len(parsed_records)}")
    print(f"output: {args.output}")
    print(f"runtime_only: {bool(args.runtime_only)}")
    _print_preview(parsed_records, max_records=args.preview_records)
    return 0


def _main_preserve_tree(args: argparse.Namespace) -> int:
    input_files = _iter_input_json_files(args.input)
    if args.max_records is not None:
        input_files = input_files[: args.max_records]

    outputs: List[Mapping[str, Any]] = []
    errors: List[Mapping[str, Any]] = []
    for idx, input_file in enumerate(input_files):
        try:
            parsed_payload = _parse_one_file(input_file, args=args)
            output_file = _mirrored_output_path(input_file, output_root=args.output_root)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with output_file.open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(parsed_payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            outputs.append({"input": str(input_file), "output": str(output_file)})
        except Exception as exc:
            errors.append({"input": str(input_file), "error": str(exc)})

    print(f"input_files: {len(input_files)}")
    print(f"written_files: {len(outputs)}")
    print(f"errors: {len(errors)}")
    print(f"output_root: {args.output_root}")
    if errors:
        print("error_preview:")
        for item in errors[:10]:
            print(f"- {item['input']}: {item['error']}")
    _print_tree_preview(outputs, max_records=args.preview_records)
    return 1 if errors else 0


def _parse_one_file(input_file: Path, *, args: argparse.Namespace) -> Any:
    records = read_input_records(
        input_file,
        input_format=args.input_format or "json",
        max_records=None,
        agent_reward_observation_chars=args.agent_reward_observation_chars,
        agent_reward_observation_policy=args.agent_reward_observation_policy,
        parse_include_chat_messages=True,
    )
    parsed_records: List[Dict[str, Any]] = []
    for idx, raw in enumerate(records):
        record_id = str(raw.get("__record_id__") or stable_record_id(raw, idx))
        raw["__record_id__"] = record_id
        parsed = parse_trace_record(raw)
        if not args.include_chat_messages:
            parsed["chat_messages"] = None
        _attach_cache_source(parsed, input_file)
        parsed_records.append(parsed.get("runtime_summary", parsed) if args.runtime_only else parsed)
    if len(parsed_records) == 1:
        return parsed_records[0]
    return parsed_records


def _attach_cache_source(parsed: Dict[str, Any], input_file: Path) -> None:
    relative = _relative_cache_path(input_file)
    metadata = parsed.setdefault("metadata", {})
    if isinstance(metadata, dict):
        metadata.setdefault("source_path", str(input_file))
        metadata.setdefault("relative_source_path", str(relative).replace("\\", "/"))
    audit = parsed.get("audit_trace")
    if isinstance(audit, dict):
        source_metadata = audit.setdefault("source_metadata", {})
        if isinstance(source_metadata, dict):
            source_metadata.setdefault("source_path", str(input_file))
            source_metadata.setdefault("relative_source_path", str(relative).replace("\\", "/"))
    runtime = parsed.get("runtime_summary")
    if isinstance(runtime, dict):
        runtime.setdefault("source", {})
        if isinstance(runtime["source"], dict):
            runtime["source"].setdefault("relative_source_path", str(relative).replace("\\", "/"))


def _iter_input_json_files(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(path)
    return sorted(item for item in path.rglob("*.json") if item.is_file())


def _mirrored_output_path(input_file: Path, *, output_root: Path) -> Path:
    return output_root / _relative_cache_path(input_file)


def _relative_cache_path(input_file: Path) -> Path:
    resolved = input_file.resolve()
    try:
        return resolved.relative_to((ROOT / "data").resolve())
    except ValueError:
        pass
    try:
        rel = resolved.relative_to(ROOT.resolve())
        if rel.parts and rel.parts[0].lower() == "data":
            return Path(*rel.parts[1:])
        return rel
    except ValueError:
        return Path(input_file.name)


def _print_tree_preview(outputs: Sequence[Mapping[str, Any]], *, max_records: int) -> None:
    if max_records <= 0:
        return
    print("preview:")
    for item in outputs[:max_records]:
        print(f"- {item.get('input')} -> {item.get('output')}")


def _print_preview(records: Sequence[Mapping[str, Any]], *, max_records: int) -> None:
    if max_records <= 0:
        return
    print("preview:")
    for record in records[:max_records]:
        summary = record.get("runtime_summary") if isinstance(record.get("runtime_summary"), Mapping) else record
        if not isinstance(summary, Mapping):
            continue
        cards = summary.get("state_cards", [])
        final_state = summary.get("final_state", {})
        print(
            f"- {summary.get('__record_id__', record.get('__record_id__'))} "
            f"outcome={summary.get('outcome')} "
            f"cards={len(cards) if isinstance(cards, list) else 0}"
        )
        print(f"  task: {_one_line(summary.get('task_instruction', ''), 180)}")
        if isinstance(final_state, Mapping):
            evidence = final_state.get("evidence_lines", [])
            if isinstance(evidence, list) and evidence:
                print(f"  final_evidence: {_one_line(evidence[0], 220)}")


def _one_line(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 16)].rstrip() + " ...[truncated]"


if __name__ == "__main__":
    raise SystemExit(main())
