from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rubric_miner.io import read_input_records  # noqa: E402
from rubric_miner.trace import parse_trace_record, stable_record_id  # noqa: E402


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run only the rubric_miner parse stage and write parsed JSON for debugging.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input trace file or AgentRewardBench trajectory root",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/debug_parse/parsed_traces.json"),
        help="Output JSON file for parsed records",
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
        help="Keep chat_messages in parsed output and downstream preview text",
    )
    parser.add_argument(
        "--agent-reward-observation-chars",
        type=int,
        default=1200,
        help="Max chars kept for each AgentRewardBench observation digest",
    )
    parser.add_argument(
        "--agent-reward-observation-policy",
        choices=["last", "last_and_errors", "all", "none"],
        default="last",
        help="Which AgentRewardBench steps keep axtree observations",
    )
    parser.add_argument("--preview-records", type=int, default=3, help="Print a short summary for the first N records")
    parser.add_argument(
        "--preview-steps",
        type=int,
        default=5,
        help="Print a short summary for the first N steps per previewed record",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
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
            parsed_records.append(parsed)
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
    print(f"parsed_records: {len(parsed_records)}")
    print(f"output: {args.output}")
    print(f"include_chat_messages: {bool(args.include_chat_messages)}")
    _print_preview(parsed_records, max_records=args.preview_records, max_steps=args.preview_steps)
    return 0


def _print_preview(records: Sequence[Mapping[str, Any]], *, max_records: int, max_steps: int) -> None:
    if max_records <= 0:
        return
    print("preview:")
    for record in records[:max_records]:
        steps = record.get("steps", [])
        metadata = record.get("metadata", {}) if isinstance(record.get("metadata"), Mapping) else {}
        print(
            f"- {record.get('__record_id__')} "
            f"outcome={record.get('outcome')} "
            f"steps={len(steps) if isinstance(steps, list) else 0} "
            f"benchmark={metadata.get('benchmark', '')} "
            f"chat_messages={'yes' if record.get('chat_messages') else 'no'}"
        )
        print(f"  task: {_one_line(record.get('task_instruction', ''), 180)}")
        if isinstance(steps, list):
            for step in steps[:max_steps]:
                if not isinstance(step, Mapping):
                    continue
                action = step.get("action_signature", {})
                action_text = ""
                if isinstance(action, Mapping):
                    action_text = " ".join(
                        str(action.get(key, "")).strip()
                        for key in ("action_type", "target_bid", "target_text")
                        if str(action.get(key, "")).strip()
                    )
                thought = _one_line(step.get("thought_process", ""), 180)
                print(f"  step {step.get('step_index')}: action={action_text or '<empty>'}")
                if thought:
                    print(f"    thought: {thought}")


def _one_line(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 16)].rstrip() + " ...[truncated]"


if __name__ == "__main__":
    raise SystemExit(main())
