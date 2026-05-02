from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


def main() -> int:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs/local_qwen3_vllm_full")
    path = out_dir / "mining_prompts.json"
    if not path.exists():
        print(f"missing: {path}")
        return 1

    records = _load_array(path)
    if not records:
        print(f"empty: {path}")
        return 0

    total_tokens = [_int(record.get("approx_total_prompt_tokens")) for record in records]
    user_tokens = [_int(record.get("approx_user_prompt_tokens")) for record in records]
    prompt_budget = [_int(record.get("prompt_token_budget")) for record in records]
    sampled = [_int(record.get("sampled_records_in_prompt")) for record in records]
    requested = [_int(record.get("requested_records_in_prompt")) for record in records]
    chars = [_int(record.get("max_chars_per_trace")) for record in records]
    requested_chars = [_int(record.get("requested_max_chars_per_trace")) for record in records]

    print(f"output_dir: {out_dir}")
    print(f"prompts: {len(records)}")
    _print_stats("approx_total_prompt_tokens", total_tokens)
    _print_stats("approx_user_prompt_tokens", user_tokens)
    _print_stats("sampled_records_in_prompt", sampled)
    _print_stats("max_chars_per_trace", chars)

    over_budget = [
        record
        for record in records
        if _int(record.get("prompt_token_budget")) and _int(record.get("approx_total_prompt_tokens")) > _int(record.get("prompt_token_budget"))
    ]
    auto_shrunk = [
        record
        for record in records
        if _int(record.get("sampled_records_in_prompt")) < _int(record.get("requested_records_in_prompt"))
        or _int(record.get("max_chars_per_trace")) < _int(record.get("requested_max_chars_per_trace"))
    ]

    print(f"over_budget: {len(over_budget)}")
    print(f"auto_shrunk: {len(auto_shrunk)}")
    print(f"sampled_record_counts: {Counter(sampled)}")
    print(f"requested_record_counts: {Counter(requested)}")
    print(f"char_budgets: {Counter(chars)}")
    print(f"requested_char_budgets: {Counter(requested_chars)}")
    if prompt_budget:
        print(f"prompt_token_budget: {Counter(prompt_budget)}")

    largest = sorted(records, key=lambda item: _int(item.get("approx_total_prompt_tokens")), reverse=True)[:5]
    print("largest_prompts:")
    for record in largest:
        print(
            "  "
            f"{record.get('cluster_id')} "
            f"tokens={record.get('approx_total_prompt_tokens')} "
            f"sampled={record.get('sampled_records_in_prompt')}/{record.get('requested_records_in_prompt')} "
            f"chars={record.get('max_chars_per_trace')}/{record.get('requested_max_chars_per_trace')}"
        )
    return 0


def _load_array(path: Path) -> list[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array")
    return [item for item in data if isinstance(item, Mapping)]


def _print_stats(name: str, values: Iterable[int]) -> None:
    values = [value for value in values if value > 0]
    if not values:
        print(f"{name}: n=0")
        return
    print(
        f"{name}: "
        f"n={len(values)} "
        f"min={min(values)} "
        f"p50={int(statistics.median(values))} "
        f"mean={int(statistics.mean(values))} "
        f"max={max(values)}"
    )


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
