from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pair_rubric_extraction.io import load_json_array, load_pair_records, parse_csv, select_pair_records, upsert, write_json  # noqa: E402
from pair_rubric_extraction.prompting import limit_response_payload, state_cards_from_record  # noqa: E402
from pair_rubric_extraction.io import load_selected_cache_record  # noqa: E402
from rubric_miner.llm import async_llm_call, build_client, extract_json_array  # noqa: E402


DEFAULT_PAIR_RUBRICS = Path("data/pair_rubric/pair_rubrics.json")
DEFAULT_PAIRS = Path("data/cache_pair_data")
DEFAULT_OUTPUT_DIR = Path("data/pair_rubric_eval")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate whether pair-level rubrics rank positive trajectories above negative trajectories.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pair-rubrics", type=Path, default=DEFAULT_PAIR_RUBRICS)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=os.getenv("RUBRIC_EVAL_MODEL", os.getenv("RUBRIC_MODEL", "qwen3-4b-instruct-2507")))
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:28000/v1"))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--pair-ids", default="", help="Comma-separated pair ids to process first/only.")
    parser.add_argument("--max-chars-per-response", type=int, default=80000)
    parser.add_argument(
        "--no-truncate",
        action="store_true",
        help="Send all cleaned state cards for each response. Use only with a sufficiently large-context API model.",
    )
    parser.add_argument(
        "--card-order",
        choices=("priority", "source"),
        default="priority",
        help="State-card ordering for prompt truncation. priority keeps terminal/output cards early; source preserves parser order.",
    )
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--refresh", action="store_true", help="Recompute pairs already present in output.")
    parser.add_argument("--dry-run", action="store_true", help="Print one judge prompt preview without calling a model.")
    parser.add_argument("--preview-chars", type=int, default=5000)
    return parser.parse_args(argv)


async def main_async(args: argparse.Namespace) -> int:
    rubric_records = load_json_array(args.pair_rubrics)
    pair_records = select_pair_records(
        load_pair_records(args.pairs),
        selected_pair_ids=parse_csv(args.pair_ids),
        max_pairs=None,
    )
    selected = select_eval_jobs(rubric_records, pair_records, selected_pair_ids=parse_csv(args.pair_ids), max_pairs=args.max_pairs)
    if not selected:
        raise ValueError("No pair rubric evaluation jobs selected.")

    if args.dry_run:
        print(f"pair_rubrics: {args.pair_rubrics}")
        print(f"pairs: {args.pairs}")
        print(f"selected_pairs: {len(selected)}")
        rubric_record, pair_record = selected[0]
        messages = judge_prompt_messages(
            rubric_record,
            pair_record,
            max_chars_per_response=effective_max_chars_per_response(args),
            card_order=args.card_order,
        )
        print(f"- {rubric_record.get('pair_id')}")
        print(messages[1]["content"][: max(0, args.preview_chars)])
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    score_path = args.output_dir / "pair_rubric_pair_scores.json"
    prompt_path = args.output_dir / "pair_rubric_eval_prompts.json"
    raw_path = args.output_dir / "pair_rubric_eval_raw_outputs.json"
    summary_path = args.output_dir / "pair_rubric_eval_summary.json"
    config_path = args.output_dir / "pair_rubric_eval_config.json"

    existing = load_json_array(score_path)
    prompt_records = load_json_array(prompt_path)
    raw_records = load_json_array(raw_path)
    done = {
        str(record.get("pair_id") or record.get("__record_id__"))
        for record in existing
        if record.get("pair_id") and not record.get("judge_error")
    } if not args.refresh else set()

    client = build_client(api_key_env=args.api_key_env, base_url=args.base_url)
    semaphore = asyncio.Semaphore(max(1, args.concurrency))

    async def process_job(rubric_record: Mapping[str, Any], pair_record: Mapping[str, Any]) -> Dict[str, Any]:
        pair_id = str(rubric_record.get("pair_id") or rubric_record.get("__record_id__"))
        if pair_id in done:
            for record in existing:
                if str(record.get("pair_id") or record.get("__record_id__")) == pair_id:
                    return dict(record)
        async with semaphore:
            try:
                messages = judge_prompt_messages(
                    rubric_record,
                    pair_record,
                    max_chars_per_response=effective_max_chars_per_response(args),
                    card_order=args.card_order,
                )
                prompt_record = {"__record_id__": pair_id, "pair_id": pair_id, "messages": messages}
                upsert(prompt_records, prompt_record, key="pair_id")
                write_json(prompt_path, prompt_records)

                raw_content = await async_llm_call(
                    client,
                    args.model,
                    messages,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
                raw_record = {"__record_id__": pair_id, "pair_id": pair_id, "model": args.model, "raw_output": raw_content}
                upsert(raw_records, raw_record, key="pair_id")
                write_json(raw_path, raw_records)

                raw_items = extract_json_array(raw_content)
                return score_result_record(rubric_record, raw_items, model=args.model)
            except Exception as exc:
                return {
                    "__record_id__": pair_id,
                    "pair_id": pair_id,
                    "domain": rubric_record.get("domain"),
                    "jobname": rubric_record.get("jobname"),
                    "judge_error": str(exc),
                    "rubric_count": len(rubric_record.get("rubrics") or []),
                }

    results = list(existing)
    tasks = [asyncio.create_task(process_job(rubric_record, pair_record)) for rubric_record, pair_record in selected]
    for future in asyncio.as_completed(tasks):
        record = await future
        upsert(results, record, key="pair_id")
        write_json(score_path, results)
        print(
            f"processed: {record.get('pair_id')} "
            f"pred={record.get('predicted_positive_index')} "
            f"correct={record.get('pair_correct')} error={record.get('judge_error', '')}"
        )

    selected_pair_ids = {str(r.get("pair_id") or r.get("__record_id__")) for r, _ in selected}
    selected_results = [record for record in results if str(record.get("pair_id") or record.get("__record_id__")) in selected_pair_ids]
    summary = build_summary(selected_results)
    write_json(summary_path, summary)
    write_json(
        config_path,
        {
            "pair_rubrics": str(args.pair_rubrics),
            "pairs": str(args.pairs),
            "output_dir": str(args.output_dir),
            "model": args.model,
            "base_url": args.base_url,
            "concurrency": args.concurrency,
            "max_pairs": args.max_pairs,
            "pair_ids": parse_csv(args.pair_ids),
            "max_chars_per_response": args.max_chars_per_response,
            "no_truncate": bool(args.no_truncate),
            "card_order": args.card_order,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "num_selected_pairs": len(selected),
        },
    )
    print(f"score_file: {score_path}")
    print(f"summary: {summary_path}")
    return 0


def select_eval_jobs(
    rubric_records: Sequence[Mapping[str, Any]],
    pair_records: Sequence[Mapping[str, Any]],
    *,
    selected_pair_ids: Sequence[str],
    max_pairs: Optional[int],
) -> List[tuple[Dict[str, Any], Dict[str, Any]]]:
    selected = set(selected_pair_ids)
    pair_by_id = {str(pair.get("pair_id") or pair.get("__record_id__")): dict(pair) for pair in pair_records}
    jobs = []
    for rubric_record in rubric_records:
        pair_id = str(rubric_record.get("pair_id") or rubric_record.get("__record_id__") or "")
        if selected and pair_id not in selected:
            continue
        rubrics = rubric_record.get("rubrics")
        if not pair_id or not isinstance(rubrics, list) or not rubrics:
            continue
        pair_record = pair_by_id.get(pair_id)
        if not pair_record:
            continue
        jobs.append((dict(rubric_record), pair_record))
    jobs.sort(key=lambda item: str(item[0].get("pair_id") or item[0].get("__record_id__")))
    if max_pairs is not None:
        jobs = jobs[: max(0, max_pairs)]
    return jobs


def judge_prompt_messages(
    rubric_record: Mapping[str, Any],
    pair_record: Mapping[str, Any],
    *,
    max_chars_per_response: int | None,
    card_order: str,
) -> List[Dict[str, str]]:
    rubrics = []
    for idx, rubric in enumerate(rubric_record.get("rubrics") or [], start=1):
        if not isinstance(rubric, Mapping):
            continue
        rubrics.append(
            {
                "rubric_index": idx,
                "dimension": rubric.get("dimension", ""),
                "criterion": rubric.get("criterion", ""),
                "severity": rubric.get("severity", "medium"),
                "positive_evidence": rubric.get("positive_evidence", []),
                "negative_evidence": rubric.get("negative_evidence", []),
                "verification_guide": rubric.get("verification_guide", {}),
            }
        )
    responses = pair_eval_responses(pair_record, max_chars_per_response=max_chars_per_response, card_order=card_order)
    schema = {
        "response_id": "response_0",
        "rubric_scores": [
            {
                "rubric_index": 1,
                "score": 0.0,
                "evidence": "short evidence from state_cards",
            }
        ],
        "total_score": 0.0,
    }
    return [
        {
            "role": "system",
            "content": (
                "You evaluate two unlabeled agent trajectories against fixed rubrics. "
                "Return only a strict JSON array with one object per response."
            ),
        },
        {
            "role": "user",
            "content": (
                "Score each response against the rubrics using only the provided state_cards. "
                "Do not infer from response order, hidden labels, validation, model identity, or agent identity. "
                "Terminal state and final-answer evidence are primary for task completion. If intermediate evidence "
                "conflicts with the terminal state or final answer, the terminal/final evidence wins. Do not award "
                "full credit for a transient intermediate state when the final state shows no result, an error, or an "
                "incomplete answer. "
                "Use score 1.0 for clearly satisfied, 0.5 for partial/ambiguous, and 0.0 for absent or violated. "
                "Set total_score to the sum of rubric scores.\n\n"
                f"JSON item schema:\n{json.dumps(schema, ensure_ascii=False)}\n\n"
                f"query: {pair_record.get('query', '')}\n\n"
                f"RUBRICS:\n{json.dumps(rubrics, ensure_ascii=False, indent=2)}\n\n"
                f"RESPONSES:\n{json.dumps(responses, ensure_ascii=False, indent=2)}"
            ),
        },
    ]


def pair_eval_responses(pair_record: Mapping[str, Any], *, max_chars_per_response: int | None, card_order: str) -> List[Dict[str, Any]]:
    responses = []
    selected_records = pair_record.get("selected_records", [])
    if isinstance(selected_records, list):
        for idx, selected in enumerate(selected_records[:2]):
            if not isinstance(selected, Mapping):
                continue
            record = load_selected_cache_record(selected)
            response = limit_response_payload(
                {"state_cards": state_cards_from_record(record)},
                max_chars=max_chars_per_response,
                card_order=card_order,
            )
            responses.append({"response_id": f"response_{idx}", **response})
    return responses


def score_result_record(rubric_record: Mapping[str, Any], raw_items: Sequence[Any], *, model: str) -> Dict[str, Any]:
    pair_id = str(rubric_record.get("pair_id") or rubric_record.get("__record_id__"))
    response_scores = normalize_response_scores(raw_items)
    score_by_index = {item["response_index"]: item["total_score"] for item in response_scores}
    positive_score = score_by_index.get(0)
    negative_score = score_by_index.get(1)
    predicted = None
    margin = None
    pair_correct = None
    if positive_score is not None and negative_score is not None:
        margin = positive_score - negative_score
        predicted = 0 if margin > 0 else 1 if margin < 0 else -1
        pair_correct = predicted == 0
    return {
        "__record_id__": pair_id,
        "pair_id": pair_id,
        "domain": rubric_record.get("domain"),
        "jobname": rubric_record.get("jobname"),
        "model": model,
        "rubric_count": len(rubric_record.get("rubrics") or []),
        "expected_positive_index": 0,
        "predicted_positive_index": predicted,
        "positive_score": positive_score,
        "negative_score": negative_score,
        "score_margin": margin,
        "pair_correct": pair_correct,
        "response_scores": response_scores,
    }


def normalize_response_scores(raw_items: Sequence[Any]) -> List[Dict[str, Any]]:
    scores = []
    for idx, item in enumerate(raw_items):
        if not isinstance(item, Mapping):
            continue
        response_id = str(item.get("response_id") or item.get("id") or f"response_{idx}")
        match = re.search(r"(\d+)$", response_id)
        response_index = int(match.group(1)) if match else idx
        rubric_scores = []
        for score_item in item.get("rubric_scores") or item.get("scores") or []:
            if not isinstance(score_item, Mapping):
                continue
            rubric_scores.append(
                {
                    "rubric_index": int(float(score_item.get("rubric_index", len(rubric_scores) + 1))),
                    "score": clamp_score(score_item.get("score")),
                    "evidence": str(score_item.get("evidence", ""))[:800],
                }
            )
        total = item.get("total_score")
        if total in (None, ""):
            total_score = sum(score["score"] for score in rubric_scores)
        else:
            total_score = clamp_score(total, upper=999.0)
        scores.append(
            {
                "response_id": response_id,
                "response_index": response_index,
                "total_score": total_score,
                "rubric_scores": rubric_scores,
            }
        )
    scores.sort(key=lambda item: item["response_index"])
    return scores


def clamp_score(value: Any, *, upper: float = 1.0) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = 0.0
    return max(0.0, min(upper, score))


def build_summary(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    completed = [record for record in records if not record.get("judge_error") and record.get("pair_correct") is not None]
    correct = [record for record in completed if record.get("pair_correct") is True]
    ties = [record for record in completed if record.get("predicted_positive_index") == -1]
    errors = [record for record in records if record.get("judge_error")]
    margins = [float(record.get("score_margin")) for record in completed if record.get("score_margin") is not None]
    return {
        "records": len(records),
        "completed": len(completed),
        "correct": len(correct),
        "ties": len(ties),
        "errors": len(errors),
        "pair_accuracy": round(len(correct) / len(completed), 4) if completed else None,
        "mean_margin": round(sum(margins) / len(margins), 4) if margins else None,
        "error_pair_ids": [record.get("pair_id") for record in errors],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        return asyncio.run(main_async(args))
    except KeyboardInterrupt:
        return 130


def effective_max_chars_per_response(args: argparse.Namespace) -> int | None:
    return None if args.no_truncate else args.max_chars_per_response


if __name__ == "__main__":
    raise SystemExit(main())
