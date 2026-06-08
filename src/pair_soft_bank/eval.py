from __future__ import annotations

import asyncio
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from pair_rubric_extraction.io import load_pair_records
from rubric_miner.llm import async_embedding_batch_call, async_llm_call, build_client, extract_json_array

from .io import load_embedding_cache, load_json_array, parse_csv, upsert, write_json
from .math_utils import cosine_distance, cosine_similarity, softmax_from_distances, top_weight_records
from .prompting import satisfaction_prompt_messages


async def evaluate_bank_v1(
    *,
    bank_path: Path,
    pairs_path: Path,
    pair_embedding_cache: Path,
    output_dir: Path,
    model: str,
    base_url: str,
    embedding_model: str,
    embedding_base_url: str,
    api_key_env: str,
    concurrency: int,
    max_pairs: Optional[int],
    pair_ids: str,
    softmax_temperature: float,
    rubric_batch_size: int,
    score_top_k: int,
    max_chars_per_response: int | None,
    card_order: str,
    max_tokens: int,
    temperature: float,
    refresh: bool,
    dry_run: bool,
    preview_chars: int,
) -> int:
    bank = load_json_array(bank_path)
    if not bank:
        raise ValueError(f"No bank records found in {bank_path}")
    pairs = select_pair_jobs(load_pair_records(pairs_path), selected_pair_ids=parse_csv(pair_ids), max_pairs=max_pairs)
    if not pairs:
        raise ValueError(f"No pair jobs selected from {pairs_path}")

    embedding_cache = load_embedding_cache(pair_embedding_cache)
    pending_embeddings = [pair_task_payload(pair) for pair in pairs if pair_task_payload(pair)["pair_id"] not in embedding_cache]
    if pending_embeddings and dry_run:
        print(f"missing_eval_task_embeddings: {len(pending_embeddings)} (dry-run will not call embedding server)")
    elif pending_embeddings:
        embed_client = build_client(api_key_env=api_key_env, base_url=embedding_base_url)
        new_embeddings = await async_embedding_batch_call(embed_client, embedding_model, [item["task_text"] for item in pending_embeddings])
        for item, embedding in zip(pending_embeddings, new_embeddings):
            embedding_cache[item["pair_id"]] = {"record_id": item["pair_id"], "task_text": item["task_text"], "embedding": embedding}

    if dry_run:
        pair = pairs[0]
        task = pair_task_payload(pair)
        weights = compute_applicability_weights(task, embedding_cache, bank, softmax_temperature=softmax_temperature)
        selected = select_scored_bank_items(bank, weights, score_top_k=score_top_k)
        print(f"bank: {bank_path} rubrics={len(bank)}")
        print(f"pairs: {pairs_path} selected={len(pairs)}")
        print(f"softmax_temperature: {softmax_temperature}")
        print(f"score_top_k: {score_top_k or 'all'}")
        print(f"first_pair: {task['pair_id']}")
        for item in top_weight_records(weights, limit=8):
            print(
                f"weight={item['applicability']:.6f} dist={item['cosine_distance']:.4f} "
                f"source={item['source_pair_id']} dim={item.get('dimension', '')[:80]}"
            )
        messages = satisfaction_prompt_messages(pair, selected[: max(1, min(len(selected), rubric_batch_size))], max_chars_per_response=max_chars_per_response, card_order=card_order)
        print(messages[1]["content"][: max(0, preview_chars)])
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    score_path = output_dir / "soft_pair_scores.json"
    weight_path = output_dir / "applicability_weights.json"
    satisfaction_path = output_dir / "satisfaction_scores.json"
    prompt_path = output_dir / "judge_prompts.json"
    raw_path = output_dir / "judge_raw_outputs.json"
    summary_path = output_dir / "summary.json"
    config_path = output_dir / "config.json"

    score_records = [] if refresh else load_json_array(score_path)
    weight_records = [] if refresh else load_json_array(weight_path)
    satisfaction_records = [] if refresh else load_json_array(satisfaction_path)
    prompt_records = [] if refresh else load_json_array(prompt_path)
    raw_records = [] if refresh else load_json_array(raw_path)
    done = {
        str(record.get("pair_id"))
        for record in score_records
        if record.get("pair_id") and not record.get("judge_error")
    } if not refresh else set()

    judge_client = build_client(api_key_env=api_key_env, base_url=base_url)
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def process_pair(pair: Mapping[str, Any]) -> Dict[str, Any]:
        task = pair_task_payload(pair)
        pair_id = task["pair_id"]
        if pair_id in done:
            for record in score_records:
                if str(record.get("pair_id")) == pair_id:
                    return dict(record)
        async with semaphore:
            try:
                weights = compute_applicability_weights(task, embedding_cache, bank, softmax_temperature=softmax_temperature)
                for item in weights:
                    upsert(weight_records, item, key="__record_id__")
                write_json(weight_path, weight_records)

                selected_bank = select_scored_bank_items(bank, weights, score_top_k=score_top_k)
                response_scores_by_index: Dict[int, Dict[str, Any]] = {
                    0: {"response_id": "response_0", "response_index": 0, "rubric_scores": []},
                    1: {"response_id": "response_1", "response_index": 1, "rubric_scores": []},
                }
                selected_by_id = {str(item.get("rubric_id") or item.get("__record_id__")): item for item in selected_bank}
                for chunk_index, chunk in enumerate(chunks(selected_bank, max(1, rubric_batch_size))):
                    messages = satisfaction_prompt_messages(pair, chunk, max_chars_per_response=max_chars_per_response, card_order=card_order)
                    prompt_id = f"{pair_id}::chunk_{chunk_index:04d}"
                    upsert(prompt_records, {"__record_id__": prompt_id, "pair_id": pair_id, "chunk_index": chunk_index, "rubric_ids": [item.get("rubric_id") for item in chunk], "messages": messages}, key="__record_id__")
                    write_json(prompt_path, prompt_records)

                    raw_content = await async_llm_call(judge_client, model, messages, temperature=temperature, max_tokens=max_tokens)
                    upsert(raw_records, {"__record_id__": prompt_id, "pair_id": pair_id, "chunk_index": chunk_index, "model": model, "raw_output": raw_content}, key="__record_id__")
                    write_json(raw_path, raw_records)

                    raw_items = extract_json_array(raw_content)
                    normalized = normalize_chunk_response_scores(raw_items, chunk)
                    for response in normalized:
                        response_index = int(response["response_index"])
                        response_scores_by_index.setdefault(response_index, {"response_id": response["response_id"], "response_index": response_index, "rubric_scores": []})
                        response_scores_by_index[response_index]["rubric_scores"].extend(response["rubric_scores"])
                        for score_item in response["rubric_scores"]:
                            upsert(
                                satisfaction_records,
                                {
                                    "__record_id__": f"{pair_id}::{response['response_id']}::{score_item['rubric_id']}",
                                    "pair_id": pair_id,
                                    "response_id": response["response_id"],
                                    "response_index": response_index,
                                    **score_item,
                                },
                                key="__record_id__",
                            )
                    write_json(satisfaction_path, satisfaction_records)

                response_scores = []
                weight_by_rubric = {str(item["rubric_id"]): float(item["applicability"]) for item in weights}
                for response_index in sorted(response_scores_by_index):
                    response = response_scores_by_index[response_index]
                    total_score = weighted_response_score(response["rubric_scores"], weight_by_rubric)
                    response_scores.append({**response, "total_score": total_score})
                return pair_score_record(pair_id, response_scores, bank_count=len(bank), scored_count=len(selected_bank), model=model)
            except Exception as exc:
                return {
                    "__record_id__": pair_id,
                    "pair_id": pair_id,
                    "judge_error": str(exc),
                    "bank_rubric_count": len(bank),
                }

    pending = [pair for pair in pairs if str(pair.get("pair_id") or pair.get("__record_id__")) not in done]
    for future in asyncio.as_completed([asyncio.create_task(process_pair(pair)) for pair in pending]):
        record = await future
        upsert(score_records, record, key="pair_id")
        write_json(score_path, score_records)
        print(
            f"processed: {record.get('pair_id')} "
            f"correct={record.get('pair_correct')} margin={record.get('score_margin')} "
            f"error={str(record.get('judge_error', ''))[:160]}"
        )

    selected_ids = {str(pair.get("pair_id") or pair.get("__record_id__")) for pair in pairs}
    selected_scores = [record for record in score_records if str(record.get("pair_id")) in selected_ids]
    summary = {
        "pairwise": pairwise_summary(selected_scores),
        "applicability": applicability_summary([record for record in weight_records if str(record.get("pair_id")) in selected_ids]),
        "config": {
            "bank": str(bank_path),
            "pairs": str(pairs_path),
            "pair_embedding_cache": str(pair_embedding_cache),
            "output_dir": str(output_dir),
            "model": model,
            "base_url": base_url,
            "embedding_model": embedding_model,
            "embedding_base_url": embedding_base_url,
            "softmax_temperature": softmax_temperature,
            "rubric_batch_size": rubric_batch_size,
            "score_top_k": score_top_k,
            "max_pairs": max_pairs,
            "pair_ids": parse_csv(pair_ids),
            "selected_pairs": len(pairs),
            "bank_rubrics": len(bank),
        },
    }
    write_json(summary_path, summary)
    write_json(config_path, summary["config"])
    print(f"soft_pair_scores: {score_path}")
    print(f"summary: {summary_path}")
    return 0


def compute_applicability_weights(
    task: Mapping[str, Any],
    embedding_cache: Mapping[str, Mapping[str, Any]],
    bank: Sequence[Mapping[str, Any]],
    *,
    softmax_temperature: float,
) -> List[Dict[str, Any]]:
    pair_id = str(task["pair_id"])
    task_embedding = embedding_cache.get(pair_id, {}).get("embedding")
    if not isinstance(task_embedding, list):
        raise ValueError(f"missing task embedding for pair_id={pair_id}")
    distances = []
    similarities = []
    for item in bank:
        label = item.get("label", {})
        label_embedding = label.get("embedding") if isinstance(label, Mapping) else None
        if not isinstance(label_embedding, list):
            distances.append(2.0)
            similarities.append(-1.0)
            continue
        distances.append(cosine_distance(task_embedding, label_embedding))
        similarities.append(cosine_similarity(task_embedding, label_embedding))
    weights = softmax_from_distances(distances, temperature=softmax_temperature)
    output = []
    for item, distance, similarity, weight in zip(bank, distances, similarities, weights):
        rubric = item.get("rubric", {}) if isinstance(item.get("rubric"), Mapping) else {}
        output.append(
            {
                "__record_id__": f"{pair_id}::{item.get('rubric_id') or item.get('__record_id__')}",
                "pair_id": pair_id,
                "task_text": task.get("task_text", ""),
                "rubric_id": item.get("rubric_id") or item.get("__record_id__"),
                "source_pair_id": item.get("source_pair_id"),
                "dimension": rubric.get("dimension", ""),
                "cosine_similarity": round(similarity, 8),
                "cosine_distance": round(distance, 8),
                "applicability": weight,
                "softmax_temperature": softmax_temperature,
            }
        )
    return output


def select_scored_bank_items(bank: Sequence[Mapping[str, Any]], weights: Sequence[Mapping[str, Any]], *, score_top_k: int) -> List[Dict[str, Any]]:
    by_id = {str(item.get("rubric_id") or item.get("__record_id__")): dict(item) for item in bank}
    ordered_weights = sorted(weights, key=lambda item: float(item.get("applicability", 0.0)), reverse=True)
    if score_top_k > 0:
        ordered_weights = ordered_weights[:score_top_k]
    selected = []
    for weight in ordered_weights:
        rubric_id = str(weight.get("rubric_id"))
        item = by_id.get(rubric_id)
        if item:
            item["applicability"] = float(weight.get("applicability", 0.0))
            item["cosine_distance"] = float(weight.get("cosine_distance", 2.0))
            selected.append(item)
    return selected


def normalize_chunk_response_scores(raw_items: Sequence[Any], chunk: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    outputs = []
    for response_idx, raw in enumerate(raw_items):
        if not isinstance(raw, Mapping):
            continue
        response_id = str(raw.get("response_id") or raw.get("id") or f"response_{response_idx}")
        match = re.search(r"(\d+)$", response_id)
        response_index = int(match.group(1)) if match else response_idx
        scores = []
        raw_scores = raw.get("rubric_scores") or raw.get("scores") or []
        for idx, item in enumerate(raw_scores if isinstance(raw_scores, list) else [], start=1):
            if not isinstance(item, Mapping):
                continue
            try:
                rubric_index = int(float(item.get("rubric_index", idx)))
            except (TypeError, ValueError):
                rubric_index = idx
            bank_item = chunk[rubric_index - 1] if 0 <= rubric_index - 1 < len(chunk) else {}
            scores.append(
                {
                    "rubric_id": bank_item.get("rubric_id") or bank_item.get("__record_id__"),
                    "source_pair_id": bank_item.get("source_pair_id"),
                    "rubric_index": rubric_index,
                    "score": clamp_score(item.get("score", item.get("rating", item.get("value", 0.0)))),
                    "evidence": str(item.get("evidence", ""))[:800],
                    "rationale": str(item.get("rationale", item.get("reason", "")))[:800],
                }
            )
        outputs.append({"response_id": response_id, "response_index": response_index, "rubric_scores": scores})
    return outputs


def weighted_response_score(rubric_scores: Sequence[Mapping[str, Any]], weight_by_rubric: Mapping[str, float]) -> float:
    total = 0.0
    weight_total = 0.0
    for item in rubric_scores:
        rubric_id = str(item.get("rubric_id"))
        weight = float(weight_by_rubric.get(rubric_id, 0.0))
        total += weight * float(item.get("score", 0.0))
        weight_total += weight
    return round(total / weight_total, 4) if weight_total else 0.0


def pair_score_record(pair_id: str, response_scores: Sequence[Mapping[str, Any]], *, bank_count: int, scored_count: int, model: str) -> Dict[str, Any]:
    score_by_index = {int(item.get("response_index", idx)): float(item.get("total_score", 0.0)) for idx, item in enumerate(response_scores)}
    positive_score = score_by_index.get(0)
    negative_score = score_by_index.get(1)
    margin = None
    predicted = None
    pair_correct = None
    if positive_score is not None and negative_score is not None:
        margin = round(positive_score - negative_score, 4)
        predicted = 0 if margin > 0 else 1 if margin < 0 else -1
        pair_correct = predicted == 0
    return {
        "__record_id__": pair_id,
        "pair_id": pair_id,
        "model": model,
        "positive_score": positive_score,
        "negative_score": negative_score,
        "score_margin": margin,
        "expected_positive_index": 0,
        "predicted_positive_index": predicted,
        "pair_correct": pair_correct,
        "bank_rubric_count": bank_count,
        "scored_rubric_count": scored_count,
        "response_scores": list(response_scores),
    }


def pairwise_summary(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    completed = [record for record in records if not record.get("judge_error") and record.get("pair_correct") is not None]
    correct = [record for record in completed if record.get("pair_correct") is True]
    ties = [record for record in completed if record.get("predicted_positive_index") == -1]
    margins = [float(record.get("score_margin")) for record in completed if record.get("score_margin") is not None]
    return {
        "records": len(records),
        "completed": len(completed),
        "correct": len(correct),
        "ties": len(ties),
        "errors": sum(1 for record in records if record.get("judge_error")),
        "pair_accuracy": round(len(correct) / len(completed), 4) if completed else None,
        "mean_margin": round(sum(margins) / len(margins), 4) if margins else None,
    }


def applicability_summary(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_pair: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_pair[str(record.get("pair_id"))].append(record)
    top_self = 0
    for pair_id, items in by_pair.items():
        top = max(items, key=lambda item: float(item.get("applicability", 0.0)), default=None)
        if top and str(top.get("source_pair_id")) == pair_id:
            top_self += 1
    max_weights = [max(float(item.get("applicability", 0.0)) for item in items) for items in by_pair.values() if items]
    return {
        "pairs": len(by_pair),
        "weight_records": len(records),
        "top1_source_pair_matches_eval_pair": top_self,
        "top1_self_match_rate": round(top_self / len(by_pair), 4) if by_pair else None,
        "mean_top1_weight": round(sum(max_weights) / len(max_weights), 6) if max_weights else None,
    }


def select_pair_jobs(records: Sequence[Mapping[str, Any]], *, selected_pair_ids: Sequence[str], max_pairs: Optional[int]) -> List[Dict[str, Any]]:
    selected = set(selected_pair_ids)
    output = []
    for record in records:
        pair_id = str(record.get("pair_id") or record.get("__record_id__") or "")
        if not pair_id:
            continue
        if selected and pair_id not in selected:
            continue
        if not pair_task_payload(record)["task_text"]:
            continue
        selected_records = record.get("selected_records")
        responses = record.get("responses")
        if not (isinstance(selected_records, list) and len(selected_records) >= 2) and not (isinstance(responses, list) and len(responses) >= 2):
            continue
        output.append(dict(record))
    output.sort(key=lambda item: str(item.get("pair_id") or item.get("__record_id__")))
    if max_pairs is not None:
        output = output[: max(0, max_pairs)]
    return output


def pair_task_payload(pair: Mapping[str, Any]) -> Dict[str, str]:
    pair_id = str(pair.get("pair_id") or pair.get("__record_id__") or "")
    task_text = ""
    for key in ("query", "task_text", "task_instruction", "instruction", "task"):
        value = pair.get(key)
        if isinstance(value, str) and value.strip():
            task_text = value.strip()
            break
    return {"pair_id": pair_id, "task_text": task_text}


def chunks(items: Sequence[Mapping[str, Any]], size: int) -> List[List[Mapping[str, Any]]]:
    return [list(items[start : start + size]) for start in range(0, len(items), max(1, size))]


def clamp_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        text = str(value).lower()
        if "satisf" in text or "pass" in text:
            score = 1.0
        elif "partial" in text or "ambiguous" in text:
            score = 0.5
        else:
            score = 0.0
    return max(0.0, min(1.0, score))

