from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from ..io import atomic_write_json_array, good_record_index, load_json_array, upsert
from ..llm import llm_json_array
from ..logging_utils import logger
from ..prompts import mining_messages
from ..schemas import MinedCluster, model_dump
from .common import parse_rubric_items, run_parallel_stage


async def mine_stage(
    groups: Mapping[str, Mapping[str, Any]],
    output_path: Path,
    client: Any,
    mining_models: Sequence[str],
    concurrency: int,
    max_records: int,
    max_chars: int,
    prompt_token_budget: int,
    max_tokens: int,
) -> List[Dict[str, Any]]:
    current_group_ids = set(map(str, groups))
    prompt_path = output_path.with_name("mining_prompts.json")
    prompt_records = [
        record
        for record in load_json_array(prompt_path)
        if str(record.get("__record_id__")) in current_group_ids
    ]
    stage_records = [
        record
        for record in load_json_array(output_path)
        if str(record.get("__record_id__")) in current_group_ids
    ]
    ok_index = good_record_index(stage_records)
    semaphore = asyncio.Semaphore(concurrency)

    async def process_group(group: Mapping[str, Any]) -> Dict[str, Any]:
        cluster_id = str(group["cluster_id"])
        current_sources = list(group.get("source_record_ids", []))
        if cluster_id in ok_index:
            cached = dict(ok_index[cluster_id])
            if list(cached.get("source_record_ids", [])) == current_sources:
                return cached
        async with semaphore:
            try:
                messages, fitted_records, fitted_chars, approx_prompt_tokens = _fit_mining_messages(
                    group,
                    max_records=max_records,
                    max_chars=max_chars,
                    token_budget=prompt_token_budget,
                )
                user_content = messages[-1].get("content", "") if messages else ""
                prompt_record = {
                    "__record_id__": cluster_id,
                    "cluster_id": cluster_id,
                    "cluster_key": str(group.get("cluster_key", "")),
                    "source_record_ids": list(group.get("source_record_ids", [])),
                    "num_source_records": len(group.get("source_record_ids", [])),
                    "sampled_records_in_prompt": fitted_records,
                    "requested_records_in_prompt": min(max_records, len(group.get("records", []))),
                    "max_chars_per_trace": fitted_chars,
                    "requested_max_chars_per_trace": max_chars,
                    "prompt_token_budget": prompt_token_budget,
                    "mining_models": list(mining_models),
                    "messages": messages,
                    "user_prompt_chars": len(user_content),
                    "approx_user_prompt_tokens": max(1, len(user_content) // 4),
                    "approx_total_prompt_tokens": approx_prompt_tokens,
                }
                upsert(prompt_records, prompt_record)
                atomic_write_json_array(prompt_path, prompt_records)
                raw_by_model = await asyncio.gather(
                    *[
                        llm_json_array(client, model, messages, temperature=0.35, max_tokens=max_tokens)
                        for model in mining_models
                    ]
                )
                rubrics_by_model = {}
                for idx, (model, raw) in enumerate(zip(mining_models, raw_by_model), start=1):
                    key = model if model not in rubrics_by_model else f"{model}#{idx}"
                    rubrics_by_model[key] = parse_rubric_items(raw)
                mined = MinedCluster(
                    __record_id__=cluster_id,
                    cluster_id=cluster_id,
                    cluster_key=str(group.get("cluster_key", "")),
                    source_record_ids=list(group.get("source_record_ids", [])),
                    mining_models=list(mining_models),
                    rubrics_by_model=rubrics_by_model,
                )
                return model_dump(mined)
            except Exception as exc:
                logger.warning("sample_failed", extra={"stage": "mine", "record_id": cluster_id, "error": str(exc)})
                return {
                    "__record_id__": cluster_id,
                    "cluster_id": cluster_id,
                    "cluster_key": str(group.get("cluster_key", "")),
                    "source_record_ids": list(group.get("source_record_ids", [])),
                    "skipped": True,
                    "mine_error": str(exc),
                }

    logger.info(
        "stage_start",
        extra={"stage": "mine", "total": len(groups), "mining_models": list(mining_models)},
    )
    return await run_parallel_stage(
        "In-cluster mining",
        list(groups.values()),
        process_group,
        stage_records,
        output_path,
    )


def _fit_mining_messages(
    group: Mapping[str, Any],
    *,
    max_records: int,
    max_chars: int,
    token_budget: int,
) -> tuple[List[Dict[str, str]], int, int, int]:
    records_available = len(group.get("records", []))
    fitted_records = max(1, min(max_records, records_available))
    fitted_chars = max(1000, max_chars)
    token_budget = max(1000, token_budget)

    while True:
        messages = mining_messages(group, max_records=fitted_records, max_chars=fitted_chars)
        approx_tokens = _approx_message_tokens(messages)
        if approx_tokens <= token_budget or (fitted_records <= 1 and fitted_chars <= 2000):
            return messages, fitted_records, fitted_chars, approx_tokens
        if fitted_chars > 6000:
            fitted_chars = max(6000, int(fitted_chars * 0.8))
        elif fitted_records > 1:
            fitted_records -= 1
        else:
            fitted_chars = max(2000, int(fitted_chars * 0.8))


def _approx_message_tokens(messages: Sequence[Mapping[str, str]]) -> int:
    chars = sum(len(str(message.get("content", ""))) for message in messages)
    return max(1, chars // 4)
