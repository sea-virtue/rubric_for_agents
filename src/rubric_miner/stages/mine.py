from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from ..io import good_record_index, load_json_array
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
    max_tokens: int,
) -> List[Dict[str, Any]]:
    stage_records = load_json_array(output_path)
    ok_index = good_record_index(stage_records)
    semaphore = asyncio.Semaphore(concurrency)

    async def process_group(group: Mapping[str, Any]) -> Dict[str, Any]:
        cluster_id = str(group["cluster_id"])
        if cluster_id in ok_index:
            return dict(ok_index[cluster_id])
        async with semaphore:
            try:
                messages = mining_messages(group, max_records=max_records, max_chars=max_chars)
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
