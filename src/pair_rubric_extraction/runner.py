from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from rubric_extraction.cli import normalize_rubric_items
from rubric_miner.llm import async_llm_call, build_client, extract_json_array

from .io import has_error, load_json_array, parse_csv, upsert, write_json
from .prompting import prompt_messages


async def extract_pair_rubrics(
    pair_records: Sequence[Mapping[str, Any]],
    *,
    output_dir: Path,
    pairs_path: Path,
    model: str,
    base_url: str,
    api_key_env: str,
    concurrency: int,
    max_pairs: int | None,
    pair_ids: str,
    max_chars_per_response: int,
    max_tokens: int,
    temperature: float,
    refresh: bool,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "pair_rubrics.json"
    prompt_path = output_dir / "pair_rubric_prompts.json"
    raw_path = output_dir / "pair_rubric_raw_outputs.json"
    config_path = output_dir / "pair_rubric_extraction_config.json"

    existing = [] if refresh else load_json_array(output_path)
    prompt_records = [] if refresh else load_json_array(prompt_path)
    raw_records = [] if refresh else load_json_array(raw_path)
    done = {
        str(record.get("pair_id") or record.get("__record_id__"))
        for record in existing
        if record.get("pair_id") and not has_error(record) and record.get("rubrics")
    }

    client = build_client(api_key_env=api_key_env, base_url=base_url)
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def process_pair(pair: Mapping[str, Any]) -> Dict[str, Any]:
        pair_id = str(pair.get("pair_id") or pair.get("__record_id__"))
        if pair_id in done:
            for record in existing:
                if str(record.get("pair_id") or record.get("__record_id__")) == pair_id:
                    return dict(record)
        async with semaphore:
            try:
                messages = prompt_messages(pair, max_chars_per_response=max_chars_per_response)
                prompt_record = {
                    "__record_id__": pair_id,
                    "pair_id": pair_id,
                    "domain": pair.get("domain"),
                    "jobname": pair.get("jobname"),
                    "messages": messages,
                }
                upsert(prompt_records, prompt_record, key="pair_id")
                write_json(prompt_path, prompt_records)

                raw_content = await async_llm_call(
                    client,
                    model,
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                raw_record = {
                    "__record_id__": pair_id,
                    "pair_id": pair_id,
                    "model": model,
                    "raw_output": raw_content,
                }
                upsert(raw_records, raw_record, key="pair_id")
                write_json(raw_path, raw_records)

                try:
                    raw_items = extract_json_array(raw_content)
                except Exception as exc:
                    excerpt = re.sub(r"\s+", " ", raw_content).strip()[:500]
                    return pair_error(pair, f"{exc}; output_excerpt={excerpt}")
                rubrics = normalize_rubric_items(raw_items)
                if not rubrics:
                    return {
                        **pair_error(pair, "LLM returned no normalizable rubric items"),
                        "raw_item_count": len(raw_items),
                    }
                return {
                    "__record_id__": pair_id,
                    "pair_id": pair_id,
                    "domain": pair.get("domain"),
                    "jobname": pair.get("jobname"),
                    "model": model,
                    "source_pair_id": pair_id,
                    "rubrics": rubrics,
                }
            except Exception as exc:
                return pair_error(pair, str(exc))

    results = list(existing)
    tasks = [asyncio.create_task(process_pair(pair)) for pair in pair_records]
    for future in asyncio.as_completed(tasks):
        record = await future
        upsert(results, record, key="pair_id")
        write_json(output_path, results)
        print(f"processed: {record.get('pair_id')} rubrics={len(record.get('rubrics', []))} error={record.get('rubric_error', '')}")

    config = {
        "pairs": str(pairs_path),
        "output_dir": str(output_dir),
        "model": model,
        "base_url": base_url,
        "concurrency": concurrency,
        "max_pairs": max_pairs,
        "pair_ids": parse_csv(pair_ids),
        "max_chars_per_response": max_chars_per_response,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "num_selected_pairs": len(pair_records),
        "outputs": {
            "rubrics": str(output_path),
            "prompts": str(prompt_path),
            "raw_outputs": str(raw_path),
            "config": str(config_path),
        },
    }
    write_json(config_path, config)
    print(f"pair_rubric_file: {output_path}")
    print(f"pairs_processed_or_cached: {len(results)}")
    return 0


def pair_error(pair: Mapping[str, Any], message: str) -> Dict[str, Any]:
    pair_id = str(pair.get("pair_id") or pair.get("__record_id__"))
    return {
        "__record_id__": pair_id,
        "pair_id": pair_id,
        "domain": pair.get("domain"),
        "jobname": pair.get("jobname"),
        "skipped": True,
        "rubric_error": message,
        "rubrics": [],
    }
