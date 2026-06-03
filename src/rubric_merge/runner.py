from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from rubric_extraction.cli import first_text, list_field, normalize_verification_guide, stringify_field_value
from rubric_miner.llm import async_llm_call, build_client

from .io import has_error, load_json_array, parse_csv, upsert, write_json
from .prompting import prompt_messages
from .selector import select_group_rubrics_mcr


async def merge_group_rubrics(
    groups: Sequence[Mapping[str, Any]],
    *,
    output_dir: Path,
    pair_rubrics_path: Path,
    pairs_path: Path,
    model: str,
    base_url: str,
    api_key_env: str,
    concurrency: int,
    grouping: str,
    group_ids: str,
    max_groups: int | None,
    min_pairs: int,
    num_categories: int,
    selection_method: str,
    embedding_model: str,
    embedding_base_url: str,
    embedding_batch_size: int,
    max_selected_rubrics: int,
    mcr_batch_size: int,
    mcr_eps: float,
    mcr_min_increment_threshold: float,
    mcr_patience: int,
    max_rubrics_per_group: int,
    max_chars_per_rubric: int,
    max_tokens: int,
    temperature: float,
    refresh: bool,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{grouping}_merged_rubrics.json"
    prompt_path = output_dir / f"{grouping}_rubric_merge_prompts.json"
    raw_path = output_dir / f"{grouping}_rubric_merge_raw_outputs.json"
    selection_path = output_dir / f"{grouping}_rubric_mcr_selection.json"
    config_path = output_dir / f"{grouping}_rubric_merge_config.json"

    existing = [] if refresh else load_json_array(output_path)
    prompt_records = [] if refresh else load_json_array(prompt_path)
    raw_records = [] if refresh else load_json_array(raw_path)
    selection_records = [] if refresh else load_json_array(selection_path)
    done = {
        str(record.get("group_id") or record.get("__record_id__"))
        for record in existing
        if record.get("group_id") and not has_error(record) and record.get("rubrics")
    }

    client = build_client(api_key_env=api_key_env, base_url=base_url)
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def process_group(group: Mapping[str, Any]) -> Dict[str, Any]:
        group_id = str(group.get("group_id") or group.get("__record_id__"))
        if group_id in done:
            for record in existing:
                if str(record.get("group_id") or record.get("__record_id__")) == group_id:
                    return dict(record)
        async with semaphore:
            try:
                prompt_group = dict(group)
                selection_info: Dict[str, Any] = {
                    "selection_method": selection_method,
                    "source_rubric_count": group.get("rubric_count", 0),
                }
                if selection_method == "mcr":
                    prompt_group, selection_info = await select_group_rubrics_mcr(
                        group,
                        embedding_model=embedding_model,
                        embedding_base_url=embedding_base_url,
                        api_key_env=api_key_env,
                        embedding_batch_size=embedding_batch_size,
                        max_chars_per_rubric=max_chars_per_rubric,
                        max_selected_rubrics=max_selected_rubrics,
                        mcr_batch_size=mcr_batch_size,
                        eps=mcr_eps,
                        min_increment_threshold=mcr_min_increment_threshold,
                        patience=mcr_patience,
                    )
                selection_record = {
                    "__record_id__": group_id,
                    "group_id": group_id,
                    "grouping": grouping,
                    **selection_info,
                }
                upsert(selection_records, selection_record, key="group_id")
                write_json(selection_path, selection_records)

                messages = prompt_messages(
                    prompt_group,
                    num_categories=num_categories,
                    max_rubrics_per_group=max_rubrics_per_group,
                    max_chars_per_rubric=max_chars_per_rubric,
                )
                prompt_record = {
                    "__record_id__": group_id,
                    "group_id": group_id,
                    "grouping": grouping,
                    "pair_count": group.get("pair_count"),
                    "rubric_count": group.get("rubric_count"),
                    "selected_rubric_count": prompt_group.get("selected_rubric_count"),
                    "source_pair_ids": group.get("source_pair_ids", []),
                    "selected_pair_ids": prompt_group.get("selected_pair_ids", []),
                    "messages": messages,
                }
                upsert(prompt_records, prompt_record, key="group_id")
                write_json(prompt_path, prompt_records)

                raw_content = await async_llm_call(
                    client,
                    model,
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                raw_record = {
                    "__record_id__": group_id,
                    "group_id": group_id,
                    "grouping": grouping,
                    "model": model,
                    "raw_output": raw_content,
                }
                upsert(raw_records, raw_record, key="group_id")
                write_json(raw_path, raw_records)

                try:
                    parsed = extract_json_object(raw_content)
                except Exception as exc:
                    excerpt = re.sub(r"\s+", " ", raw_content).strip()[:500]
                    return group_error(group, grouping, f"{exc}; output_excerpt={excerpt}")

                rubrics = normalize_merged_categories(parsed)
                if not rubrics:
                    return {
                        **group_error(group, grouping, "LLM returned no normalizable merged rubric categories"),
                        "raw_output_keys": sorted(parsed.keys()),
                    }
                return {
                    "__record_id__": group_id,
                    "group_id": group_id,
                    "grouping": grouping,
                    "domain": group.get("domain"),
                    "model": model,
                    "pair_count": group.get("pair_count"),
                    "source_rubric_count": group.get("rubric_count"),
                    "selected_rubric_count": prompt_group.get("selected_rubric_count", group.get("rubric_count")),
                    "source_pair_ids": group.get("source_pair_ids", []),
                    "selected_pair_ids": prompt_group.get("selected_pair_ids", group.get("source_pair_ids", [])),
                    "rubric_selection": selection_info,
                    "missing_rubric_pair_ids": group.get("missing_rubric_pair_ids", []),
                    "num_categories_requested": num_categories,
                    "rubrics": rubrics,
                    "reason": stringify_field_value(parsed.get("reason")),
                }
            except Exception as exc:
                return group_error(group, grouping, str(exc))

    results = list(existing)
    for future in asyncio.as_completed([asyncio.create_task(process_group(group)) for group in groups]):
        record = await future
        upsert(results, record, key="group_id")
        write_json(output_path, results)
        print(f"processed: {record.get('group_id')} rubrics={len(record.get('rubrics', []))} error={record.get('rubric_merge_error', '')}")

    config = {
        "pair_rubrics": str(pair_rubrics_path),
        "pairs": str(pairs_path),
        "output_dir": str(output_dir),
        "model": model,
        "base_url": base_url,
        "concurrency": concurrency,
        "grouping": grouping,
        "group_ids": parse_csv(group_ids),
        "max_groups": max_groups,
        "min_pairs": min_pairs,
        "num_categories": num_categories,
        "selection_method": selection_method,
        "embedding_model": embedding_model if selection_method == "mcr" else None,
        "embedding_base_url": embedding_base_url if selection_method == "mcr" else None,
        "embedding_batch_size": embedding_batch_size if selection_method == "mcr" else None,
        "max_selected_rubrics": max_selected_rubrics if selection_method == "mcr" else None,
        "mcr_batch_size": mcr_batch_size if selection_method == "mcr" else None,
        "mcr_eps": mcr_eps if selection_method == "mcr" else None,
        "mcr_min_increment_threshold": mcr_min_increment_threshold if selection_method == "mcr" else None,
        "mcr_patience": mcr_patience if selection_method == "mcr" else None,
        "max_rubrics_per_group": max_rubrics_per_group,
        "max_chars_per_rubric": max_chars_per_rubric,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "num_selected_groups": len(groups),
        "outputs": {
            "merged_rubrics": str(output_path),
            "prompts": str(prompt_path),
            "raw_outputs": str(raw_path),
            "mcr_selection": str(selection_path),
            "config": str(config_path),
        },
    }
    write_json(config_path, config)
    print(f"merged_rubric_file: {output_path}")
    print(f"groups_processed_or_cached: {len(results)}")
    return 0


def group_error(group: Mapping[str, Any], grouping: str, message: str) -> Dict[str, Any]:
    group_id = str(group.get("group_id") or group.get("__record_id__"))
    return {
        "__record_id__": group_id,
        "group_id": group_id,
        "grouping": grouping,
        "domain": group.get("domain"),
        "pair_count": group.get("pair_count", 0),
        "source_rubric_count": group.get("rubric_count", 0),
        "skipped": True,
        "rubric_merge_error": message,
        "rubrics": [],
    }


def extract_json_object(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.DOTALL | re.I)
    if fenced:
        cleaned = fenced.group(1).strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            return {"categories": parsed}
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    candidates: List[Dict[str, Any]] = []
    for match in re.finditer(r"[{\[]", cleaned):
        try:
            parsed_any, _ = decoder.raw_decode(cleaned[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed_any, dict):
            candidates.append(parsed_any)
        elif isinstance(parsed_any, list):
            candidates.append({"categories": parsed_any})
    if candidates:
        candidates.sort(key=json_object_score, reverse=True)
        return candidates[0]
    raise ValueError("LLM output does not contain a valid JSON object or array")


def json_object_score(item: Mapping[str, Any]) -> int:
    score = 0
    categories = item.get("categories") or item.get("rubrics") or item.get("items")
    if isinstance(categories, list):
        score += 20 + len(categories)
    if item.get("reason"):
        score += 3
    return score


def normalize_merged_categories(parsed: Mapping[str, Any]) -> List[Dict[str, Any]]:
    raw_categories = first_list(parsed, "categories", "rubrics", "rubric_items", "items", "criteria")
    output = []
    for item in iter_candidates(raw_categories):
        if not isinstance(item, Mapping):
            if stringify_field_value(item):
                output.append({"theme": stringify_field_value(item), "tips": []})
            continue
        theme = first_text(
            item,
            "theme",
            "dimension",
            "category",
            "title",
            "criterion",
            "criteria",
            "rubric",
            "description",
        )
        tips = list_field(
            item,
            "tips",
            "guidance",
            "checks",
            "criteria",
            "rubrics",
            "items",
            "requirements",
            "evaluation_points",
        )
        if not tips:
            criterion = first_text(item, "criterion", "criteria", "rubric", "description")
            tips = [criterion] if criterion and criterion != theme else []
        if not theme and tips:
            theme = tips[0]
            tips = tips[1:]
        if not theme:
            continue
        category = {
            "theme": theme,
            "tips": tips,
        }
        verification_guide = normalize_verification_guide(
            item.get("verification_guide")
            or item.get("verification")
            or item.get("verification_protocol")
            or item.get("how_to_verify")
        )
        if verification_guide:
            category["verification_guide"] = verification_guide
        source_pair_ids = list_field(item, "source_pair_ids", "pair_ids", "sources")
        if source_pair_ids:
            category["source_pair_ids"] = source_pair_ids
        output.append(category)
    return output


def first_list(item: Mapping[str, Any], *keys: str) -> List[Any]:
    for key in keys:
        value = item.get(key)
        if isinstance(value, list):
            return value
    return []


def iter_candidates(values: Sequence[Any]) -> Iterable[Any]:
    for value in values:
        if isinstance(value, Mapping):
            nested = first_list(value, "categories", "rubrics", "items")
            if nested and not value.get("theme"):
                yield from iter_candidates(nested)
                continue
        yield value
