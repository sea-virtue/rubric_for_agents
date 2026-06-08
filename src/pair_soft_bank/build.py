from __future__ import annotations

import asyncio
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from rubric_miner.llm import async_embedding_batch_call, build_client

from .io import embedding_cache_metadata, load_embedding_cache, load_json_array, write_json


def normalize_rubric(item: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "dimension": str(item.get("dimension") or item.get("theme") or item.get("category") or "trajectory_quality").strip(),
        "criterion": str(item.get("criterion") or item.get("criteria") or item.get("description") or item.get("rubric") or "").strip(),
        "positive_evidence": text_list(item.get("positive_evidence")),
        "negative_evidence": text_list(item.get("negative_evidence")),
        "severity": normalize_severity(str(item.get("severity") or "medium")),
        "rationale": str(item.get("rationale") or item.get("reason") or "").strip(),
        "verification_guide": item.get("verification_guide") if isinstance(item.get("verification_guide"), Mapping) else {},
    }


async def build_bank_v1(
    *,
    pair_rubrics_path: Path,
    pair_embedding_cache: Path,
    output_dir: Path,
    output_file: Path | None,
    embedding_model: str,
    embedding_base_url: str,
    api_key_env: str,
    embedding_batch_size: int,
    refresh_embeddings: bool,
    dry_run: bool,
) -> int:
    pair_rubrics = load_json_array(pair_rubrics_path)
    embedding_cache = {} if refresh_embeddings else load_embedding_cache(pair_embedding_cache)
    cache_meta = embedding_cache_metadata(pair_embedding_cache)

    missing_pairs = []
    for record in pair_rubrics:
        pair_id = str(record.get("pair_id") or record.get("__record_id__") or "")
        task_text = pair_task_text(record)
        if pair_id and task_text and pair_id not in embedding_cache:
            missing_pairs.append({"pair_id": pair_id, "task_text": task_text})

    if missing_pairs and dry_run:
        print(f"missing_embeddings: {len(missing_pairs)} (dry-run will not call embedding server)")
    elif missing_pairs:
        client = build_client(api_key_env=api_key_env, base_url=embedding_base_url)
        for start in range(0, len(missing_pairs), max(1, embedding_batch_size)):
            batch = missing_pairs[start : start + max(1, embedding_batch_size)]
            embeddings = await async_embedding_batch_call(client, embedding_model, [item["task_text"] for item in batch])
            for item, embedding in zip(batch, embeddings):
                embedding_cache[item["pair_id"]] = {
                    "record_id": item["pair_id"],
                    "task_text": item["task_text"],
                    "embedding": embedding,
                }
            print(f"embedded_missing: {min(start + len(batch), len(missing_pairs))}/{len(missing_pairs)}")

    bank = []
    skipped_missing_embedding = []
    for record in pair_rubrics:
        pair_id = str(record.get("pair_id") or record.get("__record_id__") or "")
        task_text = pair_task_text(record)
        cache_item = embedding_cache.get(pair_id)
        if not cache_item or not isinstance(cache_item.get("embedding"), list):
            skipped_missing_embedding.append(pair_id)
            continue
        rubrics = record.get("rubrics", [])
        if not isinstance(rubrics, list):
            continue
        for idx, raw_rubric in enumerate(rubrics):
            if not isinstance(raw_rubric, Mapping):
                continue
            rubric = normalize_rubric(raw_rubric)
            if not rubric["criterion"]:
                continue
            rubric_id = stable_rubric_id(pair_id, idx, rubric)
            bank.append(
                {
                    "__record_id__": rubric_id,
                    "rubric_id": rubric_id,
                    "bank_version": "bank_v1_pair_task_embedding",
                    "source_pair_id": pair_id,
                    "source_pair_index": idx,
                    "domain": record.get("domain"),
                    "jobname": record.get("jobname"),
                    "task_text": task_text or str(cache_item.get("task_text") or ""),
                    "label": {
                        "type": "task_embedding",
                        "embedding_model": cache_meta.get("embedding_model") or embedding_model,
                        "embedding_base_url": cache_meta.get("embedding_base_url") or embedding_base_url,
                        "embedding": list(cache_item["embedding"]),
                    },
                    "rubric": rubric,
                }
            )

    output_path = output_file or output_dir / "rubric_bank_v1.json"
    config = {
        "pair_rubrics": str(pair_rubrics_path),
        "pair_embedding_cache": str(pair_embedding_cache),
        "output_file": str(output_path),
        "embedding_model": cache_meta.get("embedding_model") or embedding_model,
        "embedding_base_url": cache_meta.get("embedding_base_url") or embedding_base_url,
        "pair_records": len(pair_rubrics),
        "bank_rubrics": len(bank),
        "missing_embeddings": skipped_missing_embedding,
    }
    if dry_run:
        print(f"pair_records: {len(pair_rubrics)}")
        print(f"bank_rubrics: {len(bank)}")
        print(f"missing_embeddings: {len(skipped_missing_embedding)}")
        if bank:
            preview = dict(bank[0])
            preview["label"] = {**preview["label"], "embedding": f"<{len(preview['label']['embedding'])} dims>"}
            print(json.dumps(preview, ensure_ascii=False, indent=2)[:2400])
        return 0
    write_json(output_path, bank)
    write_json(output_dir / "rubric_bank_v1_config.json", config)
    print(f"rubric_bank_v1: {output_path}")
    print(f"bank_rubrics: {len(bank)}")
    if skipped_missing_embedding:
        print(f"missing_embeddings: {len(skipped_missing_embedding)}")
    return 0


def pair_task_text(record: Mapping[str, Any]) -> str:
    for key in ("query", "task_text", "task_instruction", "instruction", "task"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def stable_rubric_id(pair_id: str, index: int, rubric: Mapping[str, Any]) -> str:
    digest = hashlib.sha1(json.dumps(rubric, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    safe_pair = re.sub(r"[^A-Za-z0-9_.-]+", "_", pair_id)[:100]
    return f"pairbank:{safe_pair}:{index:03d}:{digest}"


def text_list(value: Any) -> List[str]:
    if value in (None, "", [], {}):
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, Mapping):
        return [str(item).strip() for item in value.values() if str(item).strip()]
    return [str(value).strip()]


def normalize_severity(value: str) -> str:
    lowered = str(value or "").lower()
    if lowered in {"high", "critical", "major", "important"}:
        return "high"
    if lowered in {"low", "minor", "optional"}:
        return "low"
    return "medium"

