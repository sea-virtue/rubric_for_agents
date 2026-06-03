from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from pair_rubric_extraction.io import load_pair_records


def load_json_array(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array")
    return [item for item in data if isinstance(item, dict)]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(tmp_path, path)


def upsert(records: List[Dict[str, Any]], record: Dict[str, Any], *, key: str) -> None:
    value = str(record.get(key) or record.get("__record_id__"))
    for idx, existing in enumerate(records):
        if str(existing.get(key) or existing.get("__record_id__")) == value:
            records[idx] = record
            return
    records.append(record)


def has_error(record: Mapping[str, Any]) -> bool:
    return any(str(key).endswith("_error") for key in record)


def parse_csv(value: str) -> List[str]:
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def load_pair_contexts(pairs_path: Path) -> Dict[str, Dict[str, Any]]:
    contexts: Dict[str, Dict[str, Any]] = {}
    for pair in load_pair_records(pairs_path):
        pair_id = str(pair.get("pair_id") or pair.get("__record_id__") or "")
        if not pair_id:
            continue
        contexts[pair_id] = {
            "pair_id": pair_id,
            "domain": pair.get("domain"),
            "jobname": pair.get("jobname"),
            "query": str(pair.get("query") or "").strip(),
        }
    return contexts


def build_domain_groups(
    pair_rubrics: Sequence[Mapping[str, Any]],
    pair_contexts: Mapping[str, Mapping[str, Any]],
    *,
    selected_group_ids: Sequence[str],
    min_pairs: int,
    max_groups: int | None,
) -> List[Dict[str, Any]]:
    selected = set(selected_group_ids)
    groups: Dict[str, Dict[str, Any]] = {}
    for pair_id, context in pair_contexts.items():
        domain = str(context.get("domain") or pair_id.split("/", 1)[0] or "unknown")
        if selected and domain not in selected:
            continue
        group = groups.setdefault(
            domain,
            {
                "__record_id__": domain,
                "group_id": domain,
                "grouping": "domain",
                "domain": domain,
                "pairs": [],
                "missing_rubric_pair_ids": [],
            },
        )
        group["pairs"].append(dict(context))

    rubrics_by_pair = {
        str(record.get("pair_id") or record.get("__record_id__")): dict(record)
        for record in pair_rubrics
        if record.get("rubrics")
    }

    output: List[Dict[str, Any]] = []
    for group in groups.values():
        pairs = sorted(group["pairs"], key=lambda item: str(item.get("pair_id")))
        enriched_pairs = []
        rubric_count = 0
        for pair in pairs:
            pair_id = str(pair.get("pair_id"))
            rubric_record = rubrics_by_pair.get(pair_id)
            if not rubric_record:
                group["missing_rubric_pair_ids"].append(pair_id)
                continue
            rubrics = [item for item in rubric_record.get("rubrics", []) if isinstance(item, Mapping)]
            if not rubrics:
                group["missing_rubric_pair_ids"].append(pair_id)
                continue
            enriched = dict(pair)
            enriched["rubrics"] = [dict(item) for item in rubrics]
            enriched["rubric_model"] = rubric_record.get("model")
            enriched_pairs.append(enriched)
            rubric_count += len(rubrics)
        if len(enriched_pairs) < min_pairs:
            continue
        group["pairs"] = enriched_pairs
        group["pair_count"] = len(enriched_pairs)
        group["source_pair_ids"] = [str(item.get("pair_id")) for item in enriched_pairs]
        group["rubric_count"] = rubric_count
        output.append(group)

    output.sort(key=lambda item: (-int(item.get("pair_count", 0)), str(item.get("group_id"))))
    if max_groups is not None:
        output = output[: max(0, max_groups)]
    return output

