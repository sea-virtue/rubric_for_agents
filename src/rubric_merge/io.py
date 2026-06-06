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


def build_cluster_groups(
    cluster_records: Sequence[Mapping[str, Any]],
    pair_rubrics: Sequence[Mapping[str, Any]],
    pair_contexts: Mapping[str, Mapping[str, Any]],
    *,
    selected_group_ids: Sequence[str],
    min_pairs: int,
    max_groups: int | None,
) -> List[Dict[str, Any]]:
    selected = set(selected_group_ids)
    rubrics_by_pair = {
        str(record.get("pair_id") or record.get("__record_id__")): dict(record)
        for record in pair_rubrics
        if record.get("rubrics")
    }
    groups: Dict[str, Dict[str, Any]] = {}
    for record in cluster_records:
        cluster_id = str(record.get("cluster_id") or "")
        pair_id = str(record.get("pair_id") or record.get("__record_id__") or "")
        if not cluster_id or not pair_id:
            continue
        if selected and cluster_id not in selected:
            continue
        group = groups.setdefault(
            cluster_id,
            {
                "__record_id__": cluster_id,
                "group_id": cluster_id,
                "cluster_id": cluster_id,
                "grouping": "cluster",
                "pairs": [],
                "missing_rubric_pair_ids": [],
            },
        )
        context = dict(pair_contexts.get(pair_id, {}))
        context.setdefault("pair_id", pair_id)
        context.setdefault("domain", record.get("domain"))
        context.setdefault("jobname", record.get("jobname"))
        context.setdefault("query", record.get("task_instruction", ""))
        context["cluster_member"] = {
            "__record_id__": record.get("__record_id__"),
            "pair_id": pair_id,
            "cluster_id": cluster_id,
            "cluster_size": record.get("cluster_size"),
            "domain": record.get("domain"),
            "jobname": record.get("jobname"),
            "task_instruction": record.get("task_instruction"),
            "source_path": record.get("source_path"),
            "relative_source_path": record.get("relative_source_path"),
            "selected_records": record.get("selected_records", []),
        }
        group["pairs"].append(context)

    output: List[Dict[str, Any]] = []
    for group in groups.values():
        pairs = sorted(group["pairs"], key=lambda item: str(item.get("pair_id")))
        enriched_pairs = []
        rubric_count = 0
        domains: Dict[str, int] = {}
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
            domain = str(enriched.get("domain") or "")
            if domain:
                domains[domain] = domains.get(domain, 0) + 1
        if len(enriched_pairs) < min_pairs:
            continue
        group["pairs"] = enriched_pairs
        group["pair_count"] = len(enriched_pairs)
        group["cluster_size"] = len(enriched_pairs)
        group["source_pair_ids"] = [str(item.get("pair_id")) for item in enriched_pairs]
        group["source_record_ids"] = [str(item.get("pair_id")) for item in enriched_pairs]
        group["rubric_count"] = rubric_count
        group["domains"] = dict(sorted(domains.items(), key=lambda item: (-item[1], item[0])))
        output.append(group)

    output.sort(key=lambda item: (-int(item.get("pair_count", 0)), str(item.get("group_id"))))
    if max_groups is not None:
        output = output[: max(0, max_groups)]
    return output
