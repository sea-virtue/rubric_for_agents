from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping


def load_json_array(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array")
    return [item for item in data if isinstance(item, dict)]


def load_json_object(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return dict(data)


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


def parse_csv(value: str) -> List[str]:
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def load_embedding_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    data = load_json_object(path)
    items = data.get("items", [])
    if not isinstance(items, list):
        return {}
    output: Dict[str, Dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, Mapping) or not isinstance(item.get("embedding"), list):
            continue
        record_id = str(item.get("record_id") or item.get("pair_id") or "")
        if record_id:
            output[record_id] = dict(item)
    return output


def embedding_cache_metadata(path: Path) -> Dict[str, Any]:
    data = load_json_object(path)
    return {
        "embedding_model": data.get("embedding_model"),
        "embedding_base_url": data.get("embedding_base_url"),
    }

