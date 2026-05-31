from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .paths import ROOT


def load_pair_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_dir():
        index_path = path / "pair_index.json"
        if index_path.exists():
            return load_json_array(index_path)
        records = []
        for pair_path in sorted(path.rglob("pair.json")):
            data = json.loads(pair_path.read_text(encoding="utf-8"))
            if isinstance(data, Mapping):
                records.append(dict(data))
        return records
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, Mapping):
        return [dict(data)]
    raise ValueError(f"{path} must contain a JSON object or array")


def select_pair_records(
    records: Sequence[Mapping[str, Any]],
    *,
    selected_pair_ids: Sequence[str],
    max_pairs: Optional[int],
) -> List[Dict[str, Any]]:
    selected = set(selected_pair_ids)
    output = []
    for record in records:
        pair_id = str(record.get("pair_id") or record.get("__record_id__") or "")
        if selected and pair_id not in selected:
            continue
        responses = record.get("responses")
        validation = record.get("validation")
        selected_records = record.get("selected_records")
        has_pair_payload = isinstance(responses, list) and len(responses) >= 2
        has_prompt_labels = isinstance(validation, list) or (isinstance(selected_records, list) and len(selected_records) >= 2)
        if not pair_id or not has_pair_payload or not has_prompt_labels:
            continue
        output.append(dict(record))
    output.sort(key=lambda item: str(item.get("pair_id") or item.get("__record_id__")))
    if max_pairs is not None:
        output = output[: max(0, max_pairs)]
    return output


def load_selected_cache_record(selected: Mapping[str, Any]) -> Dict[str, Any]:
    source_path = Path(str(selected.get("source_path") or ""))
    if not source_path.exists():
        source_path = ROOT / source_path
    if not source_path.exists():
        return {}
    data = json.loads(source_path.read_text(encoding="utf-8"))
    record_id = str(selected.get("__record_id__", ""))
    if isinstance(data, list):
        for item in data:
            if isinstance(item, Mapping) and str(item.get("__record_id__", "")) == record_id:
                return dict(item)
        for item in data:
            if isinstance(item, Mapping):
                return dict(item)
    if isinstance(data, Mapping):
        return dict(data)
    return {}


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
