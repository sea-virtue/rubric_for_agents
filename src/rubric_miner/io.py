from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .dataloader import TraceDataLoader
from .schemas import has_error


def good_record_index(records: Sequence[Mapping[str, Any]]) -> Dict[str, Mapping[str, Any]]:
    return {
        str(record.get("__record_id__")): record
        for record in records
        if record.get("__record_id__") is not None and not has_error(record)
    }


def upsert(records: List[Dict[str, Any]], record: Dict[str, Any]) -> None:
    record_id = str(record["__record_id__"])
    for idx, existing in enumerate(records):
        if str(existing.get("__record_id__")) == record_id:
            records[idx] = record
            return
    records.append(record)


def atomic_write_json_array(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(list(records), handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def load_json_array(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array")
    return [item if isinstance(item, dict) else {"value": item} for item in data]


def read_input_records(
    path: Path,
    *,
    input_format: Optional[str] = None,
    field_map: Optional[Mapping[str, str]] = None,
    csv_group_by: Optional[str] = None,
    max_records: Optional[int] = None,
    agent_reward_observation_chars: int = 1200,
    agent_reward_observation_policy: str = "last",
    agent_reward_sample_per_bucket: Optional[int] = None,
    agent_reward_sample_seed: int = 13,
) -> List[Dict[str, Any]]:
    loader = TraceDataLoader(
        input_format=input_format,
        field_map=field_map,
        csv_group_by=csv_group_by,
        max_records=max_records,
        agent_reward_observation_chars=agent_reward_observation_chars,
        agent_reward_observation_policy=agent_reward_observation_policy,
        agent_reward_sample_per_bucket=agent_reward_sample_per_bucket,
        agent_reward_sample_seed=agent_reward_sample_seed,
    )
    return loader.load(path)
