from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from .builder import infer_outcome, pick_task_instruction


def load_candidates(input_root: Path) -> List[Dict[str, Any]]:
    if not input_root.exists():
        raise FileNotFoundError(input_root)
    paths = [input_root] if input_root.is_file() else sorted(input_root.rglob("*.json"))
    candidates: List[Dict[str, Any]] = []
    for path in paths:
        source_info = parse_source_path(path, input_root)
        if not source_info:
            continue
        payloads = list(iter_json_payloads(path))
        if not payloads:
            continue
        for item_idx, payload in enumerate(payloads):
            record = dict(payload)
            outcome = infer_outcome(record)
            record_id = str(record.get("__record_id__") or f"{path.stem}_{item_idx}")
            candidates.append(
                {
                    **source_info,
                    "__record_id__": record_id,
                    "outcome": outcome,
                    "task_instruction": pick_task_instruction(record),
                    "record": record,
                }
            )
    return candidates


def parse_source_path(path: Path, input_root: Path) -> Dict[str, Any] | None:
    try:
        relative = path.resolve().relative_to(input_root.resolve())
    except ValueError:
        return None
    parts = relative.parts
    if len(parts) < 4:
        return None
    domain = parts[0]
    model_name = parts[1]
    job_with_model = parts[2]
    job_file = parts[-1]
    jobname = Path(job_file).stem
    return {
        "domain": domain,
        "model_name": model_name,
        "job_with_model": job_with_model,
        "jobname": jobname,
        "source_path": str(path),
        "relative_source_path": str(relative).replace("\\", "/"),
    }


def iter_json_payloads(path: Path) -> Iterable[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        for item in data:
            if isinstance(item, Mapping):
                yield item
    elif isinstance(data, Mapping):
        yield data
