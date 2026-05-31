from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .loader import iter_json_payloads
from .paths import ROOT


def write_pair_outputs(
    pair_records: Sequence[Mapping[str, Any]],
    report_records: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    *,
    output_root: Path,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    for pair in pair_records:
        domain = safe_path_part(str(pair["domain"]))
        jobname = safe_path_part(str(pair["jobname"]))
        pair_dir = output_root / domain / jobname
        pair_dir.mkdir(parents=True, exist_ok=True)
        write_json(pair_dir / "pair.json", pair)

        selected = pair.get("selected_records", [])
        if isinstance(selected, list):
            for item in selected:
                if not isinstance(item, Mapping):
                    continue
                record = load_selected_record(item)
                if not record:
                    continue
                record = attach_pair_metadata(record, pair, item)
                filename = safe_path_part(str(item.get("job_with_model") or item.get("__record_id__"))) + ".json"
                write_json(pair_dir / filename, record)

    write_json(output_root / "pair_index.json", list(pair_records))
    write_json(output_root / "pair_report.json", list(report_records))
    write_json(output_root / "pair_summary.json", summary)


def load_selected_record(item: Mapping[str, Any]) -> Dict[str, Any]:
    source_path = Path(str(item.get("source_path") or ""))
    if not source_path.exists():
        source_path = ROOT / source_path
    if not source_path.exists():
        return {}
    payloads = list(iter_json_payloads(source_path))
    record_id = str(item.get("__record_id__", ""))
    for payload in payloads:
        if str(payload.get("__record_id__", "")) == record_id:
            return dict(payload)
    return dict(payloads[0]) if payloads else {}


def attach_pair_metadata(
    record: Dict[str, Any],
    pair: Mapping[str, Any],
    selected_item: Mapping[str, Any],
) -> Dict[str, Any]:
    output = dict(record)
    output["_pair_cache"] = {
        "pair_id": pair.get("pair_id"),
        "domain": pair.get("domain"),
        "jobname": pair.get("jobname"),
        "pair_role": selected_item.get("pair_role"),
        "label_rank": selected_item.get("label_rank"),
        "source_path": selected_item.get("source_path"),
        "relative_source_path": selected_item.get("relative_source_path"),
    }
    return output


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(tmp_path, path)


def safe_path_part(value: str) -> str:
    value = str(value or "").strip()
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value)
    value = value.rstrip(". ")
    return value or "unknown"
