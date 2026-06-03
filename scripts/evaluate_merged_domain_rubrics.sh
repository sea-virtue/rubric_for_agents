#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-local}"

MERGED_RUBRICS_FILE="${MERGED_RUBRICS_FILE:-data/rubric_merge/domain_merged_rubrics.json}"
PAIR_CACHE_ROOT="${PAIR_CACHE_ROOT:-data/cache_pair_data}"
CACHE_ROOT="${CACHE_ROOT:-data/cache_data}"
ADAPTER_OUTPUT_DIR="${ADAPTER_OUTPUT_DIR:-data/rubric_merge_eval/adapters}"
PAIR_EVAL_OUTPUT_DIR="${PAIR_EVAL_OUTPUT_DIR:-data/rubric_merge_eval/pair_eval}"
DOMAIN_EVAL_OUTPUT_DIR="${DOMAIN_EVAL_OUTPUT_DIR:-data/rubric_merge_eval/domain_eval}"

mkdir -p "$ADAPTER_OUTPUT_DIR"

PAIR_ADAPTER_FILE="${ADAPTER_OUTPUT_DIR}/domain_merged_as_pair_rubrics.json"
DOMAIN_RUBRIC_ADAPTER_FILE="${ADAPTER_OUTPUT_DIR}/domain_merged_as_cluster_rubrics.json"
DOMAIN_CLUSTER_ADAPTER_FILE="${ADAPTER_OUTPUT_DIR}/domain_cache_clusters.json"

python - "$MERGED_RUBRICS_FILE" "$PAIR_CACHE_ROOT" "$CACHE_ROOT" "$PAIR_ADAPTER_FILE" "$DOMAIN_RUBRIC_ADAPTER_FILE" "$DOMAIN_CLUSTER_ADAPTER_FILE" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

merged_path = Path(sys.argv[1])
pair_root = Path(sys.argv[2])
cache_root = Path(sys.argv[3])
pair_output = Path(sys.argv[4])
domain_rubric_output = Path(sys.argv[5])
domain_cluster_output = Path(sys.argv[6])


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_json_array(path: Path) -> list[dict[str, Any]]:
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array")
    return [item for item in data if isinstance(item, dict)]


def load_pair_records(path: Path) -> list[dict[str, Any]]:
    if path.is_dir():
        index = path / "pair_index.json"
        if index.exists():
            return load_json_array(index)
        return [load_json(p) for p in sorted(path.rglob("pair.json")) if isinstance(load_json(p), dict)]
    data = load_json(path)
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        return [data]
    raise ValueError(f"unsupported pair cache payload: {path}")


def text_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value in (None, "", [], {}):
        return []
    return [str(value).strip()]


def normalize_rubrics(items: Any) -> list[dict[str, Any]]:
    output = []
    if not isinstance(items, list):
        return output
    for item in items:
        if not isinstance(item, Mapping):
            continue
        theme = str(
            item.get("theme")
            or item.get("dimension")
            or item.get("criterion")
            or item.get("description")
            or ""
        ).strip()
        tips = text_list(item.get("tips"))
        if not theme and tips:
            theme = tips[0]
            tips = tips[1:]
        if not theme:
            continue
        criterion = theme
        if tips:
            criterion += "\nTips:\n" + "\n".join(f"- {tip}" for tip in tips)
        guide = item.get("verification_guide")
        if not isinstance(guide, Mapping):
            guide = {}
        positive = tips or text_list(guide.get("checks")) or text_list(guide.get("what_to_extract"))
        output.append(
            {
                "dimension": theme[:160],
                "criterion": criterion,
                "positive_evidence": positive,
                "negative_evidence": text_list(item.get("negative_evidence")),
                "severity": str(item.get("severity") or "medium").lower(),
                "rationale": str(item.get("rationale") or item.get("reason") or ""),
                "verification_guide": dict(guide),
                "source_pair_ids": text_list(item.get("source_pair_ids")),
            }
        )
    return output


def infer_domain_from_path(path: Path, known_domains: set[str]) -> str | None:
    parts = set(path.parts)
    for domain in known_domains:
        if domain in parts:
            return domain
    return None


def infer_outcome(record: Mapping[str, Any]) -> str:
    runtime = record.get("runtime_summary")
    if isinstance(runtime, Mapping) and runtime.get("outcome"):
        return str(runtime.get("outcome"))
    if record.get("outcome"):
        return str(record.get("outcome"))
    validation = runtime.get("validation", {}) if isinstance(runtime, Mapping) else record.get("validation", {})
    if isinstance(validation, Mapping):
        try:
            reward = float(validation.get("reward"))
        except (TypeError, ValueError):
            reward = None
        if reward is not None:
            return "success" if reward > 0 else "failure"
    return "unknown"


def task_instruction(record: Mapping[str, Any]) -> str:
    runtime = record.get("runtime_summary")
    if isinstance(runtime, Mapping):
        value = runtime.get("task_instruction")
        if value:
            return str(value)
    return str(record.get("task_instruction") or record.get("task") or "")


merged_records = load_json_array(merged_path)
merged_by_domain = {
    str(record.get("domain") or record.get("group_id") or record.get("cluster_id")): record
    for record in merged_records
}
domain_rubrics = {
    domain: normalize_rubrics(record.get("rubrics"))
    for domain, record in merged_by_domain.items()
}

pair_records = load_pair_records(pair_root)
pair_adapter = []
for pair in pair_records:
    pair_id = str(pair.get("pair_id") or pair.get("__record_id__") or "")
    domain = str(pair.get("domain") or pair_id.split("/", 1)[0])
    rubrics = domain_rubrics.get(domain, [])
    if not pair_id or not rubrics:
        continue
    pair_adapter.append(
        {
            "__record_id__": pair_id,
            "pair_id": pair_id,
            "domain": domain,
            "jobname": pair.get("jobname"),
            "source_group_id": domain,
            "rubrics": rubrics,
        }
    )

known_domains = set(domain_rubrics)
cluster_members_by_domain: dict[str, list[dict[str, Any]]] = {domain: [] for domain in known_domains}
for path in sorted(cache_root.rglob("*.json")):
    domain = infer_domain_from_path(path, known_domains)
    if not domain:
        continue
    try:
        payload = load_json(path)
    except Exception:
        continue
    records = payload if isinstance(payload, list) else [payload]
    for record in records:
        if not isinstance(record, Mapping):
            continue
        record_id = str(record.get("__record_id__") or "")
        if not record_id:
            continue
        cluster_members_by_domain[domain].append(
            {
                "__record_id__": record_id,
                "cluster_id": domain,
                "task_instruction": task_instruction(record),
                "source_path": path.as_posix(),
                "relative_source_path": path.as_posix(),
                "outcome": infer_outcome(record),
                "metadata": {"benchmark": domain},
            }
        )

domain_cluster_records = []
domain_rubric_adapter = []
for domain, rubrics in sorted(domain_rubrics.items()):
    members = cluster_members_by_domain.get(domain, [])
    for member in members:
        member["cluster_size"] = len(members)
        domain_cluster_records.append(member)
    domain_rubric_adapter.append(
        {
            "__record_id__": domain,
            "cluster_id": domain,
            "cluster_size": len(members),
            "source_record_ids": [member["__record_id__"] for member in members],
            "source_group_id": domain,
            "rubrics": rubrics,
        }
    )

for path, payload in (
    (pair_output, pair_adapter),
    (domain_rubric_output, domain_rubric_adapter),
    (domain_cluster_output, domain_cluster_records),
):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

print(f"pair_adapter: {pair_output} records={len(pair_adapter)}")
print(f"domain_rubric_adapter: {domain_rubric_output} records={len(domain_rubric_adapter)}")
print(f"domain_cluster_adapter: {domain_cluster_output} records={len(domain_cluster_records)}")
for domain in sorted(known_domains):
    print(
        f"domain={domain} "
        f"rubrics={len(domain_rubrics.get(domain, []))} "
        f"pairs={sum(1 for item in pair_adapter if item.get('domain') == domain)} "
        f"cache_records={len(cluster_members_by_domain.get(domain, []))}"
    )
PY

if [[ "${SKIP_PAIR_EVAL:-0}" != "1" ]]; then
  pair_args=(
    --pair-rubrics "$PAIR_ADAPTER_FILE"
    --pairs "$PAIR_CACHE_ROOT"
    --output-dir "$PAIR_EVAL_OUTPUT_DIR"
    --model "${RUBRIC_EVAL_MODEL:-${RUBRIC_MODEL:-gpt-5.4-mini}}"
    --base-url "${OPENAI_BASE_URL:-https://api.gpt.ge/v1/}"
    --concurrency "${PAIR_EVAL_CONCURRENCY:-1}"
    --max-tokens "${PAIR_EVAL_MAX_TOKENS:-4096}"
  )
  if [[ "${PAIR_EVAL_NO_TRUNCATE:-1}" == "1" ]]; then
    pair_args+=(--no-truncate)
  else
    pair_args+=(--max-chars-per-response "${PAIR_EVAL_MAX_CHARS_PER_RESPONSE:-80000}")
  fi
  if [[ -n "${PAIR_IDS:-}" ]]; then
    pair_args+=(--pair-ids "$PAIR_IDS")
  fi
  if [[ -n "${PAIR_EVAL_MAX_PAIRS:-}" ]]; then
    pair_args+=(--max-pairs "$PAIR_EVAL_MAX_PAIRS")
  fi
  if [[ "${REFRESH:-0}" == "1" ]]; then
    pair_args+=(--refresh)
  fi
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    pair_args+=(--dry-run)
  fi
  python -m pair_rubric_evaluation.cli "${pair_args[@]}"
fi

if [[ "${SKIP_DOMAIN_EVAL:-0}" != "1" ]]; then
  domain_args=(
    --rubrics "$DOMAIN_RUBRIC_ADAPTER_FILE"
    --clusters "$DOMAIN_CLUSTER_ADAPTER_FILE"
    --output-dir "$DOMAIN_EVAL_OUTPUT_DIR"
    --model "${RUBRIC_EVAL_MODEL:-${RUBRIC_MODEL:-gpt-5.4-mini}}"
    --base-url "${OPENAI_BASE_URL:-https://api.gpt.ge/v1/}"
    --concurrency "${DOMAIN_EVAL_CONCURRENCY:-1}"
    --max-records-per-cluster "${DOMAIN_EVAL_MAX_RECORDS_PER_CLUSTER:-0}"
    --max-chars-per-record "${DOMAIN_EVAL_MAX_CHARS_PER_RECORD:-6000}"
    --max-tokens "${DOMAIN_EVAL_MAX_TOKENS:-2048}"
  )
  if [[ -n "${DOMAIN_IDS:-}" ]]; then
    domain_args+=(--cluster-ids "$DOMAIN_IDS")
  fi
  if [[ "${REFRESH:-0}" == "1" ]]; then
    domain_args+=(--refresh)
  fi
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    domain_args+=(--dry-run)
  fi
  python -m rubric_evaluation.cli "${domain_args[@]}"
fi
