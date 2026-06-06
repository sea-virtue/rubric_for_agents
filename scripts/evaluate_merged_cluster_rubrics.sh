#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-local}"

MERGED_RUBRICS_FILE="${MERGED_RUBRICS_FILE:-data/rubric_merge/cluster_merged_rubrics.json}"
CLUSTERS_FILE="${CLUSTERS_FILE:-data/cluster/task_clusters.json}"
PAIR_CACHE_ROOT="${PAIR_CACHE_ROOT:-data/cache_pair_data}"
ADAPTER_OUTPUT_DIR="${ADAPTER_OUTPUT_DIR:-data/rubric_merge_eval/cluster_adapters}"
PAIR_EVAL_OUTPUT_DIR="${PAIR_EVAL_OUTPUT_DIR:-data/rubric_merge_eval/cluster_pair_eval}"

mkdir -p "$ADAPTER_OUTPUT_DIR"

PAIR_ADAPTER_FILE="${ADAPTER_OUTPUT_DIR}/cluster_merged_as_pair_rubrics.json"

python - "$MERGED_RUBRICS_FILE" "$CLUSTERS_FILE" "$PAIR_ADAPTER_FILE" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

merged_path = Path(sys.argv[1])
clusters_path = Path(sys.argv[2])
pair_output = Path(sys.argv[3])


def load_json_array(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array")
    return [item for item in data if isinstance(item, dict)]


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
        output.append(
            {
                "dimension": theme[:160],
                "criterion": criterion,
                "positive_evidence": tips or text_list(guide.get("checks")) or text_list(guide.get("what_to_extract")),
                "negative_evidence": text_list(item.get("negative_evidence")),
                "severity": str(item.get("severity") or "medium").lower(),
                "rationale": str(item.get("rationale") or item.get("reason") or ""),
                "verification_guide": dict(guide),
                "source_pair_ids": text_list(item.get("source_pair_ids")),
            }
        )
    return output


merged_records = load_json_array(merged_path)
cluster_records = load_json_array(clusters_path)
rubrics_by_cluster = {
    str(record.get("cluster_id") or record.get("group_id") or record.get("__record_id__")): normalize_rubrics(record.get("rubrics"))
    for record in merged_records
    if record.get("rubrics")
}

adapter = []
missing_clusters = set()
for member in cluster_records:
    pair_id = str(member.get("pair_id") or member.get("__record_id__") or "")
    cluster_id = str(member.get("cluster_id") or "")
    if not pair_id or not cluster_id:
        continue
    rubrics = rubrics_by_cluster.get(cluster_id)
    if not rubrics:
        missing_clusters.add(cluster_id)
        continue
    adapter.append(
        {
            "__record_id__": pair_id,
            "pair_id": pair_id,
            "domain": member.get("domain"),
            "jobname": member.get("jobname"),
            "cluster_id": cluster_id,
            "source_group_id": cluster_id,
            "rubrics": rubrics,
        }
    )

pair_output.parent.mkdir(parents=True, exist_ok=True)
pair_output.write_text(json.dumps(adapter, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"pair_adapter: {pair_output} records={len(adapter)}")
print(f"merged_clusters={len(rubrics_by_cluster)} missing_clusters={len(missing_clusters)}")
if missing_clusters:
    print("missing_cluster_ids=" + ",".join(sorted(missing_clusters)[:20]))
PY

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
