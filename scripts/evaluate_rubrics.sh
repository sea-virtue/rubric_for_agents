#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}src"
export OPENAI_API_KEY="${OPENAI_API_KEY:-local}"

default_args=(
  --rubrics "${RUBRICS_FILE:-data/rubric/cluster_rubrics.json}"
  --clusters "${CLUSTERS_FILE:-data/cluster/task_clusters.json}"
  --output-dir "${RUBRIC_EVAL_OUTPUT_DIR:-data/rubric_eval}"
  --model "${RUBRIC_EVAL_MODEL:-${RUBRIC_MODEL:-qwen3-4b-instruct-2507}}"
  --base-url "${OPENAI_BASE_URL:-http://127.0.0.1:18000/v1}"
  --max-records-per-cluster "${EVAL_MAX_RECORDS_PER_CLUSTER:-4}"
  --max-chars-per-record "${EVAL_MAX_CHARS_PER_RECORD:-6000}"
  --max-tokens "${EVAL_MAX_TOKENS:-1024}"
  --concurrency "${EVAL_CONCURRENCY:-1}"
)

python -m rubric_evaluation.cli "${default_args[@]}" "$@"
