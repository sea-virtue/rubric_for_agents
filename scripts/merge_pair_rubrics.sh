#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-local}"

default_args=(
  --pair-rubrics "${PAIR_RUBRIC_FILE:-data/pair_rubric/pair_rubrics.json}"
  --pairs "${PAIR_CACHE_ROOT:-data/cache_pair_data}"
  --output-dir "${RUBRIC_MERGE_OUTPUT_DIR:-data/rubric_merge}"
  --model "${RUBRIC_MODEL:-gpt-5.4-mini}"
  --base-url "${OPENAI_BASE_URL:-https://api.gpt.ge/v1/}"
  --selection-method "${RUBRIC_SELECTION_METHOD:-mcr}"
  --embedding-model "${RUBRIC_EMBEDDING_MODEL:-qwen3-embedding-8b}"
  --embedding-base-url "${OPENAI_EMBEDDING_BASE_URL:-${EMBEDDING_BASE_URL:-http://127.0.0.1:8001/v1}}"
  --embedding-batch-size "${EMBEDDING_BATCH_SIZE:-16}"
  --max-selected-rubrics "${MAX_SELECTED_RUBRICS:-80}"
  --mcr-batch-size "${MCR_BATCH_SIZE:-5}"
  --mcr-eps "${MCR_EPS:-0.1}"
  --mcr-min-increment-threshold "${MCR_MIN_INCREMENT_THRESHOLD:-0.001}"
  --mcr-patience "${MCR_PATIENCE:-3}"
  --num-categories "${MERGE_NUM_CATEGORIES:-8}"
  --max-rubrics-per-group "${MAX_RUBRICS_PER_GROUP:-180}"
  --max-chars-per-rubric "${MAX_CHARS_PER_RUBRIC:-1200}"
  --max-tokens "${MAX_TOKENS:-4096}"
  --concurrency "${CONCURRENCY:-1}"
)

python -m rubric_merge.cli "${default_args[@]}" "$@"
