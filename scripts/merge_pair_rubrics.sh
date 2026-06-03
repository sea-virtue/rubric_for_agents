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
  --num-categories "${MERGE_NUM_CATEGORIES:-8}"
  --max-rubrics-per-group "${MAX_RUBRICS_PER_GROUP:-180}"
  --max-chars-per-rubric "${MAX_CHARS_PER_RUBRIC:-1200}"
  --max-tokens "${MAX_TOKENS:-4096}"
  --concurrency "${CONCURRENCY:-1}"
)

python -m rubric_merge.cli "${default_args[@]}" "$@"
