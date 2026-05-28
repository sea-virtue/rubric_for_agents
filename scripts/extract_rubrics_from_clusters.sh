#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}src"
export OPENAI_API_KEY="${OPENAI_API_KEY:-local}"

default_args=(
  --model "${RUBRIC_MODEL:-qwen3-4b-instruct-2507}"
  --base-url "${OPENAI_BASE_URL:-http://127.0.0.1:28000/v1}"
  --max-records-per-cluster "${MAX_RECORDS_PER_CLUSTER:-12}"
  --max-chars-per-record "${MAX_CHARS_PER_RECORD:-6000}"
  --max-tokens "${MAX_TOKENS:-2048}"
  --concurrency "${CONCURRENCY:-1}"
)

python -m rubric_extraction.cli "${default_args[@]}" "$@"
