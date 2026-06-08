#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-local}"

default_args=(
  --bank "${PAIR_BANK_FILE:-outputs/pair/bank_v1/rubric_bank_v1.json}"
  --pairs "${PAIR_CACHE_ROOT:-data/cache_pair_data}"
  --pair-embedding-cache "${PAIR_EMBEDDING_CACHE:-outputs/pair/cluster_pair_v3_looser/pair_task_embeddings.json}"
  --output-dir "${PAIR_BANK_EVAL_OUTPUT_DIR:-outputs/pair/bank_v1_eval}"
  --model "${RUBRIC_EVAL_MODEL:-${RUBRIC_MODEL:-qwen3-4b-instruct-2507}}"
  --base-url "${OPENAI_BASE_URL:-http://127.0.0.1:28000/v1}"
  --embedding-model "${TRACE_EMBEDDING_MODEL:-qwen3-embedding-8b}"
  --embedding-base-url "${EMBEDDING_BASE_URL:-http://127.0.0.1:8001/v1}"
  --concurrency "${CONCURRENCY:-1}"
  --softmax-temperature "${SOFTMAX_TEMPERATURE:-0.05}"
  --rubric-batch-size "${RUBRIC_BATCH_SIZE:-32}"
  --score-top-k "${SCORE_TOP_K:-32}"
  --max-tokens "${MAX_TOKENS:-4096}"
)

if [[ "${PAIR_EVAL_NO_TRUNCATE:-1}" == "1" ]]; then
  default_args+=(--no-truncate)
else
  default_args+=(--max-chars-per-response "${PAIR_EVAL_MAX_CHARS_PER_RESPONSE:-80000}")
fi
if [[ -n "${PAIR_IDS:-}" ]]; then
  default_args+=(--pair-ids "$PAIR_IDS")
fi
if [[ -n "${MAX_PAIRS:-}" ]]; then
  default_args+=(--max-pairs "$MAX_PAIRS")
fi
if [[ "${REFRESH:-0}" == "1" ]]; then
  default_args+=(--refresh)
fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  default_args+=(--dry-run)
fi

python -m pair_soft_bank.cli evaluate-bank-v1 "${default_args[@]}" "$@"
