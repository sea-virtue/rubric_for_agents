#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-local}"

default_args=(
  --pair-rubrics "${PAIR_RUBRICS_FILE:-outputs/pair/pair_rubric/pair_rubrics.json}"
  --pair-embedding-cache "${PAIR_EMBEDDING_CACHE:-outputs/pair/cluster_pair_v3_looser/pair_task_embeddings.json}"
  --output-dir "${PAIR_BANK_OUTPUT_DIR:-outputs/pair/bank_v1}"
  --embedding-model "${TRACE_EMBEDDING_MODEL:-qwen3-embedding-8b}"
  --embedding-base-url "${EMBEDDING_BASE_URL:-http://127.0.0.1:8001/v1}"
  --embedding-batch-size "${EMBEDDING_BATCH_SIZE:-16}"
)

if [[ "${REFRESH_EMBEDDINGS:-0}" == "1" ]]; then
  default_args+=(--refresh-embeddings)
fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  default_args+=(--dry-run)
fi

python -m pair_soft_bank.cli build-bank-v1 "${default_args[@]}" "$@"

