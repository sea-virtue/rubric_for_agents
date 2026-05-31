#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

default_args=(
  --pair-rubrics "${PAIR_RUBRICS_FILE:-data/pair_rubric/pair_rubrics.json}"
  --pairs "${PAIR_CACHE_ROOT:-data/cache_pair_data}"
  --output-dir "${PAIR_RUBRIC_EVAL_OUTPUT_DIR:-data/pair_rubric_eval}"
  --model "${RUBRIC_EVAL_MODEL:-${RUBRIC_MODEL:-qwen3-4b-instruct-2507}}"
)

python -m pair_rubric_evaluation.cli "${default_args[@]}" "$@"
