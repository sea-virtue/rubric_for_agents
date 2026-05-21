#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONDONTWRITEBYTECODE=1

python scripts/parse_traces_to_cache.py \
  --input data/agent-reward-bench/trajectories/cleaned \
  --input-format json \
  --preserve-tree \
  --output-root data/cache_data \
  --preview-records 5