#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}src"

if [ "$#" -eq 0 ]; then
  set -- \
    --input data/agent-reward-bench/trajectories/cleaned \
    --input-format json \
    --preserve-tree \
    --output-root data/cache_data \
    --preview-records 5
fi

python -m parse_cache.cli "$@"
