#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}src"
export OPENAI_API_KEY="${OPENAI_API_KEY:-local}"

python -m task_clustering.cli "$@"
