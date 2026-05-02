#!/usr/bin/env bash
set -euo pipefail

export OPENAI_API_KEY="${OPENAI_API_KEY:-local}"
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,0.0.0.0,::1}"
export no_proxy="${no_proxy:-localhost,127.0.0.1,0.0.0.0,::1}"

CONFIG="${CONFIG:-configs/local_qwen3_vllm_full.json}"
OUT_DIR="${OUT_DIR:-outputs/local_qwen3_vllm_full}"

mkdir -p "${OUT_DIR}"

echo "Running full Rubric Miner"
echo "  CONFIG=${CONFIG}"
echo "  OUT_DIR=${OUT_DIR}"
echo "  OPENAI_API_KEY=${OPENAI_API_KEY}"
echo "  NO_PROXY=${NO_PROXY}"

python src/miner.py \
  --config "${CONFIG}" \
  --out-dir "${OUT_DIR}"
