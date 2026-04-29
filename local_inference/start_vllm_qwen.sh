#!/usr/bin/env bash
set -euo pipefail

# Edit these defaults for your server. You can still override any of them with
# environment variables, e.g. MODEL=/path/to/model bash local_inference/start_vllm_qwen.sh
MODEL="${MODEL:-local_inference/models/qwen3-4b-instruct-2507}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-4b-instruct-2507}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
HF_HOME="${HF_HOME:-$(pwd)/local_inference/hf_cache}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"

export HF_HOME
export HF_HUB_CACHE
export TRANSFORMERS_CACHE
mkdir -p "${HF_HUB_CACHE}" "${TRANSFORMERS_CACHE}"

if [[ ! -e "${MODEL}" ]]; then
  echo "Model path '${MODEL}' does not exist."
  echo "Download it first, for example:"
  echo "  bash local_inference/download_hf_model.sh"
  echo
  echo "Or override MODEL with a Hugging Face repo id if you want vLLM to download automatically:"
  echo "  MODEL='Qwen/Qwen3-4B-Instruct-2507' bash local_inference/start_vllm_qwen.sh"
  exit 1
fi

echo "Starting vLLM:"
echo "  MODEL=${MODEL}"
echo "  SERVED_MODEL_NAME=${SERVED_MODEL_NAME}"
echo "  PORT=${PORT}"
echo "  HF_HOME=${HF_HOME}"

python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
