#!/usr/bin/env bash
set -euo pipefail

# OpenAI-compatible embedding server for Qwen3-Embedding-8B.
# Keep this on a different port from the chat vLLM server.
MODEL="${MODEL:-local_inference/models/qwen3-embedding-8b}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-embedding-8b}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8001}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
HF_HOME="${HF_HOME:-$(pwd)/local_inference/hf_cache}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"

detect_visible_gpu_count() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    local visible="${CUDA_VISIBLE_DEVICES// /}"
    if [[ -z "${visible}" || "${visible}" == "-1" ]]; then
      echo 0
      return
    fi
    awk -F',' '{print NF}' <<< "${visible}"
    return
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' '
    return
  fi

  echo 1
}

TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-$(detect_visible_gpu_count)}"
if [[ "${TENSOR_PARALLEL_SIZE}" -lt 1 ]]; then
  echo "No visible GPU was detected. Set CUDA_VISIBLE_DEVICES or TENSOR_PARALLEL_SIZE."
  exit 1
fi

export HF_HOME
export HF_HUB_CACHE
export TRANSFORMERS_CACHE
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,0.0.0.0,::1}"
export no_proxy="${no_proxy:-localhost,127.0.0.1,0.0.0.0,::1}"
mkdir -p "${HF_HUB_CACHE}" "${TRANSFORMERS_CACHE}"

if [[ ! -e "${MODEL}" ]]; then
  echo "Model path '${MODEL}' does not exist."
  echo "Download it first:"
  echo "  bash local_inference/download_qwen3_embedding.sh"
  echo
  echo "Or allow vLLM to download automatically by setting:"
  echo "  MODEL='Qwen/Qwen3-Embedding-8B' bash local_inference/start_vllm_qwen3_embedding.sh"
  exit 1
fi

echo "Starting vLLM embedding server:"
echo "  MODEL=${MODEL}"
echo "  SERVED_MODEL_NAME=${SERVED_MODEL_NAME}"
echo "  PORT=${PORT}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "  TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE}"
echo "  HF_HOME=${HF_HOME}"
echo "  NO_PROXY=${NO_PROXY}"

python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --runner pooling \
  --convert embed
