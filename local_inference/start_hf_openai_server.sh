#!/usr/bin/env bash
set -euo pipefail

CHAT_MODEL="${CHAT_MODEL:-}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-BAAI/bge-m3}"
SERVED_CHAT_MODEL_NAME="${SERVED_CHAT_MODEL_NAME:-}"
SERVED_EMBEDDING_MODEL_NAME="${SERVED_EMBEDDING_MODEL_NAME:-}"
DEVICE="${DEVICE:-auto}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8001}"

args=(
  "local_inference/hf_openai_server.py"
  "--host" "${HOST}"
  "--port" "${PORT}"
  "--device" "${DEVICE}"
)

if [[ -n "${CHAT_MODEL}" ]]; then
  args+=("--chat-model" "${CHAT_MODEL}")
fi
if [[ -n "${EMBEDDING_MODEL}" ]]; then
  args+=("--embedding-model" "${EMBEDDING_MODEL}")
fi
if [[ -n "${SERVED_CHAT_MODEL_NAME}" ]]; then
  args+=("--served-chat-model-name" "${SERVED_CHAT_MODEL_NAME}")
fi
if [[ -n "${SERVED_EMBEDDING_MODEL_NAME}" ]]; then
  args+=("--served-embedding-model-name" "${SERVED_EMBEDDING_MODEL_NAME}")
fi

python "${args[@]}"
