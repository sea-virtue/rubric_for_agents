#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"
MODEL="${MODEL:-qwen3-4b-instruct-2507}"
API_KEY="${OPENAI_API_KEY:-local}"

export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,0.0.0.0,::1}"
export no_proxy="${no_proxy:-localhost,127.0.0.1,0.0.0.0,::1}"

echo "Checking models at ${BASE_URL}/models"
curl --noproxy "*" -sS "${BASE_URL}/models"
echo

echo "Checking chat completion for ${MODEL}"
curl --noproxy "*" -sS "${BASE_URL}/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${API_KEY}" \
  -d "{
    \"model\": \"${MODEL}\",
    \"messages\": [{\"role\": \"user\", \"content\": \"Return only: []\"}],
    \"max_tokens\": 16,
    \"temperature\": 0
  }"
echo
