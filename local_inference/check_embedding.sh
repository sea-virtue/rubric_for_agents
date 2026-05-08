#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8001/v1}"
MODEL="${MODEL:-qwen3-embedding-8b}"
API_KEY="${OPENAI_API_KEY:-local}"

export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,0.0.0.0,::1}"
export no_proxy="${no_proxy:-localhost,127.0.0.1,0.0.0.0,::1}"

echo "Checking models at ${BASE_URL}/models"
curl --noproxy "*" -sS "${BASE_URL}/models"
echo

echo "Checking embeddings for ${MODEL}"
response="$(curl --noproxy "*" -sS "${BASE_URL}/embeddings" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${API_KEY}" \
  -d "{
    \"model\": \"${MODEL}\",
    \"input\": \"Represent this short agent trace for clustering.\"
  }")"

RESPONSE="${response}" python - <<'PY'
import json
import os

data = json.loads(os.environ["RESPONSE"])
vec = data["data"][0]["embedding"]
print({"model": data.get("model"), "dimension": len(vec), "first3": vec[:3]})
PY
