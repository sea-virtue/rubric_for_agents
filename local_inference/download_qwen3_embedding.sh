#!/usr/bin/env bash
set -euo pipefail

export MODEL_ID="${MODEL_ID:-Qwen/Qwen3-Embedding-8B}"
export LOCAL_DIR="${LOCAL_DIR:-local_inference/models/qwen3-embedding-8b}"
export REVISION="${REVISION:-main}"
export HF_HOME="${HF_HOME:-$(pwd)/local_inference/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"

mkdir -p "$(dirname "${LOCAL_DIR}")"
mkdir -p "${HF_HUB_CACHE}" "${TRANSFORMERS_CACHE}"

python - <<'PY'
import os
from huggingface_hub import snapshot_download

model_id = os.environ["MODEL_ID"]
local_dir = os.environ["LOCAL_DIR"]
revision = os.environ.get("REVISION", "main")
token = os.environ.get("HF_TOKEN") or None

print(f"Downloading {model_id} -> {local_dir} (revision={revision})")
snapshot_download(
    repo_id=model_id,
    revision=revision,
    local_dir=local_dir,
    token=token,
    local_dir_use_symlinks=False,
    resume_download=True,
)
print("Done.")
PY
