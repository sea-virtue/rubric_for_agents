#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-8B-Instruct}"
LOCAL_DIR="${LOCAL_DIR:-models/qwen3-8b-instruct}"
REVISION="${REVISION:-main}"

mkdir -p "$(dirname "${LOCAL_DIR}")"

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
