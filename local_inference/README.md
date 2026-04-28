# Local Inference Helpers

This folder contains helper scripts for running local models behind
OpenAI-compatible APIs so `rubric_miner` can call them without changing the
pipeline code.

Two common setups are supported:

```text
Option A: vLLM for chat, no embeddings
  miner -> vLLM /v1/chat/completions
  miner -> local fallback clustering

Option B: vLLM for chat + HF/SentenceTransformers for embeddings
  miner -> vLLM /v1/chat/completions
  miner -> HF server /v1/embeddings
```

vLLM can serve embedding models if the model itself is an embedding model, but a
Qwen3 Instruct chat model is not an embedding model. For Qwen3-Instruct, keep
`embedding_model` empty or run the included HF embedding server with a dedicated
embedding model.

## vLLM Chat Server

Install vLLM in a suitable CUDA environment:

```bash
pip install vllm
```

You can either let vLLM download the model automatically into the Hugging Face
cache, or pre-download it into this project's ignored `models/` directory.

Automatic download:

```bash
MODEL="Qwen/Qwen3-7B-Instruct" \
SERVED_MODEL_NAME="qwen3-7b-instruct" \
PORT=8000 \
local_inference/start_vllm_qwen.sh
```

Pre-download to `models/`:

```bash
pip install huggingface-hub
chmod +x local_inference/download_hf_model.sh

MODEL_ID="Qwen/Qwen3-7B-Instruct" \
LOCAL_DIR="models/qwen3-7b-instruct" \
local_inference/download_hf_model.sh
```

Then run vLLM from the local path:

```bash
MODEL="models/qwen3-7b-instruct" \
SERVED_MODEL_NAME="qwen3-7b-instruct" \
PORT=8000 \
local_inference/start_vllm_qwen.sh
```

Start Qwen3 7B Instruct:

```bash
chmod +x local_inference/start_vllm_qwen.sh
MODEL="Qwen/qwen3-8b-instruct" \
SERVED_MODEL_NAME="qwen3-8b-instruct" \
PORT=8000 \
local_inference/start_vllm_qwen.sh
```

Start Qwen3 14B Instruct:

```bash
MODEL="Qwen/Qwen3-14B-Instruct" \
SERVED_MODEL_NAME="qwen3-14b-instruct" \
PORT=8000 \
local_inference/start_vllm_qwen.sh
```

Then point miner at the server:

```bash
export OPENAI_API_KEY="local"
python src/miner.py --config configs/local_qwen3_vllm.json
```

## HF OpenAI-Compatible Server

Install optional dependencies:

```bash
pip install -r local_inference/requirements.txt
```

Run a local embedding server:

```bash
chmod +x local_inference/start_hf_openai_server.sh
EMBEDDING_MODEL="BAAI/bge-m3" \
SERVED_EMBEDDING_MODEL_NAME="bge-m3" \
PORT=8001 \
local_inference/start_hf_openai_server.sh
```

Run a small direct HF chat server too:

```bash
CHAT_MODEL="Qwen/qwen3-8b-instruct" \
SERVED_CHAT_MODEL_NAME="qwen3-8b-instruct" \
EMBEDDING_MODEL="" \
PORT=8000 \
local_inference/start_hf_openai_server.sh
```

The HF chat server is meant as a simple fallback. For production throughput,
prefer vLLM for chat generation.

## Miner Config

Use `configs/local_qwen3_vllm.json` for vLLM-only chat.

Use `configs/local_qwen3_vllm_hf_embedding.json` when chat and embeddings are
served on different local ports.
