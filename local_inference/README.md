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

Option C: vLLM for chat + vLLM Qwen3-Embedding for embeddings
  miner -> vLLM chat server /v1/chat/completions
  miner -> vLLM embedding server /v1/embeddings
```

vLLM can serve embedding models if the model itself is an embedding model, but a
Qwen3 Instruct chat model is not an embedding model. For Qwen3-Instruct, keep
`embedding_model` empty or run the included HF embedding server with a dedicated
embedding model.

## vLLM Qwen3 Embedding Server

Download Qwen3-Embedding-8B to the ignored `local_inference/models/` directory:

```bash
pip install huggingface-hub
chmod +x local_inference/download_qwen3_embedding.sh
bash local_inference/download_qwen3_embedding.sh
```

Start an OpenAI-compatible vLLM embedding server on port 8001:

```bash
chmod +x local_inference/start_vllm_qwen3_embedding.sh
bash local_inference/start_vllm_qwen3_embedding.sh
```

The script uses vLLM's pooling/embed mode:

```bash
--runner pooling --convert embed
```

Older vLLM examples may show `--task embed`; your installed vLLM may reject
that flag if it has switched to the newer runner/convert arguments.

If the chat server already uses all GPUs, start chat and embedding servers with
separate `CUDA_VISIBLE_DEVICES` values. For example:

```bash
CUDA_VISIBLE_DEVICES=0,1 TENSOR_PARALLEL_SIZE=2 bash local_inference/start_vllm_qwen.sh
CUDA_VISIBLE_DEVICES=2,3 TENSOR_PARALLEL_SIZE=2 bash local_inference/start_vllm_qwen3_embedding.sh
```

Check the embedding endpoint:

```bash
chmod +x local_inference/check_embedding.sh
bash local_inference/check_embedding.sh
```

## vLLM Chat Server

Install vLLM in a suitable CUDA environment:

```bash
pip install vllm
```

Download the default Qwen3 4B Instruct 2507 model to the ignored
`local_inference/models/` directory:

```bash
pip install huggingface-hub
chmod +x local_inference/download_hf_model.sh
bash local_inference/download_hf_model.sh
```

Start vLLM with the default local model path:

```bash
chmod +x local_inference/start_vllm_qwen.sh
bash local_inference/start_vllm_qwen.sh
```

Then point miner at the server:

```bash
export OPENAI_API_KEY="local"
python src/miner.py --config configs/local_qwen3_vllm.json
```

To use a different model, edit `MODEL` and `SERVED_MODEL_NAME` near the top of
`start_vllm_qwen.sh`, or override them once:

```bash
MODEL="local_inference/models/qwen3-14b-instruct" \
SERVED_MODEL_NAME="qwen3-14b-instruct" \
bash local_inference/start_vllm_qwen.sh
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
CHAT_MODEL="Qwen/Qwen3-4B-Instruct-2507" \
SERVED_CHAT_MODEL_NAME="qwen3-4b-instruct-2507" \
EMBEDDING_MODEL="" \
PORT=8000 \
local_inference/start_hf_openai_server.sh
```

The HF chat server is meant as a simple fallback. For production throughput,
prefer vLLM for chat generation.

## Miner Config

Use `configs/local_qwen3_vllm.json` for vLLM-only chat.

Use `configs/local_qwen3_vllm_qwen3_embedding_full.json` for the current
recommended setup: vLLM chat on port 8000 plus Qwen3-Embedding on port 8001 with
DBSCAN clustering.

Use `configs/local_qwen3_vllm_balanced_debug.json` for a smaller balanced debug
run.
