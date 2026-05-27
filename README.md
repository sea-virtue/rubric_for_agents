# Rubric For Agents

This project currently focuses on the first three independent modules for
mining rubrics from agent trajectories:

```text
raw trajectories
  -> parsed cache
  -> task embedding clusters
  -> cluster-level rubric extraction
```

The old full pipeline has been removed. The active flow now reads explicit
intermediate files under `data/` so each module can be inspected and rerun
independently.

## Layout

```text
src/
  parse_cache/         # raw traces -> parsed/cache summaries
  task_clustering/     # task_instruction embeddings -> data/cluster
  rubric_extraction/   # data/cluster + parsed cache -> data/rubric
  rubric_miner/        # shared parser/LLM/text utilities

scripts/
  parse_traces_to_cache.sh
  cluster_cached_tasks.sh
  extract_rubrics_from_clusters.sh

local_inference/
  start_vllm_qwen.sh
  start_vllm_qwen3_embedding.sh
  check_vllm.sh
  check_embedding.sh

docs/
  parse_traces_to_cache.md
  cluster_cached_tasks.md
  extract_rubrics_from_clusters.md
```

`local_inference/` is only for local model serving. Python implementation code
lives under `src/`. Shell scripts under `scripts/` are thin entrypoints.

## Install

Use Python 3.10+ for project code when possible.

```bash
pip install -r requirements.txt
```

For local vLLM/HF model servers:

```bash
pip install -r local_inference/requirements.txt
```

On shared GPU servers, vLLM and PyTorch are CUDA-sensitive. Prefer the vLLM
build that matches the server's CUDA/PyTorch environment.

## Run

Parse raw trajectories into cache:

```bash
./scripts/parse_traces_to_cache.sh
```

Start the embedding server and cluster cached tasks:

```bash
bash local_inference/start_vllm_qwen3_embedding.sh

./scripts/cluster_cached_tasks.sh \
  --embedding-base-url http://127.0.0.1:8001/v1 \
  --embedding-model qwen3-embedding-8b
```

Start the chat model and extract rubrics from clusters:

```bash
bash local_inference/start_vllm_qwen.sh

./scripts/extract_rubrics_from_clusters.sh \
  --model qwen3-4b-instruct-2507 \
  --base-url http://127.0.0.1:18000/v1 \
  --max-records-per-cluster 12 \
  --max-chars-per-record 6000
```

## Outputs

Parsed cache:

```text
data/cache_data/...
```

Task clusters:

```text
data/cluster/task_clusters.json
data/cluster/task_cluster_summary.json
data/cluster/task_cluster_config.json
data/cluster/task_instruction_embeddings.json
```

Rubrics:

```text
data/rubric/cluster_rubrics.json
data/rubric/rubric_prompts.json
data/rubric/rubric_extraction_config.json
```
