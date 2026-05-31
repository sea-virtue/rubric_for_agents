# Rubric For Agents

This project currently focuses on independent, inspectable modules for mining
rubrics from agent trajectories:

```text
raw trajectories
  -> parsed cache
  -> positive/negative cache pairs
  -> pair-level rubric extraction
  -> task embedding clusters
  -> cluster-level rubric extraction
  -> rubric quality evaluation
```

The old full pipeline has been removed. The active flow now reads explicit
intermediate files under `data/` so each module can be inspected and rerun
independently.

## Layout

```text
src/
  parse_cache/         # raw traces -> parsed/cache summaries
  pair_cache/          # parsed/cache summaries -> positive/negative pair cache
  pair_rubric_extraction/ # pair cache -> pair-level candidate rubrics
  task_clustering/     # task_instruction embeddings -> data/cluster
  rubric_extraction/   # data/cluster + parsed cache -> data/rubric
  rubric_evaluation/   # data/rubric + clusters/cache -> data/rubric_eval
  rubric_miner/        # shared parser/LLM/text utilities

scripts/
  parse_traces_to_cache.sh
  build_cache_pairs.sh
  extract_pair_rubrics.sh
  cluster_cached_tasks.sh
  extract_rubrics_from_clusters.sh
  evaluate_rubrics.sh

local_inference/
  start_vllm_qwen.sh
  start_vllm_qwen3_embedding.sh
  check_vllm.sh
  check_embedding.sh

docs/
  parse_traces_to_cache.md
  build_cache_pairs.md
  extract_pair_rubrics.md
  cluster_cached_tasks.md
  extract_rubrics_from_clusters.md
  evaluate_rubrics.md
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

Build positive/negative pair cache data:

```bash
./scripts/build_cache_pairs.sh
```

Preview or extract pair-level rubrics:

```bash
./scripts/extract_pair_rubrics.sh --dry-run --max-pairs 2

./scripts/extract_pair_rubrics.sh \
  --base-url http://127.0.0.1:28000/v1 \
  --concurrency 2
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
  --base-url http://127.0.0.1:28000/v1 \
  --max-records-per-cluster 12 \
  --max-chars-per-record 6000
```

## Outputs

Parsed cache:

```text
data/cache_data/...
```

Positive/negative pair cache:

```text
data/cache_pair_data/pair_index.json
data/cache_pair_data/pair_report.json
data/cache_pair_data/pair_summary.json
data/cache_pair_data/<domain>/<jobname>/pair.json
data/cache_pair_data/<domain>/<jobname>/<jobnamewithmodelname>.json
```

Pair-level rubrics:

```text
data/pair_rubric/pair_rubrics.json
data/pair_rubric/pair_rubric_prompts.json
data/pair_rubric/pair_rubric_extraction_config.json
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

Rubric evaluation:

```text
data/rubric_eval/rubric_static_audit.json
data/rubric_eval/trajectory_rubric_scores.json
data/rubric_eval/rubric_eval_summary.json
```
