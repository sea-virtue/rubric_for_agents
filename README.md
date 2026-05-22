# Rubric Miner

Rubric Miner is a coarse-to-fine pipeline for mining evaluation rubrics from
agent execution traces. It accepts messy trace datasets, normalizes them into a
common internal format, clusters similar trajectories, mines rubrics with
multiple LLMs, conservatively merges them, refines them with success/failure
contrasts, and exports a rubric library with metadata.

This README describes only this project. The local `RubricHub-main/` directory
is not part of the project package.

## Pipeline

```text
Raw traces
  -> Step 0: Trace segmentation and feature extraction
  -> Step 1: Trace clustering
  -> Step 2: Intra-cluster N-model rubric mining
  -> Step 3: Conservative merge and cross-cluster generalization
  -> Step 4: Contrastive refinement
  -> Step 5: Calibration and export
```

Intermediate results are written as JSON arrays under `out_dir`. Each stage is
resume-friendly: records with an existing `__record_id__` and no `*_error` field
are skipped. Writes are atomic via `.tmp` files and rename.

## Project Layout

```text
src/
  parse_cache/                     # active module: raw traces -> parsed cache
  task_clustering/                 # active module: task embedding + consensus clustering
  diagnostics/                     # output inspection commands
  miner.py                         # older full-pipeline CLI entrypoint
  rubric_miner/                    # older full-pipeline package and shared utilities
configs/
  example_task.json
  local_qwen3_vllm.json
  local_qwen3_vllm_balanced_debug.json
  local_qwen3_vllm_full.json
  local_qwen3_vllm_qwen3_embedding_full.json
local_inference/
  README.md
  download_hf_model.sh
  start_vllm_qwen.sh
  start_hf_openai_server.sh
  hf_openai_server.py
scripts/
  parse_traces_to_cache.sh         # shell entrypoint for parse_cache
  cluster_cached_tasks.sh          # shell entrypoint for task_clustering
  run_local_qwen3_full.sh          # shell entrypoint for the older full pipeline
docs/
  parse_traces_to_cache.md
  cluster_cached_tasks.md
```

The current first two modules are intentionally independent at the entrypoint
level. `parse_cache` still reuses stable shared code from `rubric_miner.io` and
`rubric_miner.trace`; deeper code movement can wait until the module APIs settle.

## Install

Use Python 3.10+.

```bash
pip install -r requirements.txt
```

`requirements.txt` covers the project code under `src/` and the shell
entrypoints under `scripts/`. It does not install local model-serving stacks.

For vLLM or the minimal HF local server, install the local inference
dependencies in the server environment:

```bash
pip install -r local_inference/requirements.txt
```

On shared GPU servers, vLLM and PyTorch are CUDA-sensitive. If the server
already has a managed CUDA/PyTorch stack, prefer the matching vLLM install used
by that environment.

## Local Models

The miner calls OpenAI-compatible APIs. This includes OpenAI, vLLM, LM Studio,
Ollama-compatible proxies, LiteLLM, and the minimal HF server in
`local_inference/`.

For local Qwen3 Instruct with vLLM:

```bash
chmod +x local_inference/start_vllm_qwen.sh
bash local_inference/start_vllm_qwen.sh

export OPENAI_API_KEY="local"
python src/miner.py --config configs/local_qwen3_vllm.json
```

For more reproducible server runs, pre-download models into the ignored
`local_inference/models/` directory:

```bash
pip install -r local_inference/requirements.txt
chmod +x local_inference/download_hf_model.sh

bash local_inference/download_hf_model.sh

bash local_inference/start_vllm_qwen.sh
```

For vLLM chat plus Qwen3 embeddings and DBSCAN clustering:

```bash
pip install -r requirements.txt
pip install -r local_inference/requirements.txt

chmod +x local_inference/download_qwen3_embedding.sh
bash local_inference/download_qwen3_embedding.sh

# Terminal 1: chat model for rubric mining.
bash local_inference/start_vllm_qwen.sh

# Terminal 2: embedding model for DBSCAN clustering.
# Use CUDA_VISIBLE_DEVICES/TENSOR_PARALLEL_SIZE if you want to reserve specific GPUs.
bash local_inference/start_vllm_qwen3_embedding.sh

export OPENAI_API_KEY="local"
python src/miner.py --config configs/local_qwen3_vllm_qwen3_embedding_full.json
```

`configs/local_qwen3_vllm_full.json` also points to the same Qwen3 embedding
server by default.

For a custom HF/SentenceTransformers embedding server, use
`local_inference/start_hf_openai_server.sh` and copy one of the Qwen3 embedding
configs as a template:

```bash
chmod +x local_inference/start_hf_openai_server.sh
EMBEDDING_MODEL="BAAI/bge-m3" \
SERVED_EMBEDDING_MODEL_NAME="bge-m3" \
PORT=8001 \
local_inference/start_hf_openai_server.sh
```

vLLM can expose embeddings only when serving an embedding model. A Qwen3
Instruct model should be treated as a chat model, not an embedding model.

## Configuration

Create one config file per task. Example:

```json
{
  "input": "data/traces.jsonl",
  "out_dir": "outputs/example_task",
  "rubric_models": ["gpt-4.1-mini", "gpt-4.1", "gpt-4.1-mini"],
  "merge_model": "gpt-4.1",
  "embedding_model": "text-embedding-3-small",
  "min_model_support": 2,
  "concurrency": 4,
  "cluster_algorithm": "dbscan",
  "cluster_threshold": 0.74,
  "min_cluster_size": 2,
  "generalization_threshold": 0.74,
  "max_records_per_cluster": 20,
  "max_chars_per_trace": 6000
}
```

Run:

```bash
export OPENAI_API_KEY="..."
python src/miner.py --config configs/example_task.json
```

You can override key config values from the CLI:

```bash
python src/miner.py --config configs/example_task.json \
  --rubric-models gpt-4.1-mini,gpt-4.1,gpt-4.1 \
  --min-model-support 2
```

## Input Data

The dataloader accepts:

- `.json`
- `.jsonl`
- `.yaml` / `.yml`
- `.csv`
- AgentRewardBench trajectory directories via `input_format:
  "agent_reward_bench"`
- a directory containing supported files

The internal canonical record format is:

```json
{
  "__record_id__": "case_001",
  "task": "user task or instruction",
  "outcome": "success",
  "trace": [
    {"role": "assistant", "action": "plan"},
    {"tool": "search", "tool_input": "query"},
    {"observation": "tool output"},
    {"error": "failed attempt"},
    {"action": "retry with corrected input"}
  ],
  "metadata": {}
}
```

For JSON/JSONL/YAML, common field names are detected automatically, including
`task`, `question`, `prompt`, `messages`, `trace`, `steps`, `success`, `score`,
and `outcome`.

For CSV, two layouts are supported:

1. One row per trace, with a `trace` or `messages` column containing JSON.
2. Multiple rows per trace, grouped by a column such as `trace_id` and ordered
   by `step`.

Example multi-row CSV config snippet:

```json
{
  "input": "data/agent_steps.csv",
  "input_format": "csv",
  "csv_group_by": "trace_id",
  "field_map": {
    "__record_id__": "trace_id",
    "task": "task",
    "outcome": "success"
  }
}
```

AgentRewardBench local snapshot config snippet:

```json
{
  "input": "data/agent-reward-bench/trajectories",
  "input_format": "agent_reward_bench",
  "agent_reward_observation_policy": "last",
  "out_dir": "outputs/agent_reward_bench_local"
}
```

For this format the loader reads `cleaned/**/*.json` as trajectories, joins
`data/annotations.csv` and benchmark metadata into each record's `metadata`, and
ignores `judgments/` unless you explicitly point the generic loader there. By
default it follows AgentRewardBench's judge prompt style: every step keeps
URL/action/reasoning, while only the last step keeps an axtree observation
digest. The digest is not a raw prefix: it balances the axtree head, task/action
relevant lines, and the tail so final-page evidence is not lost behind global
navigation boilerplate. Set `agent_reward_observation_policy` to
`last_and_errors`, `all`, or `none` to change this tradeoff, and adjust
`agent_reward_observation_chars` to tune the per-observation digest budget.

## Outputs

Given `out_dir = outputs/example_task`, the pipeline writes:

```text
parsed_traces.json
clusters.json
mined.json
merged.json
generalized.json
refined.json
rubrics.json
miner.log.jsonl
```

The final rubric library is `rubrics.json`.

## Notes

- `rubric_models` can contain any number of mining models.
- `min_model_support` controls how many mining models must support a criterion
  before it survives conservative merge.
- `cluster_threshold` is a similarity threshold. Higher values generally create
  finer clusters; lower values merge more traces together.
- Failed samples are kept with a `*_error` field and `skipped: true`, so later
  runs can continue without corrupting previous results.
