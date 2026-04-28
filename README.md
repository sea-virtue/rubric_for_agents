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
  miner.py                         # CLI entrypoint
  rubric_miner/
    cli.py                         # command-line orchestration
    config.py                      # task config loading
    dataloader.py                  # CSV/JSON/JSONL/YAML normalization
    schemas.py                     # Pydantic models
    llm.py                         # async LLM/embedding calls with retry
    trace.py                       # trace segmentation and features
    text.py                        # similarity utilities
    calibration.py                 # export metadata scores
    stages/
      parse.py
      cluster.py
      mine.py
      merge.py
      generalize.py
      refine.py
      export.py
configs/
  example_task.json
  csv_multirow_task.json
  local_qwen3_vllm.json
  local_qwen3_vllm_hf_embedding.json
local_inference/
  README.md
  start_vllm_qwen.sh
  start_hf_openai_server.sh
  hf_openai_server.py
```

## Install

Use Python 3.10+.

```bash
pip install openai tenacity rich pydantic
```

Optional dependencies:

```bash
pip install pyyaml scikit-learn
```

`pyyaml` is needed for YAML inputs/configs. `scikit-learn` enables DBSCAN; if it
is missing, clustering falls back to a simple connected-components strategy.

## Local Models

The miner calls OpenAI-compatible APIs. This includes OpenAI, vLLM, LM Studio,
Ollama-compatible proxies, LiteLLM, and the minimal HF server in
`local_inference/`.

For local Qwen3 Instruct with vLLM:

```bash
chmod +x local_inference/start_vllm_qwen.sh
MODEL="Qwen/Qwen3-7B-Instruct" \
SERVED_MODEL_NAME="qwen3-7b-instruct" \
PORT=8000 \
local_inference/start_vllm_qwen.sh

export OPENAI_API_KEY="local"
python src/miner.py --config configs/local_qwen3_vllm.json
```

For vLLM chat plus local HF embeddings:

```bash
chmod +x local_inference/start_hf_openai_server.sh
EMBEDDING_MODEL="BAAI/bge-m3" \
SERVED_EMBEDDING_MODEL_NAME="bge-m3" \
PORT=8001 \
local_inference/start_hf_openai_server.sh

python src/miner.py --config configs/local_qwen3_vllm_hf_embedding.json
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

Example multi-row CSV config:

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

AgentRewardBench local snapshot config:

```json
{
  "input": "data/agent-reward-bench/trajectories",
  "input_format": "agent_reward_bench",
  "out_dir": "outputs/agent_reward_bench_local"
}
```

For this format the loader reads `cleaned/**/*.json` as trajectories, joins
`data/annotations.csv` and benchmark metadata into each record's `metadata`, and
ignores `judgments/` unless you explicitly point the generic loader there.

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
