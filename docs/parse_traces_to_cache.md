# Parse Traces To Cache

`src/parse_cache/cli.py` is the standalone parsed-cache stage for trajectory data. It reads raw traces, builds compact parsed records, and writes rubric-ready cache files without running clustering or rubric extraction. Use `scripts/parse_traces_to_cache.sh` as the normal shell entrypoint.

The parser keeps the shared trajectory structure as the primary abstraction. `runtime_summary.state_cards` uses universal evidence-card types rather than benchmark-specific domain cards:

- `task_context`
- `initial_observation`
- `action_transition`
- `evidence_observation`
- `final_observation`
- `output_or_answer`
- `media_reference`
- `risk_or_error`

## Mirror A Dataset Tree

From the repo root:

```bash
./scripts/parse_traces_to_cache.sh \
  --input data/agent-reward-bench/trajectories/cleaned \
  --input-format json \
  --preserve-tree \
  --output-root data/cache_data \
  --preview-records 5
```

This mirrors source files such as:

```text
data/agent-reward-bench/trajectories/cleaned/workarena/.../<task>.json
```

to:

```text
data/cache_data/agent-reward-bench/trajectories/cleaned/workarena/.../<task>.json
```

## Shell Script

The repo already includes `scripts/parse_traces_to_cache.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}src"

python -m parse_cache.cli \
  --input data/agent-reward-bench/trajectories/cleaned \
  --input-format json \
  --preserve-tree \
  --output-root data/cache_data \
  --preview-records 5
```

Then run:

```bash
chmod +x scripts/parse_traces_to_cache.sh
./scripts/parse_traces_to_cache.sh
```

## Single File Or Aggregate Output

For one file or an aggregate JSON array, omit `--preserve-tree`:

```bash
./scripts/parse_traces_to_cache.sh \
  --input data/agent-reward-bench/trajectories/cleaned/workarena/GenericAgent-Qwen_Qwen2.5-VL-72B-Instruct/GenericAgent-Qwen_Qwen2.5-VL-72B-Instruct_on_workarena.servicenow/workarena.servicenow.all-menu.json \
  --input-format json \
  --output outputs/debug_parse/workarena_all_menu.parsed.json
```

## Runtime-Only Cache

Use `--runtime-only` when downstream code only needs `runtime_summary`:

```bash
./scripts/parse_traces_to_cache.sh \
  --input data/agent-reward-bench/trajectories/cleaned \
  --input-format json \
  --preserve-tree \
  --output-root data/cache_data \
  --runtime-only \
  --preview-records 5
```

## Main Output Fields

Each full parsed record contains:

```json
{
  "__record_id__": "...",
  "task_instruction": "...",
  "outcome": "success|failure|unknown",
  "steps": [],
  "validation": {},
  "metadata": {},
  "chat_messages": null,
  "runtime_summary": {
    "schema_version": "rubric_ready_runtime_v2",
    "goal_terms": [],
    "action_sequence": [],
    "final_state": {},
    "state_cards": [],
    "risk_signals": {}
  },
  "audit_trace": {}
}
```

For later rubric extraction, use `runtime_summary` as the primary input. `audit_trace` is for provenance checks and debugging the lossy reduction.
