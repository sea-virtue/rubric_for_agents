# Debug Parse Traces

`scripts/debug_parse_traces.py` runs only the `rubric_miner` parse logic and writes the parsed records to JSON. It is meant for checking whether `task_instruction`, `steps`, `thought_process`, `action_signature`, `obs_snapshot`, `error_signal`, `validation`, `metadata`, and optional `chat_messages` look right before running clustering or rubric mining.

## Basic Usage

From the repo root:

```bash
python scripts/debug_parse_traces.py \
  --input data/agent-reward-bench/trajectories/workarena.servicenow.basic-filter-problems-and-mark-duplicates-medium-l2.json \
  --output outputs/debug_parse/workarena_sample.parsed.json
```

On Windows PowerShell:

```powershell
python scripts/debug_parse_traces.py `
  --input data\agent-reward-bench\trajectories\workarena.servicenow.basic-filter-problems-and-mark-duplicates-medium-l2.json `
  --output outputs\debug_parse\workarena_sample.parsed.json
```

## Rubric-Ready Cache Parse

Use `scripts/parse_traces_to_cache.py` when you want a standalone parsed dataset for later clustering or rubric extraction experiments. It writes the normal parsed record shape plus:

- `runtime_summary`: reasoning-free, rule-based evidence summary for later rubric extraction.
- `audit_trace`: source pointers and reduction notes for checking what the parser kept or downweighted.
- `runtime_summary.state_cards`: candidate entry/operation/final/risk cards built from actions, URLs, UI cues, errors, and screenshot paths.

Example:

```powershell
python scripts\parse_traces_to_cache.py `
  --input data\agent-reward-bench\trajectories\workarena.servicenow.all-menu.json `
  --input-format json `
  --output data\cache_data\workarena_all_menu.rubric_ready.parsed.json
```

For AgentRewardBench directories, keep `--agent-reward-observation-policy all` unless you are intentionally making a smaller cache; intermediate state cards need compressed observations from more than the last step.

To mirror the source dataset tree instead of writing one aggregate JSON array:

```powershell
python scripts\parse_traces_to_cache.py `
  --input data\agent-reward-bench\trajectories\cleaned\workarena\GenericAgent-Qwen_Qwen2.5-VL-72B-Instruct `
  --input-format json `
  --preserve-tree `
  --output-root data\cache_data
```

This writes files such as:

```text
data/cache_data/agent-reward-bench/trajectories/cleaned/workarena/GenericAgent-Qwen_Qwen2.5-VL-72B-Instruct/.../<task>.json
```

## Include Chat Messages

The debug script internally loads `chat_messages` when present because some compact AgentRewardBench JSON files store the action history there. By default, the full `chat_messages` object is still removed from the final parsed JSON to keep the output compact.

Use this when you also want the full `chat_messages` object kept in parsed output:

```bash
python scripts/debug_parse_traces.py \
  --input data/agent-reward-bench/trajectories/workarena.servicenow.basic-filter-problems-and-mark-duplicates-medium-l2.json \
  --output outputs/debug_parse/workarena_sample.with_chat.parsed.json \
  --include-chat-messages
```

## Useful Options

- `--max-records 10`: parse only the first 10 loaded records.
- `--preview-records 5`: print summaries for the first 5 parsed records.
- `--preview-steps 8`: print summaries for the first 8 steps of each previewed record.
- `--input-format agent_reward_bench`: force AgentRewardBench directory loading.
- `--agent-reward-observation-policy last|last_and_errors|all|none`: control how much accessibility-tree evidence is kept.
- `--agent-reward-observation-chars 2000`: increase observation digest size.

## Output

The output is a JSON array. Each parsed record is a plain JSON object, not a Python class instance. The main shape is:

```json
{
  "__record_id__": "...",
  "task_instruction": "...",
  "outcome": "success|failure|unknown",
  "steps": [
    {
      "step_index": 1,
      "thought_process": "...",
      "action_signature": {},
      "obs_snapshot": {},
      "error_signal": {}
    }
  ],
  "validation": {},
  "metadata": {},
  "chat_messages": null,
  "skipped": false
}
```
