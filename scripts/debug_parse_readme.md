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
