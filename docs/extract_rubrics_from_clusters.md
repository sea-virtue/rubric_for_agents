# Extract Rubrics From Clusters

`src/rubric_extraction/cli.py` reads task clusters from
`data/cluster/task_clusters.json`, loads the corresponding parsed-cache records
from `data/cache_data`, and asks a chat model to extract rubric items for each
cluster.

It writes rubric artifacts under `data/rubric`.

## Run

Start the chat model server first, for example:

```bash
bash local_inference/start_vllm_qwen.sh
```

Then run a small preview:

```bash
./scripts/extract_rubrics_from_clusters.sh --dry-run --max-clusters 2
```

Run extraction:

```bash
./scripts/extract_rubrics_from_clusters.sh \
  --model qwen3-4b-instruct-2507 \
  --base-url http://127.0.0.1:28000/v1 \
  --max-records-per-cluster 12 \
  --max-chars-per-record 6000 \
  --max-tokens 2048 \
  --concurrency 1
```

By default, prompts use `runtime_summary.state_cards`, final-state evidence,
validation, and risk signals. `runtime_summary.action_sequence` is omitted so a
cluster can include more sampled records. Add `--include-action-sequence` only
when you need explicit action-order context.

## Outputs

```text
data/rubric/cluster_rubrics.json
data/rubric/rubric_prompts.json
data/rubric/rubric_extraction_config.json
```

`cluster_rubrics.json` contains one record per cluster with extracted rubric
items. `rubric_prompts.json` stores the exact prompts for inspection.
