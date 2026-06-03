# Merge Pair Rubrics

`src/rubric_merge/cli.py` merges pair-level rubrics into broader grouped
rubrics. It follows the OpenJudge iterative-rubric aggregation pattern:
representative rubric selection with embedding/MCR-style coding-rate gain,
then optional LLM categorization into compact `Theme + Tips` rubrics.

Because cluster results may not exist yet, the current default grouping is
`domain`, using the four top-level categories under `data/cache_pair_data`:

```text
assistantbench
visualwebarena
webarena
workarena
```

The script reads:

```text
data/pair_rubric/pair_rubrics.json
data/cache_pair_data/pair_index.json
```

It uses pair cache only for lightweight context such as `query`, `domain`, and
`jobname`; it does not reload full trajectories.

## Run

Preview the domain groups and prompt shape without calling a model:

```bash
./scripts/merge_pair_rubrics.sh --dry-run
```

Merge all four domain groups:

```bash
bash local_inference/start_vllm_qwen3_embedding.sh

./scripts/merge_pair_rubrics.sh \
  --model gpt-5.4-mini \
  --base-url https://api.gpt.ge/v1/ \
  --concurrency 1
```

The defaults match the API configuration used for the current pair-rubric
generation run: `gpt-5.4-mini` with `https://api.gpt.ge/v1/`. Set
`OPENAI_API_KEY` on the server before running the non-dry-run command.

By default the merge stage also runs MCR-style selection over source rubrics
before sending them to the chat API. This selection uses the project's Qwen
embedding endpoint:

```text
RUBRIC_EMBEDDING_MODEL=qwen3-embedding-8b
OPENAI_EMBEDDING_BASE_URL=http://127.0.0.1:8001/v1
```

Disable this representative-selection step only for debugging:

```bash
./scripts/merge_pair_rubrics.sh --selection-method none
```

Run only one domain:

```bash
./scripts/merge_pair_rubrics.sh \
  --group-ids webarena \
  --num-categories 8
```

## Outputs

```text
data/rubric_merge/domain_merged_rubrics.json
data/rubric_merge/domain_rubric_merge_prompts.json
data/rubric_merge/domain_rubric_merge_raw_outputs.json
data/rubric_merge/domain_rubric_mcr_selection.json
data/rubric_merge/domain_rubric_merge_config.json
```

Each merged record contains:

```json
{
  "group_id": "webarena",
  "grouping": "domain",
  "pair_count": 37,
  "source_rubric_count": 127,
  "rubrics": [
    {
      "theme": "Final task completion is supported by the terminal state or final answer",
      "tips": [
        "Verify that the final visible state or user-facing answer satisfies the task parameters.",
        "Do not credit intermediate progress when the final state contradicts completion."
      ],
      "verification_guide": {
        "what_to_extract": ["final answer", "terminal state", "task parameters"],
        "checks": ["compare final evidence against the requested target and constraints"]
      },
      "source_pair_ids": ["webarena/example"]
    }
  ],
  "reason": "..."
}
```

## Important Parameters

- `--grouping`: currently supports `domain`; later this can be extended to
  consume real cluster files.
- `--group-ids`: comma-separated domain names to process.
- `--num-categories`: maximum merged Theme-Tips categories per group.
- `--selection-method`: `mcr` selects a representative, high-diversity subset
  before LLM categorization; `none` sends the rendered source rubrics directly.
- `--embedding-model` and `--embedding-base-url`: OpenAI-compatible embedding
  endpoint used by MCR selection.
- `--max-selected-rubrics`: maximum source rubrics kept after MCR selection.
- `--mcr-batch-size`, `--mcr-eps`, `--mcr-min-increment-threshold`,
  `--mcr-patience`: coding-rate selection controls.
- `--max-rubrics-per-group`: cap on source pair-level rubric items placed in a
  prompt. The default `180` covers the current largest domain.
- `--max-chars-per-rubric`: truncation cap per rendered source rubric item.
- `--refresh`: recompute groups already present in the output.
