# Merge Pair Rubrics

`src/rubric_merge/cli.py` merges pair-level rubrics into broader grouped
rubrics. It follows the OpenJudge aggregation style: many scattered source
rubrics are deduplicated into a compact `Theme + Tips` structure.

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
./scripts/merge_pair_rubrics.sh \
  --model gpt-5.4-mini \
  --base-url https://api.gpt.ge/v1/ \
  --concurrency 1
```

The defaults match the API configuration used for the current pair-rubric
generation run: `gpt-5.4-mini` with `https://api.gpt.ge/v1/`. Set
`OPENAI_API_KEY` on the server before running the non-dry-run command.

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
- `--max-rubrics-per-group`: cap on source pair-level rubric items placed in a
  prompt. The default `180` covers the current largest domain.
- `--max-chars-per-rubric`: truncation cap per rendered source rubric item.
- `--refresh`: recompute groups already present in the output.
