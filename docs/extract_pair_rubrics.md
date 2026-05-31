# Extract Pair Rubrics

`src/pair_rubric_extraction/cli.py` reads positive/negative pair cache records,
constructs a cleaned prompt payload, and asks a chat model to extract
pair-level candidate rubrics. It does not rewrite the pair cache itself.

The miner prompt uses only:

```json
{
  "query": "task instruction",
  "responses": [
    {
      "state_cards": []
    },
    {
      "state_cards": []
    }
  ],
  "validation": [
    {
      "reward": 1.0,
      "outcome": "success",
      "pair_role": "positive",
      "response_index": 0
    },
    {
      "reward": 0.0,
      "outcome": "failure",
      "pair_role": "negative",
      "response_index": 1
    }
  ]
}
```

Agent/model/model-args/source metadata may remain in `pair.json` for provenance
and file layout, but they are not part of the prompt payload. The extraction
script loads the selected parsed records, keeps only cleaned
`runtime_summary.state_cards` as `responses`, and builds `validation` for the
two trajectories. The prompt explicitly tells the model not to turn reward,
label, response index, model identity, or agent identity into rubric content.

## Run

Preview prompts without calling a model:

```bash
./scripts/extract_pair_rubrics.sh --dry-run --max-pairs 2 --preview-chars 6000
```

Extract rubrics:

```bash
./scripts/extract_pair_rubrics.sh \
  --base-url http://127.0.0.1:28000/v1 \
  --concurrency 2
```

## Outputs

```text
data/pair_rubric/pair_rubrics.json
data/pair_rubric/pair_rubric_prompts.json
data/pair_rubric/pair_rubric_raw_outputs.json
data/pair_rubric/pair_rubric_extraction_config.json
```

`pair_rubrics.json` contains one record per pair. These pair-level rubrics are
intended for the later embedding, MCR/deduplication, and merge stage.

## Important Parameters

- `--pairs`: pair cache root, `pair_index.json`, or a single `pair.json`.
- `--output-dir`: pair-rubric output directory.
- `--pair-ids`: comma-separated pair ids to process first/only.
- `--max-pairs`: process only the first N selected pairs.
- `--max-chars-per-response`: cap state-card payload size per response.
- `--preview-chars`: characters of each dry-run prompt to print.
- `--refresh`: recompute pairs already present in output.
- `--dry-run`: print prompt previews without calling a model.
