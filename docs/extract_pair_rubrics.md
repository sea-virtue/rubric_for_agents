# Extract Pair Rubrics

`src/pair_rubric_extraction/cli.py` reads positive/negative pair cache records,
constructs a cleaned prompt payload, and asks a chat model to extract
pair-level candidate rubrics. It does not rewrite the pair cache itself.

Implementation is split across:

```text
src/pair_rubric_extraction/
  cli.py       # command-line entrypoint, dry-run preview, orchestration
  paths.py     # default paths
  io.py        # pair-cache loading, pair selection, selected-record loading, JSON writes
  sanitize.py  # prompt-only metadata/noise filtering
  prompting.py # cleaned pair prompt construction
  runner.py    # chat-model calls and rubric artifact writing
```

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

## OpenAI-Compatible Transit API

For a transit API or any OpenAI-compatible endpoint, keep the API key out of
the repository and put it in environment variables on the server:

```bash
export OPENAI_BASE_URL="https://api.gpt.ge/v1/"
export OPENAI_API_KEY="your_api_key"
```

The chat-completions API requires a model name, so pass the model supported by
the transit provider:

```bash
bash ./scripts/extract_pair_rubrics.sh \
  --model "gpt-5.4-mini" \
  --max-pairs 1 \
  --concurrency 1
```

After the one-pair smoke test succeeds, remove `--max-pairs 1` to run all
available pairs:

```bash
bash ./scripts/extract_pair_rubrics.sh \
  --model "gpt-5.4-mini" \
  --concurrency 1
```

You can also pass the endpoint explicitly instead of using `OPENAI_BASE_URL`:

```bash
bash ./scripts/extract_pair_rubrics.sh \
  --base-url "https://api.gpt.ge/v1/" \
  --model "gpt-5.4-mini" \
  --concurrency 1
```

`--concurrency` controls how many pair requests are sent at the same time. Use
`--concurrency 1` for the safest transit-API run; increase it only if the
provider's rate limits are stable. Successful pair outputs are cached, so a
later run skips pairs that already have rubrics unless `--refresh` is used.

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
- `--refresh`: recompute selected pairs already present in output and upsert them.
- `--dry-run`: print prompt previews without calling a model.
