# Evaluate Pair Rubrics

`src/pair_rubric_evaluation/cli.py` tests whether mined pair-level rubrics can
rank the selected positive trajectory above the selected negative trajectory.
It is a discriminative sanity check before rubric merging.

The judge prompt hides validation labels. It shows:

- the task query,
- the pair-level rubrics mined for that pair,
- two unlabeled responses with cleaned `state_cards`.

The script asks the judge model to score each response against each rubric,
sums the scores, and counts the pair as correct if `response_0` scores higher
than `response_1`. In the current pair cache, `response_0` is the selected
positive trajectory and `response_1` is the selected negative trajectory, but
that fact is not shown in the prompt.

## Run

Preview one judge prompt without calling a model:

```bash
./scripts/evaluate_pair_rubrics.sh --dry-run --max-pairs 1 --preview-chars 8000
```

Run a one-pair smoke test:

```bash
./scripts/evaluate_pair_rubrics.sh \
  --model "gpt-5.4-mini" \
  --max-pairs 1 \
  --concurrency 1
```

Run all available pair rubrics:

```bash
./scripts/evaluate_pair_rubrics.sh \
  --model "gpt-5.4-mini" \
  --concurrency 1
```

For an OpenAI-compatible transit API:

```bash
export OPENAI_BASE_URL="https://api.gpt.ge/v1/"
export OPENAI_API_KEY="your_api_key"

./scripts/evaluate_pair_rubrics.sh \
  --model "gpt-5.4-mini" \
  --concurrency 1
```

## Outputs

```text
data/pair_rubric_eval/pair_rubric_pair_scores.json
data/pair_rubric_eval/pair_rubric_eval_prompts.json
data/pair_rubric_eval/pair_rubric_eval_raw_outputs.json
data/pair_rubric_eval/pair_rubric_eval_summary.json
data/pair_rubric_eval/pair_rubric_eval_config.json
```

`pair_rubric_eval_summary.json` contains aggregate accuracy, ties, errors, and
mean score margin.

## Important Parameters

- `--pair-rubrics`: mined pair-rubric file.
- `--pairs`: pair cache root, `pair_index.json`, or a single `pair.json`.
- `--pair-ids`: comma-separated pair ids to process first/only.
- `--max-pairs`: process only the first N selected pairs.
- `--max-chars-per-response`: cap state-card payload size per response.
- `--concurrency`: concurrent judge requests.
- `--refresh`: recompute selected pairs already present in output and upsert them.
- `--dry-run`: print a prompt preview without calling a model.
