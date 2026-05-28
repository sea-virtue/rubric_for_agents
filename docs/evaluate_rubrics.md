# Evaluate Rubrics

`src/rubric_evaluation/cli.py` is an independent quality-check stage for mined
cluster rubrics. It reads existing cluster assignments, parsed cache files, and
rubric outputs. It does not modify parse, clustering, or extraction artifacts.

The module has two parts:

- Static audit: checks field coverage, duplicate dimensions, instance-value
  leakage, singleton clusters, and too many/few rubric items.
- Judge scoring: uses a chat model to score cached trajectories against their
  own cluster rubrics, then compares weighted rubric scores with
  success/failure labels using AUC and score gaps.

Judge prompts hide label fields by default: `outcome`, `validation.reward`, and
other validation fields are not shown to the model. Labels are only used locally
after scoring to compute metrics. Use `--include-label-fields` only for
debugging leakage-sensitive behavior.

## Run Static Audit Only

```bash
./scripts/evaluate_rubrics.sh --static-only
```

Outputs:

```text
data/rubric_eval/rubric_static_audit.json
data/rubric_eval/rubric_eval_summary.json
```

## Preview Judge Prompt

```bash
./scripts/evaluate_rubrics.sh --dry-run --max-clusters 1
```

## Judge-Based Evaluation

Start the chat model server first, then run a small evaluation:

```bash
./scripts/evaluate_rubrics.sh \
  --max-clusters 5 \
  --max-records-per-cluster 4
```

Full run:

```bash
./scripts/evaluate_rubrics.sh \
  --max-records-per-cluster 4 \
  --max-tokens 2048 \
  --concurrency 1
```

The judge stage is resumable by default. Without `--refresh`, existing
successful records in `trajectory_rubric_scores.json` are skipped, while records
with `judge_error` are retried. Use `--refresh` only when you want to recompute
all selected scores from scratch.

Outputs:

```text
data/rubric_eval/rubric_static_audit.json
data/rubric_eval/trajectory_rubric_scores.json
data/rubric_eval/judge_raw_outputs.json
data/rubric_eval/rubric_eval_summary.json
data/rubric_eval/rubric_eval_config.json
```

`global_auc` measures whether trajectories with success labels receive higher
rubric scores than failure trajectories. It is not a best-of-n metric and does
not require multiple trajectories for the same task instance.
