# Build Cache Pairs

`src/pair_cache/cli.py` builds pairwise positive/negative cache data from
parsed trajectory caches. It is intended as the bridge from trajectory parsing
to pairwise rubric mining.

Use `scripts/build_cache_pairs.sh` as the normal shell entrypoint.

Implementation is split across:

```text
src/pair_cache/
  cli.py      # command-line entrypoint and orchestration
  paths.py    # default paths
  loader.py   # parsed-cache file discovery and candidate loading
  builder.py  # grouping, outcome inference, pair sampling, response rendering
  writer.py   # output files and selected parsed-record copies
```

## Default Input And Output

Input:

```text
data/cache_data/agent-reward-bench/trajectories/cleaned
```

The expected parsed-cache layout is:

```text
domain/model-name/jobnamewithmodelname/jobname.json
```

For each `domain + jobname`, the builder collects all model results. If the
group contains at least one `success` and at least one `failure`, it samples
one positive and one negative record with a fixed random seed. The default seed
is `42`.

Output:

```text
data/cache_pair_data/
  pair_index.json
  pair_report.json
  pair_summary.json
  <domain>/<jobname>/pair.json
  <domain>/<jobname>/<jobnamewithmodelname>.json
```

The per-task directory keeps the selected full parsed records in the requested
layout:

```text
domain/jobname/jobnamewithmodelname.json
```

It also writes `pair.json`, which is the compact pairwise record intended for
later rubric-mining prompts.

## Run

From the repo root:

```bash
./scripts/build_cache_pairs.sh
```

For a read-only preview:

```bash
./scripts/build_cache_pairs.sh --dry-run
```

## Pair Record Format

Each `pair.json` contains:

```json
{
  "__record_id__": "domain/jobname",
  "pair_id": "domain/jobname",
  "domain": "workarena",
  "jobname": "workarena.servicenow.some-task",
  "query": "original task instruction",
  "label_rank": [1, 2],
  "responses": [
    "positive trajectory evidence rendered from runtime_summary",
    "negative trajectory evidence rendered from runtime_summary"
  ],
  "selected_records": [],
  "candidate_count": 4,
  "outcomes": {
    "failure": 2,
    "success": 2
  }
}
```

`responses[0]` is always the selected positive record and receives
`label_rank=1`. `responses[1]` is always the selected negative record and
receives `label_rank=2`.

The response text is rendered from `runtime_summary` by default. It includes
validation, final-state evidence, state cards, and risk signals. Add
`--include-action-sequence` if the pairwise rubric miner needs explicit action
order. Pair-level rubric prompts may apply their own cleaning layer without
changing this cache/provenance format.

## Report Files

`pair_report.json` contains groups that cannot form a positive/negative pair.
The main skip reasons are:

- `all_success`: every available model result succeeded.
- `all_failure`: every available model result failed.
- `no_success_or_failure`: outcomes were unavailable or unknown.

`pair_summary.json` stores aggregate counts by outcome, skip reason, and domain.

## Important Parameters

- `--input-root`: parsed-cache root to read.
- `--output-root`: pair-cache root to write.
- `--seed`: random seed used when multiple positives or failures exist.
- `--max-pairs`: write only the first N valid pairs, useful for debugging.
- `--include-action-sequence`: include `runtime_summary.action_sequence`.
- `--include-steps`: include compact parsed steps only when no runtime summary is available.
- `--dry-run`: preview grouping and pair counts without writing files.
