# Source Layout

The active modules are organized by pipeline responsibility:

```text
src/
  parse_cache/       # module 1: raw/cache trajectory parsing entrypoint
  pair_cache/        # module 2: parsed cache -> positive/negative pair cache
  pair_rubric_extraction/ # module 3: pair cache -> pair-level candidate rubrics
  task_clustering/   # module 4: embedding + hierarchical consensus clustering
  rubric_extraction/ # module 5: clusters + parsed cache -> rubrics
  rubric_evaluation/ # module 6: independent rubric quality checks
  rubric_miner/      # shared parser/LLM/text utilities
```

`scripts/` should stay thin: shell scripts set environment variables and call
the Python modules in `src/`.

`local_inference/` is only for model-server setup, downloads, and endpoint
checks.

For now, pipeline modules reuse shared helpers from `rubric_miner` where useful
so the existing parsed-cache behavior stays stable. Future refactors can rename
that shared package once the module APIs settle.

## Module Internals

`pair_cache/` is split by responsibility:

```text
pair_cache/
  cli.py      # command-line entrypoint and orchestration
  paths.py    # default paths
  loader.py   # parsed-cache file discovery and candidate loading
  builder.py  # outcome inference, grouping, pair sampling, pair record rendering
  writer.py   # pair-cache output writing and selected-record provenance copies
```

`pair_rubric_extraction/` follows the same pattern:

```text
pair_rubric_extraction/
  cli.py       # command-line entrypoint, dry-run preview, orchestration
  paths.py     # default paths
  io.py        # pair-cache loading, pair selection, selected-record loading, JSON writes
  sanitize.py  # prompt-only metadata/noise filtering
  prompting.py # cleaned pair prompt construction
  runner.py    # chat-model calls and rubric artifact writing
```
