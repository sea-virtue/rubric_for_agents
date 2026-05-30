# Source Layout

The active modules are organized by pipeline responsibility:

```text
src/
  parse_cache/       # module 1: raw/cache trajectory parsing entrypoint
  pair_cache/        # module 2: parsed cache -> positive/negative pair cache
  task_clustering/   # module 3: embedding + hierarchical consensus clustering
  rubric_extraction/ # module 4: clusters + parsed cache -> rubrics
  rubric_evaluation/ # module 5: independent rubric quality checks
  rubric_miner/      # shared parser/LLM/text utilities
```

`scripts/` should stay thin: shell scripts set environment variables and call
the Python modules in `src/`.

`local_inference/` is only for model-server setup, downloads, and endpoint
checks.

For now, pipeline modules reuse shared helpers from `rubric_miner` where useful
so the existing parsed-cache behavior stays stable. Future refactors can rename
that shared package once the module APIs settle.
