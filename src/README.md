# Source Layout

The active modules are organized by pipeline responsibility:

```text
src/
  parse_cache/       # module 1: raw/cache trajectory parsing entrypoint
  task_clustering/   # module 2: embedding + hierarchical consensus clustering
  rubric_extraction/ # module 3: clusters + parsed cache -> rubrics
  rubric_miner/      # shared parser/LLM/text utilities
```

`scripts/` should stay thin: shell scripts set environment variables and call
the Python modules in `src/`.

`local_inference/` is only for model-server setup, downloads, and endpoint
checks.

For now, `parse_cache`, `task_clustering`, and `rubric_extraction` reuse
shared helpers from `rubric_miner` so the existing parsed-cache behavior stays
stable. Future refactors can rename that shared package once the module APIs
settle.
