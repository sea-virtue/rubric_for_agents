# Source Layout

The active modules are organized by pipeline responsibility:

```text
src/
  parse_cache/       # module 1: raw/cache trajectory parsing entrypoint
  task_clustering/   # module 2: embedding + hierarchical consensus clustering
  diagnostics/       # small output inspection commands
  rubric_miner/      # older full-pipeline package and shared utilities
  miner.py           # older full-pipeline CLI shim
```

`scripts/` should stay thin: shell scripts set environment variables and call
the Python modules in `src/`.

`local_inference/` is only for model-server setup, downloads, and endpoint
checks.

For now, `parse_cache` reuses `rubric_miner.io` and `rubric_miner.trace` so the
existing parsed-cache behavior stays stable. Future refactors can move those
shared internals only after the module boundaries are settled.
