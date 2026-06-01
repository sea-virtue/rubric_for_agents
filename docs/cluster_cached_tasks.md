# Cluster Pair Tasks

`src/task_clustering/cli.py` is a standalone clustering module for positive /
negative pair cache records. Each pair is one clustering unit. The module reads
the pair task query, embeds that text with an OpenAI-compatible embedding
endpoint, runs an ensemble of average-linkage hierarchical clusterings, then
performs a final consensus clustering over the co-clustering matrix.

Use `scripts/cluster_cached_tasks.sh` as the normal shell entrypoint.

It only produces task cluster files under `data/cluster`.

## Default Input And Output

Input:

```text
data/cache_pair_data
```

Output:

```text
data/cluster/task_clusters.json
data/cluster/task_cluster_summary.json
data/cluster/task_cluster_config.json
data/cluster/pair_task_embeddings.json
```

The default input resolves to `data/cache_pair_data/pair_index.json` when that
file exists. You can also pass `--pairs` pointing at a pair cache root, a
`pair_index.json`, or a single `pair.json`.

## Run

Start the embedding server first, for example:

```bash
bash local_inference/start_vllm_qwen3_embedding.sh
```

Then run:

```bash
export OPENAI_API_KEY="local"
./scripts/cluster_cached_tasks.sh \
  --embedding-base-url http://127.0.0.1:8001/v1 \
  --embedding-model qwen3-embedding-8b
```

For a quick local read check without embedding calls:

```bash
./scripts/cluster_cached_tasks.sh --dry-run --max-records 5
```

## Important Parameters

- `--thresholds`: comma-separated cosine-distance thresholds used for the
  hierarchical clustering ensemble. Defaults are wider than the old
  trajectory-level clustering defaults because there are currently only about
  one hundred pair units.
- `--consensus-threshold`: final co-clustering fraction required for consensus
  grouping.
- `--min-cluster-size`: clusters smaller than this are split into singletons.
- `--refresh-embeddings`: ignore the existing embedding cache and recompute.
- `--embedding-instruction`: optional instruction prefix for embedding models
  that benefit from `Instruct:` / `Query:` input format.
