from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path


def load_json(path: Path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs/local_qwen3_vllm_full")
    print(f"output_dir: {base}")
    for name in [
        "parsed_traces.json",
        "clusters.json",
        "mined.json",
        "merged.json",
        "generalized.json",
        "refined.json",
        "rubrics.json",
    ]:
        path = base / name
        rows = load_json(path)
        size_mb = os.path.getsize(path) / 1024 / 1024 if path.exists() else 0
        print(f"{name}: rows={len(rows)} size_mb={size_mb:.2f}")

    parsed = load_json(base / "parsed_traces.json")
    clusters = load_json(base / "clusters.json")
    rubrics = load_json(base / "rubrics.json")

    if parsed:
        print("bench/outcome:")
        print(Counter((r.get("metadata", {}).get("benchmark"), r.get("outcome")) for r in parsed))

    if parsed and clusters:
        by_id = {r["__record_id__"]: r for r in parsed}
        cluster_info = defaultdict(lambda: {"bench": set(), "agent": set(), "outcome": Counter(), "n": 0})
        for cluster in clusters:
            record = by_id.get(cluster.get("__record_id__"))
            if not record:
                continue
            meta = record.get("metadata", {})
            info = cluster_info[cluster.get("cluster_id")]
            info["bench"].add(meta.get("benchmark"))
            info["agent"].add(meta.get("agent"))
            info["outcome"][record.get("outcome")] += 1
            info["n"] += 1
        print("clusters:")
        for cluster_id, info in sorted(cluster_info.items(), key=lambda kv: (-kv[1]["n"], str(kv[0])))[:30]:
            print(
                cluster_id,
                "n=", info["n"],
                "bench=", sorted(info["bench"]),
                "agents=", len(info["agent"]),
                "outcome=", dict(info["outcome"]),
            )

    if rubrics:
        print("rubric_summary:")
        print("sets=", len(rubrics))
        print("with_signals=", sum(bool(r.get("discriminative_signals")) for r in rubrics))
        print("single_source_sets=", sum(len(r.get("source_record_ids", [])) == 1 for r in rubrics))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
