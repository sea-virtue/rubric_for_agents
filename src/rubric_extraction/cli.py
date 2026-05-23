from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rubric_miner.llm import build_client, llm_json_array  # noqa: E402


DEFAULT_CLUSTER_FILE = Path("data/cluster/task_clusters.json")
DEFAULT_OUTPUT_DIR = Path("data/rubric")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract rubrics from task clusters and parsed-cache samples.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--clusters", type=Path, default=DEFAULT_CLUSTER_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=os.getenv("RUBRIC_MODEL", "qwen3-4b-instruct-2507"))
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-clusters", type=int, default=None)
    parser.add_argument("--cluster-ids", default="", help="Comma-separated cluster ids to process first/only.")
    parser.add_argument("--min-cluster-size", type=int, default=1)
    parser.add_argument("--max-records-per-cluster", type=int, default=16)
    parser.add_argument("--max-chars-per-record", type=int, default=3500)
    parser.add_argument(
        "--include-action-sequence",
        action="store_true",
        help="Include runtime_summary.action_sequence in prompts. Disabled by default to favor more samples.",
    )
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--refresh", action="store_true", help="Recompute clusters already present in output.")
    parser.add_argument("--dry-run", action="store_true", help="Load clusters/cache and print a prompt preview.")
    return parser.parse_args(argv)


async def main_async(args: argparse.Namespace) -> int:
    cluster_records = load_cluster_records(args.clusters)
    groups = build_cluster_groups(
        cluster_records,
        selected_cluster_ids=parse_cluster_ids(args.cluster_ids),
        min_cluster_size=args.min_cluster_size,
        max_clusters=args.max_clusters,
    )
    if not groups:
        raise ValueError("No cluster groups selected for rubric extraction.")

    if args.dry_run:
        print(f"clusters_file: {args.clusters}")
        print(f"selected_clusters: {len(groups)}")
        for group in groups[:3]:
            samples = load_cache_samples(group, max_records=args.max_records_per_cluster)
            print(f"- {group['cluster_id']} size={group['cluster_size']} samples={len(samples)}")
            print(prompt_messages(group, samples, args.max_chars_per_record, args.include_action_sequence)[1]["content"][:1200])
            print("---")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "cluster_rubrics.json"
    prompt_path = args.output_dir / "rubric_prompts.json"
    config_path = args.output_dir / "rubric_extraction_config.json"

    existing = [] if args.refresh else load_json_array(output_path)
    prompt_records = [] if args.refresh else load_json_array(prompt_path)
    done = {
        str(record.get("cluster_id") or record.get("__record_id__"))
        for record in existing
        if record.get("cluster_id") and not has_error(record)
    }

    client = build_client(api_key_env=args.api_key_env, base_url=args.base_url)
    semaphore = asyncio.Semaphore(max(1, args.concurrency))

    async def process_group(group: Mapping[str, Any]) -> Dict[str, Any]:
        cluster_id = str(group["cluster_id"])
        if cluster_id in done:
            for record in existing:
                if str(record.get("cluster_id")) == cluster_id:
                    return dict(record)
        async with semaphore:
            try:
                samples = load_cache_samples(group, max_records=args.max_records_per_cluster)
                messages = prompt_messages(
                    group,
                    samples,
                    args.max_chars_per_record,
                    args.include_action_sequence,
                )
                prompt_record = {
                    "__record_id__": cluster_id,
                    "cluster_id": cluster_id,
                    "cluster_size": group["cluster_size"],
                    "sampled_records": [sample["__record_id__"] for sample in samples],
                    "messages": messages,
                }
                upsert(prompt_records, prompt_record, key="cluster_id")
                write_json(prompt_path, prompt_records)

                raw_items = await llm_json_array(
                    client,
                    args.model,
                    messages,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
                rubrics = normalize_rubric_items(raw_items)
                return {
                    "__record_id__": cluster_id,
                    "cluster_id": cluster_id,
                    "cluster_size": group["cluster_size"],
                    "model": args.model,
                    "source_record_ids": [str(item.get("__record_id__")) for item in group["members"]],
                    "sampled_record_ids": [sample["__record_id__"] for sample in samples],
                    "rubrics": rubrics,
                }
            except Exception as exc:
                return {
                    "__record_id__": cluster_id,
                    "cluster_id": cluster_id,
                    "cluster_size": group.get("cluster_size", 0),
                    "skipped": True,
                    "rubric_error": str(exc),
                }

    pending = list(groups)
    results = list(existing)
    for future in asyncio.as_completed([asyncio.create_task(process_group(group)) for group in pending]):
        record = await future
        upsert(results, record, key="cluster_id")
        write_json(output_path, results)
        print(f"processed: {record.get('cluster_id')} rubrics={len(record.get('rubrics', []))} error={record.get('rubric_error', '')}")

    config = {
        "clusters": str(args.clusters),
        "output_dir": str(args.output_dir),
        "model": args.model,
        "base_url": args.base_url,
        "concurrency": args.concurrency,
        "min_cluster_size": args.min_cluster_size,
        "max_records_per_cluster": args.max_records_per_cluster,
        "max_chars_per_record": args.max_chars_per_record,
        "include_action_sequence": args.include_action_sequence,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "num_selected_clusters": len(groups),
        "outputs": {
            "rubrics": str(output_path),
            "prompts": str(prompt_path),
            "config": str(config_path),
        },
    }
    write_json(config_path, config)
    print(f"rubric_file: {output_path}")
    print(f"clusters_processed_or_cached: {len(results)}")
    return 0


def load_cluster_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array")
    return [item for item in data if isinstance(item, dict) and item.get("cluster_id")]


def build_cluster_groups(
    records: Sequence[Mapping[str, Any]],
    *,
    selected_cluster_ids: Sequence[str],
    min_cluster_size: int,
    max_clusters: Optional[int],
) -> List[Dict[str, Any]]:
    selected = set(selected_cluster_ids)
    groups: Dict[str, Dict[str, Any]] = {}
    for record in records:
        cluster_id = str(record.get("cluster_id", ""))
        if selected and cluster_id not in selected:
            continue
        group = groups.setdefault(cluster_id, {"cluster_id": cluster_id, "members": []})
        group["members"].append(dict(record))
    output = []
    for group in groups.values():
        members = sorted(group["members"], key=lambda item: str(item.get("relative_source_path", item.get("source_path", ""))))
        if len(members) < min_cluster_size:
            continue
        group["members"] = members
        group["cluster_size"] = len(members)
        output.append(group)
    output.sort(key=lambda item: (-int(item["cluster_size"]), str(item["cluster_id"])))
    if max_clusters is not None:
        output = output[: max(0, max_clusters)]
    return output


def load_cache_samples(group: Mapping[str, Any], *, max_records: int) -> List[Dict[str, Any]]:
    members = list(group.get("members", []))
    selected_members = balanced_member_sample(members, max_records=max_records)
    samples = []
    for member in selected_members:
        sample = load_one_cache_record(member)
        if sample:
            samples.append(sample)
    return samples


def balanced_member_sample(members: Sequence[Mapping[str, Any]], *, max_records: int) -> List[Mapping[str, Any]]:
    if max_records <= 0 or len(members) <= max_records:
        return list(members)
    by_outcome: Dict[str, List[Mapping[str, Any]]] = {}
    for member in members:
        by_outcome.setdefault(str(member.get("outcome", "unknown")), []).append(member)
    selected: List[Mapping[str, Any]] = []
    for outcome in ("success", "failure", "unknown"):
        bucket = by_outcome.get(outcome, [])
        if bucket:
            selected.append(bucket[0])
    remaining = [member for member in members if member not in selected]
    head_count = max(0, (max_records - len(selected) + 1) // 2)
    tail_count = max(0, max_records - len(selected) - head_count)
    selected.extend(remaining[:head_count])
    if tail_count:
        selected.extend(remaining[-tail_count:])
    deduped = []
    seen = set()
    for member in selected:
        key = str(member.get("__record_id__")) + str(member.get("source_path"))
        if key in seen:
            continue
        deduped.append(member)
        seen.add(key)
    return deduped[:max_records]


def load_one_cache_record(member: Mapping[str, Any]) -> Dict[str, Any]:
    source_path = Path(str(member.get("source_path") or member.get("relative_source_path") or ""))
    if not source_path.exists():
        source_path = ROOT / source_path
    if not source_path.exists():
        return {
            "__record_id__": str(member.get("__record_id__", "")),
            "load_error": f"missing source_path: {member.get('source_path')}",
            "task_instruction": member.get("task_instruction", ""),
        }
    data = json.loads(source_path.read_text(encoding="utf-8"))
    record_id = str(member.get("__record_id__", ""))
    if isinstance(data, list):
        for item in data:
            if isinstance(item, Mapping) and str(item.get("__record_id__")) == record_id:
                return dict(item)
        for item in data:
            if isinstance(item, Mapping):
                return dict(item)
    if isinstance(data, Mapping):
        return dict(data)
    return {
        "__record_id__": record_id,
        "load_error": f"unsupported cache payload in {source_path}",
        "task_instruction": member.get("task_instruction", ""),
    }


def prompt_messages(
    group: Mapping[str, Any],
    samples: Sequence[Mapping[str, Any]],
    max_chars_per_record: int,
    include_action_sequence: bool,
) -> List[Dict[str, str]]:
    schema = {
        "dimension": "short capability or failure mode name",
        "criterion": "observable criterion for evaluating trajectories in this cluster",
        "positive_evidence": ["what strong trajectories do"],
        "negative_evidence": ["what weak trajectories do or omit"],
        "severity": "low|medium|high",
        "rationale": "why this criterion matters for this cluster",
    }
    return [
        {
            "role": "system",
            "content": (
                "You extract evaluation rubrics from clustered agent trajectory summaries. "
                "Return only a strict JSON array. Do not use markdown, prose, or a wrapper object."
            ),
        },
        {
            "role": "user",
            "content": (
                "Extract concise rubrics for this cluster. Criteria must be observable from "
                "trajectory evidence, not generic advice. Prefer criteria that can separate "
                "successful trajectories from failed or weak trajectories.\n\n"
                f"JSON item schema:\n{json.dumps(schema, ensure_ascii=False)}\n\n"
                f"cluster_id: {group.get('cluster_id')}\n"
                f"cluster_size: {group.get('cluster_size')}\n\n"
                f"SAMPLED_PARSED_CACHE_RECORDS:\n{render_samples(samples, max_chars_per_record, include_action_sequence)}"
            ),
        },
    ]


def render_samples(
    samples: Sequence[Mapping[str, Any]],
    max_chars_per_record: int,
    include_action_sequence: bool,
) -> str:
    chunks = []
    for sample in samples:
        chunks.append(trim_text(render_one_sample(sample, include_action_sequence), max_chars_per_record))
    return "\n\n---\n\n".join(chunks)


def render_one_sample(sample: Mapping[str, Any], include_action_sequence: bool) -> str:
    runtime = sample.get("runtime_summary")
    if isinstance(runtime, Mapping):
        return render_runtime_summary(sample, runtime, include_action_sequence)
    lines = [
        f"__record_id__: {sample.get('__record_id__')}",
        f"task_instruction: {sample.get('task_instruction', sample.get('task', ''))}",
        f"outcome: {sample.get('outcome', 'unknown')}",
    ]
    steps = sample.get("steps", [])
    if isinstance(steps, list):
        lines.append("steps:")
        for step in steps[:24]:
            if isinstance(step, Mapping):
                lines.append(render_step(step))
    return "\n".join(line for line in lines if line)


def render_runtime_summary(
    sample: Mapping[str, Any],
    runtime: Mapping[str, Any],
    include_action_sequence: bool,
) -> str:
    lines = [
        f"__record_id__: {sample.get('__record_id__', runtime.get('__record_id__'))}",
        f"task_instruction: {runtime.get('task_instruction', sample.get('task_instruction', ''))}",
        f"outcome: {runtime.get('outcome', sample.get('outcome', 'unknown'))}",
    ]
    validation = runtime.get("validation", {})
    if isinstance(validation, Mapping):
        lines.append(f"validation: {json.dumps(validation, ensure_ascii=False)}")
    action_sequence = runtime.get("action_sequence", [])
    if include_action_sequence and isinstance(action_sequence, list) and action_sequence:
        lines.append("action_sequence:")
        for action in action_sequence[:16]:
            lines.append(f"- {json.dumps(action, ensure_ascii=False)}")
    final_state = runtime.get("final_state", {})
    if isinstance(final_state, Mapping):
        lines.append("final_state:")
        for item in final_state.get("evidence_lines", [])[:12]:
            lines.append(f"- {item}")
    cards = runtime.get("state_cards", [])
    if isinstance(cards, list) and cards:
        lines.append("state_cards:")
        for card in cards[:10]:
            if not isinstance(card, Mapping):
                continue
            lines.append(
                f"- {card.get('state_id', '')} | role={card.get('rubric_role', card.get('evidence_role', ''))}"
            )
            for evidence in list(card.get("evidence_lines", []))[:8]:
                lines.append(f"  evidence: {evidence}")
    risk = runtime.get("risk_signals", {})
    if isinstance(risk, Mapping) and (risk.get("errors") or risk.get("repeated_action_runs")):
        lines.append("risk_signals:")
        lines.append(json.dumps(risk, ensure_ascii=False)[:2000])
    return "\n".join(line for line in lines if line)


def render_step(step: Mapping[str, Any]) -> str:
    parts = [f"- step {step.get('step_index', '')}"]
    action = step.get("action_signature", {})
    if isinstance(action, Mapping) and action.get("raw"):
        parts.append(f"  action: {action.get('raw')}")
    obs = step.get("obs_snapshot", {})
    if isinstance(obs, Mapping):
        for cue in list(obs.get("task_relevant_cues", []))[:6]:
            parts.append(f"  cue: {cue}")
    error = step.get("error_signal", {})
    if isinstance(error, Mapping) and error.get("has_error"):
        parts.append(f"  error: {error.get('message', '')}")
    return "\n".join(parts)


def normalize_rubric_items(raw_items: Sequence[Any]) -> List[Dict[str, Any]]:
    rubrics = []
    for item in raw_items:
        if not isinstance(item, Mapping):
            continue
        criterion = first_text(item, "criterion", "description", "rubric", "rule")
        if not criterion:
            continue
        severity = first_text(item, "severity", "importance", default="medium").lower()
        if severity not in {"low", "medium", "high"}:
            severity = "medium"
        rubrics.append(
            {
                "dimension": first_text(item, "dimension", "category", "capability", default="trajectory_quality"),
                "criterion": criterion,
                "positive_evidence": list_field(item, "positive_evidence", "positive", "success_evidence"),
                "negative_evidence": list_field(item, "negative_evidence", "negative", "failure_evidence"),
                "severity": severity,
                "rationale": first_text(item, "rationale", "reason", "why", default=""),
            }
        )
    return rubrics


def first_text(item: Mapping[str, Any], *keys: str, default: str = "") -> str:
    for key in keys:
        value = item.get(key)
        if value not in (None, ""):
            return str(value).strip()
    return default


def list_field(item: Mapping[str, Any], *keys: str) -> List[str]:
    for key in keys:
        value = item.get(key)
        if value in (None, ""):
            continue
        if isinstance(value, list):
            return [str(part).strip() for part in value if str(part).strip()]
        return [str(value).strip()]
    return []


def parse_cluster_ids(value: str) -> List[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def load_json_array(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array")
    return [item for item in data if isinstance(item, dict)]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(tmp_path, path)


def upsert(records: List[Dict[str, Any]], record: Dict[str, Any], *, key: str) -> None:
    value = str(record.get(key) or record.get("__record_id__"))
    for idx, existing in enumerate(records):
        if str(existing.get(key) or existing.get("__record_id__")) == value:
            records[idx] = record
            return
    records.append(record)


def has_error(record: Mapping[str, Any]) -> bool:
    return any(str(key).endswith("_error") for key in record)


def trim_text(text: str, limit: int) -> str:
    text = re.sub(r"\n{3,}", "\n\n", str(text)).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 40)].rstrip() + "\n...[truncated]..."


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        return asyncio.run(main_async(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
