from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .trace import trim_text


def trace_snippets(records: Sequence[Mapping[str, Any]], max_records: int, max_chars: int) -> str:
    chunks: List[str] = []
    for record in records[:max_records]:
        chunks.append(
            "\n".join(
                [
                    f"__record_id__: {record.get('__record_id__')}",
                    f"task: {trim_text(str(record.get('task', '')), 600)}",
                    f"outcome: {record.get('outcome', 'unknown')}",
                    f"trace: {trim_text(str(record.get('trace_text', '')), max_chars)}",
                ]
            )
        )
    return "\n\n---\n\n".join(chunks)


def mining_messages(group: Mapping[str, Any], max_records: int, max_chars: int) -> List[Dict[str, str]]:
    schema = {
        "dimension": "short capability or failure mode name",
        "criterion": "observable rubric criterion; one criterion per item",
        "positive_evidence": ["what successful traces do"],
        "negative_evidence": ["what failed or weak traces do"],
        "severity": "low|medium|high",
        "rationale": "why this criterion matters",
    }
    return [
        {
            "role": "system",
            "content": (
                "You mine evaluation rubrics from agent trajectories. "
                "Return only a strict JSON array. Do not use markdown, prose, comments, "
                "or an enclosing object."
            ),
        },
        {
            "role": "user",
            "content": (
                "Mine coarse-to-fine rubric items for the trace cluster below.\n"
                "Each item must be concrete, observable in a trajectory, and useful for "
                "distinguishing high-quality agent behavior from weak behavior.\n"
                f"JSON item schema example:\n{json.dumps(schema, ensure_ascii=False)}\n\n"
                f"cluster_id: {group['cluster_id']}\n"
                f"cluster_key: {group.get('cluster_key', '')}\n\n"
                f"TRACES:\n{trace_snippets(group['records'], max_records, max_chars)}"
            ),
        },
    ]


def merge_messages(mined: Mapping[str, Any]) -> List[Dict[str, str]]:
    rubrics_by_model = mined.get("rubrics_by_model")
    if not isinstance(rubrics_by_model, dict):
        rubrics_by_model = {
            str(mined.get("model_a", "model_a")): mined.get("rubrics_a", []),
            str(mined.get("model_b", "model_b")): mined.get("rubrics_b", []),
        }
    return [
        {
            "role": "system",
            "content": (
                "You conservatively merge independent rubric drafts from multiple models. "
                "Return only a strict JSON array. Do not use markdown or prose. "
                "Keep an item only when enough drafts clearly support the same meaning."
            ),
        },
        {
            "role": "user",
            "content": (
                "Merge the rubric arrays into a concise consensus array.\n"
                "Rules:\n"
                "- Preserve only criteria supported by enough independent mining models.\n"
                "- Treat support as semantic similarity > 0.8 or literal overlap > 70%.\n"
                "- Prefer precise observable criteria over broad advice.\n"
                "- Output items with fields: dimension, criterion, positive_evidence, "
                "negative_evidence, severity, rationale.\n\n"
                f"cluster_id: {mined.get('cluster_id')}\n"
                f"cluster_key: {mined.get('cluster_key', '')}\n\n"
                f"RUBRICS_BY_MODEL:\n{json.dumps(rubrics_by_model, ensure_ascii=False, indent=2)}"
            ),
        },
    ]


def refine_messages(
    merged: Mapping[str, Any],
    success: Mapping[str, Any],
    failure: Mapping[str, Any],
) -> List[Dict[str, str]]:
    schema = {
        "signal": "discriminative behavior visible in trajectories",
        "success_indicator": "what the successful trace does",
        "failure_indicator": "what the failed trace does instead or omits",
        "why_it_matters": "how this separates pass/fail outcomes",
    }
    return [
        {
            "role": "system",
            "content": (
                "You compare successful and failed agent trajectories to refine rubrics. "
                "Return only a strict JSON array. No markdown, no prose, no wrapper object."
            ),
        },
        {
            "role": "user",
            "content": (
                "Given consensus rubrics and a success/failure trajectory pair, produce "
                "discriminative_signals. Each signal must identify an observable contrast "
                "that explains why the success passed and the failure failed.\n"
                f"JSON item schema example:\n{json.dumps(schema, ensure_ascii=False)}\n\n"
                f"cluster_id: {merged.get('cluster_id')}\n"
                f"rubrics:\n{json.dumps(merged.get('rubrics', []), ensure_ascii=False, indent=2)}\n\n"
                f"SUCCESS_TRACE:\n{trace_snippets([success], 1, 7000)}\n\n"
                f"FAILURE_TRACE:\n{trace_snippets([failure], 1, 7000)}"
            ),
        },
    ]


def generalize_messages(bucket: Mapping[str, Any]) -> List[Dict[str, str]]:
    schema = {
        "dimension": "short shared or cluster-specific dimension",
        "criterion": "binary, observable, trajectory-verifiable criterion",
        "positive_evidence": ["observable behavior that satisfies the criterion"],
        "negative_evidence": ["observable behavior that violates the criterion"],
        "severity": "low|medium|high",
        "rationale": "why this dimension should be retained",
    }
    return [
        {
            "role": "system",
            "content": (
                "You are an Expert Rubric Designer and QA Specialist. "
                "Return only a strict JSON array. No markdown, no prose, no wrapper object."
            ),
        },
        {
            "role": "user",
            "content": (
                "Aggregate and generalize rubric items across homogeneous trace clusters.\n\n"
                "Strict protocol inspired by RubricHub:\n"
                "1. Keep criteria relevant to the represented agent strategies.\n"
                "2. Apply a conservative merge: merge only if semantic meaning, required action, "
                "and scope are identical.\n"
                "3. Do not merge items that differ in granularity, thresholds, parameters, or implied method.\n"
                "4. Preserve cluster-specific dimensions when they are not safely mergeable.\n"
                "5. Prefer concise, binary, verifiable wording.\n\n"
                f"JSON item schema example:\n{json.dumps(schema, ensure_ascii=False)}\n\n"
                f"bucket_id: {bucket.get('bucket_id')}\n"
                f"scope_hint: {bucket.get('scope')}\n"
                f"source_cluster_ids: {json.dumps(bucket.get('source_cluster_ids', []), ensure_ascii=False)}\n\n"
                f"RUBRICS_BY_CLUSTER:\n{json.dumps(bucket.get('cluster_rubrics', []), ensure_ascii=False, indent=2)}"
            ),
        },
    ]
