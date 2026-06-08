from __future__ import annotations

import math
from typing import Any, List, Mapping, Sequence


def normalize_vector(vector: Sequence[Any]) -> List[float]:
    values = [float(value) for value in vector]
    norm = math.sqrt(sum(value * value for value in values))
    if not norm:
        return [0.0 for _ in values]
    return [value / norm for value in values]


def cosine_similarity(left: Sequence[Any], right: Sequence[Any]) -> float:
    left_norm = normalize_vector(left)
    right_norm = normalize_vector(right)
    size = min(len(left_norm), len(right_norm))
    if size == 0:
        return 0.0
    return sum(left_norm[idx] * right_norm[idx] for idx in range(size))


def cosine_distance(left: Sequence[Any], right: Sequence[Any]) -> float:
    return max(0.0, min(2.0, 1.0 - cosine_similarity(left, right)))


def softmax_from_distances(distances: Sequence[float], *, temperature: float) -> List[float]:
    if not distances:
        return []
    temp = max(float(temperature), 1e-8)
    logits = [-(float(distance) / temp) for distance in distances]
    max_logit = max(logits)
    exps = [math.exp(logit - max_logit) for logit in logits]
    denom = sum(exps)
    if denom <= 0:
        return [1.0 / len(distances) for _ in distances]
    return [value / denom for value in exps]


def top_weight_records(records: Sequence[Mapping[str, Any]], *, limit: int) -> List[Mapping[str, Any]]:
    return sorted(records, key=lambda item: float(item.get("applicability", 0.0)), reverse=True)[: max(0, limit)]

