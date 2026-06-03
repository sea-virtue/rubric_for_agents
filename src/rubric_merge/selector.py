from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from sklearn.decomposition import PCA

from rubric_miner.llm import async_embedding_batch_call, build_client

from .prompting import format_rubric, trim_text


def flatten_rubric_sources(group: Mapping[str, Any], *, max_chars_per_rubric: int) -> List[Dict[str, Any]]:
    sources: List[Dict[str, Any]] = []
    for pair in group.get("pairs", []):
        if not isinstance(pair, Mapping):
            continue
        pair_id = str(pair.get("pair_id") or "")
        query = str(pair.get("query") or "").strip()
        rubrics = pair.get("rubrics", [])
        if not isinstance(rubrics, list):
            continue
        for idx, rubric in enumerate(rubrics):
            if not isinstance(rubric, Mapping):
                continue
            rubric_id = f"{pair_id}#rubric_{idx}"
            text = (
                f"query: {trim_text(query, 500)}\n"
                f"pair_id: {pair_id}\n"
                f"rubric: {trim_text(format_rubric(rubric), max_chars_per_rubric)}"
            )
            sources.append(
                {
                    "rubric_id": rubric_id,
                    "pair_id": pair_id,
                    "query": query,
                    "rubric_index": idx,
                    "rubric": dict(rubric),
                    "text": text,
                }
            )
    return sources


async def select_group_rubrics_mcr(
    group: Mapping[str, Any],
    *,
    embedding_model: str,
    embedding_base_url: str,
    api_key_env: str,
    embedding_batch_size: int,
    max_chars_per_rubric: int,
    max_selected_rubrics: int,
    mcr_batch_size: int,
    eps: float,
    min_increment_threshold: float,
    patience: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    sources = flatten_rubric_sources(group, max_chars_per_rubric=max_chars_per_rubric)
    if not sources:
        return dict(group), {"selection_method": "mcr", "error": "no source rubrics", "selected_rubric_ids": []}
    if max_selected_rubrics <= 0 or len(sources) <= max_selected_rubrics:
        selected_indices = list(range(len(sources)))
        selected_group = filter_group_to_sources(group, sources, selected_indices)
        return selected_group, {
            "selection_method": "mcr",
            "skipped_selection": True,
            "reason": "source count is within max_selected_rubrics",
            "source_rubric_count": len(sources),
            "selected_rubric_count": len(selected_indices),
            "selected_rubric_ids": [sources[idx]["rubric_id"] for idx in selected_indices],
        }

    client = build_client(api_key_env=api_key_env, base_url=embedding_base_url)
    texts = [source["text"] for source in sources]
    embeddings: List[List[float]] = []
    for start in range(0, len(texts), max(1, embedding_batch_size)):
        batch = texts[start : start + max(1, embedding_batch_size)]
        embeddings.extend(await async_embedding_batch_call(client, embedding_model, batch))

    X = preprocess_embeddings(np.array(embeddings, dtype=float))
    selected_indices, mcr_info = greedy_mcr_selection(
        X,
        max_selected=max_selected_rubrics,
        batch_size=mcr_batch_size,
        eps=eps,
        min_increment_threshold=min_increment_threshold,
        patience=patience,
    )
    selected_group = filter_group_to_sources(group, sources, selected_indices)
    mcr_info.update(
        {
            "selection_method": "mcr",
            "embedding_model": embedding_model,
            "embedding_base_url": embedding_base_url,
            "source_rubric_count": len(sources),
            "selected_rubric_count": len(selected_indices),
            "selected_rubric_ids": [sources[idx]["rubric_id"] for idx in selected_indices],
            "selected_pair_ids": sorted({sources[idx]["pair_id"] for idx in selected_indices}),
        }
    )
    return selected_group, mcr_info


def preprocess_embeddings(X: np.ndarray) -> np.ndarray:
    if X.ndim != 2 or X.shape[0] == 0:
        return X
    n_samples, original_dim = X.shape
    max_components = min(n_samples, original_dim, 100)
    if original_dim > max_components and max_components > 0:
        X = PCA(n_components=max_components, random_state=42).fit_transform(X)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)


def greedy_mcr_selection(
    X: np.ndarray,
    *,
    max_selected: int,
    batch_size: int,
    eps: float,
    min_increment_threshold: float,
    patience: int,
) -> Tuple[List[int], Dict[str, Any]]:
    selected: List[int] = []
    candidates = list(range(X.shape[0]))
    coding_rate_history = [0.0]
    increment_history: List[float] = []
    batch_history: List[Dict[str, Any]] = []
    low_increment_count = 0
    batch_num = 0

    while candidates and len(selected) < max_selected:
        batch_num += 1
        chosen_batch = []
        target_batch_size = min(max(1, batch_size), max_selected - len(selected), len(candidates))
        for _ in range(target_batch_size):
            current_rate = coding_rate(X[selected], eps) if selected else 0.0
            best_idx = None
            best_delta = -float("inf")
            for idx in candidates:
                candidate_rate = coding_rate(X[[*selected, idx]], eps)
                delta = candidate_rate - current_rate
                if delta > best_delta:
                    best_delta = delta
                    best_idx = idx
            if best_idx is None:
                break
            selected.append(best_idx)
            candidates.remove(best_idx)
            chosen_batch.append(best_idx)

        if not chosen_batch:
            break
        previous_rate = coding_rate_history[-1]
        current_rate = coding_rate(X[selected], eps)
        increment = current_rate - previous_rate
        coding_rate_history.append(current_rate)
        increment_history.append(increment)
        batch_history.append(
            {
                "batch_num": batch_num,
                "batch_indices": chosen_batch,
                "increment": increment,
                "coding_rate": current_rate,
                "cumulative_samples": len(selected),
            }
        )
        if increment < min_increment_threshold:
            low_increment_count += 1
            if low_increment_count >= max(1, patience):
                break
        else:
            low_increment_count = 0

    return selected, {
        "final_coding_rate": coding_rate_history[-1],
        "coding_rate_history": coding_rate_history,
        "increment_history": increment_history,
        "batch_history": batch_history,
        "configuration": {
            "max_selected_rubrics": max_selected,
            "batch_size": batch_size,
            "eps": eps,
            "min_increment_threshold": min_increment_threshold,
            "patience": patience,
            "method": "greedy_mcr_coding_rate",
        },
    }


def coding_rate(X: np.ndarray, eps: float) -> float:
    if X.size == 0:
        return 0.0
    n = X.shape[0]
    try:
        _, singular_values, _ = np.linalg.svd(X, full_matrices=False)
    except np.linalg.LinAlgError:
        return 0.0
    if singular_values.size == 0:
        return 0.0
    energy = np.cumsum(singular_values**2) / (np.sum(singular_values**2) + 1e-8)
    k = min(np.searchsorted(energy, 0.95) + 1, len(singular_values))
    main_values = singular_values[:k]
    log_det_approx = 2 * np.sum(np.log(1 + main_values**2 / (eps**2 * n) + 1e-8))
    return float(0.5 * log_det_approx)


def filter_group_to_sources(
    group: Mapping[str, Any],
    sources: Sequence[Mapping[str, Any]],
    selected_indices: Sequence[int],
) -> Dict[str, Any]:
    selected_by_pair: Dict[str, List[Mapping[str, Any]]] = {}
    for idx in selected_indices:
        source = sources[idx]
        selected_by_pair.setdefault(str(source.get("pair_id")), []).append(source)

    selected_group = dict(group)
    selected_pairs = []
    for pair in group.get("pairs", []):
        if not isinstance(pair, Mapping):
            continue
        pair_id = str(pair.get("pair_id") or "")
        selected_sources = selected_by_pair.get(pair_id, [])
        if not selected_sources:
            continue
        selected_pair = dict(pair)
        selected_pair["rubrics"] = [dict(source["rubric"]) for source in selected_sources]
        selected_pairs.append(selected_pair)

    selected_group["pairs"] = selected_pairs
    selected_group["selected_pair_count"] = len(selected_pairs)
    selected_group["selected_rubric_count"] = sum(len(pair.get("rubrics", [])) for pair in selected_pairs)
    selected_group["selected_pair_ids"] = [str(pair.get("pair_id")) for pair in selected_pairs]
    return selected_group
