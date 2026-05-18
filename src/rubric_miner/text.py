from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Iterable, List, Mapping, Sequence


TOKEN_RE = re.compile(r"[\u4e00-\u9fff]|[a-zA-Z0-9_]+")
STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "you",
    "are",
    "was",
    "were",
    "has",
    "have",
    "had",
    "not",
    "but",
    "all",
    "can",
    "will",
    "should",
    "must",
    "请",
    "的",
    "了",
    "是",
    "在",
    "和",
    "或",
    "对",
    "从",
}


def tokenize(text: str) -> List[str]:
    tokens = [token.lower() for token in TOKEN_RE.findall(text)]
    return [token for token in tokens if token not in STOPWORDS and token.strip()]


def token_counter(text: str) -> Counter:
    return Counter(tokenize(text))


def cosine_counts(left: Counter, right: Counter) -> float:
    if not left or not right:
        return 0.0
    dot = sum(left[token] * right.get(token, 0) for token in left)
    norm_left = math.sqrt(sum(value * value for value in left.values()))
    norm_right = math.sqrt(sum(value * value for value in right.values()))
    if not norm_left or not norm_right:
        return 0.0
    return dot / (norm_left * norm_right)


def literal_overlap(left: str, right: str) -> float:
    left_tokens = set(tokenize(left))
    right_tokens = set(tokenize(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(1, min(len(left_tokens), len(right_tokens)))


def semantic_similarity(left: str, right: str) -> float:
    """Local similarity proxy used for conservative support filtering."""

    return max(cosine_counts(token_counter(left), token_counter(right)), literal_overlap(left, right))


def cosine_vectors(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right:
        return 0.0
    size = min(len(left), len(right))
    dot = sum(float(left[i]) * float(right[i]) for i in range(size))
    norm_left = math.sqrt(sum(float(value) * float(value) for value in left[:size]))
    norm_right = math.sqrt(sum(float(value) * float(value) for value in right[:size]))
    if not norm_left or not norm_right:
        return 0.0
    return dot / (norm_left * norm_right)


def sequence_similarity(left: Sequence[str], right: Sequence[str]) -> float:
    if not left or not right:
        return 0.0
    matches = sum(1 for a, b in zip(left, right) if a == b)
    prefix_score = matches / max(len(left), len(right))
    overlap_score = len(set(left) & set(right)) / max(1, len(set(left) | set(right)))
    return max(prefix_score, overlap_score)


def cluster_text(record: Mapping[str, Any]) -> str:
    task = str(record.get("task_instruction", record.get("task", "")))
    steps = record.get("steps", [])
    step_texts = []
    if isinstance(steps, list):
        for step in steps[:32]:
            if not isinstance(step, Mapping):
                continue
            action = step.get("action_signature", {})
            action_text = ""
            if isinstance(action, Mapping):
                action_text = " ".join(
                    part
                    for part in (
                        str(action.get("action_type", "")),
                        str(action.get("target_bid", "")),
                        str(action.get("target_text", "")),
                    )
                    if part.strip()
                ).strip()
            obs = step.get("obs_snapshot", {})
            obs_bits = []
            if isinstance(obs, Mapping):
                for key in ("page_title", "focused_element"):
                    value = str(obs.get(key, "")).strip()
                    if value:
                        obs_bits.append(f"{key}: {value}")
                cues = obs.get("task_relevant_cues", [])
                if isinstance(cues, list) and cues:
                    obs_bits.extend([str(cue) for cue in cues[:4]])
            pieces = [
                f"step {step.get('step_index', '')}",
                str(step.get("thought_process", "")).strip(),
                action_text,
                " | ".join(obs_bits),
            ]
            step_texts.append("\n".join(part for part in pieces if part))
    chat_messages = record.get("chat_messages", None)
    chat_text = ""
    if chat_messages not in (None, ""):
        chat_text = f"chat_messages: {str(chat_messages)[:8000]}"
    validation = record.get("validation", {})
    validation_text = ""
    if isinstance(validation, Mapping):
        validation_text = " ".join(
            f"{key}:{validation.get(key)}"
            for key in ("reward", "n_steps", "terminated", "truncated", "has_error", "outcome")
            if validation.get(key) not in ("", None, False)
        )
    return "\n".join(part for part in [task, validation_text, chat_text, "\n\n".join(step_texts)] if part)


def top_keywords(texts: Iterable[str], limit: int = 8) -> str:
    counts: Counter = Counter()
    for text in texts:
        counts.update(tokenize(text))
    words = [word for word, _ in counts.most_common(limit)]
    return ", ".join(words) if words else "misc"
