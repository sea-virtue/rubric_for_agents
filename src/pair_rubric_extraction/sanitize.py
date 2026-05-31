from __future__ import annotations

from typing import Any, Dict, Mapping


PROMPT_NOISE_KEYS = {
    "agent",
    "agent_name",
    "agent_args",
    "agent_config",
    "model",
    "model_name",
    "model_args",
    "llm",
    "llm_config",
    "experiment",
    "source_path",
    "relative_source_path",
    "screenshot_path",
    "screenshot_paths",
    "success_prior",
}


def clean_prompt_payload(value: Any) -> Any:
    if isinstance(value, Mapping):
        cleaned: Dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            key_lower = key_text.lower()
            if key_lower in PROMPT_NOISE_KEYS:
                continue
            if "model_args" in key_lower or key_lower.startswith("agent_") or key_lower.startswith("model_"):
                continue
            if key_lower == "source_fields" and isinstance(item, list):
                item = [entry for entry in item if not is_prompt_noise_text(entry)]
            cleaned_item = clean_prompt_payload(item)
            if cleaned_item in (None, "", [], {}):
                continue
            cleaned[key_text] = cleaned_item
        return cleaned
    if isinstance(value, list):
        cleaned_list = [clean_prompt_payload(item) for item in value]
        return [item for item in cleaned_list if item not in (None, "", [], {})]
    if isinstance(value, str):
        return clean_prompt_text(value)
    return value


def is_prompt_noise_text(value: Any) -> bool:
    lowered = str(value or "").strip().lower()
    if not lowered:
        return True
    return lowered in {"validation"} or any(token in lowered for token in PROMPT_NOISE_KEYS)


def clean_prompt_text(value: str) -> str:
    text = value.strip()
    return text.replace("Task instruction hints and dataset-level outcome supervision.", "Task instruction hints.")
