from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import Field

from .schemas import CompatModel


class MinerConfig(CompatModel):
    input: Optional[Path] = None
    input_format: Optional[str] = None
    field_map: Dict[str, str] = Field(default_factory=dict)
    csv_group_by: Optional[str] = None
    max_input_records: Optional[int] = None
    agent_reward_observation_chars: int = 1200
    agent_reward_observation_policy: str = "last"
    agent_reward_sample_per_bucket: Optional[int] = None
    agent_reward_sample_seed: int = 13
    out_dir: Path = Path("outputs/rubric_miner")
    log_file: Optional[Path] = None
    verbose: bool = False

    api_key_env: str = "OPENAI_API_KEY"
    base_url: Optional[str] = None
    embedding_api_key_env: str = "OPENAI_API_KEY"
    embedding_base_url: Optional[str] = None
    embedding_instruction: str = (
        "Represent this agent trajectory for clustering by task intent, strategy, tools, observations, "
        "error recovery, and success/failure behavior."
    )
    embedding_max_chars: int = 20000
    embedding_batch_size: int = 16

    rubric_models: List[str] = Field(default_factory=lambda: ["gpt-4.1-mini", "gpt-4.1-mini"])
    merge_model: str = "gpt-4.1"
    embedding_model: str = ""

    concurrency: int = 4
    cluster_threshold: float = 0.28
    cluster_algorithm: str = "dbscan"
    cluster_partition_metadata_keys: List[str] = Field(default_factory=list)
    min_cluster_size: int = 2
    generalization_threshold: float = 0.74
    min_model_support: int = 2
    max_records_per_cluster: int = 8
    max_chars_per_trace: int = 6000
    mining_prompt_token_budget: int = 26000
    llm_max_tokens: int = 2048


def load_config(path: Optional[Path]) -> MinerConfig:
    config = MinerConfig()
    if path:
        data = _load_config_file(path)
        config = _validate_config(data)
    return _apply_env_defaults(config)


def _load_config_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("YAML config requires PyYAML. Use JSON or install pyyaml.") from exc
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    else:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("config file must contain an object")
    return data


def _validate_config(data: Dict[str, Any]) -> MinerConfig:
    if hasattr(MinerConfig, "model_validate"):
        return MinerConfig.model_validate(data)
    return MinerConfig.parse_obj(data)


def _apply_env_defaults(config: MinerConfig) -> MinerConfig:
    if not config.rubric_models:
        env_models = os.getenv("RUBRIC_MODELS", "")
        if env_models:
            config.rubric_models = [model.strip() for model in env_models.split(",") if model.strip()]
    if not config.rubric_models:
        a = os.getenv("RUBRIC_MODEL_A", "gpt-4.1-mini")
        b = os.getenv("RUBRIC_MODEL_B", a)
        config.rubric_models = [a, b]
    if config.merge_model == "gpt-4.1":
        config.merge_model = os.getenv("MERGE_MODEL", config.merge_model)
    if not config.base_url:
        config.base_url = os.getenv("OPENAI_BASE_URL") or None
    if not config.embedding_base_url:
        config.embedding_base_url = os.getenv("EMBEDDING_BASE_URL") or config.base_url
    if not config.embedding_model:
        config.embedding_model = os.getenv("TRACE_EMBEDDING_MODEL", "")
    config.min_model_support = max(1, min(config.min_model_support, len(config.rubric_models)))
    return config
