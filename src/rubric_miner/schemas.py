from __future__ import annotations

from typing import Any, Dict, List, Mapping

from pydantic import BaseModel, Field

try:
    from pydantic import ConfigDict
except ImportError:  # pragma: no cover - pydantic v1
    ConfigDict = None


if ConfigDict is not None:
    class CompatModel(BaseModel):
        model_config = ConfigDict(
            populate_by_name=True,
            protected_namespaces=(),
            extra="allow",
        )

else:

    class CompatModel(BaseModel):
        class Config:
            allow_population_by_field_name = True
            extra = "allow"


class CompactStep(CompatModel):
    step_index: int
    thought_process: str = ""
    action_signature: Dict[str, Any] = Field(default_factory=dict)
    obs_snapshot: Dict[str, Any] = Field(default_factory=dict)
    error_signal: Dict[str, Any] = Field(default_factory=dict)


class TraceParsed(CompatModel):
    record_id: str = Field(alias="__record_id__")
    task_instruction: str = ""
    outcome: str = "unknown"
    steps: List[CompactStep] = Field(default_factory=list)
    validation: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    chat_messages: Any = None
    skipped: bool = False


def model_validate(cls: Any, data: Any) -> Any:
    if hasattr(cls, "model_validate"):
        return cls.model_validate(data)
    return cls.parse_obj(data)


def model_dump(obj: Any) -> Dict[str, Any]:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(by_alias=True)
    return obj.dict(by_alias=True)


def has_error(record: Mapping[str, Any]) -> bool:
    return any(key.endswith("_error") for key in record)
