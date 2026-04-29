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


class TraceParsed(CompatModel):
    record_id: str = Field(alias="__record_id__")
    task: str = ""
    outcome: str = "unknown"
    trace_text: str
    compact_trace: Dict[str, Any] = Field(default_factory=dict)
    structured_sequence: List[Dict[str, Any]] = Field(default_factory=list)
    features: Dict[str, Any] = Field(default_factory=dict)
    raw: Any
    metadata: Dict[str, Any] = Field(default_factory=dict)
    skipped: bool = False


class ClusterAssignment(CompatModel):
    record_id: str = Field(alias="__record_id__")
    cluster_id: str
    cluster_key: str
    similarity: float = 0.0
    skipped: bool = False


class RubricItem(CompatModel):
    dimension: str
    criterion: str
    positive_evidence: List[str] = Field(default_factory=list)
    negative_evidence: List[str] = Field(default_factory=list)
    severity: str = "medium"
    rationale: str = ""


class DiscriminativeSignal(CompatModel):
    signal: str
    success_indicator: str
    failure_indicator: str
    why_it_matters: str = ""


class MinedCluster(CompatModel):
    record_id: str = Field(alias="__record_id__")
    cluster_id: str
    cluster_key: str
    source_record_ids: List[str]
    mining_models: List[str]
    rubrics_by_model: Dict[str, List[RubricItem]]
    skipped: bool = False


class MergedCluster(CompatModel):
    record_id: str = Field(alias="__record_id__")
    cluster_id: str
    cluster_key: str
    source_record_ids: List[str]
    merge_model: str
    rubrics: List[RubricItem]
    support_summary: Dict[str, Any] = Field(default_factory=dict)
    skipped: bool = False


class GeneralizedRubricSet(CompatModel):
    record_id: str = Field(alias="__record_id__")
    generalized_id: str
    scope: str = "cluster_specific"
    cluster_key: str = ""
    source_cluster_ids: List[str]
    source_record_ids: List[str]
    merge_model: str
    rubrics: List[RubricItem]
    support_summary: Dict[str, Any] = Field(default_factory=dict)
    skipped: bool = False


class RefinedCluster(CompatModel):
    record_id: str = Field(alias="__record_id__")
    cluster_id: str = ""
    generalized_id: str = ""
    scope: str = "cluster_specific"
    cluster_key: str
    source_cluster_ids: List[str] = Field(default_factory=list)
    source_record_ids: List[str]
    rubrics: List[RubricItem]
    discriminative_signals: List[DiscriminativeSignal] = Field(default_factory=list)
    support_summary: Dict[str, Any] = Field(default_factory=dict)
    skipped: bool = False


class ExportCluster(CompatModel):
    record_id: str = Field(alias="__record_id__")
    cluster_id: str = ""
    generalized_id: str = ""
    scope: str = "cluster_specific"
    cluster_key: str
    source_cluster_ids: List[str] = Field(default_factory=list)
    source_record_ids: List[str]
    rubrics: List[RubricItem]
    discriminative_signals: List[DiscriminativeSignal]
    support_summary: Dict[str, Any] = Field(default_factory=dict)
    calibration: Dict[str, Any] = Field(default_factory=dict)


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
