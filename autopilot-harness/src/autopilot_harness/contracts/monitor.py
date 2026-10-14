"""Serializable Monitor contracts. All timestamps are UTC Unix seconds."""

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TrainingSpecification:
    values: dict[str, Any]
    missing_fields: list[str]
    source_ref: str


@dataclass
class CurrentConfiguration:
    values: dict[str, Any]
    status: str  # missing | recorded | confirmed
    source_ref: str | None
    configuration_id: str | None
    confirmation: dict[str, Any] | None = None


@dataclass
class Observation:
    metric: str
    value: float | None
    unit: str | None
    entity: dict[str, str]
    observed_at: float | None
    available_at: float
    timestamp_basis: str
    window: dict[str, Any]
    coverage: dict[str, Any]
    basis: str
    source_ref: str
    logging_step: int | None = None
    train_step: int | None = None
    missing_reason: str | None = None


@dataclass
class SourceReport:
    source: str
    mode: str  # live | fixture
    status: str  # ok | empty | partial | error
    reference: str
    details: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


@dataclass
class SourceBatch:
    observations: list[Observation]
    report: SourceReport
    recorded_configuration: dict[str, Any] | None = None


@dataclass
class MonitorSnapshot:
    schema_version: str
    snapshot_id: str
    run_id: str
    segment_id: str
    collection_started_at: float
    collected_at: float
    training_specification: TrainingSpecification
    current_configuration: CurrentConfiguration
    runtime_observations: dict[str, Any]
    execution_history: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
