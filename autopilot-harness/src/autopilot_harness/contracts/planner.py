"""Planner input and result contracts, reusing Monitor context structures."""

from dataclasses import asdict, dataclass, field
from typing import Any

from .monitor import CurrentConfiguration, MonitorSnapshot, TrainingSpecification


@dataclass(frozen=True)
class PlannerInput:
    current_configuration: CurrentConfiguration
    runtime_observations: dict[str, Any]
    training_specification: TrainingSpecification | None = None
    execution_history: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_snapshot(cls, snapshot: MonitorSnapshot) -> "PlannerInput":
        """Expose the existing Monitor context without initializing a Planner."""
        return cls(
            current_configuration=snapshot.current_configuration,
            runtime_observations=snapshot.runtime_observations,
            training_specification=snapshot.training_specification,
            execution_history=snapshot.execution_history,
        )


@dataclass(frozen=True)
class PlannerResult:
    config: dict[str, Any]
    changed: bool
    search_status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
