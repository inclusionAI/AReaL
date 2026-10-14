"""Contracts for checking candidate configurations against harness-owned policy."""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ParameterRule:
    path: tuple[str, ...]
    kind: str
    nullable: bool = False
    minimum: int | float | None = None
    maximum: int | float | None = None
    exclusive_minimum: bool = False
    exclusive_maximum: bool = False
    choices: tuple[Any, ...] | None = None
    active_backend: str | None = None
    source_ref: str = ""


@dataclass(frozen=True)
class ValidationPolicy:
    name: str
    policy_id: str
    runtime_revision: str
    parameters: tuple[ParameterRule, ...]
    checks: tuple[str, ...]
    deferred_paths: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class ConfigurationChange:
    # Segments avoid confusing a literal dotted key with a nested path.
    path: tuple[str, ...]
    operation: str  # add | remove | replace


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    path: tuple[str, ...]
    message: str
    severity: str = "error"  # error | unknown
    source_ref: str | None = None


@dataclass(frozen=True)
class ValidationResult:
    status: str  # passed | rejected | undetermined
    policy_id: str
    runtime_revision: str | None
    baseline_id: str | None
    candidate_id: str | None
    changes: tuple[ConfigurationChange, ...]
    issues: tuple[ValidationIssue, ...]
    checked_rules: tuple[str, ...]
    scope: str = "configuration_and_transition"
    runtime_applicability: str = "undetermined"
    context_id: str | None = None
    state_id: str | None = None
    evidence_refs: tuple[str, ...] = ()
    execution_id: str | None = None
    transition_json: str | None = None

    @property
    def passed(self) -> bool:
        """All required checks passed for the exact configurations and context."""
        return self.status == "passed"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
