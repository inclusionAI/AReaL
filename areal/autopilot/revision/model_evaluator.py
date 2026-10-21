# SPDX-License-Identifier: Apache-2.0

"""Fixed historical measurements for offline cost-model revision acceptance.

Prediction receives only HistoricalInput. Calibration/validation/final partitions,
labels, scoring and acceptance belong to the evaluator, not the candidate.
"""

from __future__ import annotations

import copy
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from areal.autopilot.contracts import WorkloadStats
from areal.autopilot.elastic.policy.basic.cost_model.config import ActorConfig
from areal.autopilot.elastic.policy.basic.cost_model.rollout import (
    RolloutConfig,
    RolloutWorkload,
)
from areal.autopilot.elastic.policy.basic.joint import JointConfig, JointScenario
from areal.autopilot.revision.contracts import digest, source_digest

JOINT_BOUNDARY = "committed_drained_window_through_shutdown"

METRICS = {
    "actor": frozenset(
        {
            "recompute_logp_seconds",
            "compute_advantage_seconds",
            "train_step_seconds",
            "actor_total_seconds",
        }
    ),
    "rollout": frozenset({"inference_seconds", "external_seconds", "rollout_seconds"}),
    "joint": frozenset(
        {
            "seconds",
            "actor_gpu_seconds",
            "rollout_gpu_seconds",
            "total_gpu_seconds",
            "queue_wait_seconds",
        }
    ),
}


def _number(name: str, value: float) -> None:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True)
class HistorySpan:
    """Inclusive original run/step interval; rank/phase aliases share this identity."""

    run_id: str
    first_step: int
    last_step: int

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError("Original run identity is required")
        if (
            any(type(n) is not int or n < 0 for n in (self.first_step, self.last_step))
            or self.last_step < self.first_step
        ):
            raise ValueError("History span requires ordered nonnegative integer steps")

    def overlaps(self, other: HistorySpan) -> bool:
        return self.run_id == other.run_id and max(
            self.first_step, other.first_step
        ) <= min(self.last_step, other.last_step)


@dataclass(frozen=True)
class HistoricalInput:
    """One actual configuration/workload, without labels or run/step identifiers.

    execution_id identifies model, hardware, runtime and unrepresented execution
    options. Timing scope is explicit. Logical mode rejects dispatch/packing
    observations; given_packing is conditional and scored separately.
    Rollout/joint inputs always declare their supplied external wait durations.
    Joint initial-state/cleanup flags require an audit of the actual trace.
    """

    kind: str
    config: ActorConfig | RolloutConfig | JointConfig
    workload: WorkloadStats | RolloutWorkload | JointScenario
    execution_id: str
    timing_boundary: str
    input_mode: str = "logical"
    initial_state_verified: bool | None = None
    shutdown_included: bool | None = None

    def __post_init__(self) -> None:
        types = {
            "actor": (ActorConfig, WorkloadStats),
            "rollout": (RolloutConfig, RolloutWorkload),
            "joint": (JointConfig, JointScenario),
        }
        if (
            self.kind not in types
            or not isinstance(self.config, types[self.kind][0])
            or not isinstance(self.workload, types[self.kind][1])
        ):
            raise ValueError("Historical kind/config/workload types disagree")
        if not all(
            isinstance(x, str) and x.strip()
            for x in (self.execution_id, self.timing_boundary)
        ):
            raise ValueError(
                "Execution identity and measured timing boundary are required"
            )
        modes = {
            "actor": {"logical", "given_packing"},
            "rollout": {"given_external"},
            "joint": {"given_external", "given_packing_and_external"},
        }
        if self.input_mode not in modes[self.kind]:
            raise ValueError("Unknown historical input mode")
        for value in (self.initial_state_verified, self.shutdown_included):
            if value is not None and type(value) is not bool:
                raise ValueError(
                    "Trace boundary assertions must be explicit booleans or unknown"
                )
        stats = (
            (self.workload,)
            if self.kind == "actor"
            else (
                tuple(batch.actor.actor_stats for batch in self.workload.batches)
                if self.kind == "joint"
                else ()
            )
        )
        if self.input_mode in {"logical", "given_external"} and any(
            s.train_microbatches is not None or s.per_dp_sequence_lengths is not None
            for s in stats
        ):
            raise ValueError(
                "Logical prediction must not receive measured rank packing/counts; use given_packing explicitly"
            )

    @property
    def config_id(self) -> str:
        return digest(
            {
                "kind": self.kind,
                "execution_id": self.execution_id,
                "config": asdict(self.config),
            }
        )

    @property
    def unsupported_reason(self) -> str:
        if self.kind == "joint" and (
            self.initial_state_verified is not True
            or self.shutdown_included is not True
            or self.timing_boundary != JOINT_BOUNDARY
        ):
            return "Joint history requires verified committed/drained initial state and measured shutdown boundary"
        return ""


@dataclass(frozen=True)
class HistoricalCase:
    sample_id: str
    span: HistorySpan
    split: str
    inputs: HistoricalInput
    measured: tuple[tuple[str, float | None], ...]
    source_id: str
    workload_group: str = "all"

    def __post_init__(self) -> None:
        object.__setattr__(self, "measured", tuple(tuple(x) for x in self.measured))
        if self.split not in {"calibration", "validation", "final_holdout"}:
            raise ValueError(
                "Explicit calibration/validation/final_holdout split required"
            )
        if not all(
            isinstance(x, str) and x.strip()
            for x in (self.sample_id, self.source_id, self.workload_group)
        ):
            raise ValueError(
                "Sample identity, measurement source and workload group are required"
            )
        if not self.measured or len(dict(self.measured)) != len(self.measured):
            raise ValueError("Historical metric labels must have unique names")
        for name, value in self.measured:
            if name not in METRICS[self.inputs.kind]:
                raise ValueError(
                    f"Unknown measured metric for {self.inputs.kind}: {name}"
                )
            if value is not None:
                _number(name, value)


@dataclass(frozen=True)
class HistoricalPrediction:
    values: tuple[tuple[str, float | None], ...] = ()
    reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", tuple(tuple(x) for x in self.values))
        if len(dict(self.values)) != len(self.values):
            raise ValueError("Duplicate predicted metric")
        for name, value in self.values:
            if value is not None:
                _number(name, value)


class HistoricalPredictor(Protocol):
    def predict(self, inputs: HistoricalInput) -> HistoricalPrediction: ...


@dataclass(frozen=True)
class HistoryRow:
    sample_id: str
    kind: str
    boundary: str
    input_mode: str
    metric: str
    config_id: str
    workload_group: str
    measured: float | None
    predicted: float | None
    reason: str = ""

    @property
    def cell(self) -> tuple[str, str]:
        return self.sample_id, self.metric

    @property
    def covered(self) -> bool:
        return self.measured is not None and self.predicted is not None

    @property
    def metric_key(self) -> tuple[str, ...]:
        return self.kind, self.boundary, self.input_mode, self.metric


@dataclass(frozen=True)
class HistoryMetric:
    kind: str
    boundary: str
    input_mode: str
    metric: str
    group_by: str
    group: str
    cells: int
    labelled: int
    covered: int
    absolute_error: float
    measured_sum: float
    wape: float | None
    mae: float | None
    p90_relative_error: float | None
    relative_error_samples: int
    normalized_error: float | None

    @property
    def key(self) -> tuple[str, ...]:
        return (
            self.kind,
            self.boundary,
            self.input_mode,
            self.metric,
            self.group_by,
            self.group,
        )


def _summaries(
    rows: tuple[HistoryRow, ...], group_by: str = "all"
) -> tuple[HistoryMetric, ...]:
    groups = defaultdict(list)
    for row in rows:
        group = "all" if group_by == "all" else getattr(row, group_by)
        groups[(*row.metric_key, group)].append(row)
    result = []
    for (*key, group), members in sorted(groups.items()):
        known = [row for row in members if row.covered]
        absolute = math.fsum(abs(row.predicted - row.measured) for row in known)
        measured = math.fsum(row.measured for row in known)
        relative = sorted(
            abs(row.predicted - row.measured) / row.measured
            for row in known
            if row.measured > 0
        )
        # Keep undefined WAPE distinct from error on all-zero labels. For scoring,
        # that special case uses MAE in the metric's native unit (seconds/GPU-s).
        result.append(
            HistoryMetric(
                *key,
                group_by,
                group,
                len(members),
                sum(r.measured is not None for r in members),
                len(known),
                absolute,
                measured,
                absolute / measured if measured else None,
                absolute / len(known) if known else None,
                relative[math.ceil(0.9 * len(relative)) - 1] if relative else None,
                len(relative),
                absolute / (measured or len(known)) if known else None,
            )
        )
    return tuple(result)


def _score(
    metrics: tuple[HistoryMetric, ...], primary: tuple[str, ...]
) -> float | None:
    selected = [m for m in metrics if m.metric in primary]
    if (
        not selected
        or {m.metric for m in selected} != set(primary)
        or any(m.normalized_error is None for m in selected)
    ):
        return None
    return math.fsum(m.normalized_error for m in selected) / len(selected)


@dataclass(frozen=True)
class ModelAcceptanceCriteria:
    min_relative_improvement: float = 0.005
    max_metric_regression: float = 0.0
    max_slice_regression: float = 0.0
    max_added_error: float = 0.1
    bootstrap_max_error: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "min_relative_improvement",
            "max_metric_regression",
            "max_slice_regression",
            "max_added_error",
        ):
            _number(name, getattr(self, name))
        if self.min_relative_improvement >= 1:
            raise ValueError("Minimum relative improvement must be below one")
        if self.bootstrap_max_error is not None:
            _number("bootstrap_max_error", self.bootstrap_max_error)


@dataclass(frozen=True)
class HistoryReport:
    candidate_version: str
    benchmark_sha256: str
    valid: bool
    score: float | None
    metrics: tuple[HistoryMetric, ...]
    slices: tuple[HistoryMetric, ...]
    rows: tuple[HistoryRow, ...]
    errors: tuple[str, ...]
    fitted_on: tuple[HistorySpan, ...]
    reserved_cases: int

    def feedback(self) -> dict:
        """Aggregate feedback; suppress small slices and absolute error statistics.

        This limits direct label disclosure, not inference from repeated queries.
        Validation is development data; final_holdout must remain untouched.
        """

        def public(metric):
            values = asdict(metric)
            for name in ("absolute_error", "measured_sum", "mae", "normalized_error"):
                values.pop(name)
            values["error_suppressed"] = metric.covered < 5
            if values["error_suppressed"]:
                values["wape"] = values["p90_relative_error"] = None
            return values

        return {
            "candidate_version": self.candidate_version,
            "benchmark_sha256": self.benchmark_sha256,
            "valid": self.valid,
            "score": self.score
            if self.metrics and all(m.covered >= 5 for m in self.metrics if m.labelled)
            else None,
            "errors": list(self.errors),
            "metrics": [public(m) for m in self.metrics],
            "slices": [public(m) for m in self.slices],
        }


@dataclass(frozen=True)
class ModelComparison:
    accepted: bool
    reason: str
    incumbent_score: float | None
    candidate_score: float | None
    retained_cells: int
    added_cells: int
    lost_cells: int


class HistoricalModelEvaluator:
    """Frozen offline evaluator; repeated validation feedback is development data.

    final_holdout is fingerprinted and protected from fitting but never predicted
    or scored here. Supplied fitted_on provenance is a trusted caller declaration,
    not proof of what arbitrary Python previously read. No fitting occurs here.
    """

    def __init__(
        self,
        cases: tuple[HistoricalCase, ...],
        *,
        primary_metrics: tuple[str, ...],
        criteria: ModelAcceptanceCriteria | None = None,
    ):
        self._cases = copy.deepcopy(tuple(sorted(cases, key=lambda c: c.sample_id)))
        self._primary_metrics = tuple(primary_metrics)
        self._criteria = criteria or ModelAcceptanceCriteria()
        if not self._cases or len({c.sample_id for c in self._cases}) != len(
            self._cases
        ):
            raise ValueError("Historical cases require unique sample IDs")
        if not self.primary_metrics or len(set(self.primary_metrics)) != len(
            self.primary_metrics
        ):
            raise ValueError("Freeze a nonempty unique set of primary metrics")
        if "external_seconds" in self.primary_metrics:
            raise ValueError(
                "Known external waits cannot be a primary prediction metric"
            )
        if "rollout_seconds" in self.primary_metrics:
            if "inference_seconds" not in self.primary_metrics:
                raise ValueError(
                    "Rollout total acceptance also requires inference_seconds"
                )
            for case in self._cases:
                labels = dict(case.measured)
                if (
                    case.split == "validation"
                    and labels.get("rollout_seconds") is not None
                    and labels.get("inference_seconds") is None
                ):
                    raise ValueError(
                        "Each scored Rollout total needs an independent inference label"
                    )
        seen = set()
        for i, case in enumerate(self._cases):
            for other in self._cases[:i]:
                if case.split != other.split and case.span.overlaps(other.span):
                    raise ValueError(
                        "Calibration/validation/final history spans overlap"
                    )
            for metric, _ in case.measured:
                key = (
                    case.span,
                    case.inputs.kind,
                    case.inputs.timing_boundary,
                    case.inputs.input_mode,
                    metric,
                )
                if key in seen:
                    raise ValueError(
                        "Duplicate original measurement under different sample IDs"
                    )
                seen.add(key)
        validation = [c for c in self._cases if c.split == "validation"]
        available = {
            name for c in validation for name, value in c.measured if value is not None
        }
        if not set(self.primary_metrics) <= available:
            raise ValueError("Every primary metric needs a measured validation label")
        # Do not claim an improvement by mixing conditional and logical modes or
        # incompatible measurement boundaries in a single acceptance decision.
        for kind in METRICS:
            scopes = {
                (c.inputs.timing_boundary, c.inputs.input_mode)
                for c in validation
                if c.inputs.kind == kind
            }
            if len(scopes) > 1:
                raise ValueError(
                    "Use separate evaluators for timing boundaries/input modes"
                )
        root = Path(__file__).parents[1]
        self._reference_files = tuple(
            root / name
            for name in (
                "revision/model_evaluator.py",
                "revision/model_adapters.py",
                "contracts.py",
                "elastic/policy/basic/joint.py",
                "elastic/policy/basic/cost_model/config.py",
                "elastic/policy/basic/cost_model/rollout.py",
            )
        )
        self._code = self._code_fingerprint()
        self._benchmark_sha256 = digest(
            {
                "cases": [asdict(c) for c in self._cases],
                "primary_metrics": self.primary_metrics,
                "criteria": asdict(self.criteria),
                "reference_code": self._code,
            }
        )
        self._issued_reports: set[str] = set()

    def _code_fingerprint(self) -> tuple[str, ...]:
        return tuple(source_digest(path.read_text()) for path in self._reference_files)

    @property
    def primary_metrics(self) -> tuple[str, ...]:
        return self._primary_metrics

    @property
    def criteria(self) -> ModelAcceptanceCriteria:
        return self._criteria

    @property
    def benchmark_sha256(self) -> str:
        return self._benchmark_sha256

    @property
    def calibration(self) -> tuple[HistoricalCase, ...]:
        return copy.deepcopy(tuple(c for c in self._cases if c.split == "calibration"))

    @property
    def prediction_inputs(self) -> tuple[HistoricalInput, ...]:
        """Ordered prediction requests; evaluator-owned unsupported traces stay out."""
        return copy.deepcopy(
            tuple(
                c.inputs
                for c in self._cases
                if c.split == "validation" and not c.inputs.unsupported_reason
            )
        )

    def validate_fit(self, fitted_on: tuple[HistorySpan, ...]) -> None:
        protected = [c for c in self._cases if c.split != "calibration"]
        if any(not isinstance(span, HistorySpan) for span in fitted_on):
            raise ValueError("Fitting provenance requires original HistorySpan values")
        if any(span.overlaps(case.span) for span in fitted_on for case in protected):
            raise ValueError("Model fit overlaps validation or final holdout history")

    def assert_unchanged(self) -> None:
        if self._code_fingerprint() != self._code:
            raise ValueError(
                "Historical evaluator/reference code changed; create a new evaluation context"
            )

    def evaluate(
        self,
        predictor: HistoricalPredictor,
        version: str,
        *,
        fitted_on: tuple[HistorySpan, ...],
    ) -> HistoryReport:
        if not isinstance(version, str) or not version.strip():
            raise ValueError("Candidate version/snapshot identity is required")
        self.assert_unchanged()
        fitted_on = tuple(fitted_on)
        self.validate_fit(fitted_on)
        rows, errors = [], []
        for case in self._cases:
            if case.split != "validation":
                continue
            values, reason = {}, ""
            try:
                prediction = (
                    HistoricalPrediction(reason=case.inputs.unsupported_reason)
                    if case.inputs.unsupported_reason
                    else predictor.predict(copy.deepcopy(case.inputs))
                )
                if not isinstance(prediction, HistoricalPrediction):
                    raise ValueError("Candidate must return HistoricalPrediction")
                # Revalidate even if trusted code mutates an otherwise frozen value.
                prediction = HistoricalPrediction(prediction.values, prediction.reason)
                values, reason = dict(prediction.values), prediction.reason
                if not values.keys() <= METRICS[case.inputs.kind]:
                    raise ValueError("Candidate returned unknown metric names")
                if (
                    "rollout_seconds" in self.primary_metrics
                    and values.get("rollout_seconds") is not None
                    and values.get("inference_seconds") is None
                ):
                    raise ValueError(
                        "Scored Rollout total requires a paired inference prediction"
                    )
            except Exception as error:
                reason = f"{type(error).__name__}: {error}"
                errors.append(f"{case.sample_id}: {reason}")
                values = {}
            for name, measured in case.measured:
                value = values.get(name)
                rows.append(
                    HistoryRow(
                        case.sample_id,
                        case.inputs.kind,
                        case.inputs.timing_boundary,
                        case.inputs.input_mode,
                        name,
                        case.inputs.config_id,
                        case.workload_group,
                        measured,
                        value,
                        (reason or "Metric prediction unavailable")
                        if value is None
                        else "",
                    )
                )
        rows = tuple(rows)
        if self._code_fingerprint() != self._code:
            errors.append(
                "Historical evaluator/reference code changed during prediction"
            )
        metrics = _summaries(rows)
        report = HistoryReport(
            version,
            self.benchmark_sha256,
            not errors,
            _score(metrics, self.primary_metrics),
            metrics,
            _summaries(rows, "config_id") + _summaries(rows, "workload_group"),
            rows,
            tuple(errors),
            fitted_on,
            sum(c.split == "final_holdout" for c in self._cases),
        )
        self._issued_reports.add(digest(asdict(report)))
        return report

    def compare(
        self, candidate: HistoryReport, incumbent: HistoryReport
    ) -> ModelComparison:
        if any(
            r.benchmark_sha256 != self.benchmark_sha256 for r in (candidate, incumbent)
        ):
            raise ValueError("Cannot compare different historical evaluation contexts")
        if self._code_fingerprint() != self._code:
            raise ValueError(
                "Historical evaluator/reference code changed before comparison"
            )
        if any(
            digest(asdict(r)) not in self._issued_reports
            for r in (candidate, incumbent)
        ):
            raise ValueError(
                "Comparison requires unchanged reports issued by this evaluator"
            )
        if not incumbent.valid:
            raise ValueError("Incumbent execution must be valid before comparison")
        old = {r.cell: r for r in incumbent.rows if r.covered}
        new = {r.cell: r for r in candidate.rows if r.covered}
        retained, added, lost = (
            old.keys() & new.keys(),
            new.keys() - old.keys(),
            old.keys() - new.keys(),
        )
        before = _score(_summaries(tuple(old.values())), self.primary_metrics)
        after = _score(
            _summaries(tuple(new[k] for k in sorted(retained))), self.primary_metrics
        )

        def decision(accepted, reason):
            return ModelComparison(
                accepted, reason, before, after, len(retained), len(added), len(lost)
            )

        if not candidate.valid:
            return decision(False, "Candidate execution or output failed")
        if lost:
            return decision(
                False, "Candidate dropped previously covered sample/metric cells"
            )
        if not old:
            after = candidate.score
            limit = self.criteria.bootstrap_max_error
            accepted = (
                limit is not None
                and candidate.score is not None
                and all(
                    m.covered == m.labelled and m.normalized_error <= limit
                    for m in (*candidate.metrics, *candidate.slices)
                    if m.labelled
                )
            )
            return decision(
                accepted,
                "Bootstrap accuracy/coverage passed"
                if accepted
                else "Empty incumbent coverage requires explicit bootstrap accuracy and full coverage",
            )
        added_rows = tuple(new[k] for k in sorted(added))
        added_metrics = tuple(
            m
            for grouping in ("all", "config_id", "workload_group")
            for m in _summaries(added_rows, grouping)
        )
        if any(
            m.normalized_error > self.criteria.max_added_error for m in added_metrics
        ):
            return decision(False, "New coverage exceeds its fixed error limit")
        for grouping, limit in (
            ("all", self.criteria.max_metric_regression),
            ("config_id", self.criteria.max_slice_regression),
            ("workload_group", self.criteria.max_slice_regression),
        ):
            old_metrics = {m.key: m for m in _summaries(tuple(old.values()), grouping)}
            new_metrics = _summaries(tuple(new[k] for k in sorted(retained)), grouping)
            if any(
                m.normalized_error > old_metrics[m.key].normalized_error + limit + 1e-12
                for m in new_metrics
            ):
                return decision(False, f"Previously covered {grouping} error regressed")
        if before is None or after is None:
            return decision(False, "Primary metrics need common covered measurements")
        improvement = before - after
        if improvement <= max(1e-12, before * self.criteria.min_relative_improvement):
            return decision(
                False, "Insufficient improvement on identical covered measurements"
            )
        return decision(
            True, "Historical prediction error improved without coverage loss"
        )
