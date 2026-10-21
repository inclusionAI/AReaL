# SPDX-License-Identifier: Apache-2.0

"""Versioned, JSON-compatible boundaries for component code revisions."""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from areal.autopilot.contracts import BatchWorkload, ElasticObservation, TrainingConfig
from areal.autopilot.elastic.policy.basic.simulator import (
    ForecastUpdate,
    HistorySnapshot,
    MeasuredComponent,
    Measurement,
    PlanStep,
    WorkflowModel,
)

Target = Literal["simulator", "optimizer", "model", "joint_optimizer", "serial_model"]


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def source_digest(source: str) -> str:
    return hashlib.sha256(source.encode()).hexdigest()


def lifecycle_profile_digest(profile: dict) -> str:
    """Executable phase state; additional agent calibration data is not fitted state."""
    return digest(
        {k: profile.get(k) for k in ("compatibility", "inference_gpus", "measurements")}
    )


def observation_from_dict(data: dict) -> ElasticObservation:
    data = dict(data)
    for key, cls in (
        ("training_config", TrainingConfig),
        ("checkpoint_config", TrainingConfig),
        ("next_batch", BatchWorkload),
    ):
        if data.get(key) is not None:
            data[key] = cls(**data[key])
    return ElasticObservation(**data)


def history_from_dict(data: dict) -> HistorySnapshot:
    return HistorySnapshot(
        tuple(BatchWorkload(**x) for x in data.get("workloads", ())),
        tuple(Measurement(**x) for x in data.get("measurements", ())),
    )


def forecast_from_dict(data: list[dict]) -> tuple[ForecastUpdate, ...]:
    return tuple(
        ForecastUpdate(BatchWorkload(**x["workload"]), x["wait_seconds"]) for x in data
    )


def plan_from_dict(data: list[dict]) -> tuple[PlanStep, ...]:
    result = []
    for step in data:
        if type(step.get("release_before", False)) is not bool:
            raise ValueError("release_before must be boolean")
        result.append(
            PlanStep(
                TrainingConfig(**step["config"]), step.get("release_before", False)
            )
        )
    return tuple(result)


def workflow_from_profile(profile: dict) -> WorkflowModel:
    measurements = tuple(Measurement(**x) for x in profile["measurements"])
    return WorkflowModel(
        **{
            phase: MeasuredComponent(phase, measurements, scale_tokens=phase == "train")
            for phase in (
                "train",
                "start",
                "restore",
                "checkpoint",
                "publish",
                "release",
            )
        },
        inference_gpus=profile.get("inference_gpus", 1),
    )


def deployment_context(profile: dict, settings: dict) -> dict:
    """Bind accepted code to the model, residency and optimizer contract."""
    return {
        "api_version": 1,
        "compatibility": profile.get("compatibility"),
        "inference_gpus": profile.get("inference_gpus", 1),
        "optimizer_settings": settings,
    }


@dataclass(frozen=True)
class Proposal:
    source: str
    rationale: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("Candidate source is empty")
        if len(self.source.encode()) > 128_000:
            raise ValueError("Candidate source exceeds 128 KB")


@dataclass(frozen=True)
class ProposalRequest:
    target: Target
    round: int
    incumbent_source: str
    incumbent_sha256: str
    calibration: dict
    settings: dict
    feedback: tuple[dict, ...]
    previous_source: str | None = None

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "api": (
                "build_joint_optimizer(settings: dict) -> object implementing optimize(scenario: JointScenario | SerialJointWindow, simulator: JointModel) -> JointSearchResult; settings.candidates are decoded JointConfig objects"
                if self.target == "joint_optimizer"
                else "build_serial_model(profile: dict) -> object implementing predict_timing(SerialTimingInput) -> SerialTimingEstimate; static profile has no validation labels"
                if self.target == "serial_model"
                else "build_model(calibration_cases: tuple[HistoricalCase, ...], settings: dict) -> object implementing predict(HistoricalInput) -> HistoricalPrediction"
                if self.target == "model"
                else "build_simulator(workflow: WorkflowModel) -> object implementing evaluate(observation, forecast, plan, history)"
                if self.target == "simulator"
                else "build_optimizer(settings: dict) -> object implementing optimize(observation, simulator, history), returning SearchResult with a complete fixed-horizon plan"
            ),
            "response_schema": {
                "source": "Python module string",
                "rationale": "short explanation string",
            },
            "component_api": (
                "JointModel is an evaluate(scenario, config) proxy to one frozen model version. Preserve the entire scenario and full configuration identities. Query budget, job time bound, GPU limits and switch_margin are enforced by the evaluator. Only result.config is used; all claimed metrics are ignored. Settings are fixed across revisions."
                if self.target == "joint_optimizer"
                else "Predict conditional phase/whole-episode seconds from full joint config and logical workload. Only calibration updates are provided. Keep coverage, GPU accounting, transition allocations and runtime fixed. Missing timing is None. Runtime wraps the timing predictor; custom evaluate() is ignored."
                if self.target == "serial_model"
                else "HistoricalCase exposes calibration labels only. HistoricalInput contains the actual config/workload, execution identity and timing/input boundary. Return missing metrics as absent or None, never invented zeros. Preserve actual configurations, logical work and coverage. Final holdout is unavailable. Prefer composing BasicHistoryPredictor and existing cost models."
                if self.target == "model"
                else "WorkflowModel contains rollout/start/restore/train/checkpoint/publish/release components. Each estimate(ComponentContext(config, workload, history)) returns Estimate(seconds, extra_gpu_seconds, feasible). Prefer composition via dataclasses.replace."
            ),
            "scope": "Return one Python module as source plus rationale. Keep runtime, evaluator, references, resource limits and factory API fixed. Only calibration data and aggregate feedback are provided.",
        }


@dataclass(frozen=True)
class EvaluationReport:
    target: Target
    candidate_sha256: str
    benchmark_sha256: str
    valid: bool
    score: float | None
    metrics: dict
    errors: tuple[str, ...] = ()


BASELINE_SOURCE = {
    "serial_model": "from areal.autopilot.elastic.policy.basic.online import serial_model_from_profile\ndef build_serial_model(profile):\n    return serial_model_from_profile(profile)\n",
    "joint_optimizer": "from areal.autopilot.elastic.policy.basic.optimizer import JointOptimizer\ndef build_joint_optimizer(settings):\n    return JointOptimizer(**settings)\n",
    "simulator": "from areal.autopilot.elastic.policy.basic.simulator import WorkflowSimulator\ndef build_simulator(workflow):\n    return WorkflowSimulator(workflow)\n",
    "optimizer": "from areal.autopilot.elastic.policy.basic.optimizer import BasicOptimizer\ndef build_optimizer(settings):\n    return BasicOptimizer(**settings)\n",
}
