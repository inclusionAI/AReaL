# SPDX-License-Identifier: Apache-2.0

"""Basic model adapters for historical evaluation; no fitting or reference labels."""

from __future__ import annotations

from dataclasses import dataclass

from areal.autopilot.domains.elastic.model_evaluator import (
    JOINT_BOUNDARY,
    HistoricalInput,
    HistoricalPrediction,
)
from areal.elastic.policy.basic.cost_model.actor import ActorCostModel
from areal.elastic.policy.basic.cost_model.rollout import RolloutCostModel
from areal.elastic.policy.basic.simulator import JointSimulator

ACTOR_BOUNDARY = "trainer_host_wall_with_stats"
ROLLOUT_BOUNDARY = "isolated_service_plus_external_wall"


@dataclass(frozen=True)
class BasicHistoryPredictor:
    """Explicitly bind a frozen model bundle to the historical execution identity.

    For different runtime/model/hardware identities use separate instances or a
    routing predictor. The adapter never substitutes a default config or model.
    The caller owns calibration and supplies its original spans to the evaluator.
    """

    execution_id: str
    actor: ActorCostModel | None = None
    rollout: RolloutCostModel | None = None
    joint: JointSimulator | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.execution_id, str) or not self.execution_id.strip():
            raise ValueError("Audited execution identity is required")

    def predict(self, inputs: HistoricalInput) -> HistoricalPrediction:
        if inputs.execution_id != self.execution_id:
            return HistoricalPrediction(
                reason="Execution/model/hardware/runtime identity mismatch"
            )
        if inputs.kind == "actor":
            if self.actor is None or inputs.timing_boundary != ACTOR_BOUNDARY:
                return HistoricalPrediction(
                    reason="Missing Actor model or unsupported timing boundary"
                )
            result = self.actor.estimate(
                inputs.config, inputs.workload, sensitivity=False
            )
            if result["static_memory_exceeds_capacity"]:
                return HistoricalPrediction(
                    reason="Actor static memory exceeds configured capacity"
                )
            stages = result["stages_seconds"]
            values = tuple(
                (name + "_seconds", stages.get(name))
                for name in ("recompute_logp", "compute_advantage", "train_step")
            )
            # Missing stage estimates do not become an invented zero/partial total.
            total = (
                sum(v for _, v in values)
                if all(v is not None for _, v in values)
                else None
            )
            return HistoricalPrediction((*values, ("actor_total_seconds", total)))
        if inputs.kind == "rollout":
            if self.rollout is None or inputs.timing_boundary != ROLLOUT_BOUNDARY:
                return HistoricalPrediction(
                    reason="Missing Rollout model or unsupported service/wait boundary"
                )
            result = self.rollout.predict(inputs.config, inputs.workload)
            if not result.supported:
                return HistoricalPrediction(reason=result.reason)
            inference = sum(gpu for gpu, _ in result.stages)
            external = sum(wait for _, wait in result.stages)
            return HistoricalPrediction(
                (
                    ("inference_seconds", inference),
                    ("external_seconds", external),
                    ("rollout_seconds", inference + external),
                )
            )
        if self.joint is None or inputs.timing_boundary != JOINT_BOUNDARY:
            return HistoricalPrediction(
                reason="Missing Joint model or unsupported window boundary"
            )
        if inputs.unsupported_reason:
            return HistoricalPrediction(reason=inputs.unsupported_reason)
        result = self.joint.evaluate(inputs.workload, inputs.config)
        if result.status != "complete":
            return HistoricalPrediction(
                reason=f"Joint {result.status}: {result.reason}"
            )
        batches = inputs.workload.batches
        expected = (
            len(batches),
            sum(b.actor.samples for b in batches),
            sum(b.actor.effective_tokens for b in batches),
            sum(b.actor.sequence_tokens for b in batches),
        )
        if (
            result.config != inputs.config
            or (
                result.completed_batches,
                result.samples,
                result.effective_tokens,
                result.sequence_tokens,
            )
            != expected
        ):
            raise ValueError(
                "Joint prediction changed historical configuration or logical progress"
            )
        return HistoricalPrediction(
            tuple(
                (name, getattr(result, name))
                for name in (
                    "seconds",
                    "actor_gpu_seconds",
                    "rollout_gpu_seconds",
                    "total_gpu_seconds",
                    "queue_wait_seconds",
                )
            )
        )
