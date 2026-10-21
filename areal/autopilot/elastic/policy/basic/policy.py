# SPDX-License-Identifier: Apache-2.0

"""Fixed idle-timeout baseline and history-based basic resource policy.

This baseline is not an agent revision target. Execution and consistency guards
remain in ElasticLoop and runtime, outside revisions to scheduling decisions.
"""

import math
from dataclasses import dataclass

from ....contracts import (
    BatchWorkload,
    Decision,
    ElasticObservation,
    ResourceDecision,
    TrainingConfig,
)
from .joint import JointScenario, JointSearchResult
from .optimizer import BasicOptimizer, JointModel, JointOptimizer, Optimizer
from .simulator import (
    HistoryWindow,
    Measurement,
    WorkflowSimulator,
)


@dataclass(frozen=True)
class IdleTimeoutPolicy:
    """Train a ready batch; release clean idle workers after the timeout."""

    idle_seconds: float = 30

    def __post_init__(self) -> None:
        if not math.isfinite(self.idle_seconds) or self.idle_seconds <= 0:
            raise ValueError("Idle timeout must be finite and positive")

    def decide(self, observation: ElasticObservation) -> Decision:
        if observation.pending_publication or not observation.committed:
            return "keep"
        if observation.batch_ready:
            return "train" if observation.gate_available else "keep"
        if (
            observation.training_active
            and observation.idle_duration >= self.idle_seconds
        ):
            return "release"
        return "keep"


class BasicPolicy:
    """Replan from a causal history window; retain timeout policy as cold fallback."""

    def __init__(
        self,
        simulator: WorkflowSimulator,
        *,
        optimizer: Optimizer | None = None,
        history: HistoryWindow | None = None,
        fallback: IdleTimeoutPolicy | None = None,
    ) -> None:
        self.simulator = simulator
        self.optimizer = optimizer if optimizer is not None else BasicOptimizer()
        self.history = history if history is not None else HistoryWindow()
        self.fallback = fallback if fallback is not None else IdleTimeoutPolicy()
        self.last_search = None

    def decide(self, observation: ElasticObservation) -> ResourceDecision:
        if (
            not observation.batch_ready
            and not observation.episode_active
            and observation.training_active
            and observation.committed
            and not observation.pending_publication
            and observation.idle_duration >= self.fallback.idle_seconds
        ):
            self.last_search = None
            return ResourceDecision(
                "release", reason="No admitted workload before idle timeout"
            )
        self.last_search = self.optimizer.optimize(
            observation, self.simulator, self.history.snapshot()
        )
        if self.last_search.prediction is not None:
            return self.last_search.decision
        if not self.last_search.fallback_allowed:
            return self.last_search.decision
        baseline = self.fallback.decide(observation)
        if baseline == "release":
            return ResourceDecision("release", reason="Cold fallback idle timeout")
        if baseline == "train":
            return ResourceDecision(
                "keep" if observation.training_active else "launch",
                observation.training_config,
                True,
                "Cold fallback trains current configuration",
            )
        return ResourceDecision("keep", reason=self.last_search.decision.reason)

    def record_workload(self, workload: BatchWorkload) -> None:
        self.history.add_workload(workload)

    def record_phase(
        self,
        phase: str,
        config: TrainingConfig | None,
        seconds: float,
        workload: BatchWorkload | None = None,
    ) -> None:
        if config is not None:
            self.history.add_measurement(
                Measurement(
                    phase,
                    config.world_size,
                    seconds,
                    0 if workload is None else workload.sequence_tokens,
                    config.profile_id,
                )
            )


class JointPolicy:
    """Offline joint planning; returns a plan, never an executable runtime action."""

    def __init__(self, simulator: JointModel, optimizer: JointOptimizer) -> None:
        self.simulator = simulator
        self.optimizer = optimizer
        self.last_search: JointSearchResult | None = None

    def plan(self, scenario: JointScenario) -> JointSearchResult:
        self.last_search = self.optimizer.optimize(scenario, self.simulator)
        return self.last_search
