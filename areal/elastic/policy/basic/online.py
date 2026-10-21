# SPDX-License-Identifier: Apache-2.0

"""Causal joint planning for the strict, one-episode-at-a-time runtime.

Rollout timing is deliberately opaque: an exact configuration/workload observation
includes environment, export and enqueue. It is not an inference throughput rate.
The asynchronous offline JointSimulator keeps its separate workload contract.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, replace

from ...contracts import (
    BatchWorkload,
    ElasticObservation,
    HistorySnapshot,
    HistoryWindow,
    Measurement,
    ResourceDecision,
    TrainingConfig,
)
from .cost_model.rollout import RolloutConfig, nonnegative, positive_int
from .cost_model.transition import (
    JointTransitionModel,
    TransitionPhase,
    TransitionProfile,
)
from .joint import JointConfig, JointPrediction
from .optimizer import JointOptimizer
from .policy import IdleTimeoutPolicy


def workload_key(workload: BatchWorkload) -> tuple:
    """Exact observed shape; never extrapolate a rate to a new token count."""
    return workload.samples, workload.effective_tokens, workload.sequence_tokens


@dataclass(frozen=True)
class EpisodeTiming:
    rollout: RolloutConfig
    workload: BatchWorkload
    seconds: float
    source_id: str

    def __post_init__(self):
        nonnegative("episode seconds", self.seconds)
        if not self.source_id:
            raise ValueError("Episode timing needs provenance")


@dataclass(frozen=True)
class JointPhaseTiming:
    config: JointConfig
    phase: str
    seconds: float
    sequence_tokens: int
    source_id: str

    def __post_init__(self):
        nonnegative("phase seconds", self.seconds)
        if self.phase not in {
            "train",
            "start",
            "restore",
            "checkpoint",
            "publish",
            "release",
        }:
            raise ValueError("Unknown joint service phase")
        if (
            type(self.sequence_tokens) is not int
            or self.sequence_tokens < 0
            or not self.source_id
        ):
            raise ValueError("Joint phase needs valid workload and provenance")


@dataclass(frozen=True)
class JointResourceDecision(ResourceDecision):
    rollout: RolloutConfig | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.config is None or self.rollout is None or not self.train:
            raise ValueError("Joint actions require both roles and a ready update")

    @property
    def target(self) -> JointConfig:
        return JointConfig(self.config, self.rollout)


@dataclass(frozen=True)
class SerialBatch:
    actor: BatchWorkload


@dataclass(frozen=True)
class SerialJointWindow:
    batches: tuple[SerialBatch, ...]
    current: JointConfig
    observation: ElasticObservation
    history: HistorySnapshot
    episodes: tuple[EpisodeTiming, ...]
    phases: tuple[JointPhaseTiming, ...]
    max_total_gpus: int


@dataclass(frozen=True)
class SerialTimingInput:
    config: JointConfig
    phase: str
    workload: BatchWorkload
    phases: tuple[JointPhaseTiming, ...] = ()
    episodes: tuple[EpisodeTiming, ...] = ()


@dataclass(frozen=True)
class SerialTimingEstimate:
    seconds: float | None
    reason: str = ""

    def __post_init__(self):
        if self.seconds is not None:
            nonnegative("predicted timing", self.seconds)


def logical_workload(workload):
    """Timing targets must not be hidden inside a workload's rollout_seconds."""
    return BatchWorkload(
        workload.samples, workload.effective_tokens, workload.sequence_tokens
    )


def serial_config(row):
    return (
        None
        if row is None
        else JointConfig(
            TrainingConfig(**row["actor"]), RolloutConfig(**row["rollout"])
        )
    )


def serial_model_from_profile(profile, *, idle_seconds=30):
    if profile.get("format_version") != "serial-joint-v1":
        raise ValueError("Unsupported serial joint profile")
    return SerialJointModel(
        tuple(
            JointPhaseTiming(**{**row, "config": serial_config(row["config"])})
            for row in profile["phases"]
        ),
        tuple(
            EpisodeTiming(
                **{
                    **row,
                    "rollout": RolloutConfig(**row["rollout"]),
                    "workload": BatchWorkload(**row["workload"]),
                }
            )
            for row in profile["episodes"]
        ),
        JointTransitionModel(
            tuple(
                TransitionProfile(
                    **{
                        **row,
                        "source": serial_config(row["source"]),
                        "target": serial_config(row["target"]),
                        "phases": tuple(TransitionPhase(**p) for p in row["phases"]),
                    }
                )
                for row in profile["transitions"]
            )
        ),
        idle_seconds=idle_seconds,
    )


def timing_input_from_dict(row):
    return SerialTimingInput(
        serial_config(row["config"]),
        row["phase"],
        logical_workload(BatchWorkload(**row["workload"])),
        tuple(
            JointPhaseTiming(**{**p, "config": serial_config(p["config"])})
            for p in row.get("phases", ())
        ),
        tuple(
            EpisodeTiming(
                **{
                    **p,
                    "rollout": RolloutConfig(**p["rollout"]),
                    "workload": BatchWorkload(**p["workload"]),
                }
            )
            for p in row.get("episodes", ())
        ),
    )


def serial_window_from_dict(row):
    from .serialization import history_from_dict, observation_from_dict

    timing = timing_input_from_dict(
        {
            "config": row["current"],
            "phase": "train",
            "workload": row["batches"][0]["actor"],
            "phases": row["phases"],
            "episodes": row["episodes"],
        }
    )
    return SerialJointWindow(
        tuple(SerialBatch(BatchWorkload(**b["actor"])) for b in row["batches"]),
        timing.config,
        observation_from_dict(row["observation"]),
        history_from_dict(row["history"]),
        timing.episodes,
        timing.phases,
        row["max_total_gpus"],
    )


class SerialJointModel:
    """Fixed timeout lifecycle, checkpoint/update and whole-episode forecasts.

    Only future episodes are charged. The first batch is already ready. Directed
    switch profiles include all restart/reload work and temporary residency;
    missing profiles reject a switch. Terminal Actor release is included, while
    Rollout stays resident throughout this finite window and release.
    """

    def __init__(
        self,
        phases: tuple[JointPhaseTiming, ...],
        episodes: tuple[EpisodeTiming, ...],
        transitions: JointTransitionModel,
        *,
        idle_seconds: float = 30,
    ):
        self.phases, self.episodes, self.transitions = (
            tuple(phases),
            tuple(episodes),
            transitions,
        )
        self.idle_seconds = IdleTimeoutPolicy(idle_seconds).idle_seconds
        self.timing_predictor = None

    def predict_timing(self, context: SerialTimingInput) -> SerialTimingEstimate:
        if context.phase == "episode":
            samples = [
                row
                for row in (*self.episodes, *context.episodes)
                if row.rollout == context.config.rollout
                and workload_key(row.workload) == workload_key(context.workload)
            ][-5:]
        else:
            samples = [
                row
                for row in (*self.phases, *context.phases)
                if row.config == context.config and row.phase == context.phase
            ]
            if context.phase == "train":
                samples = [row for row in samples if row.sequence_tokens > 0]
        if not samples:
            return SerialTimingEstimate(
                None,
                f"No {context.phase} profile for the complete joint configuration/workload",
            )
        if context.phase != "train":
            return SerialTimingEstimate(
                sum(row.seconds for row in samples[-5:]) / len(samples[-5:])
            )
        latest = min(
            reversed(samples),
            key=lambda row: abs(
                math.log(context.workload.sequence_tokens / row.sequence_tokens)
            ),
        )
        ratio = context.workload.sequence_tokens / latest.sequence_tokens
        return (
            SerialTimingEstimate(latest.seconds * ratio)
            if 0.25 <= ratio <= 4
            else SerialTimingEstimate(
                None, "Actor workload outside the measured joint phase range"
            )
        )

    def estimate_timing(self, context: SerialTimingInput) -> SerialTimingEstimate:
        context = replace(context, workload=logical_workload(context.workload))
        coverage = self.predict_timing(context)
        if coverage.seconds is None or self.timing_predictor is None:
            return coverage
        # A candidate receives detached CPU inputs; fixed runtime accounting and
        # coverage are never delegated to candidate.evaluate().
        first = self.timing_predictor.predict_timing(copy.deepcopy(context))
        second = self.timing_predictor.predict_timing(copy.deepcopy(context))
        if not isinstance(first, SerialTimingEstimate) or first != second:
            raise ValueError("Serial timing predictor must be typed and repeatable")
        return SerialTimingEstimate(first.seconds, first.reason)

    def evaluate(
        self, window: SerialJointWindow, config: JointConfig
    ) -> JointPrediction:
        if not isinstance(window, SerialJointWindow):
            raise TypeError("Serial joint model requires a strict runtime window")
        if config.world_size > window.max_total_gpus:
            return JointPrediction(
                "infeasible", config, reason="Joint GPU pool exceeded"
            )
        source = window.current
        identity = ("model_id", "hardware_id", "runtime_id", "sampling_id", "backend")
        if any(
            getattr(source.rollout, k) != getattr(config.rollout, k) for k in identity
        ):
            return JointPrediction(
                "infeasible", config, reason="Rollout identity changed"
            )
        if config.rollout.max_inflight_batches != 1:
            return JointPrediction(
                "infeasible", config, reason="Strict runtime needs one inflight batch"
            )
        seconds = actor_cost = rollout_cost = 0.0
        sources = []
        active = window.observation.training_active
        completed = 0

        def charge(duration, actor_gpus, rollout_gpus):
            nonlocal seconds, actor_cost, rollout_cost
            seconds += duration
            actor_cost += duration * actor_gpus
            rollout_cost += duration * rollout_gpus

        def phase(name, workload):
            estimate = self.estimate_timing(
                SerialTimingInput(
                    config, name, workload, window.phases, window.episodes
                )
            )
            if estimate.seconds is None:
                raise ValueError(estimate.reason)
            return estimate.seconds

        try:
            if source != config:
                profile = self.transitions.lookup("reconfigure", source, config)
                if profile is None:
                    raise ValueError("Missing directed joint switch profile")
                for interval in profile.phases:
                    if (
                        interval.actor_gpus + interval.rollout_gpus
                        > window.max_total_gpus
                    ):
                        return JointPrediction(
                            "infeasible", config, reason="Switch exceeds shared pool"
                        )
                    charge(interval.seconds, interval.actor_gpus, interval.rollout_gpus)
                sources.append(profile.source_id)
                # The executor activates on an Actor or placement change. A
                # Rollout-only capacity change preserves a suspended Actor.
                active = (
                    active
                    or source.actor != config.actor
                    or source.rollout.world_size != config.rollout.world_size
                )
            for index, batch in enumerate(window.batches):
                if index:
                    wait = phase("episode", batch.actor)
                    if active and wait >= self.idle_seconds:
                        release = phase("release", batch.actor)
                        # Release may overlap the episode, but update waits for both.
                        charge(
                            self.idle_seconds + release,
                            config.actor.world_size,
                            config.rollout.world_size,
                        )
                        charge(
                            max(0, wait - self.idle_seconds - release),
                            0,
                            config.rollout.world_size,
                        )
                        active = False
                    else:
                        charge(
                            wait,
                            config.actor.world_size if active else 0,
                            config.rollout.world_size,
                        )
                if not active:
                    name = (
                        "restore"
                        if index or window.observation.has_checkpoint
                        else "start"
                    )
                    charge(
                        phase(name, batch.actor),
                        config.actor.world_size,
                        config.rollout.world_size,
                    )
                    active = True
                for name in ("train", "checkpoint", "publish"):
                    charge(
                        phase(name, batch.actor),
                        config.actor.world_size,
                        config.rollout.world_size,
                    )
                completed += 1
            charge(
                phase("release", window.batches[-1].actor),
                config.actor.world_size,
                config.rollout.world_size,
            )
        except ValueError as error:
            return JointPrediction("unsupported", config, reason=str(error))
        return JointPrediction(
            "complete",
            config,
            seconds,
            actor_cost,
            rollout_cost,
            completed,
            sum(b.actor.samples for b in window.batches),
            sum(b.actor.effective_tokens for b in window.batches),
            sum(b.actor.sequence_tokens for b in window.batches),
            assumptions=(
                "Strict serial updates; actual ready batch plus repeated causal workload",
                "Exact Rollout config and aggregate workload shape; whole-episode durations, no TP speedup inference",
                "Idle-timeout release, checkpoint each update, terminal Actor release; Rollout remains resident",
            ),
            source_ids=tuple(dict.fromkeys(sources)),
        )


class OnlineJointPolicy:
    """Offline serial joint reference; native topology changes are unsupported."""

    def __init__(
        self,
        simulator: SerialJointModel,
        optimizer: JointOptimizer,
        *,
        horizon: int = 2,
    ):
        positive_int("joint horizon", horizon)
        if horizon < 2:
            raise ValueError("Joint runtime horizon needs at least one future episode")
        self.simulator, self.optimizer, self.horizon = simulator, optimizer, horizon
        self.fallback = IdleTimeoutPolicy(simulator.idle_seconds)
        self.history = HistoryWindow()
        self.episodes: list[EpisodeTiming] = []
        self.phases: list[JointPhaseTiming] = []
        self.last_search, self.last_window = None, None
        self.through_version = 0
        self._boundary_sequence = 0

    def decide(self, observation: ElasticObservation):
        """Cold/idle path retains the current configuration and timeout behavior."""
        self.last_search, self.last_window = None, None
        return self.fallback.decide(observation)

    def validate_executor(self, executor):
        """Reject an incompatible allowlist before starting any services."""
        if not callable(getattr(executor, "placement", None)):
            raise ValueError(
                "Online joint planning requires a joint placement executor"
            )
        for config in self.optimizer.candidates:
            executor.validate(config)

    def decide_joint(self, observation, current, max_total_gpus):
        fallback = self.decide(observation)
        if (
            not observation.batch_ready
            or observation.next_batch is None
            or not observation.gate_available
            or observation.pending_publication
            or not observation.committed
        ):
            return fallback
        if self.through_version > observation.model_version:
            raise ValueError("Joint history is newer than runtime weights")
        snapshot = self.history.snapshot()
        if not snapshot.workloads:
            return fallback
        latest = snapshot.workloads[-1]
        # Measured rank packing belongs to its source configuration. This local
        # fixture only consumes logical counts, never rank observations.
        batches = (SerialBatch(observation.next_batch),) + tuple(
            SerialBatch(replace(latest, actor_stats=None))
            for _ in range(self.horizon - 1)
        )
        self.last_window = SerialJointWindow(
            batches,
            current,
            observation,
            snapshot,
            tuple(self.episodes),
            tuple(self.phases),
            max_total_gpus,
        )
        self.last_search = self.optimizer.optimize(self.last_window, self.simulator)
        target = self.last_search.config
        if target is None:
            if self.last_search.candidates and all(
                row.prediction.status == "unsupported"
                for row in self.last_search.candidates
            ):
                return fallback
            # A deadline/infeasibility failure cannot be bypassed by training.
            return ResourceDecision("keep", reason=self.last_search.reason)
        return JointResourceDecision(
            "launch"
            if not observation.training_active
            else "reconfigure"
            if target != current
            else "keep",
            target.actor,
            True,
            self.last_search.reason,
            target.rollout,
        )

    def record_committed(self, row):
        version = row["weights_version"]
        if version <= self.through_version:
            return
        for item in row["measurements"]:
            if item["config"] is not None:
                self.history.add_measurement(
                    Measurement(
                        item["phase"],
                        item["config"]["world_size"],
                        item["seconds"],
                        0
                        if item["workload"] is None
                        else item["workload"]["sequence_tokens"],
                        item["config"]["profile_id"],
                    )
                )
        episode = row.get("episode")
        config = row.get("joint_config")
        if config is not None:
            joint = JointConfig(
                TrainingConfig(**config["actor"]), RolloutConfig(**config["rollout"])
            )
            for item in row["measurements"]:
                # Switch profiles own subphases; do not relabel source release
                # or a partial target restore as steady-state target work.
                if item["config"] == config["actor"] and (
                    item["phase"] in {"train", "checkpoint", "publish"}
                    or row.get("config_receipt") is None
                ):
                    self.phases.append(
                        JointPhaseTiming(
                            joint,
                            item["phase"],
                            item["seconds"],
                            0
                            if item["workload"] is None
                            else item["workload"]["sequence_tokens"],
                            f"committed-phase-v{version}",
                        )
                    )
        self.phases = self.phases[-1536:]
        if episode is not None:
            timing = EpisodeTiming(
                RolloutConfig(**episode["rollout"]),
                BatchWorkload(**episode["workload"]),
                episode["seconds"],
                f"committed-episode-v{version}",
            )
            self.episodes = [*self.episodes, timing][-256:]
            self.history.add_workload(timing.workload)
        self.through_version = version

    def runtime_state(self):
        return {
            "format_version": 1,
            "phases": [asdict(p) for p in self.phases],
            "episodes": [asdict(e) for e in self.episodes],
            "through_version": self.through_version,
            "boundary_sequence": self._boundary_sequence,
        }

    def restore_joint_state(self, state, *, history=None):
        if state["format_version"] != 1 or any(
            type(state[k]) is not int or state[k] < 0
            for k in ("through_version", "boundary_sequence")
        ):
            raise ValueError("Invalid joint runtime state")
        self.phases = [
            JointPhaseTiming(**{**p, "config": serial_config(p["config"])})
            for p in state["phases"]
        ]
        self.episodes = [
            EpisodeTiming(
                **{
                    **e,
                    "rollout": RolloutConfig(**e["rollout"]),
                    "workload": BatchWorkload(**e["workload"]),
                }
            )
            for e in state["episodes"]
        ]
        self.through_version = state["through_version"]
        self._boundary_sequence = state["boundary_sequence"]
        if history is not None:
            self.history = history

    def restore_runtime(self, snapshot, *, history=None):
        self.history = HistoryWindow()
        self.episodes, self.phases = [], []
        self.through_version = self._boundary_sequence = 0
        for row in snapshot["updates"]:
            self.record_committed(row)
        for row in snapshot.get("boundaries", ()):
            self.record_boundary(row)
        if history is not None:
            self.history = history

    def record_boundary(self, row):
        if row["sequence"] <= self._boundary_sequence:
            return
        self.history.add_measurement(
            Measurement(
                row["phase"],
                row["config"]["world_size"],
                row["seconds"],
                row["sequence_tokens"],
                row["config"]["profile_id"],
            )
        )
        joint = row["joint_config"]
        self.phases = [
            *self.phases,
            JointPhaseTiming(
                JointConfig(
                    TrainingConfig(**joint["actor"]), RolloutConfig(**joint["rollout"])
                ),
                row["phase"],
                row["seconds"],
                row["sequence_tokens"],
                f"boundary-{row['sequence']}",
            ),
        ][-1536:]
        self._boundary_sequence = row["sequence"]
