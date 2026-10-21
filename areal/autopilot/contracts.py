# SPDX-License-Identifier: Apache-2.0

"""Shared, CPU-only observations and lifecycle decisions for AutoPilot."""

from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import dataclass
from typing import Literal, Protocol

Decision = Literal["train", "keep", "release"]


@dataclass(frozen=True, order=True)
class TrainingConfig:
    """Resource identity; empty profile_id preserves the pure-DP runtime contract.

    Named profiles identify complete planning configs and require an executor
    that explicitly validates that profile. Equal card counts are not identity.
    """

    world_size: int
    profile_id: str = ""

    def __post_init__(self) -> None:
        if type(self.world_size) is not int or self.world_size < 1:
            raise ValueError("world_size must be a positive integer")
        if (
            not isinstance(self.profile_id, str)
            or self.profile_id.strip() != self.profile_id
        ):
            raise ValueError("profile_id must be a string without outer whitespace")


@dataclass(frozen=True)
class BatchWorkload:
    """One logical update; counts never include replicated rank copies."""

    samples: int
    effective_tokens: int
    sequence_tokens: int
    rollout_seconds: float = 0.0
    actor_stats: WorkloadStats | None = None

    def __post_init__(self) -> None:
        if any(
            type(x) is not int or x < 1
            for x in (self.samples, self.effective_tokens, self.sequence_tokens)
        ):
            raise ValueError("Workload counts must be positive integers")
        if self.effective_tokens > self.sequence_tokens:
            raise ValueError("Effective tokens exceed sequence tokens")
        if not math.isfinite(self.rollout_seconds) or self.rollout_seconds < 0:
            raise ValueError("Invalid rollout duration")
        if self.actor_stats is not None:
            stats = self.actor_stats
            if isinstance(stats, dict):
                stats = WorkloadStats(**stats)
                object.__setattr__(self, "actor_stats", stats)
            if not isinstance(stats, WorkloadStats):
                raise ValueError("actor_stats must be WorkloadStats or its JSON object")
            if not math.isclose(stats.sequence_tokens, self.sequence_tokens, abs_tol=1):
                raise ValueError("Actor stats and workload sequence tokens disagree")
            if stats.loss_tokens is not None and not math.isclose(
                stats.loss_tokens, self.effective_tokens, abs_tol=1
            ):
                raise ValueError("Actor loss tokens and effective_tokens disagree")


@dataclass(frozen=True)
class ResourceDecision:
    """Resource ownership and consuming a ready update are separate decisions."""

    action: Literal["keep", "launch", "release", "reconfigure"]
    config: TrainingConfig | None = None
    train: bool = False
    reason: str = ""

    def __post_init__(self) -> None:
        if self.action not in {"keep", "launch", "release", "reconfigure"}:
            raise ValueError("Invalid resource action")
        if self.action in {"launch", "reconfigure"} and self.config is None:
            raise ValueError("Resource action requires a target configuration")
        if self.action == "release" and self.config is not None:
            raise ValueError("Release cannot select a target configuration")
        if self.action == "release" and self.train:
            raise ValueError("Cannot release and train in one decision")


@dataclass(frozen=True)
class ElasticObservation:
    state: str
    model_version: int
    serving_version: int | None
    training_active: bool
    batch_ready: bool
    idle_duration: float
    gate_available: bool
    pending_publication: bool
    committed: bool
    observed_at: float
    training_config: TrainingConfig | None = None
    next_batch: BatchWorkload | None = None
    committed_samples: int = 0
    committed_tokens: int = 0
    max_train_gpus: int = 0
    has_checkpoint: bool = False
    episode_active: bool = False
    episode_elapsed: float = 0.0
    checkpoint_config: TrainingConfig | None = None


# Decision preserves the original string-based baseline policy interface.


def _positive_stat(name: str, value: float) -> None:
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class WorkloadStats:
    """One accepted actor batch, after trajectory/turn expansion, before PPO splitting.

    sequence_tokens are attention_mask tokens, not outer n_tokens or loss tokens.
    Exact lengths and actual synchronized microbatch count are optional refinements.
    The scalar mb count is only unambiguous for ppo_n_minibatches=1.
    Optional trajectory_group_sizes correspond to TrajBatchMeta.traj_group_sizes.
    Per-DP lengths bypass unknown atomic trajectory dispatch; neither is required.
    """

    n_seqs: int
    sequence_tokens: float
    max_seq_len: float | None = None
    min_seq_len: float | None = None
    loss_tokens: float | None = None
    outer_padded_tokens: float | None = None
    sequence_lengths: tuple[int, ...] | None = None
    train_microbatches: int | None = None
    trajectory_group_sizes: tuple[int, ...] | None = None
    trajectory_dispatch_weights: tuple[float, ...] | None = None
    per_dp_sequence_lengths: tuple[tuple[int, ...], ...] | None = None

    def __post_init__(self) -> None:
        for name in (
            "sequence_lengths",
            "trajectory_group_sizes",
            "trajectory_dispatch_weights",
        ):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, tuple(getattr(self, name)))
        if self.per_dp_sequence_lengths is not None:
            object.__setattr__(
                self,
                "per_dp_sequence_lengths",
                tuple(tuple(rank) for rank in self.per_dp_sequence_lengths),
            )
        if type(self.n_seqs) is not int or self.n_seqs < 1:
            raise ValueError("n_seqs must be a positive actual actor sequence count")
        _positive_stat("sequence_tokens", self.sequence_tokens)
        mean = self.sequence_tokens / self.n_seqs
        if mean < 1:
            raise ValueError("Each actor sequence must contain at least one token")
        for name in (
            "min_seq_len",
            "max_seq_len",
            "loss_tokens",
            "outer_padded_tokens",
        ):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.min_seq_len is not None and not 1 <= self.min_seq_len <= mean:
            raise ValueError("min_seq_len must be between one and mean length")
        if self.max_seq_len is not None and self.max_seq_len < mean:
            raise ValueError("max_seq_len is smaller than mean length")
        if self.loss_tokens is not None and self.loss_tokens > self.sequence_tokens + 1:
            raise ValueError("Loss tokens cannot exceed attended tokens")
        if (
            self.outer_padded_tokens is not None
            and self.outer_padded_tokens + 1 < self.sequence_tokens
        ):
            raise ValueError(
                "Outer padded tokens cannot be smaller than attended tokens"
            )
        if self.sequence_lengths is not None:
            if len(self.sequence_lengths) != self.n_seqs or any(
                type(n) is not int or n < 1 for n in self.sequence_lengths
            ):
                raise ValueError("sequence_lengths must have n_seqs positive integers")
            if not math.isclose(
                sum(self.sequence_lengths), self.sequence_tokens, abs_tol=1
            ):
                raise ValueError("Exact lengths disagree with sequence_tokens")
            if (
                self.max_seq_len is not None
                and max(self.sequence_lengths) != self.max_seq_len
            ):
                raise ValueError("Exact lengths disagree with max_seq_len")
            if (
                self.min_seq_len is not None
                and min(self.sequence_lengths) != self.min_seq_len
            ):
                raise ValueError("Exact lengths disagree with min_seq_len")
        if self.train_microbatches is not None and (
            type(self.train_microbatches) is not int or self.train_microbatches < 1
        ):
            raise ValueError(
                "train_microbatches must be a positive synchronized per-rank count"
            )
        if self.trajectory_group_sizes is not None:
            if (
                any(type(n) is not int or n < 1 for n in self.trajectory_group_sizes)
                or sum(self.trajectory_group_sizes) != self.n_seqs
            ):
                raise ValueError(
                    "trajectory_group_sizes must partition n_seqs into nonempty atomic trajectories"
                )
        if self.trajectory_dispatch_weights is not None:
            if self.trajectory_group_sizes is None or len(
                self.trajectory_dispatch_weights
            ) != len(self.trajectory_group_sizes):
                raise ValueError(
                    "Dispatch weights require matching trajectory_group_sizes"
                )
            for weight in self.trajectory_dispatch_weights:
                _positive_stat("trajectory_dispatch_weight", weight)
        if self.per_dp_sequence_lengths is not None:
            flat = [n for rank in self.per_dp_sequence_lengths for n in rank]
            if (
                any(not rank for rank in self.per_dp_sequence_lengths)
                or len(flat) != self.n_seqs
                or any(type(n) is not int or n < 1 for n in flat)
                or not math.isclose(sum(flat), self.sequence_tokens, abs_tol=1)
            ):
                raise ValueError(
                    "per_dp_sequence_lengths must contain all attended sequences on nonempty DP ranks"
                )
            if (
                self.max_seq_len is not None
                and max(flat) != self.max_seq_len
                or self.min_seq_len is not None
                and min(flat) != self.min_seq_len
            ):
                raise ValueError("Per-DP lengths disagree with summary extrema")
            if self.sequence_lengths is not None and sorted(flat) != sorted(
                self.sequence_lengths
            ):
                raise ValueError("Per-DP lengths disagree with global sequence_lengths")

    @classmethod
    def from_areal_metrics(
        cls, metrics: dict, prefix: str = "ppo_actor"
    ) -> WorkloadStats:
        """Consume one AReaL exported stats row; no GPU access or new counters.

        Deliberately never use outer n_tokens as attended tokens. update/n_tokens
        is also not used implicitly because it may aggregate repeated exposure.
        """

        def get(key: str):
            return metrics.get(f"{prefix}/{key}")

        count, avg = get("n_seqs"), get("seq_len/avg")
        if count is None or avg is None:
            raise ValueError(
                "Need ppo_actor/n_seqs and ppo_actor/seq_len/avg from one stats row"
            )
        if not math.isfinite(count) or int(count) != count:
            raise ValueError("n_seqs must be an integer global DP sum")
        return cls(
            n_seqs=int(count),
            sequence_tokens=count * avg,
            max_seq_len=get("seq_len/max"),
            min_seq_len=get("seq_len/min"),
            loss_tokens=get("n_valid_tokens"),
            outer_padded_tokens=get("n_tokens"),
        )

    def lengths(self, *, tail: bool = False, capacity: int | None = None) -> list[int]:
        """Deterministic surrogate when logs lack lengths; never label it observed.

        The nominal surrogate has the smallest variance compatible with summary
        extrema. tail=True redistributes to bounded extremes for sensitivity tests.
        """
        if self.sequence_lengths is not None:
            return list(self.sequence_lengths)
        if self.per_dp_sequence_lengths is not None:
            return [n for rank in self.per_dp_sequence_lengths for n in rank]
        total = round(self.sequence_tokens)
        low = max(1, math.floor(self.min_seq_len or 1))
        high = max(
            math.ceil(self.sequence_tokens / self.n_seqs),
            math.ceil(self.max_seq_len or total - self.n_seqs + 1),
        )
        if capacity is not None:
            high = min(high, capacity)
        if high * self.n_seqs < total:
            raise ValueError("Mean sequence length exceeds microbatch capacity")
        if self.max_seq_len is not None and self.max_seq_len > high:
            raise ValueError("Maximum sequence length exceeds microbatch capacity")
        fixed = []
        if self.max_seq_len is not None:
            fixed.append(math.ceil(self.max_seq_len))
        if self.min_seq_len is not None and len(fixed) < self.n_seqs:
            fixed.append(low)
        free = self.n_seqs - len(fixed)
        remaining = total - sum(fixed) - free * low
        if remaining < 0 or remaining > free * (high - low):
            raise ValueError("Summary extrema cannot describe this actor batch")
        if not free:
            return fixed
        if tail:
            result = [low] * free
            for i in range(free):
                extra = min(high - low, remaining)
                result[i] += extra
                remaining -= extra
        else:
            quotient, extra = divmod(remaining, free)
            result = [low + quotient + (i < extra) for i in range(free)]
        return fixed + result


@dataclass(frozen=True)
class Measurement:
    phase: str
    world_size: int
    seconds: float
    sequence_tokens: int = 0
    profile_id: str = ""

    def __post_init__(self) -> None:
        TrainingConfig(self.world_size, self.profile_id)
        if not math.isfinite(self.seconds) or self.seconds < 0:
            raise ValueError("Measurement duration must be finite and nonnegative")
        if self.sequence_tokens < 0:
            raise ValueError("Measurement token count must be nonnegative")


@dataclass(frozen=True)
class HistorySnapshot:
    workloads: tuple[BatchWorkload, ...] = ()
    measurements: tuple[Measurement, ...] = ()


class HistoryWindow:
    """Bounded completed observations; no future trace or wall-clock oracle."""

    def __init__(self, capacity: int = 32):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("History capacity must be positive")
        self._workloads = deque(maxlen=capacity)
        self._measurements = deque(maxlen=capacity * 16)
        self._lock = threading.RLock()

    def add_workload(self, workload: BatchWorkload) -> None:
        with self._lock:
            self._workloads.append(workload)

    def add_measurement(self, measurement: Measurement) -> None:
        with self._lock:
            self._measurements.append(measurement)

    def snapshot(self) -> HistorySnapshot:
        with self._lock:
            return HistorySnapshot(tuple(self._workloads), tuple(self._measurements))


@dataclass(frozen=True)
class ForecastUpdate:
    workload: BatchWorkload
    wait_seconds: float


@dataclass(frozen=True)
class PlanStep:
    config: TrainingConfig
    release_before: bool = False


@dataclass(frozen=True)
class SimulationResult:
    feasible: bool
    seconds: float = 0.0
    training_gpu_seconds: float = 0.0
    inference_gpu_seconds: float = 0.0
    extra_gpu_seconds: float = 0.0
    samples: int = 0
    effective_tokens: int = 0
    reason: str = ""
    events: tuple[tuple[str, int, float], ...] = ()
    unmeasured: bool = False
    profile_events: tuple[tuple[str, str, float], ...] = ()

    @property
    def total_gpu_seconds(self) -> float:
        return (
            self.training_gpu_seconds
            + self.inference_gpu_seconds
            + self.extra_gpu_seconds
        )


@dataclass(frozen=True)
class SearchResult:
    decision: ResourceDecision
    prediction: SimulationResult | None
    evaluated: int
    rejected: tuple[str, ...] = ()
    plan: tuple[PlanStep, ...] = ()
    fallback_allowed: bool = False


class SchedulingPolicy(Protocol):
    def decide(
        self, observation: ElasticObservation
    ) -> Decision | ResourceDecision: ...
