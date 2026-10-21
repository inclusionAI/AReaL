# SPDX-License-Identifier: Apache-2.0

"""Typed offline inputs. These are not launcher configs or a benchmark runner."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ....contracts import WorkloadStats
from .core import Estimate, Evidence, fingerprint, integer, nonnegative
from .cost_model._actor.config import ActorConfig
from .cost_model.advantage import AdvantageWorkload
from .cost_model.rollout import RolloutConfig


@dataclass(frozen=True)
class Request:
    name: str
    input_tokens: int
    output_tokens: int
    predecessors: tuple[str, ...] = ()
    cached_prefix_tokens: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "predecessors", tuple(self.predecessors))
        integer("input_tokens", self.input_tokens, minimum=1)
        integer("output_tokens", self.output_tokens, minimum=1)
        integer("cached_prefix_tokens", self.cached_prefix_tokens)
        if not self.name or self.cached_prefix_tokens >= self.input_tokens:
            raise ValueError(
                "Request needs a name and at least one uncached input token"
            )


@dataclass(frozen=True)
class ExternalCall:
    name: str
    pool: str
    duration: Estimate
    predecessors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "predecessors", tuple(self.predecessors))
        if not self.name or not self.pool:
            raise ValueError("External call name and pool are required")


@dataclass(frozen=True)
class Group:
    group_id: str
    nodes: tuple[Request | ExternalCall, ...]
    workload: WorkloadStats
    samples: int
    accepted: bool = True
    postprocess: Estimate = field(
        default_factory=lambda: Estimate(
            0.0, "assumed", notes=("Explicit zero group postprocessing prior",)
        )
    )
    available_at: float = 0.0
    dispatch_weight: float | None = None
    task_id: int | None = None  # Defaults to scenario input order, not group_id text.

    def __post_init__(self) -> None:
        object.__setattr__(self, "nodes", tuple(self.nodes))
        integer("samples", self.samples, minimum=1)
        nonnegative("available_at", self.available_at)
        if self.dispatch_weight is not None:
            nonnegative("dispatch_weight", self.dispatch_weight, positive=True)
        if self.task_id is not None:
            integer("task_id", self.task_id)
        if not self.group_id or type(self.accepted) is not bool:
            raise ValueError("Group identity and explicit outcome are required")
        names = {n.name for n in self.nodes}
        if len(names) != len(self.nodes) or not names:
            raise ValueError("Group nodes must be nonempty and uniquely named")
        seen = set()
        while len(seen) < len(names):
            ready = {
                n.name
                for n in self.nodes
                if n.name not in seen and set(n.predecessors) <= seen
            }
            if not ready:
                raise ValueError("Group has cyclic or unknown dependencies")
            seen.update(ready)
        if (
            self.workload.per_dp_sequence_lengths is not None
            or self.workload.train_microbatches is not None
        ):
            raise ValueError("Group workload must precede config-specific DP dispatch")


@dataclass(frozen=True)
class ExecutionConfig:
    actor: ActorConfig
    rollout: RolloutConfig
    actor_devices: tuple[str, ...]
    rollout_devices: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "actor_devices", tuple(self.actor_devices))
        object.__setattr__(self, "rollout_devices", tuple(self.rollout_devices))
        devices = self.actor_devices + self.rollout_devices
        if (
            len(self.actor_devices) != self.actor.parallel.world_size
            or len(self.rollout_devices) != self.rollout.world_size
        ):
            raise ValueError("Device identities must match each configured world size")
        if len(set(devices)) != len(devices) or any(not d for d in devices):
            raise ValueError(
                "This simulator requires disjoint actor/rollout device identities"
            )

    @property
    def key(self) -> str:
        return fingerprint(self)


@dataclass(frozen=True)
class PlanStep:
    config: ExecutionConfig
    checkpoint: bool = False
    release_after: bool = False
    inflight: Literal["retain", "drain", "cancel"] = "retain"
    update_succeeds: bool = True
    publish_succeeds: bool = True
    after_publish: Estimate = field(
        default_factory=lambda: Estimate(
            0.0, "assumed", notes=("Explicit zero save/eval/resume overhead prior",)
        )
    )
    auxiliary_stages: tuple[tuple[str, Estimate], ...] = ()
    advantage_workload: AdvantageWorkload | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "auxiliary_stages", tuple(tuple(x) for x in self.auxiliary_stages)
        )
        if self.inflight not in {"retain", "drain", "cancel"}:
            raise ValueError("Unknown in-flight transition policy")
        for name in (
            "checkpoint",
            "release_after",
            "update_succeeds",
            "publish_succeeds",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        if self.release_after and not self.checkpoint:
            raise ValueError("Release after an update requires its checkpoint")


@dataclass(frozen=True)
class RequestProgress:
    node: str
    replica: int
    prefilled: int
    generated: int

    def __post_init__(self) -> None:
        for name in ("replica", "prefilled", "generated"):
            integer(name, getattr(self, name))


@dataclass(frozen=True)
class RunningGroup:
    group_id: str
    started_at: float
    completed_nodes: tuple[str, ...] = ()
    requests: tuple[RequestProgress, ...] = ()
    external_remaining: tuple[tuple[str, float], ...] = ()
    postprocess_remaining: float | None = None
    response_remaining: tuple[tuple[str, float], ...] = ()
    replica: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "completed_nodes", tuple(self.completed_nodes))
        object.__setattr__(self, "requests", tuple(self.requests))
        object.__setattr__(
            self, "external_remaining", tuple(tuple(x) for x in self.external_remaining)
        )
        object.__setattr__(
            self, "response_remaining", tuple(tuple(x) for x in self.response_remaining)
        )
        nonnegative("started_at", self.started_at)
        if self.replica is not None:
            integer("replica", self.replica)
        for _, seconds in self.external_remaining:
            nonnegative("external_remaining", seconds)
        for _, seconds in self.response_remaining:
            nonnegative("response_remaining", seconds)
        if self.postprocess_remaining is not None:
            nonnegative("postprocess_remaining", self.postprocess_remaining)


@dataclass(frozen=True)
class InitialState:
    config: ExecutionConfig
    time: float = 0.0
    actor_active: bool = True
    provider_version: int = 0
    training_version: int = 0
    serving_version: int = 0
    accepted_total: int = 0
    rejected_total: int = 0
    ready_groups: tuple[tuple[str, float, float], ...] = ()  # id, started, accepted
    running: tuple[RunningGroup, ...] = ()
    checkpoint_config: ExecutionConfig | None = None
    complete: bool = True

    def __post_init__(self) -> None:
        nonnegative("initial time", self.time)
        if type(self.complete) is not bool or type(self.actor_active) is not bool:
            raise ValueError(
                "Initial completeness and actor activity must be explicit booleans"
            )
        for name in (
            "provider_version",
            "training_version",
            "serving_version",
            "accepted_total",
            "rejected_total",
        ):
            integer(name, getattr(self, name))
        object.__setattr__(
            self, "ready_groups", tuple(tuple(x) for x in self.ready_groups)
        )
        object.__setattr__(self, "running", tuple(self.running))
        if (
            self.provider_version != self.serving_version
            or self.training_version != self.serving_version
        ):
            raise ValueError(
                "Prediction starts at a committed boundary, without a partial publication"
            )
        if self.accepted_total < len(self.ready_groups):
            raise ValueError("Accepted offset cannot be smaller than the ready queue")
        ids = [g[0] for g in self.ready_groups] + [g.group_id for g in self.running]
        if len(set(ids)) != len(ids):
            raise ValueError("Initial ready/running groups overlap")
        for _, start, accepted in self.ready_groups:
            if not 0 <= start <= accepted <= self.time:
                raise ValueError("Initial queue timestamps are inconsistent")
        if any(g.started_at > self.time for g in self.running):
            raise ValueError("Initial running group has not started")


@dataclass(frozen=True)
class Scenario:
    groups: tuple[Group, ...]
    initial: InitialState
    groups_per_batch: int
    consumer_batch_size: int
    samples_per_group: int
    max_concurrent_groups: int
    runner_concurrency: int
    pending_limit: int
    runner_queue_limit: int
    staleness: int
    external_pools: tuple[tuple[str, int], ...] = ()
    selection: Literal["start_time", "task_id"] = "start_time"
    dynamic_batch: bool = False
    terminal: Literal["drain", "cancel"] = "drain"
    max_events: int = 1_000_000
    shuffle_seed: int = 0
    workflow: Literal["eager", "v2_inline"] = "eager"
    request_routing: Literal["request_load", "group_affinity"] = "request_load"

    def __post_init__(self) -> None:
        object.__setattr__(self, "groups", tuple(self.groups))
        object.__setattr__(
            self, "external_pools", tuple(tuple(x) for x in self.external_pools)
        )
        for name in (
            "groups_per_batch",
            "consumer_batch_size",
            "samples_per_group",
            "max_concurrent_groups",
            "runner_concurrency",
            "pending_limit",
            "runner_queue_limit",
            "max_events",
        ):
            integer(name, getattr(self, name), minimum=1)
        integer("staleness", self.staleness)
        integer("shuffle_seed", self.shuffle_seed)
        if self.selection not in {"start_time", "task_id"} or self.terminal not in {
            "drain",
            "cancel",
        }:
            raise ValueError("Invalid selection or terminal policy")
        if type(self.dynamic_batch) is not bool:
            raise ValueError("dynamic_batch must be boolean")
        if self.workflow not in {"eager", "v2_inline"}:
            raise ValueError("Unknown workflow submission protocol")
        if self.request_routing not in {"request_load", "group_affinity"}:
            raise ValueError("Unknown request routing assumption")
        if (
            self.workflow == "v2_inline"
            and self.runner_queue_limit < self.groups_per_batch
        ):
            raise ValueError("V2 runner queue must hold at least one logical batch")
        by_id = {g.group_id: g for g in self.groups}
        if len(by_id) != len(self.groups):
            raise ValueError("Group IDs must be unique")
        task_ids = [
            g.task_id if g.task_id is not None else i for i, g in enumerate(self.groups)
        ]
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("Explicit/default task IDs must be unique")
        initial_ids = {g[0] for g in self.initial.ready_groups} | {
            g.group_id for g in self.initial.running
        }
        if not initial_ids <= by_id.keys():
            raise ValueError("Missing initial group definitions")
        if any(g.samples != self.samples_per_group for g in self.groups):
            raise ValueError("samples_per_group is fixed within a comparison")
        pools = dict(self.external_pools)
        if len(pools) != len(self.external_pools):
            raise ValueError("Duplicate external pool")
        for name, capacity in self.external_pools:
            integer(name, capacity, minimum=1)
        if any(
            n.pool not in pools
            for g in self.groups
            for n in g.nodes
            if isinstance(n, ExternalCall)
        ):
            raise ValueError(
                "Every external service needs an explicit concurrency limit"
            )


@dataclass(frozen=True)
class Event:
    time: float
    kind: str
    subject: str = ""
    details: tuple[tuple[str, object], ...] = ()


@dataclass(frozen=True)
class ExpertResult:
    status: Literal["complete", "incomplete", "unsupported"]
    reason: str
    seconds: float
    actor_gpu_seconds: float
    rollout_gpu_seconds: float
    completed_updates: int
    attempted_updates: int
    accepted_total: int
    rejected_total: int
    cancelled_groups: int
    queued_groups: int
    running_groups: int
    queue_wait_seconds: float | None
    events: tuple[Event, ...]
    evidence: Evidence
    snapshot_id: str
    remaining_devices: tuple[str, ...]
    notes: tuple[str, ...] = ()

    @property
    def gpu_hours(self) -> float:
        return (self.actor_gpu_seconds + self.rollout_gpu_seconds) / 3600
