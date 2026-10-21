# SPDX-License-Identifier: Apache-2.0

"""Deterministic asynchronous execution of supplied plans; no configuration search."""

from __future__ import annotations

import heapq
import random
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

from ...contracts import WorkloadStats
from .core import Estimate, UnsupportedEstimate, code_revision, evidence_of, fingerprint
from .cost_model._rollout.scheduling import policy_for
from .cost_model.actor import ExpertActorModel
from .cost_model.rollout import ExpertRolloutModel, ServiceState
from .cost_model.rpc import RPCTimingModel
from .cost_model.transition import ExpertTransitionModel
from .scenario import (
    Event,
    ExpertResult,
    ExternalCall,
    Group,
    PlanStep,
    Request,
    Scenario,
)


@dataclass(frozen=True)
class ExpertModels:
    actor: ExpertActorModel
    rollout: ExpertRolloutModel
    transition: ExpertTransitionModel
    rpc_timing: RPCTimingModel | None = None

    @property
    def snapshot_id(self) -> str:
        return fingerprint(code_revision(), self)


class AllocationLedger:
    """Reference-count device ownership; overlapping phases do not double bill."""

    def __init__(self, time: float = 0.0):
        self.time = time
        self.owners: dict[tuple[str, str], int] = {}
        self.seconds = {"actor": 0.0, "rollout": 0.0}

    def advance(self, time: float) -> None:
        if time < self.time:
            raise ValueError("Allocation clock cannot move backwards")
        for role in self.seconds:
            self.seconds[role] += len(self.devices(role)) * (time - self.time)
        self.time = time

    def devices(self, role: str) -> set[str]:
        return {
            device
            for (owner, device), count in self.owners.items()
            if owner == role and count
        }

    def change(self, action: str, role: str, devices: tuple[str, ...]) -> None:
        if role not in self.seconds or action not in {"acquire", "release"}:
            raise ValueError("Invalid allocation operation")
        other = "rollout" if role == "actor" else "actor"
        if len(set(devices)) != len(devices):
            raise ValueError("Duplicate devices in allocation operation")
        if action == "acquire" and self.devices(other).intersection(devices):
            raise ValueError("Actor and rollout must use disjoint devices")
        if action == "release" and any(
            self.owners.get((role, d), 0) == 0 for d in devices
        ):
            raise ValueError("Cannot release an unowned device")
        for device in devices:
            key = (role, device)
            self.owners[key] = self.owners.get(key, 0) + (
                1 if action == "acquire" else -1
            )


@dataclass
class _GroupState:
    group: Group
    started: float
    finished: set[str] = field(default_factory=set)
    started_nodes: set[str] = field(default_factory=set)
    finishing: bool = False
    replica: int = -1


@dataclass
class _RequestState:
    group_id: str
    request: Request
    order: int
    replica: int = -1
    prefilled: int = 0
    generated: int = 0
    response_seconds: float = 0.0

    @property
    def kv_tokens(self) -> int:
        return self.prefilled + max(0, self.generated - 1)


class ExpertSimulator:
    def __init__(self, models: ExpertModels):
        self.models = models

    def simulate(
        self, scenario: Scenario, supplied_plan: tuple[PlanStep, ...]
    ) -> ExpertResult:
        """Evaluate a fixed plan. Unsupported coverage returns partial cost explicitly."""
        if not supplied_plan:
            raise ValueError("A supplied plan must contain at least one update")
        return _Simulation(self.models, scenario, tuple(supplied_plan)).run()


class _Simulation:
    def __init__(
        self, models: ExpertModels, scenario: Scenario, plan: tuple[PlanStep, ...]
    ):
        self.models, self.scenario, self.plan = models, scenario, plan
        self.now = scenario.initial.time
        self.config = scenario.initial.config
        self.actor_active = scenario.initial.actor_active
        self.checkpoint_config = scenario.initial.checkpoint_config
        self.provider = scenario.initial.provider_version
        self.training = scenario.initial.training_version
        self.serving = scenario.initial.serving_version
        self.accepted = scenario.initial.accepted_total
        self.rejected = scenario.initial.rejected_total
        self.completed = self.attempted = self.cancelled = self.index = 0
        self.queue_wait = 0.0
        self.groups = {g.group_id: g for g in scenario.groups}
        self.task_ids = {
            g.group_id: g.task_id if g.task_id is not None else i
            for i, g in enumerate(scenario.groups)
        }
        self.ready = {
            gid: (start, accepted)
            for gid, start, accepted in scenario.initial.ready_groups
        }
        self.results = dict(self.ready)  # Completed accepted AND rejected results.
        self.undelivered: dict[str, tuple[float, float]] = {}
        self.producer_due = self.now
        self.final_admission_pass = False
        self.batch: list[str] = []
        self.batch_collected = False
        self.active: dict[str, _GroupState] = {}
        self.pending: deque[str] = deque()
        self.submitted: deque[str] = deque()
        self.seen = set(self.ready) | {g.group_id for g in scenario.initial.running}
        self.requests: dict[tuple[str, str], _RequestState] = {}
        self.responses: dict[tuple[str, str], int | None] = {}
        self.rpc_profiles = {}
        self.external_live: set[tuple[str, str]] = set()
        self.external_counts = dict.fromkeys(dict(scenario.external_pools), 0)
        self.replica_busy = [False] * self.config.rollout.replicas
        self.replica_cursor = 0
        self.events: list[Event] = []
        self.heap: list[tuple[float, int, int, Callable[[], None]]] = []
        self.sequence = self.request_sequence = 0
        self.phase = "idle"
        self.blocks: set[str] = set()
        self.admission_paused = self.freeze_generation = False
        self.drain_submitted = False
        self.terminal_started = self.done = False
        self.evidence = []
        self.notes = [
            "Fixed trajectory content/outcomes; this model does not predict reward",
            "Inference uses deterministic decode-first batching and full-length KV reservation",
            "Publication waits for current service iterations, then preserves group/request progress",
        ]
        if models.rpc_timing is None:
            self.notes.append("No RPC timing model: explicit zero response-tail prior")
        if scenario.workflow == "v2_inline":
            self.notes.append(
                "V2 inline batch enqueue/1s wait and paused result delivery; zero thread handoff latency prior, not exact OS scheduling"
            )
        if scenario.request_routing == "group_affinity":
            self.notes.append(
                "Assumed group affinity: least active groups, rotating ties; fixed replica through external cleanup, not a reconstruction of runtime routing"
            )
        self.ledger = AllocationLedger(self.now)
        self.ledger.change("acquire", "rollout", self.config.rollout_devices)
        if self.actor_active:
            self.ledger.change("acquire", "actor", self.config.actor_devices)
        self.rng = random.Random(scenario.shuffle_seed)
        self.snapshot = fingerprint(models.snapshot_id, scenario, plan)

    def emit(self, kind: str, subject: str = "", **details) -> None:
        self.events.append(Event(self.now, kind, subject, tuple(details.items())))

    def schedule(
        self, delay: float, callback: Callable[[], None], priority: int = 2
    ) -> None:
        if delay < 0:
            raise ValueError("Cannot schedule a negative delay")
        self.sequence += 1
        heapq.heappush(self.heap, (self.now + delay, priority, self.sequence, callback))

    def duration(self, estimate: Estimate) -> float:
        self.evidence.append(estimate.evidence)
        self.notes.extend(estimate.notes)
        return estimate.require_seconds()

    @property
    def running(self) -> int:
        return len(self.active) + len(self.submitted)

    def raw_capacity(self) -> int:
        s = self.scenario
        return min(
            s.max_concurrent_groups - self.running,
            (s.staleness + self.provider + 1) * s.consumer_batch_size
            - self.accepted
            - self.running,
        )

    def run(self) -> ExpertResult:
        if not self.scenario.initial.complete:
            return self.result(
                "unsupported", "Initial queue/version/in-flight state is incomplete"
            )
        try:
            self.restore_running()
            for group in self.scenario.groups:
                if group.available_at > self.now:
                    self.schedule(group.available_at - self.now, lambda: None)
            count = 0
            self.tick()
            while not self.done:
                if not self.heap:
                    return self.result(
                        "incomplete",
                        "No event can make progress: insufficient groups, capacity, or KV space",
                    )
                time = self.heap[0][0]
                self.ledger.advance(time)
                self.now = time
                # All completions at one timestamp precede the next dispatch pass.
                while self.heap and self.heap[0][0] == self.now:
                    _, _, _, callback = heapq.heappop(self.heap)
                    callback()
                    count += 1
                    if count >= self.scenario.max_events:
                        return self.result("incomplete", "Event budget exhausted")
                self.tick()
            return self.result(
                "complete", "Supplied plan and terminal cleanup completed"
            )
        except UnsupportedEstimate as exc:
            return self.result("unsupported", str(exc))

    def result(self, status: str, reason: str) -> ExpertResult:
        remaining = tuple(
            sorted(self.ledger.devices("actor") | self.ledger.devices("rollout"))
        )
        return ExpertResult(
            status,
            reason,
            self.now - self.scenario.initial.time,
            self.ledger.seconds["actor"],
            self.ledger.seconds["rollout"],
            self.completed,
            self.attempted,
            self.accepted,
            self.rejected,
            self.cancelled,
            len(self.ready)
            + sum(self.groups[gid].accepted for gid in self.undelivered),
            self.running,
            self.queue_wait,
            tuple(self.events),
            "unsupported"
            if status == "unsupported"
            else evidence_of(tuple(self.evidence)),
            self.snapshot,
            remaining,
            tuple(dict.fromkeys(self.notes)),
        )

    def tick(self) -> None:
        if self.done:
            return
        if not self.admission_paused and "admission" not in self.blocks:
            for gid, timing in tuple(self.undelivered.items()):
                self.deliver_group(gid, timing)
            self.undelivered.clear()
        self.produce()
        self.workflow()
        self.admit()
        self.activate()
        self.dispatch_nodes()
        self.service()
        self.final_admission_pass = False
        self.workflow()

    def produce(self) -> None:
        """Main-thread active_submit_and_wait, separate from background commit."""
        s = self.scenario
        if (
            s.workflow != "v2_inline"
            or self.phase != "idle"
            or self.index >= len(self.plan)
            or self.admission_paused
            or self.now < self.producer_due
        ):
            return
        capacity = min(
            s.pending_limit - len(self.pending),
            s.runner_queue_limit - len(self.submitted) - s.groups_per_batch,
        )
        count = 0
        for group in s.groups:
            if count >= min(s.groups_per_batch, capacity):
                break
            if group.group_id not in self.seen and group.available_at <= self.now:
                self.pending.append(group.group_id)
                self.seen.add(group.group_id)
                self.emit("group_enqueued", group.group_id)
                count += 1
        # A timeout wakes the main producer even without service completions.
        # Stale timer events are harmless: produce checks the current deadline.
        self.producer_due = self.now + 1.0
        if any(g.group_id not in self.seen for g in s.groups):
            self.schedule(1.0, lambda: None)

    def admit(self) -> None:
        if (
            self.admission_paused
            or "admission" in self.blocks
            or (self.index >= len(self.plan) and not self.final_admission_pass)
        ):
            return
        if self.scenario.workflow == "eager":
            for group in self.scenario.groups:
                if len(self.pending) >= self.scenario.pending_limit:
                    break
                if group.group_id not in self.seen and group.available_at <= self.now:
                    self.pending.append(group.group_id)
                    self.seen.add(group.group_id)
                    self.emit("group_enqueued", group.group_id)
        while (
            self.pending
            and self.raw_capacity() > 0
            and len(self.submitted) < self.scenario.runner_queue_limit
        ):
            gid = self.pending.popleft()
            self.submitted.append(gid)
            self.emit(
                "group_submitted",
                gid,
                provider_version=self.provider,
                running=self.running,
                accepted_total=self.accepted,
            )

    def activate(self) -> None:
        if "admission" in self.blocks or (
            self.admission_paused and not self.drain_submitted
        ):
            return
        while self.submitted and len(self.active) < self.scenario.runner_concurrency:
            gid = self.submitted.popleft()
            replica = -1
            if self.scenario.request_routing == "group_affinity":
                replicas = self.config.rollout.replicas
                counts = [
                    sum(state.replica == i for state in self.active.values())
                    for i in range(replicas)
                ]
                replica = min(
                    range(replicas),
                    key=lambda i: (counts[i], (i - self.replica_cursor) % replicas),
                )
                self.replica_cursor = (replica + 1) % replicas
            self.active[gid] = _GroupState(self.groups[gid], self.now, replica=replica)
            if replica >= 0:
                self.emit("group_started", gid, replica=replica)
            else:
                self.emit("group_started", gid)

    def dispatch_nodes(self) -> None:
        pools = dict(self.scenario.external_pools)
        for gid, state in list(self.active.items()):
            if state.finishing:
                continue
            if len(state.finished) == len(state.group.nodes):
                state.finishing = True
                self.schedule(
                    self.duration(state.group.postprocess),
                    lambda gid=gid: self.finish_group(gid),
                )
                continue
            for node in state.group.nodes:
                if (
                    node.name in state.started_nodes
                    or not set(node.predecessors) <= state.finished
                ):
                    continue
                key = (gid, node.name)
                if isinstance(node, ExternalCall):
                    if self.external_counts[node.pool] >= pools[node.pool]:
                        continue
                    self.external_counts[node.pool] += 1
                    self.external_live.add(key)
                    state.started_nodes.add(node.name)
                    self.emit("external_start", f"{gid}/{node.name}", pool=node.pool)
                    self.schedule(
                        self.duration(node.duration),
                        lambda key=key, pool=node.pool: self.finish_external(key, pool),
                    )
                else:
                    if (
                        node.cached_prefix_tokens
                        and self.config.rollout.cache_mode != "prefix"
                    ):
                        raise UnsupportedEstimate(
                            "Cached prefix supplied for cache_mode=none"
                        )
                    state.started_nodes.add(node.name)
                    self.request_sequence += 1
                    self.requests[key] = _RequestState(
                        gid,
                        node,
                        self.request_sequence,
                        prefilled=node.cached_prefix_tokens,
                        response_seconds=self.response_delay(node),
                    )
                    self.emit("request_queued", f"{gid}/{node.name}")

    def response_delay(self, request: Request) -> float:
        timing = self.models.rpc_timing
        if timing is None:
            return 0.0
        return self.duration(
            timing.response_delay(
                self.rpc_profile(),
                request.input_tokens,
                request.output_tokens,
            )
        )

    def rpc_profile(self) -> str:
        config = self.config.rollout
        if config not in self.rpc_profiles:
            self.rpc_profiles[config] = RPCTimingModel.profile_for(
                self.models.rollout, config
            )
        return self.rpc_profiles[config]

    def finish_external(self, key: tuple[str, str], pool: str) -> None:
        if key not in self.external_live:
            return
        self.external_live.remove(key)
        self.external_counts[pool] -= 1
        self.active[key[0]].finished.add(key[1])
        self.emit("external_complete", "/".join(key))

    def finish_group(self, gid: str) -> None:
        state = self.active.pop(gid, None)
        if state is None:
            return
        if state.group.accepted:
            self.accepted += 1
            self.emit(
                "group_accepted",
                gid,
                accepted_total=self.accepted,
                queued=len(self.ready)
                + sum(self.groups[g].accepted for g in self.undelivered)
                + 1,
            )
        else:
            self.rejected += 1
            self.emit("group_rejected", gid, rejected_total=self.rejected)
        timing = (state.started, self.now)
        if self.scenario.workflow == "v2_inline" and self.admission_paused:
            self.undelivered[gid] = timing
        else:
            self.deliver_group(gid, timing)

    def deliver_group(self, gid: str, timing: tuple[float, float]) -> None:
        self.results[gid] = timing
        if self.groups[gid].accepted:
            self.ready[gid] = timing
        self.emit("group_result_ready", gid)

    def service(self) -> None:
        if self.freeze_generation or "generation" in self.blocks:
            return
        cfg = self.config.rollout
        policy = policy_for(cfg.backend)
        assigned = [
            [r for r in self.requests.values() if r.replica == i]
            for i in range(cfg.replicas)
        ]
        for request in sorted(self.requests.values(), key=lambda r: r.order):
            if request.replica >= 0:
                continue
            fixed = (
                self.active[request.group_id].replica
                if self.scenario.request_routing == "group_affinity"
                else None
            )
            replica = policy.choose_replica(cfg, request, assigned, fixed)
            if replica is None:
                continue
            request.replica = replica
            assigned[replica].append(request)
            self.emit(
                "request_assigned",
                f"{request.group_id}/{request.request.name}",
                replica=replica,
            )
        for replica, rows in enumerate(assigned):
            if self.replica_busy[replica] or not rows:
                continue
            slices, keys = policy.batch(cfg, rows)
            estimate = self.models.rollout.estimate(
                cfg, ServiceState(sum(r.kv_tokens for r in rows)), tuple(slices)
            )
            duration = self.duration(
                self.models.rpc_timing.service_duration(
                    self.rpc_profile(), estimate.duration
                )
                if self.models.rpc_timing is not None
                else estimate.duration
            )
            self.replica_busy[replica] = True
            version = self.serving
            self.emit(
                "service_start", str(replica), version=version, requests=tuple(keys)
            )
            self.schedule(
                duration,
                lambda replica=replica,
                keys=tuple(keys),
                slices=tuple(slices),
                version=version: self.finish_service(replica, keys, slices, version),
            )

    def finish_service(self, replica: int, keys, slices, version: int) -> None:
        self.replica_busy[replica] = False
        for key, item in zip(keys, slices, strict=True):
            request = self.requests[key]
            if item.kind == "prefill":
                request.prefilled += item.query_tokens
            if item.emit_token:
                request.generated += 1
                self.emit(
                    "token_generated",
                    "/".join(key),
                    version=version,
                    output_index=request.generated,
                )
            if request.generated == request.request.output_tokens:
                del self.requests[key]
                self.responses[key] = version
                if request.response_seconds:
                    self.emit(
                        "response_wait",
                        "/".join(key),
                        seconds=request.response_seconds,
                        version=version,
                    )
                    self.schedule(
                        request.response_seconds,
                        lambda key=key: self.finish_response(key),
                    )
                else:
                    self.finish_response(key)
        self.emit("service_complete", str(replica), version=version)

    def finish_response(self, key: tuple[str, str]) -> None:
        if key not in self.responses:
            return
        version = self.responses.pop(key)
        self.active[key[0]].finished.add(key[1])
        self.emit("request_complete", "/".join(key), version=version)

    def restore_running(self) -> None:
        for running in self.scenario.initial.running:
            group = self.groups[running.group_id]
            nodes = {n.name: n for n in group.nodes}
            finished = set(running.completed_nodes)
            if (
                len(finished) != len(running.completed_nodes)
                or not finished <= nodes.keys()
            ):
                raise ValueError("Invalid initial completed-node set")
            if any(not set(nodes[name].predecessors) <= finished for name in finished):
                raise ValueError("Initial completed nodes violate dependencies")
            state = _GroupState(group, running.started_at, finished, set(finished))
            if self.scenario.request_routing == "group_affinity":
                replicas = {p.replica for p in running.requests}
                if running.replica is not None:
                    replicas.add(running.replica)
                if len(replicas) != 1:
                    raise ValueError(
                        "Initial group affinity requires one known replica"
                    )
                state.replica = replicas.pop()
                if state.replica >= self.config.rollout.replicas:
                    raise ValueError("Initial group replica is outside the layout")
            self.active[group.group_id] = state
            for progress in running.requests:
                node = nodes.get(progress.node)
                if not isinstance(node, Request) or node.name in state.started_nodes:
                    raise ValueError("Invalid/duplicate initial request")
                if (
                    not set(node.predecessors) <= finished
                    or progress.replica >= self.config.rollout.replicas
                ):
                    raise ValueError("Initial request dependency or replica is invalid")
                if (
                    not node.cached_prefix_tokens
                    <= progress.prefilled
                    <= node.input_tokens
                    or progress.generated >= node.output_tokens
                ):
                    raise ValueError(
                        "Initial request progress is outside its token budget"
                    )
                if progress.generated and progress.prefilled != node.input_tokens:
                    raise ValueError("Decode cannot precede complete prefill")
                if not progress.generated and progress.prefilled == node.input_tokens:
                    raise ValueError(
                        "A completed prefill must include its first output token"
                    )
                state.started_nodes.add(node.name)
                self.request_sequence += 1
                self.requests[(group.group_id, node.name)] = _RequestState(
                    group.group_id,
                    node,
                    self.request_sequence,
                    progress.replica,
                    progress.prefilled,
                    progress.generated,
                    self.response_delay(node),
                )
            for name, remaining in running.response_remaining:
                node = nodes.get(name)
                if (
                    not isinstance(node, Request)
                    or name in state.started_nodes
                    or not set(node.predecessors) <= finished
                ):
                    raise ValueError("Invalid/duplicate initial response")
                state.started_nodes.add(name)
                key = (group.group_id, name)
                # Residual delivery time is known; the token version is not.
                self.responses[key] = None
                self.schedule(remaining, lambda key=key: self.finish_response(key))
            for name, remaining in running.external_remaining:
                node = nodes.get(name)
                if (
                    not isinstance(node, ExternalCall)
                    or name in state.started_nodes
                    or not set(node.predecessors) <= finished
                ):
                    raise ValueError("Invalid initial external call")
                state.started_nodes.add(name)
                self.external_counts[node.pool] += 1
                key = (group.group_id, name)
                self.external_live.add(key)
                self.schedule(
                    remaining,
                    lambda key=key, pool=node.pool: self.finish_external(key, pool),
                )
            if running.postprocess_remaining is not None:
                if finished != nodes.keys():
                    raise ValueError("Group postprocessing requires all nodes complete")
                state.finishing = True
                self.schedule(
                    running.postprocess_remaining,
                    lambda gid=group.group_id: self.finish_group(gid),
                )
        if len(self.active) > self.scenario.runner_concurrency:
            raise ValueError("Initial state exceeds runner concurrency")
        if any(
            self.external_counts[name] > limit
            for name, limit in self.scenario.external_pools
        ):
            raise ValueError("Initial external calls exceed pool concurrency")
        for replica in range(self.config.rollout.replicas):
            rows = [r for r in self.requests.values() if r.replica == replica]
            if (
                len(rows) > self.config.rollout.max_batch_sequences
                or sum(
                    r.request.input_tokens + r.request.output_tokens - 1 for r in rows
                )
                > self.config.rollout.kv_capacity_tokens
            ):
                raise ValueError("Initial requests exceed replica capacity")

    def transition(self, action: str, source, target, done: Callable[[], None]) -> None:
        estimate = self.models.transition.estimate(
            action, source.key if source else None, target.key if target else None
        )
        duration = self.duration(estimate.duration)
        self.phase = "transition"
        self.blocks = set(estimate.blocks)
        self.emit(
            "transition_start",
            action,
            source=source.key if source else None,
            target=target.key if target else None,
        )
        changes = {}
        for offset, change in estimate.allocation_events:
            changes.setdefault(offset, []).append(change)
        for offset, batch in changes.items():
            self.schedule(
                offset,
                lambda batch=tuple(batch): self.allocation_batch(batch),
                priority=0,
            )

        def finish():
            self.blocks.clear()
            self.phase = "idle"
            self.emit("transition_complete", action)
            done()

        self.schedule(duration, finish)

    def allocation_batch(self, changes) -> None:
        # Simultaneous acquire/release also handles zero-duration allocations.
        owners = dict(self.ledger.owners)
        for change in changes:
            for device in change.devices:
                key = (change.role, device)
                owners[key] = owners.get(key, 0) + (
                    1 if change.action == "acquire" else -1
                )
        if any(count < 0 for count in owners.values()):
            raise ValueError("Transition releases resources it does not own")
        actor = {d for (role, d), n in owners.items() if role == "actor" and n}
        rollout = {d for (role, d), n in owners.items() if role == "rollout" and n}
        if actor & rollout:
            raise ValueError("Transition violates disjoint GPU pools")
        self.ledger.owners = owners
        for change in changes:
            self.emit(
                "allocation_" + change.action, change.role, devices=change.devices
            )

    def validate_allocation(self, actor_active: bool) -> None:
        expected_actor = set(self.config.actor_devices) if actor_active else set()
        if self.ledger.devices("actor") != expected_actor or self.ledger.devices(
            "rollout"
        ) != set(self.config.rollout_devices):
            raise ValueError(
                "Transition allocation effects do not match target configuration"
            )
        if any(count > 1 for count in self.ledger.owners.values()):
            raise ValueError("Transition left temporary allocation references live")

    def workflow(self) -> None:
        if (
            self.done
            or self.phase == "transition"
            or self.phase in {"training", "post_publish"}
        ):
            return
        if self.phase == "await_publish":
            if not any(self.replica_busy):
                self.transition("publish", self.config, self.config, self.published)
            return
        if self.index >= len(self.plan):
            if not self.final_admission_pass:
                self.terminal()
            return
        step = self.plan[self.index]
        if step.config != self.config:
            different_rollout = (step.config.rollout, step.config.rollout_devices) != (
                self.config.rollout,
                self.config.rollout_devices,
            )
            if different_rollout and step.inflight == "retain":
                raise UnsupportedEstimate(
                    "A rollout layout change requires drain or explicit cancellation"
                )
            action = "reconfigure" if self.actor_active else "reconfigure_suspended"
            blocked = self.models.transition.estimate(
                action, self.config.key, step.config.key
            ).blocks
            self.admission_paused = (
                different_rollout or step.inflight != "retain" or "admission" in blocked
            )
            self.drain_submitted = step.inflight == "drain"
            if step.inflight == "drain" and self.running:
                return
            if step.inflight == "cancel" and self.running:
                self.freeze_generation = True
                if not any(self.replica_busy):
                    self.transition(
                        "cancel", self.config, self.config, self.cancel_running
                    )
                return
            self.drain_submitted = False
            self.freeze_generation = different_rollout or "generation" in blocked
            if self.freeze_generation and any(self.replica_busy):
                return

            def reconfigured():
                self.config = step.config
                self.actor_active = True
                if different_rollout:
                    self.replica_busy = [False] * self.config.rollout.replicas
                self.validate_allocation(True)
                self.admission_paused = self.freeze_generation = False

            self.transition(action, self.config, step.config, reconfigured)
            return
        s = self.scenario
        # Mirror wait_results(count): wait for the requested number of completed
        # attempts, select/order them, then filter rejected results. Fixed batches
        # retain accepted partial collections while waiting for replacements.
        while not self.batch_collected:
            count = s.groups_per_batch - len(self.batch)
            if len(self.results) < count:
                return
            key = (
                (lambda gid: self.task_ids[gid])
                if s.selection == "task_id"
                else (lambda gid: (self.results[gid][0], self.task_ids[gid]))
            )
            arrived = sorted(self.results, key=key)[:count]
            if s.selection != "task_id":
                self.rng.shuffle(arrived)
            for gid in arrived:
                del self.results[gid]
                if self.groups[gid].accepted:
                    self.batch.append(gid)
            self.batch_collected = (
                s.dynamic_batch or len(self.batch) == s.groups_per_batch
            )
            if s.workflow == "v2_inline" and not self.batch_collected:
                self.producer_due = self.now
                self.produce()
        if not self.batch:
            raise UnsupportedEstimate(
                "Dynamic batch contains no accepted actor workload"
            )
        if not self.actor_active:
            source = self.checkpoint_config

            def started():
                self.actor_active = True
                self.validate_allocation(True)

            self.transition(
                "restore" if source else "start", source, self.config, started
            )
            return
        selected = self.batch
        workload = self.merge_workload(selected)
        inputs = {"auxiliary_stages": step.auxiliary_stages}
        if step.advantage_workload is not None:
            inputs["advantage_workload"] = step.advantage_workload
        estimate = self.models.actor.estimate(self.config.actor, workload, **inputs)
        # Check all phases before consuming the batch, preserving partial-result accounting.
        durations = [(name, self.duration(value)) for name, value in estimate.stages]
        self.queue_wait += sum(self.now - self.ready[gid][1] for gid in selected)
        for gid in selected:
            del self.ready[gid]
        self.batch = []
        self.batch_collected = False
        self.phase = "training"
        self.emit(
            "actor_start",
            str(self.index),
            groups=tuple(selected),
            accepted_total=self.accepted,
            sequence_tokens=workload.sequence_tokens,
        )
        elapsed = 0.0
        for name, duration in durations:
            self.schedule(
                elapsed, lambda name=name: self.emit("actor_phase_start", name)
            )
            elapsed += duration
            self.schedule(
                elapsed, lambda name=name: self.emit("actor_phase_complete", name)
            )
        self.schedule(elapsed, self.actor_complete)

    def merge_workload(self, ids: list[str]) -> WorkloadStats:
        groups = [self.groups[gid] for gid in ids]
        rows = [g.workload for g in groups]
        lengths = (
            tuple(n for row in rows for n in row.sequence_lengths)
            if all(row.sequence_lengths is not None for row in rows)
            else None
        )
        sizes = tuple(
            n for row in rows for n in (row.trajectory_group_sizes or (row.n_seqs,))
        )
        weights = []
        for group in groups:
            row = group.workload
            if row.trajectory_dispatch_weights is not None:
                weights.extend(row.trajectory_dispatch_weights)
            elif (
                group.dispatch_weight is not None
                and len(row.trajectory_group_sizes or (row.n_seqs,)) == 1
            ):
                weights.append(group.dispatch_weight)
            else:
                weights = None
                break
        return WorkloadStats(
            sum(r.n_seqs for r in rows),
            sum(r.sequence_tokens for r in rows),
            max_seq_len=max(r.max_seq_len for r in rows)
            if all(r.max_seq_len is not None for r in rows)
            else None,
            min_seq_len=min(r.min_seq_len for r in rows)
            if all(r.min_seq_len is not None for r in rows)
            else None,
            loss_tokens=sum(r.loss_tokens for r in rows)
            if all(r.loss_tokens is not None for r in rows)
            else None,
            sequence_lengths=lengths,
            trajectory_group_sizes=sizes,
            trajectory_dispatch_weights=tuple(weights) if weights is not None else None,
        )

    def actor_complete(self) -> None:
        step = self.plan[self.index]
        self.attempted += 1
        self.completed += int(step.update_succeeds)
        self.training = self.provider + 1
        self.phase = "await_publish"
        self.admission_paused = self.freeze_generation = True
        self.emit(
            "actor_complete",
            str(self.index),
            successful=step.update_succeeds,
            training_version=self.training,
            provider_version=self.provider,
        )

    def published(self) -> None:
        step = self.plan[self.index]
        if not step.publish_succeeds:
            raise UnsupportedEstimate(
                "Publication failed; provider and serving versions were not advanced"
            )
        self.serving = self.provider = self.training
        self.emit("weights_published", version=self.serving)
        self.emit("provider_version", version=self.provider)
        self.freeze_generation = False
        if step.checkpoint:

            def saved():
                self.checkpoint_config = self.config
                self.emit("checkpoint_saved", version=self.training)
                self.post_publish()

            self.transition("checkpoint", self.config, self.config, saved)
        else:
            self.post_publish()

    def post_publish(self) -> None:
        self.phase = "post_publish"
        self.schedule(
            self.duration(self.plan[self.index].after_publish), self.after_publish
        )

    def after_publish(self) -> None:
        step = self.plan[self.index]
        self.final_admission_pass = (
            self.scenario.workflow == "v2_inline" and self.index + 1 == len(self.plan)
        )
        self.admission_paused = (
            self.index + 1 >= len(self.plan) and not self.final_admission_pass
        )
        if not self.admission_paused:
            self.emit("admission_resume", version=self.provider)
        if step.release_after:

            def released():
                self.actor_active = False
                self.validate_allocation(False)
                self.finish_step()

            self.transition("release", self.config, None, released)
        else:
            self.finish_step()

    def finish_step(self) -> None:
        self.index += 1
        self.phase = "idle"
        self.producer_due = self.now

    def cancel_running(self) -> None:
        self.cancelled += self.running
        for gid in (*self.active.keys(), *self.submitted):
            self.emit("group_cancelled", gid)
        self.active.clear()
        self.submitted.clear()
        self.requests.clear()
        self.responses.clear()
        self.external_live.clear()
        self.external_counts = dict.fromkeys(self.external_counts, 0)
        self.freeze_generation = False

    def terminal(self) -> None:
        self.admission_paused = True
        self.drain_submitted = self.scenario.terminal == "drain"
        if self.running:
            if self.scenario.terminal == "cancel":
                self.freeze_generation = True
                if not any(self.replica_busy):
                    self.transition(
                        "cancel", self.config, self.config, self.cancel_running
                    )
            return
        if self.terminal_started:
            return
        self.terminal_started = True
        self.freeze_generation = True

        def cleaned():
            if self.ledger.devices("actor") or self.ledger.devices("rollout"):
                raise ValueError("Terminal transition left allocated devices")
            self.done = True
            self.emit(
                "simulation_complete",
                attempted_updates=self.attempted,
                completed_updates=self.completed,
            )

        self.transition(
            "shutdown" if self.actor_active else "shutdown_suspended",
            self.config,
            None,
            cleaned,
        )
