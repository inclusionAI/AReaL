# SPDX-License-Identifier: Apache-2.0

"""Composable, CPU-only workflow estimates and causal workload replay.

Components describe elapsed time and *additional* GPU work. The simulator owns
training/inference residency, including idle and transition time, so parallel
components cannot accidentally charge the same training group twice.
"""

from __future__ import annotations

import heapq
import math
from collections import deque
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Protocol

from ...contracts import BatchWorkload, ElasticObservation, TrainingConfig
from ...contracts import ForecastUpdate as ForecastUpdate
from ...contracts import HistorySnapshot as HistorySnapshot
from ...contracts import HistoryWindow as HistoryWindow
from ...contracts import Measurement as Measurement
from ...contracts import PlanStep as PlanStep
from ...contracts import SimulationResult as SimulationResult
from .cost_model.rollout import RolloutCostModel
from .cost_model.transition import JointTransitionModel
from .joint import JointConfig, JointEvent, JointPrediction, JointScenario

if TYPE_CHECKING:
    from .actor_cost import ActorCatalog


@dataclass(frozen=True)
class ComponentContext:
    config: TrainingConfig
    workload: BatchWorkload
    history: HistorySnapshot
    source_config: TrainingConfig | None = None


@dataclass(frozen=True)
class Estimate:
    seconds: float = 0.0
    extra_gpu_seconds: float = 0.0
    feasible: bool = True
    reason: str = ""
    unmeasured: bool = False

    def __post_init__(self) -> None:
        if any(
            not math.isfinite(x) or x < 0
            for x in (self.seconds, self.extra_gpu_seconds)
        ):
            raise ValueError("Component estimates must be finite and nonnegative")


class Component(Protocol):
    def estimate(self, context: ComponentContext) -> Estimate: ...


@dataclass(frozen=True)
class ConstantComponent:
    seconds: float = 0.0
    extra_gpu_seconds: float = 0.0

    def estimate(self, context: ComponentContext) -> Estimate:
        return Estimate(self.seconds, self.extra_gpu_seconds)


@dataclass(frozen=True)
class SequenceComponent:
    components: tuple[Component, ...]

    def estimate(self, context: ComponentContext) -> Estimate:
        estimates = [component.estimate(context) for component in self.components]
        return Estimate(
            sum(x.seconds for x in estimates),
            sum(x.extra_gpu_seconds for x in estimates),
            all(x.feasible for x in estimates),
            "; ".join(x.reason for x in estimates if not x.feasible),
            all(x.unmeasured for x in estimates if not x.feasible)
            and any(not x.feasible for x in estimates),
        )


@dataclass(frozen=True)
class ParallelComponent:
    """Children overlap; workflow authors must ensure their resources can overlap."""

    components: tuple[Component, ...]

    def estimate(self, context: ComponentContext) -> Estimate:
        estimates = [component.estimate(context) for component in self.components]
        return Estimate(
            max((x.seconds for x in estimates), default=0.0),
            sum(x.extra_gpu_seconds for x in estimates),
            all(x.feasible for x in estimates),
            "; ".join(x.reason for x in estimates if not x.feasible),
            all(x.unmeasured for x in estimates if not x.feasible)
            and any(not x.feasible for x in estimates),
        )


@dataclass(frozen=True)
class MeasuredComponent:
    """Same full-profile measurements only; optional local token scaling for train.

    No linear GPU speedup assumption. Missing profiles and distant workloads are
    explicitly infeasible. Seed measurements can be supplied by calibration.
    """

    phase: str
    seed: tuple[Measurement, ...] = ()
    scale_tokens: bool = False
    max_token_ratio: float = 4.0

    def estimate(self, context: ComponentContext) -> Estimate:
        samples = [
            x
            for x in (*self.seed, *context.history.measurements)
            if x.phase == self.phase
            and (x.world_size, x.profile_id)
            == (context.config.world_size, context.config.profile_id)
        ]
        if self.scale_tokens:
            samples = [x for x in samples if x.sequence_tokens > 0]
        if not samples:
            return Estimate(
                feasible=False,
                reason=f"No {self.phase} profile for {context.config.world_size} GPUs",
                unmeasured=True,
            )
        if not self.scale_tokens:
            # Prefer the latest window over the calibration seed.
            selected = samples[-5:]
            return Estimate(sum(x.seconds for x in selected) / len(selected))
        tokens = context.workload.sequence_tokens
        selected = min(
            reversed(samples), key=lambda x: abs(math.log(tokens / x.sequence_tokens))
        )
        ratio = tokens / selected.sequence_tokens
        if not 1 / self.max_token_ratio <= ratio <= self.max_token_ratio:
            return Estimate(
                feasible=False, reason=f"Workload outside {self.phase} profile range"
            )
        return Estimate(selected.seconds * ratio)


@dataclass(frozen=True)
class HistoricalRolloutComponent:
    """The supplied duration is a forecast from past completed episodes."""

    def estimate(self, context: ComponentContext) -> Estimate:
        return Estimate(context.workload.rollout_seconds)


@dataclass(frozen=True)
class WorkflowModel:
    """Each phase may itself be a custom component or a nested composition."""

    train: Component
    start: Component
    restore: Component
    checkpoint: Component
    publish: Component
    release: Component
    inference_gpus: float = 1.0
    rollout: Component = field(default_factory=HistoricalRolloutComponent)

    def __post_init__(self) -> None:
        if not math.isfinite(self.inference_gpus) or self.inference_gpus < 0:
            raise ValueError("Inference residency must be finite and nonnegative")


class WorkflowSimulator:
    def __init__(self, workflow: WorkflowModel):
        self.workflow = workflow

    def evaluate(
        self,
        observation: ElasticObservation,
        forecast: tuple[ForecastUpdate, ...],
        plan: tuple[PlanStep, ...],
        history: HistorySnapshot,
        *,
        terminal_release: bool = True,
    ) -> SimulationResult:
        if not forecast or len(forecast) != len(plan):
            raise ValueError("One plan step is required per forecast update")
        current = observation.training_config if observation.training_active else None
        checkpoint = observation.has_checkpoint
        checkpoint_config = observation.checkpoint_config or (
            observation.training_config if checkpoint else None
        )
        seconds = training = extra = 0.0
        samples = tokens = 0
        events = []
        profile_events = []

        def run(phase: str, config: TrainingConfig, workload: BatchWorkload) -> float:
            nonlocal seconds, training, extra
            estimate = getattr(self.workflow, phase).estimate(
                ComponentContext(
                    config,
                    workload,
                    history,
                    checkpoint_config if phase == "restore" else None,
                )
            )
            if not estimate.feasible:
                raise InfeasiblePlan(estimate.reason, unmeasured=estimate.unmeasured)
            seconds += estimate.seconds
            training += config.world_size * estimate.seconds
            extra += estimate.extra_gpu_seconds
            events.append((phase, config.world_size, estimate.seconds))
            profile_events.append((phase, config.profile_id, estimate.seconds))
            return estimate.seconds

        try:
            for future, step in zip(forecast, plan, strict=True):
                target = step.config
                if target.world_size > observation.max_train_gpus:
                    raise InfeasiblePlan(
                        "Configuration exceeds available training pool"
                    )
                rollout = self.workflow.rollout.estimate(
                    ComponentContext(
                        target,
                        replace(future.workload, rollout_seconds=future.wait_seconds),
                        history,
                    )
                )
                if not rollout.feasible:
                    raise InfeasiblePlan(rollout.reason, unmeasured=rollout.unmeasured)
                wait = rollout.seconds
                extra += rollout.extra_gpu_seconds
                if not math.isfinite(wait) or wait < 0:
                    raise ValueError("Forecast wait must be finite and nonnegative")
                if current is not None and step.release_before:
                    release = run("release", current, future.workload)
                    # Release and rollout overlap; training residency ends at release.
                    seconds += max(0.0, wait - release)
                    current = None
                else:
                    seconds += wait
                    if current is not None:
                        training += current.world_size * wait
                if current is not None and current != target:
                    run("release", current, future.workload)
                    current = None
                if current is None:
                    run("restore" if checkpoint else "start", target, future.workload)
                    current = target
                for phase in ("train", "checkpoint", "publish"):
                    run(phase, current, future.workload)
                checkpoint = True
                checkpoint_config = current
                samples += future.workload.samples
                tokens += future.workload.effective_tokens
            # Every candidate ends suspended: no free terminal allocation.
            if terminal_release:
                run("release", current, forecast[-1].workload)
        except InfeasiblePlan as error:
            return SimulationResult(
                False, reason=str(error), unmeasured=error.unmeasured
            )
        return SimulationResult(
            True,
            seconds,
            training,
            self.workflow.inference_gpus * seconds,
            extra,
            samples,
            tokens,
            events=tuple(events),
            profile_events=tuple(profile_events),
        )


class InfeasiblePlan(Exception):
    def __init__(self, reason: str, *, unmeasured: bool = False):
        super().__init__(reason)
        self.unmeasured = unmeasured


def forecast_window(
    observation: ElasticObservation,
    history: HistorySnapshot,
    horizon: int,
    *,
    preserve_workload: bool = False,
) -> tuple[ForecastUpdate, ...]:
    """Use empirical window means as future workload, without reading ahead.

    The ready head is known input. During an episode, estimate remaining time
    from historical durations conditional on not having completed yet. Outside
    the observed tail, use the historical mean duration as the remaining wait.
    """
    if horizon < 1:
        raise ValueError("Forecast horizon must be positive")
    workloads = history.workloads
    typical = None
    if workloads:
        typical = BatchWorkload(
            math.ceil(sum(x.samples for x in workloads) / len(workloads)),
            math.ceil(sum(x.effective_tokens for x in workloads) / len(workloads)),
            math.ceil(sum(x.sequence_tokens for x in workloads) / len(workloads)),
            sum(x.rollout_seconds for x in workloads) / len(workloads),
        )
        if preserve_workload:
            observed = next(
                (x for x in reversed(workloads) if x.actor_stats is not None),
                observation.next_batch or workloads[-1],
            )
            typical = replace(observed, rollout_seconds=typical.rollout_seconds)
    if observation.next_batch is None and not workloads:
        return ()
    forecast = []
    if observation.next_batch is not None:
        forecast.append(ForecastUpdate(observation.next_batch, 0.0))
    for _ in range(horizon - len(forecast)):
        workload = typical if typical is not None else observation.next_batch
        wait = workload.rollout_seconds
        if not forecast and observation.episode_active:
            durations = [
                x.rollout_seconds
                for x in workloads
                if x.rollout_seconds > observation.episode_elapsed
            ]
            wait = (
                sum(durations) / len(durations) - observation.episode_elapsed
                if durations
                else typical.rollout_seconds
            )
        forecast.append(ForecastUpdate(workload, wait))
    return tuple(forecast)


class JointSimulator:
    """Fixed-config finite windows with one shared aggregate inference channel.

    Admission/staleness are deliberately in whole logical batches. This is a
    coarse Basic model, not a reconstruction of group-level controller capacity.
    Initial state must be committed and drained of in-flight work; ready batches
    and cumulative counters are explicit. All candidates finish with cleanup.
    """

    def __init__(
        self,
        actors: ActorCatalog,
        rollout: RolloutCostModel,
        transitions: JointTransitionModel,
    ):
        self.actors, self.rollout, self.transitions = actors, rollout, transitions

    def evaluate(self, scenario: JointScenario, config: JointConfig) -> JointPrediction:
        return _JointRun(self, scenario, config).run()


class _JointRun:
    def __init__(
        self, model: JointSimulator, scenario: JointScenario, config: JointConfig
    ):
        self.model, self.scenario, self.config = model, scenario, config
        self.now = self.actor_cost = self.rollout_cost = self.queue_wait = 0.0
        self.provider, self.accepted = (
            scenario.provider_version,
            scenario.accepted_batches,
        )
        self.completed = self.sequence = 0
        self.ready = dict.fromkeys(range(scenario.ready_batches), 0.0)
        self.next_input = scenario.ready_batches
        self.running = set()
        self.gpu_queue = deque()
        self.heap = []
        self.events = []
        self.actor_times, self.rollout_times, self.profiles = [], {}, {}
        self.sources = []
        self.notes = [
            "Fixed logical workload and trajectory outcomes; no reward prediction",
            "Batch-level FIFO and staleness proxy; max_inflight_batches is not group-level K",
            "One shared aggregate inference channel; external waits release that channel",
            "Replica speedup is capped by stage parallel requests and assumes balanced token work; no TP scaling",
            "Publication waits for the current aggregate GPU stage; external waits continue",
            "Initial state is committed with no in-flight work; supplied ready batches retain their cost boundary",
            "Actor memory and rollout KV checks do not certify complete resource feasibility",
        ]
        self.active_transition = None
        self.actor_busy = self.gpu_busy = self.pending_publish = self.done = False
        self.allocation = (config.actor.world_size, config.rollout.world_size)

    def emit(self, kind: str, batch: int = -1) -> None:
        self.events.append(
            JointEvent(self.now, kind, batch, self.provider, self.accepted)
        )

    def schedule(self, seconds: float, kind: str, data=None) -> None:
        self.sequence += 1
        heapq.heappush(self.heap, (self.now + seconds, self.sequence, kind, data))

    def within_limits(self, actor: int, rollout: int) -> bool:
        s = self.scenario
        return (
            actor <= s.max_actor_gpus
            and rollout <= s.max_rollout_gpus
            and actor + rollout <= s.max_total_gpus
        )

    def preflight(self) -> tuple[str, str] | None:
        s, c = self.scenario, self.config
        if not self.within_limits(c.actor.world_size, c.rollout.world_size):
            return "infeasible", "Joint configuration exceeds GPU pool limits"
        if s.current and (
            s.current.rollout.model_id,
            s.current.rollout.hardware_id,
            s.current.rollout.sampling_id,
        ) != (c.rollout.model_id, c.rollout.hardware_id, c.rollout.sampling_id):
            return (
                "infeasible",
                "Joint comparison changes model, hardware cost basis or sampling semantics",
            )
        try:
            if s.current is not None:
                self.model.actors.actor_config(s.current.actor)
            self.model.actors.actor_config(c.actor)
            for index, batch in enumerate(s.batches):
                prediction = self.model.actors.predict(
                    c.actor, batch.stats_for(c.actor)
                )
                if prediction.static_memory_exceeds_capacity:
                    return (
                        "infeasible",
                        "Actor static memory lower bound exceeds capacity",
                    )
                if not math.isfinite(prediction.seconds) or prediction.seconds < 0:
                    raise ValueError("Actor returned invalid service time")
                self.actor_times.append(prediction.seconds)
                self.notes.extend(prediction.assumptions)
                if batch.inferred_lengths:
                    self.notes.append(
                        "Actor lengths are a fixed shared surrogate inferred before candidate search"
                    )
                if index >= s.ready_batches:
                    estimate = self.model.rollout.predict(c.rollout, batch.rollout)
                    if not estimate.supported:
                        return "unsupported", estimate.reason
                    self.rollout_times[index] = estimate.stages
                    self.sources.append(estimate.source_id)
                    self.notes.append(
                        f"Rollout rate evidence: {estimate.evidence} ({estimate.source_id})"
                    )
        except ValueError as exc:
            return "unsupported", str(exc)
        actions = [("publish", c, c), ("shutdown", c, None)]
        if s.checkpoint_every and any(
            (s.provider_version + i + 1) % s.checkpoint_every == 0
            for i in range(len(s.batches))
        ):
            actions.append(("checkpoint", c, c))
        if s.current != c:
            actions.append(
                ("start" if s.current is None else "reconfigure", s.current, c)
            )
        for action, source, target in actions:
            profile = self.model.transitions.lookup(action, source, target)
            if profile is None:
                return "unsupported", f"Missing directed {action} transition profile"
            if any(
                not self.within_limits(p.actor_gpus, p.rollout_gpus)
                for p in profile.phases
            ):
                return "infeasible", f"{action} temporary allocation exceeds GPU limits"
            self.profiles[action] = profile
            self.sources.append(profile.source_id)
        return None

    def transition(self, action: str) -> None:
        profile = self.profiles[action]
        self.active_transition = action
        self.emit(action + "_start")
        self.allocation = (profile.phases[0].actor_gpus, profile.phases[0].rollout_gpus)
        elapsed = 0.0
        for index, phase in enumerate(profile.phases):
            if index:
                self.schedule(
                    elapsed, "allocation", (phase.actor_gpus, phase.rollout_gpus)
                )
            elapsed += phase.seconds
        self.schedule(elapsed, "transition_end", action)

    def pump(self) -> None:
        if self.active_transition or self.done:
            return
        if self.pending_publish:
            if not self.gpu_busy:
                self.transition("publish")
            return
        s = self.scenario
        if self.completed == len(s.batches):
            self.transition("shutdown")
            return
        if not self.actor_busy and self.completed in self.ready:
            self.queue_wait += self.now - self.ready.pop(self.completed)
            self.actor_busy = True
            self.emit("actor_start", self.completed)
            self.schedule(self.actor_times[self.completed], "actor_end", self.completed)
        # accepted is cumulative; dequeuing a batch never grants fresh capacity.
        while (
            self.next_input < len(s.batches)
            and min(
                self.config.rollout.max_inflight_batches - len(self.running),
                s.staleness + self.provider + 1 - self.accepted - len(self.running),
            )
            > 0
        ):
            index = self.next_input
            self.next_input += 1
            self.running.add(index)
            self.gpu_queue.append((index, 0))
            self.emit("rollout_admitted", index)
        if not self.gpu_busy and self.gpu_queue:
            index, stage = self.gpu_queue.popleft()
            self.gpu_busy = True
            self.emit("rollout_service_start", index)
            self.schedule(
                self.rollout_times[index][stage][0], "gpu_end", (index, stage)
            )

    def dispatch(self, kind: str, data) -> None:
        if kind == "allocation":
            self.allocation = data
        elif kind == "gpu_end":
            self.gpu_busy = False
            index, stage = data
            self.emit("rollout_service_end", index)
            self.schedule(self.rollout_times[index][stage][1], "external_end", data)
        elif kind == "external_end":
            index, stage = data
            if stage + 1 < len(self.rollout_times[index]):
                self.gpu_queue.append((index, stage + 1))
            else:
                self.running.remove(index)
                self.ready[index] = self.now
                self.accepted += 1
                self.emit("batch_ready", index)
        elif kind == "actor_end":
            self.pending_publish = True
            self.emit("actor_end", data)
        elif kind == "transition_end":
            self.active_transition = None
            self.allocation = (
                self.config.actor.world_size,
                self.config.rollout.world_size,
            )
            self.emit(data + "_end")
            if data == "publish":
                self.provider += 1
                self.completed += 1
                self.actor_busy = self.pending_publish = False
                self.emit("provider_version", self.completed - 1)
                if (
                    self.scenario.checkpoint_every
                    and self.provider % self.scenario.checkpoint_every == 0
                ):
                    self.transition("checkpoint")
            elif data == "shutdown":
                self.allocation = (0, 0)
                self.done = True

    def result(self, status: str, reason: str) -> JointPrediction:
        consumed = self.scenario.batches[: self.completed]
        return JointPrediction(
            status,
            self.config,
            self.now,
            self.actor_cost,
            self.rollout_cost,
            self.completed,
            sum(b.actor.samples for b in consumed),
            sum(b.actor.effective_tokens for b in consumed),
            sum(b.actor.sequence_tokens for b in consumed),
            self.queue_wait,
            reason,
            tuple(self.events),
            tuple(dict.fromkeys(self.notes)),
            tuple(dict.fromkeys(self.sources)),
        )

    def run(self) -> JointPrediction:
        error = self.preflight()
        if error:
            return self.result(*error)
        if self.scenario.current != self.config:
            self.transition("start" if self.scenario.current is None else "reconfigure")
        self.pump()
        while not self.done:
            if not self.heap:
                return self.result(
                    "incomplete",
                    "No progress: initial counters or batch admission capacity block the fixed window",
                )
            next_time = self.heap[0][0]
            delta = next_time - self.now
            self.actor_cost += self.allocation[0] * delta
            self.rollout_cost += self.allocation[1] * delta
            self.now = next_time
            # Complete all same-time work before allowing a new service/admission.
            while self.heap and self.heap[0][0] == self.now:
                _, _, kind, data = heapq.heappop(self.heap)
                self.dispatch(kind, data)
            self.pump()
        return self.result(
            "complete",
            "Fixed joint configuration completed the same window and terminal cleanup",
        )
