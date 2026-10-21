# SPDX-License-Identifier: Apache-2.0

"""Bridge actor timing estimates into the lifecycle simulator's component API."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from ...contracts import BatchWorkload, TrainingConfig
from .cost_model import ActorConfig, ActorCostModel, WorkloadStats
from .simulator import (
    Component,
    ComponentContext,
    Estimate,
    WorkflowModel,
    WorkflowSimulator,
)

if TYPE_CHECKING:
    from .policy import BasicPolicy


@dataclass(frozen=True)
class ActorPrediction:
    stages: tuple[tuple[str, float], ...]
    static_memory_exceeds_capacity: bool
    assumptions: tuple[str, ...]

    @property
    def seconds(self) -> float:
        return sum(value for _, value in self.stages)


class ActorCatalog:
    """Full configuration identity; aliases of one trial layout share one profile.

    Cache is bounded and keyed by complete stats and configuration, never just GPU
    count or token total. Changes to timing coefficients do not mutate workloads.
    """

    def __init__(
        self,
        model: ActorCostModel,
        configurations: Mapping[str, ActorConfig],
        *,
        cache_size: int = 512,
    ):
        if not configurations or type(cache_size) is not int or cache_size < 1:
            raise ValueError(
                "Actor catalog requires candidates and a positive cache size"
            )
        self.model, self.cache_size = model, cache_size
        self._actors: dict[TrainingConfig, ActorConfig] = {}
        self.labels: dict[str, TrainingConfig] = {}
        self._cache: OrderedDict[tuple[TrainingConfig, str], ActorPrediction] = (
            OrderedDict()
        )
        semantics = None
        for label, actor in configurations.items():
            if not isinstance(label, str) or not label.strip():
                raise ValueError("Candidate labels must be nonempty strings")
            # Replanning must not change the number or meaning of optimizer updates
            # on a queued batch. Parallelism/checkpoint/packing can vary independently.
            current = (
                actor.ppo_n_minibatches,
                actor.dtype,
                actor.use_decoupled_loss,
                actor.prox_logp_method,
                actor.recompute_logprob,
            )
            if semantics is not None and semantics != current:
                raise ValueError(
                    "Candidate configurations change training semantics on the same data"
                )
            semantics = current
            identity = {
                "actor": asdict(actor),
                "model": asdict(model.model),
                "hardware": asdict(model.hardware),
            }
            digest = hashlib.sha256(
                json.dumps(identity, sort_keys=True).encode()
            ).hexdigest()
            config = TrainingConfig(actor.parallel.world_size, "actor-" + digest)
            if config in self._actors and self._actors[config] != actor:
                raise ValueError("Conflicting actor configuration identity")
            self._actors[config] = actor
            self.labels[label] = config

    @property
    def configurations(self) -> tuple[TrainingConfig, ...]:
        return tuple(self._actors)

    def actor_config(self, config: TrainingConfig) -> ActorConfig:
        try:
            return self._actors[config]
        except KeyError as exc:
            raise ValueError(
                f"Unknown full actor profile {config.profile_id!r} on {config.world_size} GPUs"
            ) from exc

    def aliases(self, config: TrainingConfig) -> tuple[str, ...]:
        return tuple(label for label, target in self.labels.items() if target == config)

    def clear_cache(self) -> None:
        """Call after replacing timing coefficients or component implementations."""
        self._cache.clear()

    def predict(self, config: TrainingConfig, stats: WorkloadStats) -> ActorPrediction:
        actor = self.actor_config(config)
        key = config, json.dumps(asdict(stats), sort_keys=True, allow_nan=False)
        if key not in self._cache:
            report = self.model.estimate(actor, stats, sensitivity=False)
            self._cache[key] = ActorPrediction(
                tuple(report["stages_seconds"].items()),
                report["static_memory_exceeds_capacity"],
                tuple(report["assumptions"]),
            )
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        self._cache.move_to_end(key)
        return self._cache[key]


def actor_workload(
    stats: WorkloadStats,
    *,
    samples: int,
    effective_tokens: int | None = None,
    rollout_seconds: float = 0,
) -> BatchWorkload:
    """samples is an explicit trajectory/sample count, NOT inferred from n_seqs."""
    if effective_tokens is None:
        if stats.loss_tokens is None:
            raise ValueError("Supply effective_tokens or observed loss_tokens")
        effective_tokens = round(stats.loss_tokens)
    return BatchWorkload(
        samples, effective_tokens, round(stats.sequence_tokens), rollout_seconds, stats
    )


@dataclass(frozen=True)
class ActorTrainComponent:
    catalog: ActorCatalog

    def estimate(self, context: ComponentContext) -> Estimate:
        stats = context.workload.actor_stats
        if stats is None:
            return Estimate(
                feasible=False,
                unmeasured=True,
                reason="Actor timing needs actual n_seqs/sequence stats; samples are not sequence count",
            )
        try:
            prediction = self.catalog.predict(context.config, stats)
        except ValueError as exc:
            return Estimate(
                feasible=False, reason=f"Actor prediction unavailable: {exc}"
            )
        if prediction.static_memory_exceeds_capacity:
            return Estimate(
                feasible=False,
                reason="Actor static state lower bound exceeds GPU memory",
            )
        # Shared GPU residency is charged ONCE by WorkflowSimulator, including idle.
        return Estimate(seconds=prediction.seconds)


@dataclass(frozen=True)
class CompatibleRestore:
    catalog: ActorCatalog
    component: Component
    allowed_transitions: frozenset[tuple[TrainingConfig, TrainingConfig]] = frozenset()

    def __post_init__(self) -> None:
        for source, target in self.allowed_transitions:
            self.catalog.actor_config(source)
            self.catalog.actor_config(target)

    def estimate(self, context: ComponentContext) -> Estimate:
        source, target = context.source_config, context.config
        try:
            self.catalog.actor_config(target)
            if source is not None:
                self.catalog.actor_config(source)
        except ValueError as exc:
            return Estimate(feasible=False, reason=str(exc))
        if source is None:
            return Estimate(
                feasible=False, reason="Checkpoint source profile is unknown"
            )
        if source != target and (source, target) not in self.allowed_transitions:
            return Estimate(
                feasible=False,
                reason="Checkpoint restore between these profiles is not declared compatible",
            )
        return self.component.estimate(context)


@dataclass(frozen=True)
class LifecycleCosts:
    """Lifecycle timing components supplied by the workflow, not invented profiles.

    ConstantComponent is appropriate for explicit what-if assumptions. Existing
    MeasuredComponent supports exact full-profile identity when data is available.
    """

    start: Component
    restore: Component
    checkpoint: Component
    publish: Component
    release: Component
    inference_gpus: float

    def workflow(self, catalog: ActorCatalog, allowed_transitions=()) -> WorkflowModel:
        return WorkflowModel(
            train=ActorTrainComponent(catalog),
            start=self.start,
            restore=CompatibleRestore(
                catalog, self.restore, frozenset(allowed_transitions)
            ),
            checkpoint=self.checkpoint,
            publish=self.publish,
            release=self.release,
            inference_gpus=self.inference_gpus,
        )


def build_actor_policy(
    catalog: ActorCatalog,
    lifecycle: LifecycleCosts,
    *,
    allowed_transitions=(),
    history=None,
    **optimizer_settings,
) -> BasicPolicy:
    """cost model -> simulator -> lifecycle optimizer -> ordinary policy decisions."""
    from .optimizer import LifecycleOptimizer
    from .policy import BasicPolicy

    return BasicPolicy(
        WorkflowSimulator(lifecycle.workflow(catalog, allowed_transitions)),
        optimizer=LifecycleOptimizer(catalog.configurations, **optimizer_settings),
        history=history,
    )
