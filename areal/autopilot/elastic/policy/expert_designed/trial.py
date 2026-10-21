# SPDX-License-Identifier: Apache-2.0

"""Compose an archived workload with frozen models, without launching engines."""

from __future__ import annotations

from dataclasses import dataclass, replace

from ....contracts import WorkloadStats
from .catalog import TrialArchive
from .core import Estimate, nonnegative
from .cost_model.actor import ExpertActorModel
from .cost_model.transition import (
    AllocationChange,
    ExpertTransitionModel,
    TransferTask,
    TransitionProfile,
)
from .scenario import ExternalCall, Group, InitialState, PlanStep, Request, Scenario
from .simulator import ExpertModels


@dataclass(frozen=True)
class DiagnosticActor:
    """Explicit legacy diagnostic fallback; the bounded model remains available."""

    raw: ExpertActorModel
    bounded: ExpertActorModel
    advantage_constant: float
    advantage_source_id: str

    def estimate(self, config, workload, **kwargs):
        strict = self.bounded.estimate(config, workload, **kwargs)
        raw = self.raw.estimate(config, workload, **kwargs)
        raw_stages = dict(raw.stages)
        stages = []
        for phase, value in strict.stages:
            if value.seconds is None:
                value = (
                    Estimate(
                        self.advantage_constant,
                        "assumed",
                        notes=(
                            "Explicit constant prior outside bounded advantage coverage",
                        ),
                        source_ids=(self.advantage_source_id,),
                    )
                    if phase == "compute_advantage"
                    else replace(
                        raw_stages[phase],
                        notes=raw_stages[phase].notes
                        + ("Explicit raw-prior fallback outside calibrated coverage",),
                    )
                )
            stages.append((phase, value))
        return replace(strict, stages=tuple(stages))


def simulation(
    trial: TrialArchive,
    *,
    cleanup_seconds: float | None = None,
    diagnostic_fallback: bool = False,
) -> tuple[ExpertModels, Scenario, tuple[PlanStep, ...]]:
    """Load stored dependencies and coefficients; never copy observed arrivals.

    Strict calibrated coverage is the default. The r10 reference comparison uses
    diagnostic_fallback=True, retaining its explicitly assumed out-of-range work.
    """
    manifest = trial.manifest()
    assembled = trial._models(manifest)
    spec = trial._read_data(manifest, "workload")
    if cleanup_seconds is not None:
        nonnegative("cleanup_seconds", cleanup_seconds)
    groups = []
    for stored in trial._read_data(manifest, "groups"):
        value = stored.copy()
        nodes = []
        for row in value.pop("nodes"):
            node = row.copy()
            kind = node.pop("kind")
            if kind == "request":
                nodes.append(Request(**node))
            elif kind == "external":
                estimate = Estimate(**node.pop("duration"))
                if cleanup_seconds is not None and node["pool"] == "cleanup":
                    estimate = replace(
                        estimate, seconds=cleanup_seconds, evidence="assumed"
                    )
                nodes.append(ExternalCall(duration=estimate, **node))
            else:
                raise ValueError("Unknown archived dependency node kind")
        groups.append(
            Group(
                nodes=tuple(nodes),
                workload=WorkloadStats(**value.pop("workload")),
                postprocess=Estimate(**value.pop("postprocess")),
                **value,
            )
        )
    scenario = Scenario(
        tuple(groups), InitialState(assembled.config), **spec["scenario"]
    )
    plan = tuple(PlanStep(assembled.config, **row) for row in spec["plan"])
    calibration = trial._read_data(manifest, "calibration")
    profiles = []
    for row in calibration["transitions"]:
        value = row.copy()
        tasks = []
        for stored in value.pop("tasks"):
            task = stored.copy()
            if task.get("measured") is not None:
                task["measured"] = Estimate(**task["measured"])
            tasks.append(TransferTask(**task))
        changes = tuple(
            AllocationChange(**change) for change in value.pop("allocation_changes")
        )
        for key in ("source", "target"):
            if value[key] == "$execution":
                value[key] = assembled.config.key
        profiles.append(
            TransitionProfile(tasks=tuple(tasks), allocation_changes=changes, **value)
        )
    actor = assembled.actor
    if diagnostic_fallback:
        fallback = calibration["diagnostic_fallback"]
        actor = DiagnosticActor(
            assembled.raw_actor,
            actor,
            fallback["advantage_seconds"],
            fallback["advantage_source_id"],
        )
    models = ExpertModels(
        actor,
        assembled.rollout,
        ExpertTransitionModel(tuple(profiles)),
        assembled.rpc_timing,
    )
    return models, scenario, plan
