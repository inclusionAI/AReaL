# SPDX-License-Identifier: Apache-2.0

"""Replay an explicit offline scenario; no GPUs, profiling or model API calls."""

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

from ...contracts import ElasticObservation
from .actor_cost import (
    ActorCatalog,
    LifecycleCosts,
    actor_workload,
    build_actor_policy,
)
from .cost_model import (
    ActorConfig,
    ActorCostModel,
    Hardware,
    WorkloadStats,
    hardware_profile,
    qwen3_30b_a3b,
    trial_config,
)
from .simulator import ConstantComponent, forecast_window


def catalog_from_scenario(data: dict) -> ActorCatalog:
    hardware = data["hardware"]
    hardware = (
        hardware_profile(hardware)
        if isinstance(hardware, str)
        else Hardware(**hardware)
    )
    actors = {}
    for label in data.get("trials", ()):
        actors[label] = replace(trial_config(label), **data.get("actor_overrides", {}))
    for label, actor in data.get("candidates", {}).items():
        if label in actors:
            raise ValueError(f"Duplicate candidate label {label}")
        actors[label] = ActorConfig.from_areal(actor)
    return ActorCatalog(ActorCostModel(qwen3_30b_a3b(), hardware), actors)


def workload_from_dict(data: dict):
    return actor_workload(
        WorkloadStats(**data["stats"]),
        samples=data["samples"],
        effective_tokens=data.get("effective_tokens"),
        rollout_seconds=data.get("rollout_seconds", 0),
    )


def run_scenario(data: dict) -> dict:
    """All scenarios share input data across candidates; timings remain estimates."""
    catalog = catalog_from_scenario(data)
    phases = ("start", "restore", "checkpoint", "publish", "release")
    timings = data["lifecycle_seconds"]
    costs = LifecycleCosts(
        **{phase: ConstantComponent(timings[phase]) for phase in phases},
        inference_gpus=data["inference_gpus"],
    )
    transitions = tuple(
        (catalog.labels[source], catalog.labels[target])
        for source, target in data.get("compatible_transitions", ())
    )
    policy = build_actor_policy(
        catalog, costs, allowed_transitions=transitions, **data.get("optimizer", {})
    )
    for workload in data.get("history", ()):
        policy.record_workload(workload_from_dict(workload))
    source = dict(data.get("observation", {}))
    current = source.pop("current_profile", None)
    checkpoint = source.pop("checkpoint_profile", None)
    active = source.pop("training_active", current is not None)
    ready = source.pop("batch_ready", "workload" in data)
    observation = ElasticObservation(
        state="ACTIVE" if active else "SUSPENDED",
        model_version=0,
        serving_version=0,
        training_active=active,
        batch_ready=ready,
        idle_duration=source.pop("idle_duration", 0),
        gate_available=source.pop("gate_available", True),
        pending_publication=source.pop("pending_publication", False),
        committed=source.pop("committed", True),
        observed_at=0,
        training_config=None if current is None else catalog.labels[current],
        next_batch=workload_from_dict(data["workload"]) if ready else None,
        max_train_gpus=data["max_train_gpus"],
        has_checkpoint=source.pop("has_checkpoint", checkpoint is not None or active),
        checkpoint_config=None if checkpoint is None else catalog.labels[checkpoint],
        episode_active=source.pop("episode_active", not ready),
        episode_elapsed=source.pop("episode_elapsed", 0),
    )
    if source:
        raise ValueError(f"Unknown observation keys: {sorted(source)}")
    decision = policy.decide(observation)
    search = policy.last_search
    forecast = forecast_window(
        observation,
        policy.history.snapshot(),
        policy.optimizer.horizon,
        preserve_workload=True,
    )
    actor_predictions = []
    stats = next(
        (
            item.workload.actor_stats
            for item in forecast
            if item.workload.actor_stats is not None
        ),
        None,
    )
    if stats is not None:
        for config in catalog.configurations:
            try:
                prediction = asdict(catalog.predict(config, stats))
            except ValueError as error:
                prediction = {"error": str(error)}
            actor_predictions.append(
                {
                    "config": asdict(config),
                    "aliases": catalog.aliases(config),
                    "actor": asdict(catalog.actor_config(config)),
                    **prediction,
                }
            )
    result = {
        "kind": "offline_model_prediction",
        "lifecycle_provenance": data["lifecycle_provenance"],
        "lifecycle_seconds": timings,
        "inference_gpus": costs.inference_gpus,
        "hardware": asdict(catalog.model.hardware),
        "decision": asdict(decision),
        "selected_aliases": ()
        if decision.config is None
        else catalog.aliases(decision.config),
        "evaluated": 0 if search is None else search.evaluated,
        "rejected": () if search is None else search.rejected,
        "plan": ()
        if search is None
        else [
            {**asdict(step), "aliases": catalog.aliases(step.config)}
            for step in search.plan
        ],
        "prediction": None
        if search is None or search.prediction is None
        else {
            **asdict(search.prediction),
            "total_gpu_seconds": search.prediction.total_gpu_seconds,
        },
        "actor_predictions": actor_predictions,
        "limitations": [
            "Actor model is sparsely calibrated; H200/Archon timings are not cluster measurements.",
            "Lifecycle seconds and transition compatibility are explicit scenario inputs.",
            "No reward or real elastic savings are validated by this simulation.",
            "Native execution uses a fixed layout; offline candidates do not imply supported GPU transitions.",
        ],
    }
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        result = run_scenario(json.loads(args.scenario.read_text()))
    except (ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    text = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text, end="")
    return 0
