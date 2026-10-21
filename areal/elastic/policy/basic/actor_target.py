# SPDX-License-Identifier: Apache-2.0

"""Explicit counterfactual recipes for the independent Expert Actor replay.

The caller supplies the frozen Expert objects. This adapter never imports Basic
or training engines and never reuses source-profile calibration for a target.
"""

import math
from dataclasses import asdict, fields, replace
from typing import Any

STAGES = ("recompute_logp", "compute_advantage", "train_step")


def actor_target(
    document: dict[str, Any], source: Any, raw_model: Any
) -> tuple[Any, Any]:
    if document.get("schema_version") != 1:
        raise ValueError("Actor target requires schema_version=1")
    allowed = {
        "schema_version",
        "actor",
        "runtime_revision",
        "efficiency",
        "auxiliary",
        "advantage_broadcast",
        "hardware",
    }
    if document.keys() - allowed:
        raise ValueError("Unknown Actor target fields")
    config = document["actor"]
    required = {
        "backend",
        "parallel",
        "optimizer_type",
        "optimizer_dtype",
        "offload_params",
        "grad_bytes",
    }
    if required - config.keys():
        raise ValueError(
            "Target must explicitly specify backend, mesh, optimizer, offload and grad_bytes"
        )
    if config.keys() - {f.name for f in fields(source)}:
        raise ValueError("Unknown Actor recipe fields")
    runtime = document.get("runtime_revision")
    if not isinstance(runtime, str) or not runtime:
        raise ValueError("Target runtime_revision is required")
    # Training workload/algorithm flags stay fixed. Precision/storage differences
    # in a backend's optimizer are explicit and still need quality validation.
    for name in (
        "dtype",
        "ppo_n_minibatches",
        "recompute_logprob",
        "use_decoupled_loss",
        "prox_logp_method",
    ):
        if name in config and config[name] != getattr(source, name):
            raise ValueError(f"Counterfactual must preserve training semantic {name}")
    values = asdict(source)
    values.update(config)
    values["parallel"] = type(source.parallel)(**config["parallel"])
    if values["backend"] != "fsdp":
        values["fsdp_parameter_tensors"] = None
    target = type(source)(**values)
    p = target.parallel
    if target.backend == "archon" and p.tp > 1 and p.ep == 1:
        raise ValueError(
            "Archon expert TP without EP requires a separate compute/communication component"
        )
    if target.backend == "archon" and p.ep > 1:
        # The frozen ActorConfig already restricts ETP to 1. Apply runtime's
        # remaining mesh rules instead of accepting every arithmetically valid EP.
        if p.ep % (p.cp * p.tp) or (p.dp * p.cp * p.tp) % p.ep:
            raise ValueError("Archon EP must divide its runtime DP/CP/TP mesh")
    model = replace(
        raw_model,
        hardware=type(raw_model.hardware)(**document["hardware"]["values"])
        if "hardware" in document
        else raw_model.hardware,
        runtime_revision=runtime,
        efficiency=type(raw_model.efficiency)(**document.get("efficiency", {})),
        calibration=(),
        advantage_model=None,
        fsdp_host_rates=raw_model.fsdp_host_rates if target.backend == "fsdp" else None,
    )
    if target.parallel.world_size > model.hardware.gpus_per_node:
        raise ValueError("This replay target currently covers a single Actor node")
    return target, model


def target_phases(prediction: Any) -> tuple[list[dict], dict]:
    """Missing phases or proven infeasibility cannot produce a numeric score."""
    stages = dict(prediction.stages)
    infeasible = any(p.feasibility == "infeasible" for p in stages.values())
    result = []
    for name in STAGES:
        value = stages.get(name)
        seconds = value.seconds if value is not None and not infeasible else None
        result.append(
            {
                "name": name,
                "seconds": seconds,
                "evidence": "assumed" if seconds is not None else "unsupported",
            }
        )
    plan = prediction.packing
    details = {
        "profile": prediction.profile,
        "feasibility": "infeasible" if infeasible else "unknown",
        "static_memory_gb": list(prediction.static_memory_gb),
        "features": dict(prediction.features),
        "notes": {name: list(value.notes) for name, value in prediction.stages},
        "packing": None
        if plan is None
        else {
            "exact_dispatch": plan.exact_dispatch,
            "exact_lengths": plan.exact_lengths,
            "train_compute_tokens": plan.train_compute_tokens,
            "train_microbatches_per_rank": [
                [len(rank) for rank in update] for update in plan.train
            ],
        },
    }
    return result, details


def target_auxiliary(document: dict[str, Any], *, final: bool) -> list[dict]:
    """Explicit conditional publication/save profiles; missing never means zero."""
    profiles = document.get("auxiliary", {})
    if profiles.keys() - {"update_weights", "save", "final_save"}:
        raise ValueError("Unknown auxiliary stage")
    result = []
    for name, key in (
        ("update_weights", "update_weights"),
        ("save", "final_save" if final else "save"),
    ):
        value = profiles.get(key)
        if value is None:
            result.append({"name": name, "seconds": None, "evidence": "unsupported"})
            continue
        seconds = value.get("seconds")
        if (
            type(seconds) not in (int, float)
            or not math.isfinite(seconds)
            or seconds < 0
            or not value.get("source")
        ):
            raise ValueError(
                "Auxiliary times require finite nonnegative seconds and source"
            )
        result.append({"name": name, "seconds": seconds, "evidence": "assumed"})
    return result
