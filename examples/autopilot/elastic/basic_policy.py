# SPDX-License-Identifier: Apache-2.0

"""Build the example workflow from measured per-configuration phase profiles."""

import json
from dataclasses import asdict
from pathlib import Path

from areal.elastic.policy.basic import BasicPolicy, IdleTimeoutPolicy
from areal.elastic.policy.basic.optimizer import BasicOptimizer
from areal.elastic.policy.basic.simulator import (
    MeasuredComponent,
    Measurement,
    WorkflowModel,
    WorkflowSimulator,
)


def measured_policy(
    profiles: Path,
    *,
    horizon=2,
    cost_scope="total",
    idle_seconds=30,
    inference_gpus=None,
    compatibility=None,
    revision_root: Path | None = None,
):
    data = json.loads(profiles.read_text())
    if data["format_version"] != 1:
        raise ValueError("Unsupported phase profile format")
    if compatibility is not None and data.get("compatibility") != compatibility:
        raise ValueError("Profiles do not match this model and training contract")
    measurements = tuple(Measurement(**record) for record in data["measurements"])
    components = {
        phase: MeasuredComponent(phase, measurements, scale_tokens=phase == "train")
        for phase in ("start", "restore", "train", "checkpoint", "publish", "release")
    }
    optimizer = BasicOptimizer(horizon=horizon, cost_scope=cost_scope)
    simulator = WorkflowSimulator(
        WorkflowModel(
            **components,
            inference_gpus=data.get("inference_gpus", 1)
            if inference_gpus is None
            else inference_gpus,
        )
    )
    if revision_root is not None:
        from areal.autopilot.domains.elastic import apply_accepted_revision

        simulator, optimizer = apply_accepted_revision(
            revision_root,
            {**data, "inference_gpus": simulator.workflow.inference_gpus},
            asdict(optimizer),
        )
    return BasicPolicy(
        simulator, optimizer=optimizer, fallback=IdleTimeoutPolicy(idle_seconds)
    )


def measured_joint_policy(profiles: Path, *, compatibility, horizon=2, idle_seconds=30):
    """Explicit profiles for the local serial loop, independent of Expert data."""
    from areal.elastic.policy.basic.online import (
        OnlineJointPolicy,
        serial_config,
        serial_model_from_profile,
    )
    from areal.elastic.policy.basic.optimizer import JointOptimizer

    data = json.loads(profiles.read_text())
    if (
        data.get("format_version") != "serial-joint-v1"
        or data.get("compatibility") != compatibility
    ):
        raise ValueError("Joint profiles do not match this runtime/training contract")

    initial = serial_config(data["initial"])
    candidates = tuple(serial_config(row) for row in data["candidates"])
    if initial not in candidates or any(row is None for row in candidates):
        raise ValueError(
            "Joint allowlist must include the complete initial configuration"
        )
    settings = data.get("search", {})
    policy = OnlineJointPolicy(
        serial_model_from_profile(data, idle_seconds=idle_seconds),
        JointOptimizer(candidates, **settings),
        horizon=horizon,
    )
    return policy, initial
