# SPDX-License-Identifier: Apache-2.0

"""Measured workflow assembly and the native fixed-layout planning boundary."""

from ...contracts import SimulationResult, TrainingConfig
from .simulator import MeasuredComponent, Measurement, WorkflowModel, WorkflowSimulator


def workflow_from_profile(profile: dict) -> WorkflowModel:
    measurements = tuple(Measurement(**x) for x in profile["measurements"])
    return WorkflowModel(
        **{
            phase: MeasuredComponent(phase, measurements, scale_tokens=phase == "train")
            for phase in (
                "train",
                "start",
                "restore",
                "checkpoint",
                "publish",
                "release",
            )
        },
        inference_gpus=profile.get("inference_gpus", 1),
    )


class FixedLayoutSimulator(WorkflowSimulator):
    """Native acceptance covers one actual ready batch and one fixed layout."""

    def __init__(self, workflow, world_size):
        super().__init__(workflow)
        self.config = TrainingConfig(world_size)

    def evaluate(self, observation, forecast, plan, history, **kwargs):
        if (
            len(forecast) != 1
            or len(plan) != 1
            or any(step.config != self.config for step in plan)
        ):
            return SimulationResult(False, reason="Outside native fixed-layout scope")
        return super().evaluate(observation, forecast, plan, history, **kwargs)


def fixed_layout_simulator(workflow, profile):
    if profile.get("execution_scope") == "native-async-v1":
        return FixedLayoutSimulator(workflow, profile["native_actor_world_size"])
    return WorkflowSimulator(workflow)
