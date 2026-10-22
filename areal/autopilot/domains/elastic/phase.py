# SPDX-License-Identifier: Apache-2.0

"""Fixed lifecycle accounting around revisable service-time components."""

from dataclasses import asdict, dataclass, replace

from areal.autopilot.domains.elastic.contracts import (
    history_from_dict,
    observation_from_dict,
)
from areal.autopilot.revision.contracts import digest
from areal.autopilot.revision.execution import load_source
from areal.elastic.policy.basic.profile import (
    fixed_layout_simulator,
    workflow_from_profile,
)
from areal.elastic.policy.basic.simulator import (
    ComponentContext,
    Estimate,
    WorkflowModel,
)

PHASES = ("start", "restore", "train", "checkpoint", "publish", "release")
SCOPE = "service-phases-v1"


@dataclass(frozen=True)
class ServiceTime:
    component: object

    def estimate(self, context):
        estimate = self.component.estimate(context)
        if (
            not isinstance(estimate, Estimate)
            or type(estimate.feasible) is not bool
            or type(estimate.unmeasured) is not bool
            or estimate.extra_gpu_seconds != 0
        ):
            raise ValueError(
                "Phase revisions may only predict service seconds/feasibility"
            )
        # Revalidate even objects constructed outside the normal dataclass path.
        return Estimate(**asdict(estimate))


def fixed_phase_simulator(candidate, workflow, profile=None):
    if not isinstance(getattr(candidate, "workflow", None), WorkflowModel):
        raise ValueError("Phase revisions must expose a WorkflowModel")
    # The evaluator owns ordering, Rollout waits and GPU residency. A candidate's
    # custom evaluate(), rollout model and inference_gpus are never executed.
    return fixed_layout_simulator(
        replace(
            workflow,
            **{
                phase: ServiceTime(getattr(candidate.workflow, phase))
                for phase in PHASES
            },
        ),
        profile or {},
    )


def build_phase_simulator(source, profile):
    workflow = workflow_from_profile(profile)
    return fixed_phase_simulator(
        load_source(source).build_simulator(workflow), workflow, profile
    )


def phase_predictions(simulator, cases):
    def predict():
        rows = []
        for case in cases:
            observation = observation_from_dict(case["observation"])
            source = observation.checkpoint_config or (
                observation.training_config if observation.has_checkpoint else None
            )
            estimate = getattr(simulator.workflow, case["phase"]).estimate(
                ComponentContext(
                    observation.training_config,
                    observation.next_batch,
                    history_from_dict(case["history"]),
                    source if case["phase"] == "restore" else None,
                )
            )
            rows.append({"prediction": asdict(estimate)})
        return rows

    rows = predict()
    if rows != predict():
        raise ValueError("Phase model is not repeatable on the same instance")
    return rows


def verify_phase_reference(simulator, reference):
    if (
        digest(phase_predictions(simulator, reference["cases"]))
        != reference["prediction_sha256"]
    ):
        raise ValueError("Phase model differs from its accepted worker predictions")
