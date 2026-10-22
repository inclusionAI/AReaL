# SPDX-License-Identifier: Apache-2.0

"""Elastic revision factory contracts and public proposal context."""

from pathlib import Path

from areal.autopilot.revision.contracts import ProposalRequest, digest
from areal.elastic.policy.basic.serialization import (
    forecast_from_dict,
    history_from_dict,
    observation_from_dict,
    plan_from_dict,
)


def lifecycle_profile_digest(profile: dict) -> str:
    """Executable phase state; additional agent calibration data is not fitted state."""
    return digest(
        {
            k: profile.get(k)
            for k in (
                "compatibility",
                "inference_gpus",
                "measurements",
                "execution_scope",
                "native_execution_id",
                "native_actor_world_size",
            )
        }
    )


def deployment_context(profile: dict, settings: dict) -> dict:
    """Bind accepted code to the model, residency and optimizer contract."""
    return {
        "api_version": 1,
        "compatibility": profile.get("compatibility"),
        "inference_gpus": profile.get("inference_gpus", 1),
        "optimizer_settings": settings,
        "execution_scope": profile.get("execution_scope", "lifecycle"),
        "native_execution_id": profile.get("native_execution_id"),
        "native_actor_world_size": profile.get("native_actor_world_size"),
    }


BASELINE_SOURCE = {
    "serial_model": "from areal.elastic.policy.basic.serial_planning import serial_model_from_profile\ndef build_serial_model(profile):\n    return serial_model_from_profile(profile)\n",
    "joint_optimizer": "from areal.elastic.policy.basic.optimizer import JointOptimizer\ndef build_joint_optimizer(settings):\n    return JointOptimizer(**settings)\n",
    "simulator": "from areal.elastic.policy.basic.simulator import WorkflowSimulator\ndef build_simulator(workflow):\n    return WorkflowSimulator(workflow)\n",
    "optimizer": "from areal.elastic.policy.basic.optimizer import BasicOptimizer\ndef build_optimizer(settings):\n    return BasicOptimizer(**settings)\n",
}


def _request_contract(target: str) -> dict:
    return {
        "api": (
            "build_joint_optimizer(settings: dict) -> object implementing optimize(scenario: JointScenario | SerialJointWindow, simulator: JointModel) -> JointSearchResult; settings.candidates are decoded JointConfig objects"
            if target == "joint_optimizer"
            else "build_serial_model(profile: dict) -> object implementing predict_timing(SerialTimingInput) -> SerialTimingEstimate; static profile has no validation labels"
            if target == "serial_model"
            else "build_model(calibration_cases: tuple[HistoricalCase, ...], settings: dict) -> object implementing predict(HistoricalInput) -> HistoricalPrediction"
            if target == "model"
            else "build_simulator(workflow: WorkflowModel) -> object implementing evaluate(observation, forecast, plan, history)"
            if target == "simulator"
            else "build_optimizer(settings: dict) -> object implementing optimize(observation, simulator, history), returning SearchResult with a complete fixed-horizon plan"
        ),
        "response_schema": {
            "source": "Python module string",
            "rationale": "short explanation string",
        },
        "component_api": (
            "JointModel is an evaluate(scenario, config) proxy to one frozen model version. Preserve the entire scenario and full configuration identities. Query budget, job time bound, GPU limits and switch_margin are enforced by the evaluator. Only result.config is used; all claimed metrics are ignored. Settings are fixed across revisions."
            if target == "joint_optimizer"
            else "Predict conditional phase/whole-episode seconds from full joint config and logical workload. Only calibration updates are provided. Keep coverage, GPU accounting, transition allocations and runtime fixed. Missing timing is None. Runtime wraps the timing predictor; custom evaluate() is ignored."
            if target == "serial_model"
            else "HistoricalCase exposes calibration labels only. HistoricalInput contains the actual config/workload, execution identity and timing/input boundary. Return missing metrics as absent or None, never invented zeros. Preserve actual configurations, logical work and coverage. Final holdout is unavailable. Prefer composing BasicHistoryPredictor and existing cost models."
            if target == "model"
            else "WorkflowModel contains rollout/start/restore/train/checkpoint/publish/release components. Each estimate(ComponentContext(config, workload, history)) returns Estimate(seconds, extra_gpu_seconds, feasible). Prefer composition via dataclasses.replace."
        ),
        "scope": "Return one Python module as source plus rationale. Keep runtime, evaluator, references, resource limits and factory API fixed. Only calibration data and aggregate feedback are provided.",
    }


DOMAIN_INSTRUCTIONS = "For a simulator, improve prediction accuracy without inventing measurements or\nassuming linear speedup across GPU counts. Prefer composing existing components.\nFor a historical model, implement build_model(calibration_cases, settings) and\npredict(HistoricalInput). Reduce historical measurement error without dropping\ncoverage. Keep timing boundaries and conditional input modes explicit.\nIf calibration declares required_joint_export, preserve the same fitted\npredictor's .joint.evaluate(JointScenario, JointConfig) model export.\nFor a serial runtime timing model, implement build_serial_model(profile) and\npredict_timing(SerialTimingInput) -> SerialTimingEstimate. Only calibration labels\nare visible. Predict phase/whole-episode seconds, preserve unknown coverage and\nfull configuration identity; fixed code owns all GPU accounting and transitions.\nFor an optimizer, reduce reference GPU cost and reference query overhead at the\nsame fixed data horizon. Keep resource, transition, publication and search guards.\nFor a joint optimizer, implement build_joint_optimizer(settings) and return a\nJointSearchResult from optimize(scenario, simulator). Keep the frozen candidates,\njob-time bound, switch margin and query budget; optimize complete joint configurations.\nThe supplied simulator is an evaluate-only reference proxy; do not inspect its internals.\n"


def proposal_context(target: str) -> dict:
    root = Path(__file__).resolve().parents[3]
    names = ["elastic/contracts.py"]
    if target == "model":
        names += [
            "autopilot/domains/elastic/" + n + ".py"
            for n in ("model_evaluator", "model_adapters", "model_execution")
        ]
    elif target == "serial_model":
        names += [
            "elastic/policy/basic/serial_planning.py",
            "elastic/policy/basic/joint.py",
        ]
    elif target == "joint_optimizer":
        names += [
            "elastic/policy/basic/joint.py",
            "elastic/policy/basic/optimizer.py",
            "autopilot/domains/elastic/joint_evaluator.py",
            "elastic/policy/basic/serial_planning.py",
        ]
    elif target in {"simulator", "optimizer"}:
        names += [
            "elastic/policy/basic/simulator.py",
            "elastic/policy/basic/optimizer.py",
        ]
    else:
        raise ValueError("Unknown Elastic revision target")
    return {
        **_request_contract(target),
        "instructions": DOMAIN_INSTRUCTIONS,
        "reference_sources": {
            "areal/" + name: (root / name).read_text() for name in names
        },
    }


def elastic_proposal_request(*args, **kwargs) -> ProposalRequest:
    target = args[0] if args else kwargs["target"]
    return ProposalRequest(*args, **{**kwargs, "context": proposal_context(target)})


__all__ = [
    "forecast_from_dict",
    "history_from_dict",
    "observation_from_dict",
    "plan_from_dict",
    "BASELINE_SOURCE",
    "deployment_context",
    "lifecycle_profile_digest",
    "proposal_context",
    "elastic_proposal_request",
]
