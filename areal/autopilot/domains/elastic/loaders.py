# SPDX-License-Identifier: Apache-2.0

"""Load accepted Elastic components under their frozen evaluation context."""

from __future__ import annotations

import json
from pathlib import Path

from areal.autopilot.revision.contracts import digest
from areal.autopilot.revision.execution import load_module
from areal.autopilot.revision.loop import RevisionStore

from .model_execution import HistoricalSourceEvaluator


def apply_accepted_revision(root: Path, profile: dict, settings: dict):
    """Load one evaluated component at construction time, leaving the other fixed."""
    from areal.autopilot.domains.elastic.contracts import (
        deployment_context,
        lifecycle_profile_digest,
        workflow_from_profile,
    )
    from areal.autopilot.revision.execution import load_source
    from areal.elastic.policy.basic.optimizer import BasicOptimizer
    from areal.elastic.policy.basic.simulator import WorkflowSimulator

    store = RevisionStore(root)
    context = json.loads((store.root / "context.json").read_text())
    if context["target"] == "model":
        raise ValueError(
            "Historical predictors use load_accepted_model, not lifecycle component loading"
        )
    if context["target"] == "joint_optimizer":
        raise ValueError(
            "Joint optimizer revisions require their fixed offline evaluator"
        )
    if context["target"] == "serial_model":
        raise ValueError(
            "Serial revisions require their matching executable joint bundle"
        )
    expected = deployment_context(profile, settings)
    if "profile_sha256" in context["deployment_context"]:
        expected["profile_sha256"] = lifecycle_profile_digest(profile)
    reference = context["deployment_context"].get("simulator_source")
    if reference is not None:
        expected["simulator_source"] = reference
    if digest(context["deployment_context"]) != digest(expected):
        raise ValueError(
            "Accepted revision does not match the model/residency/search contract"
        )
    active = store.active()
    if active is None:
        raise ValueError("No accepted component revision is available")
    module = load_module(active[0])
    workflow = workflow_from_profile(profile)
    simulator = (
        WorkflowSimulator(workflow)
        if reference is None
        else load_source(reference).build_simulator(workflow)
    )
    optimizer = BasicOptimizer(**settings)
    if context["target"] == "simulator":
        simulator = module.build_simulator(workflow)
    else:
        optimizer = module.build_optimizer(dict(settings))
    if not callable(getattr(simulator, "evaluate", None)) or not callable(
        getattr(optimizer, "optimize", None)
    ):
        raise ValueError("Accepted revision does not implement its factory contract")
    return simulator, optimizer


def load_accepted_model(root: Path, evaluator: HistoricalSourceEvaluator):
    """Load one accepted offline predictor in its exact historical factory context."""
    evaluator.assert_unchanged()
    store = RevisionStore(root)
    context = json.loads((store.root / "context.json").read_text())
    if (
        context["target"] != "model"
        or context["benchmark_sha256"] != evaluator.benchmark_sha256
    ):
        raise ValueError(
            "Accepted model belongs to a different historical evaluation context"
        )
    active = store.active()
    if active is None:
        raise ValueError("No accepted historical model is available")
    predictor = load_module(active[0]).build_model(
        evaluator.history.calibration, evaluator.settings
    )
    evaluator.assert_unchanged()
    if not callable(getattr(predictor, "predict", None)):
        raise ValueError("Accepted model does not implement predict")
    accepted_report = json.loads((active[0].parent / "evaluation.json").read_text())
    evaluator.verify_loaded(
        predictor, accepted_report["metrics"].get("prediction_sha256")
    )
    return predictor
