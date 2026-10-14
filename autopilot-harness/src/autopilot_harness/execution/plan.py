"""Bind a completed validation to one live runtime and one configuration pair."""

import json

from autopilot_harness.contracts.execution import ApplyPlan
from autopilot_harness.contracts.validator import ValidationResult
from autopilot_harness.validator.checker import configuration_id


def build_apply_plan(
    baseline: dict, candidate: dict, validation: ValidationResult, *, execution: dict
) -> ApplyPlan:
    """No training side effects. Reject unbound/offline validation results.

    execution comes from RestartCoordinator.execution_context(), never Planner.
    The Validator must have received that exact context before this call.
    """
    if not validation.passed or validation.runtime_applicability not in {
        "supported",
        "not_required",
    }:
        raise ValueError("Apply requires a passed configuration/transition validation")
    if (
        validation.baseline_id != configuration_id(baseline)
        or validation.candidate_id != configuration_id(candidate)
        or validation.execution_id != configuration_id(execution)
        or execution.get("configuration_id") != validation.baseline_id
        or execution.get("runtime_revision") != validation.runtime_revision
        or validation.transition_json is None
    ):
        raise ValueError("Validation is not bound to this configuration pair/runtime")
    document = dict(
        schema_version=1,
        execution=execution,
        baseline=baseline,
        candidate=candidate,
        validation=validation.to_dict(),
        transition=json.loads(validation.transition_json),
    )
    return ApplyPlan(
        json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False)
    )


def check_apply_plan(plan: ApplyPlan, execution: dict) -> dict:
    """Check provenance/freshness only; do not rerun validity after pausing."""
    if not isinstance(plan, ApplyPlan):
        raise TypeError("apply requires an ApplyPlan")
    value = plan.to_dict()
    validation = value["validation"]
    if (
        value.get("schema_version") != 1
        or value["execution"] != execution
        or validation["status"] != "passed"
        or validation["execution_id"] != configuration_id(execution)
        or validation["baseline_id"] != configuration_id(value["baseline"])
        or validation["candidate_id"] != configuration_id(value["candidate"])
        or validation["runtime_revision"] != execution["runtime_revision"]
        or validation["baseline_id"] != execution["configuration_id"]
        or value["transition"] != json.loads(validation["transition_json"])
    ):
        raise ValueError("Apply plan is stale or its validated inputs were changed")
    return value
