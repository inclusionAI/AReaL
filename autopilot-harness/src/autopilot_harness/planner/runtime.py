"""Placeholder planning call; no initialization, prediction, or search."""

from copy import deepcopy

from autopilot_harness.contracts.planner import PlannerInput, PlannerResult


def plan(inputs: PlannerInput) -> PlannerResult:
    """Return the current configuration without altering the supplied context.

    The result makes no claim about configuration validity or application state;
    validation remains the caller's responsibility.
    """
    return PlannerResult(
        config=deepcopy(inputs.current_configuration.values),
        changed=False,
        search_status="not_run",
    )
