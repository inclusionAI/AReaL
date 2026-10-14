"""Organize validation: read inputs, compute diffs, bind evidence and collect results.

Concrete admissibility rules live in constraints.py.

No AReaL imports, GPU access, configuration mutation, or runtime application.
The caller owns the baseline, policy, and runtime revision; the Planner supplies
only the candidate configuration.
"""

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from autopilot_harness.contracts.validator import (
    ConfigurationChange,
    ParameterRule,
    ValidationIssue,
    ValidationPolicy,
    ValidationResult,
)

from . import constraints


def _unique_object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def read_document(path: Path) -> dict:
    """Read JSON without silently accepting duplicate fields."""
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream, object_pairs_hook=_unique_object)
    if not isinstance(value, dict):
        raise ValueError("validation inputs must be JSON objects")
    return value


def configuration_id(value: Any) -> str:
    """Content identity used to bind harness evidence to exact configurations."""
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return "sha256:" + hashlib.sha256(data.encode()).hexdigest()


def _policy_path(value: str) -> tuple[str, ...]:
    if not isinstance(value, str) or not re.fullmatch(
        r"[A-Za-z_][A-Za-z_0-9]*(\.[A-Za-z_][A-Za-z_0-9]*)*", value
    ):
        raise ValueError(
            "policy paths must name exact fields, without wildcards or array indexes"
        )
    return tuple(value.split("."))


def parse_policy(value: dict) -> ValidationPolicy:
    allowed = {
        "schema_version",
        "name",
        "runtime_revision",
        "parameters",
        "checks",
        "deferred_paths",
        "notes",
    }
    if not isinstance(value, dict) or set(value) - allowed:
        raise ValueError("unknown validation policy fields")
    if type(value.get("schema_version")) is not int or value["schema_version"] != 1:
        raise ValueError("validation policy schema_version must be 1")
    if not isinstance(value.get("name"), str) or not value["name"]:
        raise ValueError("validation policy requires a name")
    revision = value.get("runtime_revision")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("validation policy requires a full AReaL Git revision")
    raw_parameters = value.get("parameters")
    if not isinstance(raw_parameters, dict):
        raise ValueError("policy.parameters must be an object")
    parameters = []
    rule_fields = {
        "type",
        "nullable",
        "minimum",
        "maximum",
        "exclusive_minimum",
        "exclusive_maximum",
        "choices",
        "active_backend",
        "source_ref",
    }
    for name, raw in raw_parameters.items():
        path = _policy_path(name)
        if not isinstance(raw, dict) or set(raw) - rule_fields:
            raise ValueError("unknown parameter rule fields")
        kind = raw.get("type")
        if not isinstance(kind, str) or kind not in constraints.PARAMETER_KINDS:
            raise ValueError(
                "parameter type must be integer, number, boolean, or string"
            )
        for flag in ("nullable", "exclusive_minimum", "exclusive_maximum"):
            if flag in raw and type(raw[flag]) is not bool:
                raise ValueError("parameter rule flags must be booleans")
        for bound in ("minimum", "maximum"):
            if bound in raw and (
                kind not in {"integer", "number"}
                or not constraints.has_parameter_type(raw[bound], kind)
            ):
                raise ValueError(
                    "parameter bounds must match the numeric type and be finite"
                )
            if raw.get("exclusive_" + bound) and bound not in raw:
                raise ValueError("exclusive bound requires its numeric bound")
        if "minimum" in raw and "maximum" in raw:
            if raw["minimum"] > raw["maximum"] or (
                raw["minimum"] == raw["maximum"]
                and (raw.get("exclusive_minimum") or raw.get("exclusive_maximum"))
            ):
                raise ValueError("parameter bounds define an empty range")
        choices = raw.get("choices")
        if "choices" in raw and (
            not isinstance(choices, list)
            or not choices
            or any(not constraints.has_parameter_type(item, kind) for item in choices)
        ):
            raise ValueError(
                "choices must be a non-empty list matching the parameter type"
            )
        backend = raw.get("active_backend")
        if backend is not None and backend not in constraints.INFERENCE_BACKENDS:
            raise ValueError("active_backend must be sglang or vllm")
        if not isinstance(raw.get("source_ref", ""), str):
            raise ValueError("rule source_ref must be a string")
        parameters.append(
            ParameterRule(
                path,
                kind,
                raw.get("nullable", False),
                raw.get("minimum"),
                raw.get("maximum"),
                raw.get("exclusive_minimum", False),
                raw.get("exclusive_maximum", False),
                tuple(choices) if choices is not None else None,
                backend,
                raw.get("source_ref", ""),
            )
        )
    paths = {rule.path for rule in parameters}
    if any(path[:index] in paths for path in paths for index in range(1, len(path))):
        raise ValueError("policy cannot allow both a container and its nested fields")
    checks = value.get("checks", [])
    if not isinstance(checks, list) or any(
        not isinstance(item, str) or item not in constraints.COMBINATION_CHECKS
        for item in checks
    ):
        raise ValueError("unknown combination check")
    if len(set(checks)) != len(checks):
        raise ValueError("duplicate combination check")
    deferred = value.get("deferred_paths", [])
    if not isinstance(deferred, list):
        raise ValueError("deferred_paths must be a list")
    deferred = tuple(_policy_path(item) for item in deferred)
    if any(path not in paths for path in deferred):
        raise ValueError("deferred paths must refer to declared parameters")
    return ValidationPolicy(
        value["name"],
        configuration_id(value),
        revision,
        tuple(parameters),
        tuple(checks),
        deferred,
    )


def load_policy(path: Path) -> ValidationPolicy:
    return parse_policy(read_document(path))


def _diff(
    baseline: Any, candidate: Any, path: tuple[str, ...] = ()
) -> list[ConfigurationChange]:
    if isinstance(baseline, dict) and isinstance(candidate, dict):
        changes = []
        for key in sorted(baseline.keys() | candidate.keys()):
            child = path + (key,)
            if key not in baseline:
                changes.append(ConfigurationChange(child, "add"))
            elif key not in candidate:
                changes.append(ConfigurationChange(child, "remove"))
            else:
                changes.extend(_diff(baseline[key], candidate[key], child))
        return changes
    return (
        []
        if constraints.values_equal(baseline, candidate)
        else [ConfigurationChange(path, "replace")]
    )


def _input_issues(
    value: Any, label: str, path: tuple[str, ...] = ()
) -> list[ValidationIssue]:
    issues = []
    if isinstance(value, dict):
        for key, item in value.items():
            if type(key) is not str:
                issues.append(
                    ValidationIssue(
                        "invalid_input", path, f"{label} object keys must be strings"
                    )
                )
            else:
                issues.extend(_input_issues(item, label, path + (key,)))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            issues.extend(_input_issues(item, label, path + (str(index),)))
    elif type(value) is float and not math.isfinite(value):
        issues.append(
            ValidationIssue(
                "invalid_input", path, f"{label} contains a non-finite number"
            )
        )
    elif isinstance(value, str) and (
        value == "???" or re.search(r"\$\{[^}]+\}", value)
    ):
        issues.append(
            ValidationIssue(
                "unresolved_input",
                path,
                f"{label} must have defaults and interpolations resolved",
            )
        )
    elif value is not None and type(value) not in (str, bool, int, float):
        issues.append(
            ValidationIssue("invalid_input", path, f"{label} contains a non-JSON value")
        )
    return issues


def _bind_context(context, *, runtime_revision, baseline_id, candidate_id):
    """Check the context envelope and bind it to this validation request."""
    path = ("context",)
    if context is None:
        return [
            ValidationIssue(
                "validation_context_missing",
                (),
                "Harness must provide model, resource, memory and transition evidence",
                "unknown",
            )
        ], None
    fields = {
        "schema_version",
        "runtime_revision",
        "baseline_id",
        "candidate_id",
        "state_id",
        "model",
        "resources",
        "backends",
        "dispatch",
        "memory",
        "transition",
        "auxiliary",
        "training_checkpoint",
    }
    try:
        constraints._record(context, path, fields)
        if (
            type(context.get("schema_version")) is not int
            or context["schema_version"] != 1
        ):
            constraints._fail(
                "invalid_context", path, "Validation context schema_version must be 1"
            )
        for key, expected in (
            ("runtime_revision", runtime_revision),
            ("baseline_id", baseline_id),
            ("candidate_id", candidate_id),
            ("state_id", None),
        ):
            value = constraints._need(context, key, path)
            if key != "state_id" and value != expected:
                constraints._fail(
                    "evidence_scope_mismatch",
                    (*path, key),
                    "Evidence belongs to another revision or configuration pair",
                    "unknown",
                )
        state_id = constraints._string(context["state_id"], path)
    except constraints.ConstraintViolation as exc:
        return [exc.issue], None
    return [], state_id


def _check_runtime_constraints(
    baseline,
    candidate,
    changes,
    context,
    *,
    runtime_revision,
    baseline_id,
    candidate_id,
    execution=None,
):
    """Order dependent checks and collect their diagnostics and provenance."""
    layouts, issues = constraints.check_layouts(baseline, candidate)
    checked, refs = ["native_layout"], set()
    applicability = "undetermined"
    binding_issues, state_id = _bind_context(
        context,
        runtime_revision=runtime_revision,
        baseline_id=baseline_id,
        candidate_id=candidate_id,
    )
    issues.extend(binding_issues)
    if binding_issues or layouts is None:
        return issues, checked, applicability, state_id, ()

    def run(name, check):
        checked.append(name)
        try:
            return check()
        except constraints.ConstraintViolation as exc:
            issues.append(exc.issue)
            return None

    resources = run(
        "resource_reservation",
        lambda: constraints.check_resources(context, layouts, refs),
    )
    run(
        "model_backend_layout",
        lambda: constraints.check_model_layouts(
            context, candidate, layouts, resources, refs
        ),
    )
    run(
        "shared_inference_configuration",
        lambda: constraints.check_shared_inference(
            context, candidate, changes, resources, refs
        ),
    )
    run(
        "batch_dispatch",
        lambda: constraints.check_batch_dispatch(context, layouts, refs),
    )
    if resources is not None:
        run(
            "memory_capacity",
            lambda: constraints.check_memory(
                context, candidate, resources, changes, refs
            ),
        )
    if execution is None:
        applicability = (
            run(
                "state_transition",
                lambda: constraints.check_transition(
                    context, baseline, candidate, changes, refs
                ),
            )
            or "undetermined"
        )
    return issues, checked, applicability, state_id, tuple(sorted(refs))


def validate_configuration(
    baseline: dict,
    candidate: dict,
    policy: ValidationPolicy,
    *,
    runtime_revision: str | None = None,
    context: dict | None = None,
    execution: dict | None = None,
) -> ValidationResult:
    """Check a full candidate against a trusted, already-resolved baseline.

    Unlisted fields and all structural additions/removals are denied. The result
    intentionally contains field paths, not potentially sensitive field values.
    Context is harness-owned evidence for this exact pair and runtime state.
    Missing evidence yields undetermined; passed covers all required checks.
    """
    issues = []
    changes = []
    baseline_id = candidate_id = None
    context_id = state_id = None
    evidence_refs = ()
    applicability = "undetermined"
    execution_id = transition_json = None
    checked = ["input_shape", "runtime_revision"]
    if not isinstance(baseline, dict) or not isinstance(candidate, dict):
        issues.append(
            ValidationIssue(
                "invalid_input",
                (),
                "Baseline and candidate must be full configuration objects",
            )
        )
    else:
        issues.extend(_input_issues(baseline, "baseline"))
        issues.extend(_input_issues(candidate, "candidate"))
    if not issues:
        checked.extend(("field_permissions", "parameter_values"))
        baseline_id, candidate_id = (
            configuration_id(baseline),
            configuration_id(candidate),
        )
        changes = _diff(baseline, candidate)
        issues.extend(constraints.check_permissions(changes, policy))
        issues.extend(constraints.check_parameter_values(candidate, changes, policy))
        checked.extend(policy.checks)
        issues.extend(
            constraints.check_combinations(baseline, candidate, policy.checks)
        )
    if not runtime_revision:
        issues.append(
            ValidationIssue(
                "runtime_revision_missing",
                (),
                "Caller must supply the trusted runtime revision",
                "unknown",
            )
        )
    elif runtime_revision != policy.runtime_revision:
        issues.append(
            ValidationIssue(
                "runtime_revision_mismatch",
                (),
                "Policy was checked against a different AReaL revision",
                "unknown",
            )
        )
    if not issues:
        if context is not None:
            issues.extend(_input_issues(context, "context", ("context",)))
            if not issues:
                context_id = configuration_id(context)
        if not issues:
            extra, rules, applicability, state_id, evidence_refs = (
                _check_runtime_constraints(
                    baseline,
                    candidate,
                    changes,
                    context,
                    runtime_revision=runtime_revision,
                    baseline_id=baseline_id,
                    candidate_id=candidate_id,
                    execution=execution,
                )
            )
            issues.extend(extra)
            checked.extend(rules)
    if not issues and execution is not None:
        checked.append("implemented_apply_adapter")
        issues.extend(_input_issues(execution, "execution", ("execution",)))
        if not issues:
            execution_refs = set(evidence_refs)
            try:
                transition = constraints.check_execution(
                    execution,
                    baseline,
                    candidate,
                    changes,
                    context,
                    runtime_revision=runtime_revision,
                    baseline_id=baseline_id,
                    refs=execution_refs,
                )
                execution_id = configuration_id(execution)
                transition_json = json.dumps(
                    transition, sort_keys=True, separators=(",", ":")
                )
                applicability = "supported" if changes else "not_required"
            except constraints.ConstraintViolation as exc:
                issues.append(exc.issue)
            except (KeyError, TypeError, ValueError, AttributeError) as exc:
                issues.append(
                    ValidationIssue(
                        "invalid_execution_evidence",
                        ("execution",),
                        f"Resolved configuration/evidence cannot be interpreted ({type(exc).__name__})",
                    )
                )
            evidence_refs = tuple(sorted(execution_refs))
    status = (
        "rejected"
        if any(issue.severity == "error" for issue in issues)
        else "undetermined"
        if issues
        else "passed"
    )
    if status != "passed":
        applicability = "unsupported" if status == "rejected" else "undetermined"
    return ValidationResult(
        status,
        policy.policy_id,
        runtime_revision,
        baseline_id,
        candidate_id,
        tuple(changes),
        tuple(issues),
        tuple(checked),
        runtime_applicability=applicability,
        context_id=context_id,
        state_id=state_id,
        evidence_refs=evidence_refs,
        execution_id=execution_id,
        transition_json=transition_json,
    )
