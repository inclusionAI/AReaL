"""Concrete configuration rules: permissions, values, layouts and runtime constraints.

checker.py owns input/context binding, check ordering and result aggregation.

Context is supplied by the harness, never by the Planner. Source references are
provenance, not executable code or certificates we fetch/verify. Missing bounds
or backend evidence are unknown, not implicit support. Memory bounds must cover
the whole fixed workload and all resident roles, not just a measured step peak.
Native layout semantics follow areal/api/alloc_mode.py at the policy revision.
"""

import math
import re
from dataclasses import dataclass
from itertools import combinations
from typing import Any

from autopilot_harness.contracts.validator import (
    ConfigurationChange,
    ParameterRule,
    ValidationIssue,
    ValidationPolicy,
)

_MISSING = object()
PARAMETER_KINDS = frozenset({"integer", "number", "boolean", "string"})
COMBINATION_CHECKS = frozenset(
    {"preserve_backend", "explicit_padding_limit", "megatron_n_mbs"}
)
INFERENCE_BACKENDS = frozenset({"sglang", "vllm"})


def has_parameter_type(value: Any, kind: str) -> bool:
    if kind == "integer":
        return type(value) is int
    if kind == "number":
        return type(value) in (int, float) and (
            type(value) is int or math.isfinite(value)
        )
    if kind == "boolean":
        return type(value) is bool
    return type(value) is str


def values_equal(left: Any, right: Any) -> bool:
    # Python would otherwise consider True == 1, also inside containers.
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            values_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            values_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def _get(config: Any, path: tuple[str, ...]) -> Any:
    for part in path:
        if not isinstance(config, dict) or part not in config:
            return _MISSING
        config = config[part]
    return config


def _value_issues(value: Any, rule: ParameterRule) -> list[ValidationIssue]:
    def issue(code: str, message: str) -> list[ValidationIssue]:
        return [ValidationIssue(code, rule.path, message, source_ref=rule.source_ref)]

    if value is None:
        return (
            []
            if rule.nullable
            else issue("null_not_allowed", "This parameter does not allow null")
        )
    if not has_parameter_type(value, rule.kind):
        return issue("invalid_type", f"Expected {rule.kind}; values are not coerced")
    if rule.choices is not None and not any(
        values_equal(value, choice) for choice in rule.choices
    ):
        return issue("invalid_choice", "Value is outside the configured choices")
    if rule.minimum is not None and (
        value < rule.minimum or (rule.exclusive_minimum and value == rule.minimum)
    ):
        return issue("below_minimum", "Value is below the permitted lower bound")
    if rule.maximum is not None and (
        value > rule.maximum or (rule.exclusive_maximum and value == rule.maximum)
    ):
        return issue("above_maximum", "Value is above the permitted upper bound")
    return []


def _backend(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    match = re.fullmatch(r"([a-z][a-z0-9_]*)(?:\[[^\]]+\])?:\s*(\S.*)", value)
    return match.group(1) if match else None


def check_combinations(
    baseline: dict, candidate: dict, checks: tuple[str, ...]
) -> list[ValidationIssue]:
    issues = []
    if "preserve_backend" in checks:
        for role in ("actor", "rollout"):
            path = (role, "backend")
            old, new = _get(baseline, path), _get(candidate, path)
            if old is _MISSING and new is _MISSING:
                continue
            if _backend(old) is None or _backend(new) is None:
                issues.append(
                    ValidationIssue(
                        "backend_format",
                        path,
                        "Backend requires an implementation prefix and layout",
                    )
                )
            elif _backend(old) != _backend(new):
                issues.append(
                    ValidationIssue(
                        "backend_substitution",
                        path,
                        "Backend implementation must remain fixed",
                    )
                )
    if (
        "explicit_padding_limit" in checks
        and _get(candidate, ("actor", "pad_to_maximum")) is True
    ):
        path = ("actor", "mb_spec", "max_tokens_per_mb")
        limit = _get(candidate, path)
        if type(limit) is not int or limit <= 0 or limit == 10**12:
            issues.append(
                ValidationIssue(
                    "padding_requires_limit",
                    path,
                    "Padding requires an explicit positive token limit; AReaL otherwise disables this setting",
                    source_ref="areal/utils/data.py:pad_mb_list",
                )
            )
    if (
        "megatron_n_mbs" in checks
        and _backend(_get(candidate, ("actor", "backend"))) == "megatron"
    ):
        path = ("actor", "mb_spec", "n_mbs")
        count = _get(candidate, path)
        if type(count) is not int or count <= 0:
            issues.append(
                ValidationIssue(
                    "megatron_requires_n_mbs",
                    path,
                    "Megatron's microbatch preparation requires a positive integer n_mbs",
                    source_ref="areal/engine/megatron_engine.py:2715",
                )
            )
    return issues


def check_permissions(
    changes: list[ConfigurationChange], policy: ValidationPolicy
) -> list[ValidationIssue]:
    """Only explicitly allowed leaf replacements may change the baseline."""
    issues = []
    rules = {rule.path: rule for rule in policy.parameters}
    for change in changes:
        if change.path not in rules:
            issues.append(
                ValidationIssue(
                    "unauthorized_change",
                    change.path,
                    "Field is not in the adjustable parameter allowlist",
                )
            )
        elif change.operation != "replace":
            issues.append(
                ValidationIssue(
                    "field_presence",
                    change.path,
                    "Full configurations must preserve field presence; use explicit null where permitted",
                )
            )
    return issues


def check_parameter_values(
    candidate: dict, changes: list[ConfigurationChange], policy: ValidationPolicy
) -> list[ValidationIssue]:
    """Check active-backend fields, required presence, types and allowed values."""
    issues = []
    changed = {change.path for change in changes}
    active_backend = _backend(_get(candidate, ("rollout", "backend")))
    for rule in policy.parameters:
        value = _get(candidate, rule.path)
        if rule.active_backend and rule.active_backend != active_backend:
            if rule.path in changed:
                issues.append(
                    ValidationIssue(
                        "inactive_backend",
                        rule.path,
                        "Parameter belongs to an inactive or unknown inference backend",
                        source_ref=rule.source_ref,
                    )
                )
            continue
        if value is _MISSING:
            issues.append(
                ValidationIssue(
                    "missing_parameter",
                    rule.path,
                    "Active parameter is missing; provide a resolved full configuration",
                    "unknown",
                    rule.source_ref,
                )
            )
            continue
        issues.extend(_value_issues(value, rule))
    return issues


class ConstraintViolation(Exception):
    def __init__(self, code, path, message, severity="error"):
        self.issue = ValidationIssue(code, tuple(path), message, severity)


def _fail(code, path, message, severity="error"):
    raise ConstraintViolation(code, path, message, severity)


def _record(value, path, fields):
    if not isinstance(value, dict) or set(value) - set(fields):
        _fail(
            "invalid_context",
            path,
            "Expected an object containing only documented evidence fields",
        )
    return value


def _need(value, key, path):
    if key not in value or value[key] is None:
        _fail(
            "evidence_missing",
            (*path, key),
            "Required validation evidence is missing",
            "unknown",
        )
    return value[key]


def _integer(value, path, minimum=1):
    if type(value) is not int or value < minimum:
        _fail(
            "invalid_context",
            path,
            "Expected an integer within the documented evidence range",
        )
    return value


def _string(value, path):
    if not isinstance(value, str) or not value.strip():
        _fail("invalid_context", path, "Expected a non-empty string")
    return value


def _boolean(value, path):
    if type(value) is not bool:
        _fail("invalid_context", path, "Expected an explicit boolean")
    return value


def _strings(value, path, *, empty=False):
    if not isinstance(value, list) or (not value and not empty):
        _fail("invalid_context", path, "Expected a list of unique strings")
    for item in value:
        _string(item, path)
    if len(value) != len(set(value)):
        _fail("invalid_context", path, "Duplicate evidence entries are not allowed")
    return value


@dataclass(frozen=True)
class Layout:
    backend: str
    dp: int = 1
    tp: int = 1
    pp: int = 1
    cp: int = 1
    ep: int = 1
    etp: int = 1

    @property
    def world_size(self):
        # EP folds over the attention world; it does not allocate extra GPUs.
        return self.dp * self.tp * self.pp * self.cp


def parse_layout(value: str) -> Layout:
    """Parse one native allocation, including attention/FFN MoE folding.

    Reject repeated dimensions (the native parser silently overwrites them).
    No imports of AReaL, torch or lark are needed on the control machine.
    """
    if not isinstance(value, str):
        raise ValueError("Layout must be a string")
    value = re.sub(r"\s+", "", value)
    match = re.fullmatch(
        r"(fsdp|megatron|archon|sglang|vllm)(?:\[[A-Za-z_][A-Za-z_0-9]*\])?:(.+)", value
    )
    if not match:
        raise ValueError("Expected a supported backend and one native parallel layout")
    backend, body = match.groups()

    def dims(raw, allowed):
        tokens = re.findall(r"([dtpce])([0-9]+)", raw)
        if not tokens or "".join(k + n for k, n in tokens) != raw:
            raise ValueError("Malformed parallel dimensions")
        result = {}
        for key, number in tokens:
            if key not in allowed or key in result or int(number) < 1:
                raise ValueError(
                    "Unsupported, duplicate or non-positive parallel dimension"
                )
            result[key] = int(number)
        return result

    folded = re.fullmatch(r"\(attn:([^|]+)\|ffn:([^|]+)\)", body)
    if folded:
        if backend in ("sglang", "vllm"):
            raise ValueError(
                "Inference layouts do not use native attention/FFN folding syntax"
            )
        a, e = dims(folded[1], "dtpc"), dims(folded[2], "dtpe")
        layout = Layout(
            backend,
            a.get("d", 1),
            a.get("t", 1),
            a.get("p", 1),
            a.get("c", 1),
            e.get("e", 1),
            e.get("t", 1),
        )
        divisor = layout.pp * layout.ep * layout.etp
        if e.get("p", layout.pp) != layout.pp or layout.world_size % divisor:
            raise ValueError("Attention and FFN pipeline/world sizes are incompatible")
        if e.get("d", layout.world_size // divisor) * divisor != layout.world_size:
            raise ValueError("Attention and FFN must have the same world size")
    else:
        d = dims(body, "dtp" if backend in ("sglang", "vllm") else "dtpce")
        layout = Layout(
            backend,
            d.get("d", 1),
            d.get("t", 1),
            d.get("p", 1),
            d.get("c", 1),
            d.get("e", 1),
        )
    if layout.world_size % (layout.pp * layout.ep * layout.etp):
        raise ValueError("Expert parallelism must fold into the allocated world")
    if backend == "fsdp" and (layout.pp != 1 or layout.ep != 1):
        raise ValueError("AReaL FSDP does not support pipeline or expert parallelism")
    return layout


def configuration_layouts(candidate):
    layouts, issues = {}, []
    for role, allowed in (
        ("actor", {"fsdp", "megatron", "archon"}),
        ("rollout", {"sglang", "vllm"}),
    ):
        try:
            layout = parse_layout(candidate.get(role, {}).get("backend"))
            if layout.backend not in allowed:
                raise ValueError("Backend does not support this role")
            layouts[role] = layout
        except (ValueError, AttributeError) as exc:
            issues.append(
                ValidationIssue(
                    "invalid_layout",
                    (role, "backend"),
                    str(exc),
                    source_ref="areal/api/alloc_mode.py",
                )
            )
    return layouts, issues


def check_layouts(baseline, candidate):
    """Both layouts must be interpretable before transition rules can run."""
    layouts, issues = configuration_layouts(candidate)
    _, baseline_issues = configuration_layouts(baseline)
    if baseline_issues:
        issues.append(
            ValidationIssue(
                "invalid_baseline_layout",
                (),
                "The current layout cannot be interpreted; transition evidence is insufficient",
                "unknown",
            )
        )
    return (layouts if len(layouts) == 2 and not baseline_issues else None), issues


def check_resources(ctx, layouts, refs):
    path = ("context", "resources")
    r = _record(
        _need(ctx, "resources", ("context",)),
        path,
        {
            "source_ref",
            "gpus",
            "role_pools",
            "rank_devices",
            "colocation_groups",
            "host_memory_bytes",
            "disk_bytes",
        },
    )
    refs.add(_string(_need(r, "source_ref", path), path))
    gpus = _need(r, "gpus", path)
    if not isinstance(gpus, dict) or not gpus:
        _fail("invalid_context", path, "GPU inventory must be a non-empty object")
    for name, raw in gpus.items():
        _string(name, path)
        gpu = _record(raw, path, {"node", "memory_bytes", "unavailable_bytes"})
        _string(_need(gpu, "node", path), path)
        capacity = _integer(_need(gpu, "memory_bytes", path), path)
        unavailable = _integer(_need(gpu, "unavailable_bytes", path), path, 0)
        if unavailable > capacity:
            _fail(
                "gpu_memory_budget", path, "Unavailable memory exceeds device capacity"
            )
    for field in ("host_memory_bytes", "disk_bytes"):
        _integer(_need(r, field, path), (*path, field), 0)
    pools, ranks = _need(r, "role_pools", path), _need(r, "rank_devices", path)
    if (
        not isinstance(pools, dict)
        or not isinstance(ranks, dict)
        or set(pools) != set(ranks)
    ):
        _fail(
            "invalid_context",
            path,
            "Role pools and rank assignments must cover the same roles",
        )
    if not {"actor", "rollout"} <= set(pools):
        _fail(
            "evidence_missing",
            path,
            "Both adjustable roles require device assignments",
            "unknown",
        )
    for role, devices in pools.items():
        _string(role, path)
        _strings(devices, path)
        _strings(ranks[role], path)
        if not set(devices) <= set(gpus) or not set(ranks[role]) <= set(devices):
            _fail(
                "gpu_reservation",
                path,
                "Device assignment exceeds the fixed GPU reservation or role pool",
            )
        if role in layouts and len(ranks[role]) != layouts[role].world_size:
            _fail(
                "gpu_world_size",
                (role, "backend"),
                "Layout world size does not match the assigned GPU count",
            )
    groups = _need(r, "colocation_groups", path)
    if not isinstance(groups, list):
        _fail("invalid_context", path, "Colocation groups must be a list")
    members = set()
    for group in groups:
        _strings(group, path)
        if (
            len(group) < 2
            or not set(group) <= set(pools)
            or members.intersection(group)
        ):
            _fail(
                "invalid_context",
                path,
                "Colocation groups require distinct known roles",
            )
        members.update(group)
        if any(set(ranks[role]) != set(ranks[group[0]]) for role in group):
            _fail(
                "colocation_layout",
                path,
                "Colocated roles must use matching device sets",
            )
    for left, right in combinations(pools, 2):
        if set(pools[left]) & set(pools[right]) and not any(
            {left, right} <= set(g) for g in groups
        ):
            _fail("gpu_overlap", path, "Separated role reservations overlap")
    return r


def check_model_layouts(
    ctx, candidate, layouts, resources, refs, *, prefix=("context",)
):
    path = (*prefix, "model")
    model = _record(
        _need(ctx, "model", prefix),
        path,
        {
            "source_ref",
            "hidden_size",
            "num_attention_heads",
            "num_key_value_heads",
            "num_hidden_layers",
            "num_experts",
        },
    )
    refs.add(_string(_need(model, "source_ref", path), path))
    for key in (
        "hidden_size",
        "num_attention_heads",
        "num_key_value_heads",
        "num_hidden_layers",
        "num_experts",
    ):
        _integer(
            _need(model, key, path), (*path, key), 0 if key == "num_experts" else 1
        )
    backend_profiles = _record(
        _need(ctx, "backends", prefix), (*prefix, "backends"), set(layouts)
    )
    for role, layout in layouts.items():
        path = (*prefix, "backends", role)
        profile = _record(
            _need(backend_profiles, role, path),
            path,
            {
                "source_ref",
                "supported_layouts",
                "kv_head_replication",
                "layer_partition",
                "max_nodes_per_instance",
            },
        )
        refs.add(_string(_need(profile, "source_ref", path), path))
        supported = _strings(_need(profile, "supported_layouts", path), path)
        try:
            permitted = [parse_layout(item) for item in supported]
        except ValueError:
            _fail(
                "invalid_context",
                path,
                "Backend support evidence contains an invalid layout",
            )
        if layout not in permitted:
            _fail(
                "unsupported_layout",
                (role, "backend"),
                "Layout is outside the backend/model support profile",
            )
        replicate_kv = _boolean(_need(profile, "kv_head_replication", path), path)
        nodes = _integer(_need(profile, "max_nodes_per_instance", path), path)
        # Inference DP counts independent TP*PP instances, not SGLang DP-attention.
        if resources is not None:
            ranks = resources["rank_devices"][role]
            instance_size = (
                layout.tp * layout.pp
                if layout.backend in INFERENCE_BACKENDS
                else layout.world_size
            )
            for start in range(0, len(ranks), instance_size):
                used_nodes = {
                    resources["gpus"][g]["node"]
                    for g in ranks[start : start + instance_size]
                }
                if len(used_nodes) > nodes:
                    _fail(
                        "unsupported_topology",
                        (role, "backend"),
                        "Placement exceeds the backend profile's node span",
                    )
        attention_tp, ep = layout.tp, layout.ep
        if layout.backend == "sglang":
            s = candidate["sglang"]
            dp = _integer(s["dp_size"], ("sglang", "dp_size"))
            ep = _integer(s["ep_size"], ("sglang", "ep_size"))
            if (not s["enable_dp_attention"] and dp != 1) or layout.tp % dp:
                _fail(
                    "sglang_dp_layout",
                    ("sglang", "dp_size"),
                    "DP-attention must be enabled and divide instance TP",
                )
            attention_tp = layout.tp // dp
            if (not s["enable_ep_moe"] and ep != 1) or layout.tp % ep:
                _fail(
                    "sglang_ep_layout",
                    ("sglang", "ep_size"),
                    "EP must be enabled and divide instance TP",
                )
            if s["enable_ep_moe"] and model["num_experts"] == 0:
                _fail(
                    "expert_model_required",
                    ("sglang", "enable_ep_moe"),
                    "Expert parallelism requires a MoE model",
                )
        if (
            model["num_attention_heads"] % attention_tp
            or model["hidden_size"] % attention_tp
        ):
            _fail(
                "attention_divisibility",
                (role, "backend"),
                "Attention heads and hidden size must divide across attention TP",
            )
        kv = model["num_key_value_heads"]
        if kv % attention_tp and not (
            replicate_kv and layout.backend != "fsdp" and attention_tp % kv == 0
        ):
            _fail(
                "kv_head_divisibility",
                (role, "backend"),
                "KV heads cannot be partitioned by this backend's supported rule",
            )
        if ep > 1 and (model["num_experts"] == 0 or model["num_experts"] % ep):
            _fail(
                "expert_divisibility",
                (role, "backend"),
                "Experts must divide across expert parallel ranks",
            )
        partition = _need(profile, "layer_partition", path)
        if partition == "uniform":
            if model["num_hidden_layers"] % layout.pp:
                _fail(
                    "pipeline_divisibility",
                    (role, "backend"),
                    "Uniform pipeline stages require divisible layer counts",
                )
        else:
            if not isinstance(partition, list) or len(partition) != layout.pp:
                _fail(
                    "invalid_context",
                    path,
                    "Provide uniform partitioning or a layer count for each pipeline stage",
                )
            for count in partition:
                _integer(count, path)
            if sum(partition) != model["num_hidden_layers"]:
                _fail(
                    "pipeline_partition",
                    (role, "backend"),
                    "Pipeline layer partition does not cover the model",
                )


def check_shared_inference(ctx, candidate, changes, resources, refs):
    """Teacher inference consumes the same SGLang/vLLM server settings.

    Its model and layout may differ from rollout. A rollout profile cannot
    prove that shared DP/EP or memory settings are legal for the teacher.
    """
    teacher = candidate.get("teacher")
    if teacher is None:
        return
    if not isinstance(teacher, dict):
        _fail(
            "invalid_teacher_configuration",
            ("teacher",),
            "Teacher configuration must be an object",
        )
    if teacher.get("engine_type") != "rollout":
        return
    try:
        layout = parse_layout(teacher["rollout"]["backend"])
    except (KeyError, TypeError, ValueError):
        _fail(
            "invalid_teacher_layout",
            ("teacher", "rollout", "backend"),
            "Teacher inference layout cannot be interpreted",
        )
    if layout.backend not in INFERENCE_BACKENDS:
        _fail(
            "invalid_teacher_layout",
            ("teacher", "rollout", "backend"),
            "Teacher inference requires SGLang or vLLM",
        )
    if not any(change.path[0] == layout.backend for change in changes):
        return
    path = ("context", "auxiliary")
    auxiliary = _record(_need(ctx, "auxiliary", ("context",)), path, {"teacher"})
    path = (*path, "teacher")
    evidence = _record(
        _need(auxiliary, "teacher", path),
        path,
        {"model", "backends", "inference_memory_fraction"},
    )
    if resources is None:
        return  # The resource check already reported why it cannot be used.
    ranks = resources["rank_devices"].get("teacher")
    if ranks is None or len(ranks) != layout.world_size:
        _fail(
            "auxiliary_placement_missing",
            ("teacher",),
            "Teacher rank placement must cover its complete layout",
            "unknown",
        )
    check_model_layouts(
        evidence, candidate, {"teacher": layout}, resources, refs, prefix=path
    )
    fraction = _need(evidence, "inference_memory_fraction", path)
    if type(fraction) not in (int, float) or not 0 < fraction <= 1:
        _fail(
            "invalid_context",
            path,
            "Teacher resolved memory fraction must be in (0, 1]",
        )
    key = (
        "mem_fraction_static"
        if layout.backend == "sglang"
        else "gpu_memory_utilization"
    )
    configured = candidate[layout.backend].get(key)
    if configured is not None and fraction != configured:
        _fail(
            "memory_fraction_scope",
            path,
            "Teacher evidence uses a different shared memory fraction",
            "unknown",
        )
    memory = _need(ctx, "memory", ("context",))
    for gpu in ranks:
        try:
            amounts = memory["devices"][gpu]["roles"]["teacher"]
            static = sum(
                _integer(amounts[field], (*path, field), 0)
                for field in ("parameters", "kv_cache")
            )
        except (KeyError, TypeError):
            _fail(
                "memory_role_coverage",
                path,
                "Teacher memory bounds are missing",
                "unknown",
            )
        if static > int(resources["gpus"][gpu]["memory_bytes"] * fraction):
            _fail(
                "inference_memory_budget",
                path,
                "Teacher parameters/KV exceed the configured memory fraction",
            )


def check_batch_dispatch(ctx, layouts, refs):
    path = ("context", "dispatch")
    dispatch = _record(
        _need(ctx, "dispatch", ("context",)),
        path,
        {"source_ref", "train_batch_size", "group_size", "allow_uneven_dp"},
    )
    refs.add(_string(_need(dispatch, "source_ref", path), path))
    batch = _integer(_need(dispatch, "train_batch_size", path), path)
    group = _integer(_need(dispatch, "group_size", path), path)
    uneven = _boolean(_need(dispatch, "allow_uneven_dp", path), path)
    dp = layouts["actor"].dp
    if batch % group or batch // group < dp or (not uneven and (batch // group) % dp):
        _fail(
            "batch_dispatch",
            ("actor", "backend"),
            "Training batch cannot be dispatched as whole non-empty groups across DP ranks",
        )


def check_memory(ctx, candidate, resources, changes, refs):
    path = ("context", "memory")
    memory = _record(
        _need(ctx, "memory", ("context",)),
        path,
        {
            "source_ref",
            "basis",
            "devices",
            "host_bytes",
            "disk_bytes",
            "inference_memory_fraction",
        },
    )
    refs.add(_string(_need(memory, "source_ref", path), path))
    if _need(memory, "basis", path) != "upper_bound":
        _fail(
            "memory_bound_required",
            path,
            "A sampled peak or prediction is not a workload-wide memory upper bound",
            "unknown",
        )
    devices = _need(memory, "devices", path)
    if not isinstance(devices, dict) or set(devices) != set(resources["gpus"]):
        _fail(
            "memory_coverage",
            path,
            "Memory bounds must cover every reserved GPU, including auxiliary roles and idle GPUs",
            "unknown",
        )
    fields = {
        "parameters",
        "gradients",
        "optimizer",
        "activations",
        "kv_cache",
        "buffers",
        "other",
    }
    inference_backend = parse_layout(candidate["rollout"]["backend"]).backend
    configured_fraction = candidate[inference_backend].get(
        "mem_fraction_static"
        if inference_backend == "sglang"
        else "gpu_memory_utilization"
    )
    fraction = _need(memory, "inference_memory_fraction", path)
    if type(fraction) not in (int, float) or not 0 < fraction <= 1:
        _fail(
            "invalid_context",
            path,
            "Resolved inference memory fraction must be in (0, 1]",
        )
    if configured_fraction is not None and fraction != configured_fraction:
        _fail(
            "memory_fraction_scope",
            path,
            "Memory evidence uses a different inference memory fraction",
            "unknown",
        )
    for gpu, raw in devices.items():
        entry = _record(raw, path, {"roles", "transition_extra"})
        roles = _need(entry, "roles", path)
        expected_roles = {
            r for r, ranks in resources["rank_devices"].items() if gpu in ranks
        }
        if not isinstance(roles, dict) or set(roles) != expected_roles:
            _fail(
                "memory_role_coverage",
                path,
                "Per-device bounds must include every resident role exactly once",
                "unknown",
            )
        peak = 0
        for role, amounts in roles.items():
            _record(amounts, path, fields)
            for field in fields:
                _integer(_need(amounts, field, path), (*path, field), 0)
            peak += sum(amounts.values())
            if role == "rollout":
                # Model and KV state must fit inside the server's reserved
                # fraction too, even if the whole device would have room.
                static = amounts["parameters"] + amounts["kv_cache"]
                if static > int(resources["gpus"][gpu]["memory_bytes"] * fraction):
                    _fail(
                        "inference_memory_budget",
                        path,
                        "Inference parameter/KV bounds exceed the configured memory fraction",
                    )
        extra = _integer(_need(entry, "transition_extra", path), path, 0)
        # Include old/new overlap, staging and recovery memory in this bound.
        if changes:
            peak += extra
        device = resources["gpus"][gpu]
        if peak > device["memory_bytes"] - device["unavailable_bytes"]:
            _fail(
                "memory_capacity",
                path,
                "Steady-state or transition memory exceeds available device capacity",
            )
    for needed, budget in (
        ("host_bytes", "host_memory_bytes"),
        ("disk_bytes", "disk_bytes"),
    ):
        value = _integer(_need(memory, needed, path), (*path, needed), 0)
        if value > resources[budget]:
            _fail(
                "auxiliary_capacity",
                (*path, needed),
                "Host-memory or checkpoint-disk requirement exceeds its reservation",
            )


def check_transition(ctx, baseline, candidate, changes, refs):
    if not changes:
        return "not_required"
    path = ("context", "transition")
    t = _record(
        _need(ctx, "transition", ("context",)),
        path,
        {
            "source_ref",
            "supported",
            "changes",
            "boundary",
            "state_mapping",
            "recovery_ref",
        },
    )
    refs.add(_string(_need(t, "source_ref", path), path))
    if not _boolean(_need(t, "supported", path), path):
        _fail(
            "unsupported_transition",
            path,
            "The runtime evidence explicitly excludes this configuration transition",
        )
    expected = {".".join(change.path) for change in changes}
    if set(_strings(_need(t, "changes", path), path)) != expected:
        _fail(
            "transition_coverage",
            path,
            "Support evidence must cover this entire combined change, not separate field updates",
            "unknown",
        )
    _string(_need(t, "boundary", path), path)
    refs.add(_string(_need(t, "recovery_ref", path), path))
    fields = {
        "parameters",
        "optimizer",
        "lr_scheduler",
        "gradients",
        "data_cursor",
        "rng",
        "ready_rollouts",
        "in_flight_rollouts",
        "policy_version",
        "device_residency",
    }
    mapping = _record(_need(t, "state_mapping", path), path, fields)
    actor_layout_changed = parse_layout(baseline["actor"]["backend"]) != parse_layout(
        candidate["actor"]["backend"]
    )
    for field in fields:
        p = (*path, "state_mapping", field)
        entry = _record(_need(mapping, field, path), p, {"method", "source_ref"})
        refs.add(_string(_need(entry, "source_ref", p), p))
        method = _need(entry, "method", p)
        allowed = {"preserve"}
        if field in {"parameters", "optimizer"}:
            allowed = {"reshard"} if actor_layout_changed else {"preserve", "reshard"}
        elif field == "gradients":
            allowed = (
                {"none_at_boundary", "reshard"}
                if actor_layout_changed
                else {"preserve", "none_at_boundary", "reshard"}
            )
        elif field == "rng":
            allowed = {"remap"} if actor_layout_changed else {"preserve", "remap"}
        elif field == "in_flight_rollouts":
            allowed = {"preserve", "drain", "migrate"}
        elif field == "device_residency":
            allowed = {"preserve", "remap"}
        if not isinstance(method, str) or method not in allowed:
            _fail(
                "state_mapping",
                p,
                "State handling does not preserve training progress under this change",
            )
    return "supported"


def check_execution(
    execution, baseline, candidate, changes, ctx, *, runtime_revision, baseline_id, refs
):
    """The implemented adapter's static checks, before any stop request.

    Unlike offline transition evidence, supported=True cannot enable a mapping.
    This adapter preserves same-layout RNGs, reseeds supported training layout
    changes deterministically, and rebuilds inference.
    """
    from autopilot_harness.contracts.execution import APPLY_ADAPTER

    path = ("execution",)
    fields = {
        "run_id",
        "segment_id",
        "nonce",
        "runtime_revision",
        "configuration_id",
        "runtime_source_id",
        "apply_adapter",
    }
    e = _record(execution, path, fields)
    for key in fields:
        _string(_need(e, key, path), (*path, key))
    if (
        e["runtime_revision"] != runtime_revision
        or e["configuration_id"] != baseline_id
    ):
        _fail(
            "execution_scope_mismatch",
            path,
            "Runtime binding differs from validation inputs",
        )
    if e["apply_adapter"] != APPLY_ADAPTER:
        _fail(
            "unsupported_apply_adapter",
            path,
            "Runtime does not implement this transition adapter",
        )
    implemented = {
        "actor.backend",
        "actor.mb_spec.n_mbs",
        "actor.mb_spec.max_tokens_per_mb",
        "actor.pad_to_maximum",
        "rollout.backend",
        "rollout.max_concurrent_rollouts",
        "sglang.enable_dp_attention",
        "sglang.dp_size",
        "sglang.enable_ep_moe",
        "sglang.ep_size",
        "sglang.max_running_requests",
        "sglang.max_prefill_tokens",
        "sglang.mem_fraction_static",
        "vllm.max_num_seqs",
    }
    changed = {".".join(change.path) for change in changes}
    if not changed <= implemented:
        _fail(
            "unsupported_apply_parameter",
            path,
            "No state mapping is implemented for this change",
        )
    actor = parse_layout(candidate["actor"]["backend"])
    rollout = parse_layout(candidate["rollout"]["backend"])
    if parse_layout(baseline["rollout"]["backend"]).backend != rollout.backend:
        _fail(
            "unsupported_backend_substitution",
            ("rollout", "backend"),
            "This adapter preserves the inference backend implementation",
        )
    if actor.backend not in {"fsdp", "megatron"}:
        _fail(
            "unsupported_snapshot_backend",
            ("actor", "backend"),
            "Snapshot runtime state supports FSDP and Megatron",
        )
    for role in ("actor", "critic"):
        settings = candidate.get(role)
        if settings:
            backend = parse_layout(settings["backend"]).backend
            if backend not in {"fsdp", "megatron"}:
                _fail(
                    "unsupported_snapshot_backend",
                    (role, "backend"),
                    "Snapshot runtime state supports FSDP and Megatron",
                )
            if backend == "megatron" and settings.get("use_lora"):
                _fail(
                    "unsupported_snapshot_lora",
                    (role, "use_lora"),
                    "Megatron LoRA does not implement model/optimizer DCP save and load",
                )
    source_actor = parse_layout(baseline["actor"]["backend"])
    training_layout = None
    if source_actor != actor:
        if source_actor.backend != actor.backend:
            _fail(
                "unsupported_backend_substitution",
                ("actor", "backend"),
                "Training backend implementation must stay fixed",
            )
        if actor.backend == "fsdp" and any(
            layout.pp != 1 or layout.ep != 1 or layout.etp != 1
            for layout in (source_actor, actor)
        ):
            _fail(
                "training_layout_mapping_unsupported",
                ("actor", "backend"),
                "The native FSDP implementation supports DP/TP/CP mapping; it does not implement pipeline or expert execution",
            )
        training_layout = dict(
            source=[
                source_actor.dp,
                source_actor.tp,
                source_actor.pp,
                source_actor.cp,
                source_actor.ep,
                source_actor.etp,
            ],
            target=[actor.dp, actor.tp, actor.pp, actor.cp, actor.ep, actor.etp],
            rng_policy="fsdp-mesh-reseed-v1",
        )
        if actor.backend == "megatron":
            training_layout.update(
                backend="megatron",
                rng_policy="megatron-model-parallel-reseed-v1",
                virtual_pipeline_size=check_megatron_mapping(
                    ctx, baseline, candidate, source_actor, actor, refs
                ),
            )
    if candidate.get("scheduler", {}).get("type") != "local":
        _fail(
            "unsupported_apply_scheduler",
            ("scheduler",),
            "This adapter requires the standard v1 LocalScheduler",
        )
    for role in ("actor", "rollout", "critic", "ref"):
        settings = candidate.get(role)
        if settings and settings.get("_version") != "v1":
            _fail(
                "unsupported_apply_controller",
                (role, "_version"),
                "All managed roles must use v1",
            )
    mode = candidate["actor"].get("weight_update_mode")
    if mode not in {"disk", "xccl", "awex"}:
        _fail(
            "weight_update_evidence_missing",
            ("actor", "weight_update_mode"),
            "Resolved weight update mode is required",
            "unknown",
        )
    if mode == "xccl" and rollout.pp > 1:
        if (
            actor.backend == "megatron"
            and rollout.backend == "sglang"
            and actor.pp != rollout.pp
        ):
            _fail(
                "weight_sync_layout",
                ("rollout", "backend"),
                "Megatron/SGLang XCCL requires matching pipeline sizes",
            )
        if actor.backend == "fsdp" and rollout.backend == "vllm":
            _fail(
                "weight_sync_layout",
                ("rollout", "backend"),
                "FSDP/vLLM XCCL pipeline groups are incompatible",
            )
    resources = ctx["resources"]
    if mode == "awex":
        if actor.backend != "megatron" or rollout.backend != "sglang":
            _fail("weight_sync_layout", path, "v1 AWEX requires Megatron and SGLang")
        ranks = resources["rank_devices"]
        if (
            actor.world_size != rollout.world_size
            or set(ranks["actor"]) != set(ranks["rollout"])
            or actor.world_size % (rollout.tp * rollout.pp)
        ):
            _fail(
                "weight_sync_placement",
                path,
                "AWEX requires the same GPU set and compatible instance size",
            )
    # LocalScheduler colocation matches worker replicas, not merely GPU totals.
    replicas = {"actor": actor.world_size, "rollout": rollout.dp}
    settings_by_role = {role: candidate[role] for role in ("actor", "rollout")}
    for role in ("critic", "ref"):
        if candidate.get(role):
            replicas[role] = parse_layout(candidate[role]["backend"]).world_size
            settings_by_role[role] = candidate[role]
    teacher = candidate.get("teacher")
    if teacher:
        kind = teacher.get("engine_type")
        settings = teacher.get(kind, {})
        layout = parse_layout(settings.get("backend"))
        if settings.get("_version") != "v1":
            _fail("unsupported_apply_controller", ("teacher",), "Teacher must use v1")
        replicas["teacher"] = layout.dp if kind == "rollout" else layout.world_size
        settings_by_role["teacher"] = settings
    for role, settings in settings_by_role.items():
        if role not in resources["rank_devices"]:
            _fail(
                "auxiliary_placement_missing",
                (role,),
                "Resource/memory evidence must include every active role",
                "unknown",
            )
        if role not in {"actor", "rollout"}:
            layout = parse_layout(settings["backend"])
            if len(resources["rank_devices"][role]) != layout.world_size:
                _fail(
                    "auxiliary_placement_size",
                    (role,),
                    "Auxiliary rank placement does not match its layout",
                )
        strategy = settings.get("scheduling_strategy", {"type": "separation"})
        if strategy["type"] == "colocation":
            target = strategy.get("target")
            if target not in replicas:
                _fail(
                    "colocation_evidence_missing",
                    (role,),
                    "Missing target worker replica information",
                    "unknown",
                )
            if replicas[role] != replicas[target]:
                _fail(
                    "colocation_replicas",
                    (role,),
                    "LocalScheduler colocated jobs require identical worker replica counts",
                )
            if not any(
                {role, target} <= set(group) for group in resources["colocation_groups"]
            ):
                _fail(
                    "colocation_placement",
                    (role,),
                    "Scheduling strategy differs from the supplied GPU sharing evidence",
                )
            if {role, target} == {"actor", "rollout"} and mode not in {"disk", "awex"}:
                _fail(
                    "weight_sync_placement",
                    (role,),
                    "Actor/rollout colocation requires disk or AWEX synchronization",
                )
        elif strategy["type"] != "separation":
            _fail(
                "unsupported_scheduling_strategy",
                (role,),
                "Unknown LocalScheduler strategy",
            )

    def limits(config):
        settings = config["rollout"]
        consumer = _integer(
            _need(settings, "consumer_batch_size", ("rollout",)),
            ("rollout", "consumer_batch_size"),
        )
        concurrent = settings.get("max_concurrent_rollouts") or consumer
        staleness = _integer(
            _need(settings, "max_head_offpolicyness", ("rollout",)),
            ("rollout", "max_head_offpolicyness"),
            0,
        )
        return [concurrent, consumer, staleness]

    source_limits, target_limits = limits(baseline), limits(candidate)
    if source_limits[1:] != target_limits[1:]:
        _fail(
            "staleness_mapping",
            ("rollout",),
            "Batch size and staleness must remain fixed",
        )
    mapping = dict(
        adapter_id=APPLY_ADAPTER,
        source_limits=source_limits,
        target_limits=target_limits,
    )
    if training_layout is not None:
        mapping["training_layout"] = training_layout
    return mapping


def check_megatron_mapping(ctx, baseline, candidate, source, target, refs):
    """Validate the portable sharding contract before requesting a snapshot.

    Schemas describe global parameter/optimizer semantics, excluding rank and
    chain ownership. They are trusted model-adapter evidence, not Planner flags.
    Memory evidence must include host gathering and target load staging.
    """
    path = ("context", "training_checkpoint")
    evidence = _record(
        _need(ctx, "training_checkpoint", ("context",)),
        path,
        {"source_ref", "format", "model_schema", "optimizer_schema"},
    )
    refs.add(_string(_need(evidence, "source_ref", path), path))
    if _need(evidence, "format", path) != "areal-megatron-fully-reshardable-v1":
        _fail(
            "training_checkpoint_format",
            path,
            "Megatron layout changes require the portable optimizer encoding",
        )
    for name in ("model_schema", "optimizer_schema"):
        pair = _record(_need(evidence, name, path), (*path, name), {"source", "target"})
        left, right = (
            _string(_need(pair, side, (*path, name)), (*path, name, side))
            for side in ("source", "target")
        )
        if left != right:
            _fail(
                "training_checkpoint_schema",
                (*path, name),
                "Global model/optimizer semantics differ; no conversion is implemented",
            )
    vpp = None
    for config, layout in ((baseline, source), (candidate, target)):
        settings = config["actor"].get("megatron", {})
        if not isinstance(settings, dict):
            _fail(
                "invalid_megatron_configuration",
                ("actor", "megatron"),
                "Megatron settings must be an object",
            )
        current_vpp = _integer(
            settings.get("virtual_pipeline_parallel_size") or 1,
            ("actor", "megatron", "virtual_pipeline_parallel_size"),
        )
        if vpp is not None and vpp != current_vpp:
            _fail(
                "training_checkpoint_vpp",
                ("actor", "megatron"),
                "The current adapter preserves the VPP setting",
            )
        vpp = current_vpp
        if settings.get("use_custom_fsdp"):
            _fail(
                "training_checkpoint_optimizer",
                ("actor", "megatron", "use_custom_fsdp"),
                "Megatron custom FSDP requires a different checkpoint encoding",
            )
        bridge = settings.get("bridge_type", "mbridge")
        if (layout.pp > 1 or vpp > 1) and bridge not in {"mbridge", "megatron-bridge"}:
            _fail(
                "training_checkpoint_pipeline",
                ("actor", "megatron", "bridge_type"),
                "The fallback model adapter does not implement pipeline construction",
            )
        if vpp > 1 and layout.pp == 1:
            _fail(
                "training_checkpoint_pipeline",
                ("actor", "backend"),
                "Virtual pipeline requires pipeline parallelism",
            )
    return vpp
