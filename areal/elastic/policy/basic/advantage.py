# SPDX-License-Identifier: Apache-2.0

"""Structural timing prior for eager token GAE at the trainer wall-clock boundary.

Independent of Basic and the external Expert archive. No timings/labels enter this
predictor. Operation counts approximate the source path, not measured CUDA kernels.
The v1 recipe covers group reward normalization and optional masked advantage
normalization without centering/scaling. Model-parallel replication requires
explicit broadcast priors. Other algorithm paths remain unsupported.
"""

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class AdvantageRates:
    device_bytes_per_second: float
    host_bytes_per_second: float
    transfer_bytes_per_second: float
    rpc_bytes_per_second: float
    launch_seconds: float
    iteration_seconds: float
    host_sync_seconds: float
    rpc_fixed_seconds: float
    rpc_rank_seconds: float
    audit_file_seconds: float
    trainer_stats_seconds: float

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
            if name.endswith("bytes_per_second") and value == 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class AdvantageOperations:
    # All byte counts are proxies for tensor passes at the named boundary.
    input_bytes_per_slot: float
    output_bytes_per_slot: float
    input_bytes_per_sequence: float
    output_bytes_per_sequence: float
    batching_bytes_per_slot: float
    batching_launches_per_group: float
    preparation_launches: float
    preparation_bytes_per_slot: float
    recurrence_ops_per_iteration: float
    recurrence_bytes_per_element: float
    finish_launches: float
    finish_bytes_per_slot: float
    reward_norm_launches_per_group: float
    reward_norm_bytes_per_sequence: float
    masked_norm_launches: float
    masked_norm_bytes_per_slot: float
    audit_launches: float
    audit_bytes_per_slot: float
    audit_syncs: float

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.recurrence_ops_per_iteration == 0:
            raise ValueError("Eager recurrence requires operations per iteration")


@dataclass(frozen=True)
class AdvantageProfile:
    execution_id: str
    device: str
    gae_mode: str
    reward_normalization: str
    masked_advantage_normalization: bool
    audit: bool
    rates: AdvantageRates
    operations: AdvantageOperations

    def __post_init__(self) -> None:
        if not isinstance(self.execution_id, str) or not self.execution_id:
            raise ValueError("An execution identity is required")
        if (
            type(self.audit) is not bool
            or type(self.masked_advantage_normalization) is not bool
        ):
            raise ValueError("Normalization and audit switches must be booleans")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AdvantageProfile":
        if data["schema_version"] != 1:
            raise ValueError("Unsupported advantage profile schema")
        return cls(
            **data["recipe"],
            rates=AdvantageRates(**data["rates"]),
            operations=AdvantageOperations(**data["operations"]),
        )


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def partition_groups(
    groups: Sequence[Sequence[int]], dp: int, weights: Sequence[float]
) -> list[list[int]]:
    """Controller-style equal-count greedy dispatch of atomic trajectory groups."""
    if not _positive_int(dp) or not groups or len(groups) % dp:
        raise ValueError("Non-empty group count must be divisible by positive DP")
    if any(not g or any(not _positive_int(n) for n in g) for g in groups):
        raise ValueError("Complete positive integer sequence lengths are required")
    if len(weights) != len(groups) or any(
        not math.isfinite(w) or w <= 0 for w in weights
    ):
        raise ValueError("One positive finite dispatch weight is required per group")
    ranks = [[] for _ in range(dp)]
    loads = [0.0] * dp
    capacity = len(groups) // dp
    for index in sorted(range(len(groups)), key=lambda i: -weights[i]):
        rank = min(
            (r for r in range(dp) if len(ranks[r]) < capacity),
            key=lambda r: (loads[r], r),
        )
        ranks[rank].append(index)
        loads[rank] += weights[index]
    return ranks


def estimate_advantage(
    groups: Sequence[Sequence[int]],
    *,
    dp: int,
    profile: AdvantageProfile,
    dispatch_weights: Sequence[float] | None = None,
    model_parallel_size: int = 1,
    broadcast_bytes_per_second: float | None = None,
    broadcast_latency_seconds: float | None = None,
) -> dict[str, Any]:
    """Return per-rank work and an explicitly assumed critical-path prediction.

    Groups remain atomic; each rank concatenates and pads its assigned groups to
    its local maximum length. Training microbatch limits and model parameter
    offload do not determine this tensor shape or the advantage execution device.
    Each model-parallel worker executes the full DP-local GAE tensor. Transfers assume
    one complete input/output materialization, without overlap or cache reuse.
    """
    groups = tuple(tuple(g) for g in groups)
    weights = (
        tuple(len(g) * max(g) if g else 0 for g in groups)
        if dispatch_weights is None
        else tuple(dispatch_weights)
    )
    assignments = partition_groups(groups, dp, weights)
    if not _positive_int(model_parallel_size):
        raise ValueError("model_parallel_size must be a positive integer")
    if model_parallel_size > 1 and (
        broadcast_bytes_per_second is None or broadcast_latency_seconds is None
    ):
        return {
            "seconds": None,
            "evidence": "unsupported",
            "reason": "Model-parallel advantage requires explicit broadcast rate/latency",
        }
    if model_parallel_size > 1 and (
        not math.isfinite(broadcast_bytes_per_second)
        or broadcast_bytes_per_second <= 0
        or not math.isfinite(broadcast_latency_seconds)
        or broadcast_latency_seconds < 0
    ):
        raise ValueError("Invalid model-parallel broadcast prior")
    if (
        profile.device not in {"cuda", "cpu"}
        or profile.gae_mode != "token_eager"
        or profile.reward_normalization not in {"group", "none"}
    ):
        return {
            "seconds": None,
            "evidence": "unsupported",
            "reason": "v1 requires token_eager, cpu/cuda and group/none reward norm",
        }
    rates, ops = profile.rates, profile.operations
    launch = rates.launch_seconds if profile.device == "cuda" else 0.0
    sync = rates.host_sync_seconds if profile.device == "cuda" else 0.0
    ranks = []
    for rank, indices in enumerate(assignments):
        lengths = [n for i in indices for n in groups[i]]
        batch, width = len(lengths), max(lengths)
        slots = batch * width
        input_slots = sum(len(groups[i]) * max(groups[i]) for i in indices)
        iterations = width - 1  # Includes padding/masked positions in eager GAE.
        elements = batch * iterations
        input_bytes = (
            input_slots * ops.input_bytes_per_slot
            + batch * ops.input_bytes_per_sequence
        )
        output_bytes = (
            slots * ops.output_bytes_per_slot + batch * ops.output_bytes_per_sequence
        )
        broadcast_bytes = input_bytes * (model_parallel_size - 1)
        broadcast_seconds = (
            (
                broadcast_bytes / broadcast_bytes_per_second
                + (model_parallel_size - 1) * broadcast_latency_seconds
            )
            if model_parallel_size > 1
            else 0.0
        )
        reward_groups = len(indices) if profile.reward_normalization == "group" else 0
        norm_launches = reward_groups * ops.reward_norm_launches_per_group
        norm_bytes = batch * ops.reward_norm_bytes_per_sequence if reward_groups else 0
        # Group mean and std each use a scalar .item(); masked norm checks mask.sum.
        norm_syncs = 2 * reward_groups
        if profile.masked_advantage_normalization:
            norm_launches += ops.masked_norm_launches
            norm_bytes += slots * ops.masked_norm_bytes_per_slot
            norm_syncs += 1
        recurrence_launches = iterations * ops.recurrence_ops_per_iteration
        components = {
            # concat/pad executes after the worker's device transfer.
            "batching": slots
            * ops.batching_bytes_per_slot
            / rates.device_bytes_per_second
            + len(indices) * ops.batching_launches_per_group * launch
            + slots * ops.preparation_bytes_per_slot / rates.device_bytes_per_second
            + ops.preparation_launches * launch,
            "recurrence": iterations * rates.iteration_seconds
            + (recurrence_launches + ops.finish_launches) * launch
            + (
                elements * ops.recurrence_bytes_per_element
                + slots * ops.finish_bytes_per_slot
            )
            / rates.device_bytes_per_second,
            "normalization": norm_launches * launch
            + norm_bytes / rates.device_bytes_per_second
            + norm_syncs * sync,
            "transport": (input_bytes + output_bytes) / rates.transfer_bytes_per_second
            + broadcast_seconds
            if profile.device == "cuda"
            else 0.0,
            "diagnostics": (
                ops.audit_launches * launch
                + slots * ops.audit_bytes_per_slot / rates.device_bytes_per_second
                + ops.audit_syncs * sync
                + rates.audit_file_seconds
            )
            if profile.audit
            else 0.0,
        }
        ranks.append(
            {
                "rank": rank,
                "groups": indices,
                "batch": batch,
                "padded_length": width,
                "sequence_tokens": sum(lengths),
                "padded_slots": slots,
                "input_group_padded_slots": input_slots,
                "iterations": iterations,
                "recurrence_elements": elements,
                "recurrence_pointwise_ops": recurrence_launches,
                "normalization_host_syncs": norm_syncs,
                "normalization_allreduces": 0,
                "input_bytes": input_bytes,
                "output_bytes": output_bytes,
                "replicated_workers": model_parallel_size,
                "broadcast_bytes": broadcast_bytes,
                "components": components,
                "seconds": sum(components.values()),
            }
        )
    critical = max(ranks, key=lambda r: r["seconds"])
    rpc_bytes = sum(r["input_bytes"] + r["output_bytes"] for r in ranks)
    shared = {
        "rpc": rates.rpc_fixed_seconds
        + dp * model_parallel_size * rates.rpc_rank_seconds
        + rpc_bytes / rates.rpc_bytes_per_second,
        "host_serialization": rpc_bytes / rates.host_bytes_per_second,
        "trainer_diagnostics": rates.trainer_stats_seconds,
    }
    return {
        "model": "structural-token-gae-v1",
        "execution_id": profile.execution_id,
        "device": profile.device,
        "worker_count": dp * model_parallel_size,
        "model_parallel_size": model_parallel_size,
        "evidence": "assumed",
        "shape_evidence": "derived_from_atomic_group_lengths_and_dispatch",
        "timing_boundary": "trainer_host_wall_with_stats_and_optional_rank_audit",
        "seconds": sum(shared.values()) + critical["seconds"],
        "critical_rank": critical["rank"],
        "critical_components": critical["components"],
        "shared_components": shared,
        "ranks": ranks,
    }


def fit_advantage_scale(
    samples: Sequence[dict[str, Any]], *, profile_sha256: str
) -> dict[str, Any]:
    """Fit one whole-stage multiplier from explicitly selected observations.

    Four aggregate wall times cannot identify the individual component rates.
    Least squares through the origin keeps those priors and the shape model
    intact. This routine receives no validation labels or unselected history.
    """
    if not samples:
        raise ValueError("Calibration requires selected observations")
    steps = [s["step"] for s in samples]
    if any(not _positive_int(step) for step in steps) or steps != sorted(set(steps)):
        raise ValueError("Calibration steps must be positive, unique and ordered")
    first = samples[0]["prediction"]
    identity = (
        first.get("execution_id"),
        first.get("device"),
        len(first.get("ranks", [])),
        first.get("model_parallel_size", 1),
    )
    labels, widths = [], []
    for sample in samples:
        prediction, actual = sample["prediction"], sample["observed_seconds"]
        if prediction.get("evidence") != "assumed" or "calibration" in prediction:
            raise ValueError("Calibration requires an uncalibrated structural prior")
        if (
            prediction.get("execution_id"),
            prediction.get("device"),
            len(prediction.get("ranks", [])),
            prediction.get("model_parallel_size", 1),
        ) != identity:
            raise ValueError("Calibration observations must share execution/device/DP")
        prior = prediction["seconds"]
        if any(v is None or not math.isfinite(v) or v <= 0 for v in (prior, actual)):
            raise ValueError("Calibration timings must be finite and positive")
        widths.append(max(r["padded_length"] for r in prediction["ranks"]))
        labels.append(
            {"step": sample["step"], "prior_seconds": prior, "observed_seconds": actual}
        )
    scale = sum(s["prior_seconds"] * s["observed_seconds"] for s in labels) / sum(
        s["prior_seconds"] ** 2 for s in labels
    )
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid fitted scale")
    return {
        "schema_version": 1,
        "status": "fitted",
        "method": "least_squares_through_origin",
        "scope": "whole_stage_scale_not_individual_component_fit",
        "profile_sha256": profile_sha256,
        "execution_id": identity[0],
        "device": identity[1],
        "dp": identity[2],
        "model_parallel_size": identity[3],
        "steps": steps,
        "scale": scale,
        "max_rank_padded_length_range": [min(widths), max(widths)],
        "samples": labels,
        "samples_sha256": hashlib.sha256(
            json.dumps(labels, sort_keys=True, allow_nan=False).encode()
        ).hexdigest(),
    }


def apply_advantage_scale(
    prediction: dict[str, Any], calibration: dict[str, Any], *, profile_sha256: str
) -> dict[str, Any]:
    """Apply a frozen multiplier; no timing labels are accepted at prediction."""
    if calibration["profile_sha256"] != profile_sha256:
        raise ValueError("Calibration belongs to another structural profile")
    if prediction["seconds"] is None:
        return prediction  # Unknown execution never becomes a numeric fallback.
    if "calibration" in prediction:
        raise ValueError("Prediction is already calibrated")
    if prediction.get("model_parallel_size", 1) != calibration.get(
        "model_parallel_size", 1
    ):
        raise ValueError("Calibration model-parallel layout differs from prediction")
    if (prediction["execution_id"], prediction["device"], len(prediction["ranks"])) != (
        calibration["execution_id"],
        calibration["device"],
        calibration["dp"],
    ):
        raise ValueError("Calibration execution/device/DP differs from prediction")
    factor = calibration["scale"]
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("Calibration scale must be finite and positive")
    width = max(r["padded_length"] for r in prediction["ranks"])
    lower, upper = calibration["max_rank_padded_length_range"]
    in_range = lower <= width <= upper
    ranks = [
        {
            **r,
            "seconds": r["seconds"] * factor,
            "components": {k: v * factor for k, v in r["components"].items()},
        }
        for r in prediction["ranks"]
    ]
    return {
        **prediction,
        "seconds": prediction["seconds"] * factor,
        "uncalibrated_seconds": prediction["seconds"],
        "evidence": "calibrated" if in_range else "assumed",
        "ranks": ranks,
        "critical_components": ranks[prediction["critical_rank"]]["components"],
        "shared_components": {
            k: v * factor for k, v in prediction["shared_components"].items()
        },
        "calibration": {
            "scale": factor,
            "status": "within_width_range" if in_range else "width_extrapolation",
            "component_times": "allocated_by_stage_scale_not_individually_fitted",
            "receipt_sha256": hashlib.sha256(
                json.dumps(calibration, sort_keys=True, allow_nan=False).encode()
            ).hexdigest(),
        },
    }
