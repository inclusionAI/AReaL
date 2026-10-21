# SPDX-License-Identifier: Apache-2.0

"""Sparse public aggregate calibration with explicit holdout and transfer limits.

Fit only a total train-step scale. Total-step labels cannot independently identify
attention, backward, communication or optimizer times. No network access at runtime.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace

from .actor import ActorCostModel
from .catalog import hardware_profile, qwen3_30b_a3b, read_data
from .config import ActorConfig, ParallelLayout
from .workload import WorkloadStats


def point_inputs(point: dict) -> tuple[ActorConfig, WorkloadStats]:
    backend, parallel = ParallelLayout.from_allocation(point["allocation"])
    count, length = point["global_batch"], point["sequence_length"]
    config = ActorConfig(
        backend,
        parallel,
        max_tokens_per_mb=length,
        n_mbs=point["microbatches_per_rank"],
        compile=True,
        expert_dispatcher="hierarchical",
    )
    workload = WorkloadStats(count, count * length, sequence_lengths=(length,) * count)
    return config, workload


@dataclass(frozen=True)
class Calibration:
    train_scale: float
    fit_ids: tuple[str, ...]
    backend: str = "megatron"
    hardware: str = "h100_sxm_reference"
    scope: str = (
        "Aggregate natural-routing BF16 optimizer step; DP16/TP1/PP1/CP1/EP16, HybridEP"
    )

    @classmethod
    def fit(cls, points: list[dict]) -> Calibration:
        if not points:
            raise ValueError("Calibration requires at least one public training point")
        if any(
            p["hardware"] != "h100_sxm_reference" or p["split"] != "fit" for p in points
        ):
            raise ValueError(
                "Holdout/protocol-stress rows cannot enter the bundled fit"
            )
        ratios = []
        for point in points:
            config, workload = point_inputs(point)
            raw = ActorCostModel(
                qwen3_30b_a3b(), hardware_profile(point["hardware"])
            ).predict(config, workload)["train_step"]
            actual = point["step_seconds"]
            if not math.isfinite(actual) or actual <= 0:
                raise ValueError("Public step label must be finite and positive")
            ratios.append(math.log(actual / raw))
        return cls(math.exp(sum(ratios) / len(ratios)), tuple(p["id"] for p in points))

    def apply(
        self,
        model: ActorCostModel,
        config: ActorConfig,
        *,
        allow_transfer: bool = False,
    ) -> ActorCostModel:
        p = config.parallel
        matching = (
            model.hardware == hardware_profile(self.hardware)
            and model.model == qwen3_30b_a3b()
            and config.backend == self.backend
            and config.expert_dispatcher == "hierarchical"
            and (p.dp, p.tp, p.pp, p.cp, p.ep, p.etp) == (16, 1, 1, 1, 16, 1)
        )
        if not matching and not allow_transfer:
            raise ValueError(
                "Public fit does not cover this hardware/layout/backend; opt into transfer explicitly"
            )
        note = self.scope + "; fit IDs=" + ",".join(self.fit_ids)
        note += (
            "; aggregate fit only"
            if matching
            else "; UNVALIDATED hardware/layout/backend transfer"
        )
        return replace(model, train_scale=self.train_scale, calibration_note=note)


def bundled_calibration() -> Calibration:
    points = read_data("public_training.json")["points"]
    return Calibration.fit([p for p in points if p["split"] == "fit"])


def public_validation() -> dict:
    data = read_data("public_training.json")
    calibration = bundled_calibration()
    rows = []
    for point in data["points"]:
        config, workload = point_inputs(point)
        model = ActorCostModel(qwen3_30b_a3b(), hardware_profile(point["hardware"]))
        raw = model.predict(config, workload)["train_step"]
        fitted = calibration.apply(model, config).predict(config, workload)[
            "train_step"
        ]
        rows.append(
            {
                **point,
                "analytical_seconds": raw,
                "calibrated_seconds": fitted,
                "relative_error": (fitted - point["step_seconds"])
                / point["step_seconds"],
                "used_for_fit": point["id"] in calibration.fit_ids,
            }
        )
    heldout = [row for row in rows if row["split"] == "holdout"]
    return {
        "model": "Qwen3-30B-A3B",
        "sources": read_data("public_sources.json")["sources"],
        "fit": asdict(calibration),
        "fit_method": "One positive multiplicative aggregate train_step correction, geometric mean ratio; no stage parameter fit",
        "dataset_sha256": hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest(),
        "holdout_count": len(heldout),
        "holdout_mape": sum(abs(row["relative_error"]) for row in heldout)
        / len(heldout)
        if heldout
        else None,
        "points": rows,
        "excluded": data["excluded"],
        "limitations": [
            "One fit point and one coarse-input holdout are insufficient to establish broad accuracy",
            "Protocol-stress point is not the natural-routing holdout",
            "No measured H200/Archon actor phase accuracy claim",
            "Subphase decomposition and advantage/logprob estimates have no independent timing labels",
        ],
    }
