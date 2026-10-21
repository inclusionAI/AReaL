# SPDX-License-Identifier: Apache-2.0

"""Replica-level mixed prefill/decode service, excluding queue and tool latency."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from ..core import (
    Estimate,
    Features,
    PhaseCalibration,
    apply_calibration,
    code_revision,
    fingerprint,
    integer,
    nonnegative,
)
from ._actor.config import Hardware, ModelShape
from ._rollout.service import iteration_cost
from ._rollout.types import ServiceSlice, ServiceState


@dataclass(frozen=True)
class RolloutConfig:
    replicas: int
    tp: int
    max_batch_sequences: int
    max_batch_tokens: int
    kv_capacity_tokens: int
    backend: str = "sglang"
    weight_bytes: int = 2
    kv_bytes: int = 2
    cache_mode: Literal["none", "prefix"] = "none"

    def __post_init__(self) -> None:
        for name in (
            "replicas",
            "tp",
            "max_batch_sequences",
            "max_batch_tokens",
            "kv_capacity_tokens",
            "weight_bytes",
            "kv_bytes",
        ):
            integer(name, getattr(self, name), minimum=1)
        if self.backend != "sglang" or self.cache_mode not in {"none", "prefix"}:
            raise ValueError(
                "This service model covers SGLang with explicit cache mode"
            )
        if self.weight_bytes != 2 or self.kv_bytes != 2:
            raise ValueError("Quantized weights/KV require a separate operator profile")

    @property
    def world_size(self) -> int:
        return self.replicas * self.tp


@dataclass(frozen=True)
class InferenceRates:
    """Explicit effective/peak fractions; no company hardware calibration implied."""

    dense: float
    attention: float
    hbm: float
    network: float
    launch_seconds: float
    collectives_per_layer: int = 2

    def __post_init__(self) -> None:
        for name in ("dense", "attention", "hbm", "network"):
            value = getattr(self, name)
            nonnegative(name, value, positive=True)
            if value > 1:
                raise ValueError("Effective rate cannot exceed specified peak")
        nonnegative("launch_seconds", self.launch_seconds)
        integer("collectives_per_layer", self.collectives_per_layer)


@dataclass(frozen=True)
class ServiceEstimate:
    profile: str
    duration: Estimate
    features: Features
    kv_tokens_after: int


@dataclass(frozen=True)
class ExpertRolloutModel:
    model: ModelShape
    hardware: Hardware
    rates: InferenceRates
    runtime_revision: str
    calibration: tuple[PhaseCalibration, ...] = ()
    version: str = "expert-rollout-v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "calibration", tuple(self.calibration))
        if not self.version or not self.runtime_revision:
            raise ValueError("Model version is required")

    def profile_key(self, config: RolloutConfig) -> str:
        return fingerprint(
            code_revision(),
            self.version,
            self.runtime_revision,
            self.model,
            self.hardware,
            self.rates,
            config,
        )

    def estimate(
        self,
        config: RolloutConfig,
        state: ServiceState,
        ready_requests: tuple[ServiceSlice, ...],
    ) -> ServiceEstimate:
        slices = tuple(ready_requests)
        if not slices or len({s.request_id for s in slices}) != len(slices):
            raise ValueError("A service iteration needs distinct requests")
        profile = self.profile_key(config)
        tokens = sum(s.query_tokens for s in slices)
        prefill = sum(s.query_tokens for s in slices if s.kind == "prefill")
        decode = sum(s.kind == "decode" for s in slices)
        pairs = sum(
            s.query_tokens * s.context_tokens
            + s.query_tokens * (s.query_tokens + 1) / 2
            for s in slices
        )
        kv_after = state.kv_tokens + tokens
        features = (
            ("prefill_tokens", float(prefill)),
            ("decode_sequences", float(decode)),
            ("attention_pairs", pairs),
            ("kv_tokens", float(state.kv_tokens)),
        )
        if (
            len(slices) > config.max_batch_sequences
            or tokens > config.max_batch_tokens
            or kv_after > config.kv_capacity_tokens
        ):
            return ServiceEstimate(
                profile,
                Estimate(
                    None,
                    "unsupported",
                    "infeasible",
                    ("Service batch exceeds sequence/token/KV limit",),
                ),
                features,
                kv_after,
            )
        m, tp = self.model, config.tp
        if m.attention_heads % tp:
            return ServiceEstimate(
                profile,
                Estimate(
                    None,
                    "unsupported",
                    "infeasible",
                    ("TP must divide query attention heads",),
                ),
                features,
                kv_after,
            )
        seconds, lower_bound = iteration_cost(
            self.model, self.hardware, self.rates, config, slices, kv_after
        )
        estimate = apply_calibration(
            seconds, profile, "service", features, self.calibration
        )
        estimate = replace(
            estimate,
            feasibility="infeasible"
            if lower_bound > self.hardware.memory_gb * 1e9
            else "unknown",
            notes=estimate.notes
            + (
                "Uniform MoE routing and mixed-batch roofline prior; target service traces required",
                "Queue, external tools and admission latency are excluded",
                "Weight plus KV memory is a lower bound; temporary buffers are not certified",
            ),
        )
        return ServiceEstimate(profile, estimate, features, kv_after)
