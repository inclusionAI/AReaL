# SPDX-License-Identifier: Apache-2.0

"""CPU-only model inputs, separate from the training/launcher config schema."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, fields
from typing import Any


def positive(name: str, value: float) -> None:
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class ParallelLayout:
    dp: int = 1
    tp: int = 1
    pp: int = 1
    cp: int = 1
    ep: int = 1
    etp: int = 1

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{field.name} must be a positive integer")
        if self.world_size % (self.pp * self.ep * self.etp):
            raise ValueError("Expert layout does not divide attention world size")

    @property
    def world_size(self) -> int:
        return self.dp * self.tp * self.pp * self.cp

    @property
    def expert_dp(self) -> int:
        return self.world_size // (self.pp * self.ep * self.etp)

    @classmethod
    def from_allocation(cls, allocation: str) -> tuple[str, ParallelLayout]:
        """Read AReaL's canonical hybrid MoE notation without importing engines."""
        text = allocation.lower().replace(" ", "")
        if ":(" in text and text.endswith(")"):
            text = text.replace(":(", ":", 1)[:-1]
        match = re.fullmatch(
            r"(megatron|archon):attn:([dptc0-9]+)\|ffn:([dpte0-9]+)", text
        )
        if match is None:
            raise ValueError(
                "Use canonical backend:(attn:d...|ffn:d...e...) allocation"
            )

        def parse(text: str) -> dict[str, int]:
            parts = re.findall(r"([a-z])(\d+)", text)
            if "".join(k + n for k, n in parts) != text or len(
                {k for k, _ in parts}
            ) != len(parts):
                raise ValueError("Malformed or repeated parallel dimension")
            return {k: int(n) for k, n in parts}

        backend, attn_text, ffn_text = match.groups()
        attn, ffn = parse(attn_text), parse(ffn_text)
        layout = cls(
            dp=attn.get("d", 1),
            tp=attn.get("t", 1),
            pp=attn.get("p", 1),
            cp=attn.get("c", 1),
            ep=ffn.get("e", 1),
            etp=ffn.get("t", 1),
        )
        if (
            ffn.get("p", layout.pp) != layout.pp
            or ffn.get("d", layout.expert_dp) != layout.expert_dp
        ):
            raise ValueError("Attention and FFN layouts describe different world sizes")
        return backend, layout


@dataclass(frozen=True)
class Hardware:
    """Per-GPU rates. Bandwidths are GB/s, one direction, not bidirectional sums.

    Inter-node bandwidth is an explicit scenario input, not inferred from GPU type.
    Named bundled presets are reference systems, not an assertion about a trial.
    """

    name: str
    bf16_tflops: float
    hbm_gbps: float
    intra_node_gbps: float
    inter_node_gbps: float
    gpus_per_node: int
    memory_gb: float
    intra_latency_us: float = 3
    inter_latency_us: float = 12

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Hardware name cannot be empty")
        for field in fields(self):
            if field.name != "name":
                positive(field.name, getattr(self, field.name))
        if type(self.gpus_per_node) is not int:
            raise ValueError("gpus_per_node must be an integer")


@dataclass(frozen=True)
class ModelShape:
    """Qwen3-MoE dimensions, loaded from a bundled, source-linked HF config."""

    hidden_size: int
    intermediate_size: int
    layers: int
    attention_heads: int
    kv_heads: int
    head_dim: int
    experts: int
    top_k: int
    vocab_size: int
    tied_embeddings: bool = False

    def __post_init__(self) -> None:
        for field in fields(self):
            if field.name != "tied_embeddings":
                value = getattr(self, field.name)
                if type(value) is not int or value < 1:
                    raise ValueError(f"{field.name} must be a positive integer")
        if self.top_k > self.experts or self.attention_heads % self.kv_heads:
            raise ValueError("Invalid top_k or grouped-query attention dimensions")

    @property
    def attention_parameters_per_layer(self) -> int:
        return (
            2
            * self.hidden_size
            * self.head_dim
            * (self.attention_heads + self.kv_heads)
        )

    @property
    def expert_parameters_per_layer(self) -> int:
        return 3 * self.hidden_size * self.intermediate_size * self.experts

    @property
    def router_parameters_per_layer(self) -> int:
        return self.hidden_size * self.experts

    @property
    def head_parameters(self) -> int:
        return self.hidden_size * self.vocab_size

    @property
    def total_parameters(self) -> int:
        norms = 2 * self.hidden_size + 2 * self.head_dim
        return (
            self.layers
            * (
                self.attention_parameters_per_layer
                + self.expert_parameters_per_layer
                + self.router_parameters_per_layer
                + norms
            )
            + (1 if self.tied_embeddings else 2) * self.head_parameters
            + self.hidden_size
        )


@dataclass(frozen=True)
class ActorConfig:
    backend: str
    parallel: ParallelLayout
    max_tokens_per_mb: int = 32768
    n_mbs: int = 1
    n_mbs_divisor: int = 1
    granularity: int = 1
    packing_algorithm: str = "ffd"
    pad_to_maximum: bool = False
    dtype: str = "bfloat16"
    gradient_checkpointing: bool = False
    checkpoint_recompute_fraction: float = 1.0
    checkpoint_note: str = "Full layer recomputation when enabled"
    expert_dispatcher: str = "alltoall"
    reshard_after_forward: bool = True
    reshard_head_after_forward: bool = False
    ppo_n_minibatches: int = 1
    recompute_logprob: bool = False
    use_decoupled_loss: bool = False
    prox_logp_method: str = "recompute"
    distributed_optimizer: bool = True
    grad_bytes: int = 4
    overlap_grad_reduce: bool = False
    compile: bool = False
    enable_tree_training: bool = False
    pipeline_layers: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.pipeline_layers is not None:
            object.__setattr__(self, "pipeline_layers", tuple(self.pipeline_layers))
        if self.backend not in {"megatron", "archon"}:
            raise ValueError("Actor cost model supports megatron and archon")
        for name in (
            "pad_to_maximum",
            "gradient_checkpointing",
            "reshard_after_forward",
            "reshard_head_after_forward",
            "recompute_logprob",
            "use_decoupled_loss",
            "distributed_optimizer",
            "overlap_grad_reduce",
            "compile",
            "enable_tree_training",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        if self.parallel.etp != 1:
            raise ValueError(
                "Expert tensor parallelism requires another compute component"
            )
        if self.expert_dispatcher not in {"alltoall", "hierarchical"}:
            raise ValueError("Supported expert_dispatcher: alltoall or hierarchical")
        if (
            not math.isfinite(self.checkpoint_recompute_fraction)
            or not 0 <= self.checkpoint_recompute_fraction <= 1
        ):
            raise ValueError(
                "checkpoint_recompute_fraction must be between zero and one"
            )
        for name in (
            "max_tokens_per_mb",
            "n_mbs",
            "n_mbs_divisor",
            "granularity",
            "ppo_n_minibatches",
            "grad_bytes",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.dtype not in {"bfloat16", "bf16", "float16", "fp16"}:
            raise ValueError(
                "This calibration supports BF16/FP16 full-parameter training"
            )
        if self.packing_algorithm != "ffd" or self.enable_tree_training:
            raise ValueError(
                "FFD packed sequences are supported; tree/KK require another packing component"
            )
        if self.prox_logp_method not in {
            "recompute",
            "metrics",
            "loglinear",
            "reuse_train_logp",
        }:
            raise ValueError("Unknown AReaL proximal logprob method")
        if (
            self.use_decoupled_loss
            and self.prox_logp_method == "reuse_train_logp"
            and self.ppo_n_minibatches != 1
        ):
            raise ValueError("reuse_train_logp requires ppo_n_minibatches=1")
        if self.pipeline_layers is not None and (
            len(self.pipeline_layers) != self.parallel.pp
            or any(type(n) is not int or n < 1 for n in self.pipeline_layers)
        ):
            raise ValueError(
                "pipeline_layers must contain one positive layer count per PP stage"
            )

    @property
    def compute_prox_logp(self) -> bool:
        return (
            self.prox_logp_method in {"recompute", "metrics"}
            if self.use_decoupled_loss
            else self.recompute_logprob
        )

    @property
    def minimum_microbatches(self) -> int:
        pipeline_min = (
            2 * self.parallel.pp if self.backend == "megatron" else self.parallel.pp
        )
        return max(self.n_mbs, pipeline_min if self.parallel.pp > 1 else 1)

    @classmethod
    def from_areal(cls, data: dict[str, Any]) -> ActorConfig:
        """Accept the resolved experiment config or just its actor section.

        Interpolations must be resolved by AReaL/OmegaConf before passing a config.
        train_dataset.batch_size is deliberately NOT interpreted as actor n_seqs.
        """
        actor = data.get("actor", data)
        backend, parallel = ParallelLayout.from_allocation(actor["backend"])
        mb = actor.get("mb_spec", {})
        megatron, archon = actor.get("megatron", {}), actor.get("archon", {})
        ddp = megatron.get("ddp", {})
        optimizer = actor.get("optimizer", {})
        if optimizer is None or optimizer.get("type", "adam") != "adam":
            raise ValueError(
                "Bundled optimizer component requires full-precision Adam states"
            )
        if backend == "archon" and actor.get("optimizer_dtype", "float32") != "float32":
            raise ValueError("BF16 optimizer states need another optimizer component")
        if backend == "megatron" and megatron.get("enable_fp32_lm_head", False):
            raise ValueError("FP32 LM head needs a separate operator profile")
        if actor.get("use_lora", False):
            raise ValueError("Bundled model supports full-parameter training, not LoRA")
        if mb.get("max_tokens_per_mb") is None:
            raise ValueError(
                "Bundled packing component requires explicit mb_spec.max_tokens_per_mb"
            )
        if (backend == "megatron" and megatron.get("fp8_config")) or (
            backend == "archon"
            and (archon.get("fp8_config") or {}).get("mode", "disabled") != "disabled"
        ):
            raise ValueError("FP8 needs a separate operator/calibration profile")
        if actor.get("enable_offload", data.get("enable_offload", False)) or archon.get(
            "offload_params", False
        ):
            raise ValueError(
                "CPU offload requires an explicit transfer component; bundled calibration excludes it"
            )
        fraction, note = 1.0, "Full layer recomputation when enabled"
        if backend == "archon":
            mode = archon.get("ac_mode", "selective")
            option = str(archon.get("selective_ac_option", "op"))
            if mode == "none":
                fraction = 0.0
            elif mode == "selective":
                fraction = (
                    1 / int(option) if option.isdigit() and int(option) > 0 else 0.25
                )
                note = "Selective checkpoint recompute fraction is an analytical approximation; override if known"
            elif mode == "memory_budget":
                fraction = 1 - float(archon.get("ac_memory_budget", 0.5))
                note = "Compiler memory budget is not a compute fraction; 1-budget is a sensitivity prior"
            elif mode != "full":
                raise ValueError("Unknown Archon checkpoint mode")
        elif megatron.get("recompute_granularity", "full") == "selective":
            fraction, note = (
                0.25,
                "Selective Megatron recompute uses a 0.25 layer-service prior",
            )
        dispatcher = "alltoall"
        if (
            backend == "megatron"
            and megatron.get("bridge_type", "mbridge") != "megatron-bridge"
        ):
            if megatron.get("moe_enable_deepep", False):
                dispatcher = "hierarchical"
            elif megatron.get("moe_token_dispatcher_type", "alltoall") != "alltoall":
                raise ValueError(
                    "Non-alltoall dispatcher needs an explicit communication component"
                )
        return cls(
            backend,
            parallel,
            **{
                key: mb[key]
                for key in (
                    "max_tokens_per_mb",
                    "n_mbs",
                    "n_mbs_divisor",
                    "granularity",
                    "packing_algorithm",
                )
                if mb.get(key) is not None
            },
            **{
                key: actor[key]
                for key in (
                    "pad_to_maximum",
                    "dtype",
                    "gradient_checkpointing",
                    "ppo_n_minibatches",
                    "recompute_logprob",
                    "use_decoupled_loss",
                    "prox_logp_method",
                    "enable_tree_training",
                )
                if key in actor
            },
            distributed_optimizer=ddp.get("use_distributed_optimizer", True),
            expert_dispatcher=dispatcher,
            checkpoint_recompute_fraction=fraction,
            checkpoint_note=note,
            reshard_head_after_forward=archon.get(
                "reshard_after_forward_policy", "default"
            )
            == "always",
            reshard_after_forward=archon.get("reshard_after_forward_policy", "default")
            != "never"
            and not (
                archon.get("reshard_after_forward_policy", "default") == "default"
                and parallel.pp > 1
            ),
            overlap_grad_reduce=ddp.get("overlap_grad_reduce", False),
            grad_bytes=(
                4 if actor.get("grad_reduce_dtype", "float32") == "float32" else 2
            )
            if backend == "archon"
            else (4 if ddp.get("grad_reduce_in_fp32", True) else 2),
            compile=archon.get("enable_compile", False)
            if backend == "archon"
            else bool(megatron.get("cuda_graph_impl")),
        )
