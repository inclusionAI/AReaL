from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from areal.experimental.models.archon import ArchonParallelDims
    from areal.experimental.models.archon.activation_checkpoint import (
        ActivationCheckpointConfig,
    )
    from areal.experimental.models.archon.base import (
        BaseModelArgs,
        BaseStateDictAdapter,
    )


# Type alias for parallelize function signature
class ParallelizeFn(Protocol):
    """Protocol for model parallelization functions.

    This protocol defines the signature for functions that apply various
    parallelization strategies to models:
    - TP (Tensor Parallelism)
    - CP (Context Parallelism / Ulysses SP)
    - EP (Expert Parallelism)
    - AC (Activation Checkpointing)
    - FSDP (Fully Sharded Data Parallelism)

    The function receives parallel_dims and internally determines which
    parallelization strategies to apply based on *_enabled flags.
    """

    def __call__(
        self,
        model: nn.Module,
        parallel_dims: ArchonParallelDims,
        param_dtype: torch.dtype = torch.bfloat16,
        reduce_dtype: torch.dtype = torch.float32,
        loss_parallel: bool = True,
        cpu_offload: bool = False,
        reshard_after_forward_policy: str = "default",
        ac_config: ActivationCheckpointConfig | None = None,
        enable_compile: bool = True,
    ) -> nn.Module: ...


@dataclass
class ModelSpec:
    """Specification for a Archon-compatible model."""

    name: str
    model_class: type[nn.Module]
    model_args_class: type[BaseModelArgs]
    state_dict_adapter_class: type[BaseStateDictAdapter]
    parallelize_fn: ParallelizeFn
    supported_model_types: frozenset[str]


_MODEL_SPECS: dict[str, ModelSpec] = {}


def register_model_spec(spec: ModelSpec) -> ModelSpec:
    """Register a ModelSpec."""
    for model_type in spec.supported_model_types:
        if model_type in _MODEL_SPECS:
            existing = _MODEL_SPECS[model_type]
            raise ValueError(
                f"model_type '{model_type}' already registered by {existing.name}"
            )
        _MODEL_SPECS[model_type] = spec
    return spec


def get_model_spec(model_type: str) -> ModelSpec:
    """Get ModelSpec by HF model_type."""
    if model_type not in _MODEL_SPECS:
        available = sorted(_MODEL_SPECS.keys())
        raise KeyError(f"Unknown model_type '{model_type}'. Available: {available}")
    return _MODEL_SPECS[model_type]


def is_supported_model(model_type: str) -> bool:
    """Check if a model_type is supported."""
    return model_type in _MODEL_SPECS


def get_supported_model_types() -> frozenset[str]:
    """Get all supported model types."""
    return frozenset(_MODEL_SPECS.keys())


__all__ = [
    "ModelSpec",
    "register_model_spec",
    "get_model_spec",
    "is_supported_model",
    "get_supported_model_types",
]
