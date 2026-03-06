"""LoRA (Low-Rank Adaptation) infrastructure for Archon engine.

Following torchtune's design patterns for FSDP2 compatibility.
This module provides custom LoRALinear implementation and utilities
for parameter-efficient fine-tuning of large language models.
"""

from areal.experimental.models.archon.lora.adapter import (
    AdapterModule,
    disable_adapter,
    enable_adapter,
    get_adapter_params,
    get_adapter_state_dict,
    set_trainable_params,
)
from areal.experimental.models.archon.lora.lora_linear import LoRALinear

__all__ = [
    "LoRALinear",
    "AdapterModule",
    "get_adapter_params",
    "get_adapter_state_dict",
    "set_trainable_params",
    "disable_adapter",
    "enable_adapter",
]
