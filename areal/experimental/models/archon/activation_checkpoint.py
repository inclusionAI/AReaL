# Adapted from torchtitan: torchtitan/distributed/activation_checkpoint.py

from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)

from areal.utils import logging

logger = logging.getLogger("ArchonActivationCheckpoint")

# Op-level selective AC: ops to save instead of recompute
_OP_SAC_SAVE_LIST = {
    torch.ops.aten.mm.default,
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
    torch.ops.aten._scaled_dot_product_cudnn_attention.default,
    torch.ops.aten._scaled_dot_product_attention_math.default,
    torch.ops.aten._scaled_dot_product_fused_attention_overrideable.default,
    torch.ops.aten.max.default,
}

# Global counter for layer-level selective AC
_layer_sac_count = 0


@dataclass
class ActivationCheckpointConfig:
    """Activation checkpointing configuration for Archon Engine.

    Attributes:
        mode: AC mode - 'none' (disabled), 'selective' (layer-based), 'full' (all layers).
        selective_ac_option: For selective mode:
            - Integer string (e.g., '1'): Apply AC every N layers
            - 'op': Use op-level selective AC (advanced)
        preserve_rng_state: Whether to preserve RNG state for deterministic output.
            Setting to True may slow down training but ensures reproducibility.
    """

    mode: Literal["selective", "full", "none"] = "none"
    selective_ac_option: str = "1"
    preserve_rng_state: bool = False

    def __post_init__(self):
        valid_modes = ("none", "full", "selective")
        if self.mode not in valid_modes:
            raise ValueError(
                f"Invalid AC mode: {self.mode}. Valid modes: {valid_modes}"
            )

        if self.mode == "selective":
            if not (
                self.selective_ac_option.isdigit() or self.selective_ac_option == "op"
            ):
                raise ValueError(
                    f"Invalid selective_ac_option: {self.selective_ac_option}. "
                    "Must be a positive integer string (e.g., '1') or 'op'."
                )


def apply_activation_checkpointing(
    model: nn.Module,
    ac_config: ActivationCheckpointConfig,
) -> None:
    """Apply activation checkpointing to the model.

    This function wraps TransformerBlocks with checkpoint wrappers based on
    the configuration. Must be called AFTER TP parallelization and BEFORE FSDP.

    Args:
        model: The model to apply AC to. Must have a `layers` attribute
               (ModuleDict of TransformerBlocks).
        ac_config: Activation checkpointing configuration.

    Raises:
        ValueError: If AC mode is invalid or model doesn't have layers attribute.
    """
    if ac_config.mode == "none":
        logger.debug("Activation checkpointing is disabled")
        return

    if not hasattr(model, "layers"):
        raise ValueError(
            "Model must have a 'layers' attribute (ModuleDict) to apply AC"
        )

    global _layer_sac_count
    _layer_sac_count = 0  # Reset counter for each model

    for layer_id, transformer_block in model.layers.items():
        if ac_config.mode == "full":
            wrapped = _apply_full_ac(transformer_block, ac_config)
        elif ac_config.mode == "selective":
            if ac_config.selective_ac_option.isdigit():
                wrapped = _apply_layer_sac(transformer_block, ac_config)
            elif ac_config.selective_ac_option == "op":
                wrapped = _apply_op_sac(transformer_block, ac_config)
            else:
                raise ValueError(
                    f"Invalid selective_ac_option: {ac_config.selective_ac_option}"
                )
        else:
            raise ValueError(f"Invalid AC mode: {ac_config.mode}")

        model.layers[layer_id] = wrapped

    logger.info(f"Applied {ac_config.mode} activation checkpointing to the model")


def _apply_full_ac(
    module: nn.Module,
    ac_config: ActivationCheckpointConfig,
) -> nn.Module:
    """Apply full activation checkpointing to the module.

    Args:
        module: The module (TransformerBlock) to wrap.
        ac_config: Activation checkpointing configuration.

    Returns:
        The wrapped module with full activation checkpointing.
    """
    return ptd_checkpoint_wrapper(
        module,
        preserve_rng_state=ac_config.preserve_rng_state,
    )


def _apply_layer_sac(
    module: nn.Module,
    ac_config: ActivationCheckpointConfig,
) -> nn.Module:
    """Apply layer-level selective activation checkpointing.

    This applies AC to every Nth layer based on selective_ac_option.
    For example, with selective_ac_option="2", AC is applied to layers 2, 4, 6, etc.

    Args:
        module: The module (TransformerBlock) to potentially wrap.
        ac_config: Activation checkpointing configuration.

    Returns:
        The module, potentially wrapped with activation checkpointing.
    """
    global _layer_sac_count
    _layer_sac_count += 1

    ac_freq = int(ac_config.selective_ac_option)
    if not ac_freq or _layer_sac_count % ac_freq == 0:
        return ptd_checkpoint_wrapper(
            module,
            preserve_rng_state=ac_config.preserve_rng_state,
        )
    return module


def _apply_op_sac(
    module: nn.Module,
    ac_config: ActivationCheckpointConfig,
) -> nn.Module:
    """Apply op-level selective activation checkpointing.

    This uses a custom policy to decide which operations to save vs recompute.
    The policy saves outputs of expensive ops (matmuls, attention) while
    recomputing cheaper ops.

    Args:
        module: The module (TransformerBlock) to wrap.
        ac_config: Activation checkpointing configuration.

    Returns:
        The wrapped module with op-level selective activation checkpointing.
    """
    from torch.utils.checkpoint import (
        CheckpointPolicy,
        create_selective_checkpoint_contexts,
    )

    def _get_custom_policy(meta):
        def _custom_policy(ctx, func, *args, **kwargs):
            # Always save CPU-to-CUDA transfers
            if (
                func == torch.ops.aten._to_copy.default
                and len(args) > 0
                and hasattr(args[0], "device")
                and "cuda" in str(args[0].device)
                and "device" in kwargs
                and str(kwargs["device"]) == "cpu"
            ):
                return CheckpointPolicy.MUST_SAVE

            mode = "recompute" if ctx.is_recompute else "forward"
            mm_count_key = f"{mode}_mm_count"
            if func == torch.ops.aten.mm.default:
                meta[mm_count_key] += 1

            # Save output of all ops in save list, except every second mm
            to_save = func in _OP_SAC_SAVE_LIST and not (
                func == torch.ops.aten.mm.default and meta[mm_count_key] % 2 == 0
            )
            return (
                CheckpointPolicy.MUST_SAVE
                if to_save
                else CheckpointPolicy.PREFER_RECOMPUTE
            )

        return _custom_policy

    def selective_checkpointing_context_fn():
        meta = defaultdict(int)
        return create_selective_checkpoint_contexts(_get_custom_policy(meta))

    return ptd_checkpoint_wrapper(
        module,
        context_fn=selective_checkpointing_context_fn,
        preserve_rng_state=ac_config.preserve_rng_state,
    )


__all__ = [
    "ActivationCheckpointConfig",
    "apply_activation_checkpointing",
]
