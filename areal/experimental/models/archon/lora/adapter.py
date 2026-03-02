"""AdapterModule protocol and utilities for LoRA parameter management.

Reference: torchtune/torchtune/modules/peft/_utils.py
Provides utilities for extracting, filtering, and managing adapter parameters.
"""

from typing import Protocol, runtime_checkable

import torch.nn as nn


@runtime_checkable
class AdapterModule(Protocol):
    """Protocol for modules that contain adapter parameters.

    Any module implementing this protocol should provide an adapter_params()
    method that returns a list of parameter names (relative to the module)
    that should be treated as trainable adapters.
    """

    def adapter_params(self) -> list[str]:
        """Return list of adapter parameter names relative to this module.

        Returns:
            List of parameter names (e.g., ["lora_a.weight", "lora_b.weight"])
        """
        ...


def get_adapter_params(model: nn.Module) -> dict[str, nn.Parameter]:
    """Extract all adapter parameters from model using AdapterModule protocol.

    Walks through all modules in the model and collects parameters from modules
    that implement the AdapterModule protocol.

    Args:
        model: Model to extract adapter parameters from

    Returns:
        Dictionary mapping fully-qualified parameter names to Parameter objects
    """
    adapter_params = {}

    for module_name, module in model.named_modules():
        # Check if module implements AdapterModule protocol
        if isinstance(module, AdapterModule):
            # Get adapter param names relative to this module
            current_adapter_params = module.adapter_params()

            # Collect adapter parameters with fully-qualified names
            for param_name, param in module.named_parameters(recurse=True):
                if param_name in current_adapter_params:
                    # Construct fully-qualified name
                    full_key = (
                        f"{module_name}.{param_name}" if module_name else param_name
                    )
                    adapter_params[full_key] = param

    return adapter_params


def set_trainable_params(model: nn.Module, adapter_param_names: set[str]) -> None:
    """Freeze all parameters except those in adapter_param_names.

    This is used to set up LoRA training where only adapter parameters
    should be trainable while base model parameters are frozen.

    Args:
        model: Model to configure
        adapter_param_names: Set of fully-qualified parameter names to keep trainable
    """
    for name, param in model.named_parameters():
        param.requires_grad_(name in adapter_param_names)


def get_adapter_state_dict(state_dict: dict, device: str = "cpu") -> dict:
    """Filter state dict to only adapter parameters.

    Used for saving adapter-only checkpoints (without base model weights).

    Args:
        state_dict: Full model state dict
        device: Device to move parameters to (default: "cpu")

    Returns:
        Filtered state dict containing only adapter parameters
    """

    def is_adapter_key(k: str) -> bool:
        """Check if key is an adapter parameter."""
        return "lora_a" in k or "lora_b" in k

    return {k: v.to(device) for k, v in state_dict.items() if is_adapter_key(k)}


def disable_adapter(model: nn.Module) -> None:
    """Disable LoRA adapters in all LoRALinear modules.

    Sets the `disabled` flag to True, causing forward passes to only use
    the base weights. This is useful for reference models in DPO/PPO.

    Args:
        model: Model containing LoRALinear modules
    """
    for module in model.modules():
        if isinstance(module, AdapterModule) and hasattr(module, "disabled"):
            module.disabled = True


def enable_adapter(model: nn.Module) -> None:
    """Enable LoRA adapters in all LoRALinear modules.

    Sets the `disabled` flag to False, enabling LoRA contributions
    during forward passes.

    Args:
        model: Model containing LoRALinear modules
    """
    for module in model.modules():
        if isinstance(module, AdapterModule) and hasattr(module, "disabled"):
            module.disabled = False
