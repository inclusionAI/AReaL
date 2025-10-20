"""Model patcher to inject custom positional encoding into LLaMA-Factory models.

This module provides the hook to replace standard positional encodings with
the custom periodic RoPE implementation during model loading.
"""

import logging
from typing import Optional, Tuple

import torch
from torch import nn
from transformers import PreTrainedModel

from custom_positional_encoding import CustomRotaryPositionalEncoding, CustomPositionalEncoding


logger = logging.getLogger(__name__)


def _retrieve_module_and_parent(root: nn.Module, name: str) -> Tuple[nn.Module, str]:
    """Helper to get parent module and attribute name from dotted path."""
    parent = root
    parts = name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def inject_custom_positional_encoding_factory(
    model: PreTrainedModel,
    max_length: Optional[int] = None,
    dropout: float = 0.0,
    learned_scaling: bool = True,
    period: int = 128000,
) -> Tuple[nn.Module, str]:
    """Replace the model's positional embedding with custom periodic encoding.
    
    This function searches for rotary positional embeddings (RoPE) in the model
    and replaces them with the custom periodic implementation that supports
    100M+ context windows via periodic encoding and dimension masking.
    
    Args:
        model: The Hugging Face pretrained model
        max_length: Maximum sequence length (defaults to model's config)
        dropout: Dropout rate for positional embeddings
        learned_scaling: Whether to use learnable alpha/beta parameters
        period: Periodicity for positional encoding (default: 128000)
        
    Returns:
        Tuple of (custom_module, replaced_name) indicating what was replaced
        
    Raises:
        RuntimeError: If no suitable positional embedding is found
    """
    config = model.config
    
    # Determine hidden size
    hidden_size = getattr(config, "hidden_size", None) or getattr(config, "n_embd", None)
    if hidden_size is None:
        raise ValueError("Unable to determine the model's hidden size for PE replacement.")

    # Determine max length
    max_length = (
        max_length
        or getattr(config, "max_position_embeddings", None)
        or getattr(config, "n_positions", None)
        or getattr(config, "n_ctx", None)
    )
    if max_length is None:
        raise ValueError("Unable to infer `max_length` for positional encoding.")

    primary_device = next(model.parameters()).device
    primary_dtype = next(model.parameters()).dtype

    # First, try to replace traditional positional embeddings (for GPT-2 style models)
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            num_embeddings = getattr(module, "num_embeddings", None)
            embedding_dim = getattr(module, "embedding_dim", None)
            if num_embeddings == max_length and embedding_dim == hidden_size:
                custom_module = CustomPositionalEncoding(
                    hidden_size=hidden_size,
                    max_length=max_length,
                    dropout=dropout,
                    learned_scaling=learned_scaling,
                ).to(device=primary_device, dtype=primary_dtype)
                
                parent_module, attribute_name = _retrieve_module_and_parent(model, name)
                setattr(parent_module, attribute_name, custom_module)
                logger.info(f"✓ Replaced positional embedding module at {name}")
                logger.info(f"  Max length: {max_length}, Hidden size: {hidden_size}")
                return custom_module, name

    # Try to replace rotary embeddings (for Qwen, LLaMA, etc.)
    for name, module in model.named_modules():
        cls_name = module.__class__.__name__.lower()
        if "rotary" in cls_name and hasattr(module, "inv_freq"):
            inv_freq = module.inv_freq.detach().to(device=primary_device, dtype=torch.float32)
            rope_max = max_length or getattr(module, "max_seq_len_cached", None)
            if rope_max is None:
                rope_max = max_length or config.max_position_embeddings

            attention_scaling = getattr(module, "attention_scaling", 1.0)
            
            custom_rotary = CustomRotaryPositionalEncoding(
                inv_freq=inv_freq,
                max_length=rope_max,
                attention_scaling=attention_scaling,
                dropout=0.0,  # Disable dropout to avoid gradient checkpointing issues
                learned_scaling=learned_scaling,
                period=period,  # Custom: periodic encoding
            ).to(device=primary_device, dtype=primary_dtype)

            parent_module, attribute_name = _retrieve_module_and_parent(model, name)
            setattr(parent_module, attribute_name, custom_rotary)
            
            logger.info(f"✓ Replaced rotary positional embedding at {name}")
            logger.info(f"  Head dim: {custom_rotary.head_dim}")
            logger.info(f"  Max seq len: {rope_max}")
            logger.info(f"  Period: {period}")
            logger.info(f"  Learned scaling: {learned_scaling}")
            logger.info(f"  Dimension masking: ~0.8 * {custom_rotary.head_dim} = {int(0.8 * custom_rotary.head_dim)}")
            
            return custom_rotary, name

    raise RuntimeError(
        "Failed to locate a positional embedding to replace. "
        "The model may not have a standard RoPE or positional embedding layer."
    )


def verify_custom_encoding_applied(model: PreTrainedModel) -> bool:
    """Verify that custom positional encoding has been applied to the model.
    
    Args:
        model: The model to check
        
    Returns:
        True if custom encoding is found, False otherwise
    """
    for name, module in model.named_modules():
        if isinstance(module, (CustomRotaryPositionalEncoding, CustomPositionalEncoding)):
            logger.info(f"✓ Verified: Custom encoding active at {name}")
            if hasattr(module, 'period'):
                logger.info(f"  Period: {module.period}")
            if hasattr(module, 'learned_scaling'):
                logger.info(f"  Learned scaling: {module.learned_scaling}")
            return True
    
    logger.warning("⚠ No custom positional encoding found in model!")
    return False


def count_custom_encoding_parameters(model: PreTrainedModel) -> int:
    """Count the number of trainable parameters in custom encodings.
    
    Args:
        model: The model to analyze
        
    Returns:
        Number of trainable parameters in custom positional encodings
    """
    total_params = 0
    for name, module in model.named_modules():
        if isinstance(module, (CustomRotaryPositionalEncoding, CustomPositionalEncoding)):
            for param_name, param in module.named_parameters():
                if param.requires_grad:
                    total_params += param.numel()
                    logger.debug(f"  {name}.{param_name}: {param.numel()} params")
    
    if total_params > 0:
        logger.info(f"Custom positional encoding trainable parameters: {total_params:,}")
    
    return total_params
