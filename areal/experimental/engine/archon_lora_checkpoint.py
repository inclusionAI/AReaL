"""LoRA adapter checkpoint I/O in PEFT format.

This module provides functions to save and load LoRA adapters in PEFT-compatible
format for HuggingFace ecosystem interoperability.

PEFT checkpoint structure:
    adapter_checkpoint/
    ├── adapter_model.safetensors  # LoRA weights only
    └── adapter_config.json         # PEFT configuration

Reference: peft/src/peft/utils/save_and_load.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file

from areal.experimental.models.archon.lora.adapter import get_adapter_params
from areal.utils import logging

if TYPE_CHECKING:
    from areal.experimental.engine.archon_engine import ArchonEngine

logger = logging.getLogger("LoRACheckpoint")


def save_lora_adapter(
    engine: "ArchonEngine",
    path: str,
    base_model_path: str | None = None,
) -> None:
    """Save LoRA adapter in PEFT format.

    Creates two files:
    - adapter_model.safetensors: LoRA weights (lora_a, lora_b)
    - adapter_config.json: PEFT configuration

    Args:
        engine: ArchonEngine instance with LoRA-enabled model
        path: Directory path to save adapter checkpoint
        base_model_path: Optional path to base model (for config reference)

    Raises:
        RuntimeError: If LoRA is not enabled on engine
    """
    if engine.lora_config is None:
        raise RuntimeError("Cannot save LoRA adapter: LoRA not enabled on engine")

    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    if rank == 0:
        os.makedirs(path, exist_ok=True)
        logger.info(f"Saving LoRA adapter to {path}")

    # Extract adapter parameters from model
    adapter_params = get_adapter_params(engine.model)

    if not adapter_params:
        logger.warning("No adapter parameters found in model")
        if rank == 0:
            logger.warning("Creating empty adapter checkpoint")

    # Convert to HF format using state dict adapter
    archon_state = {k: v.cpu().detach().clone() for k, v in adapter_params.items()}
    hf_state = engine.state_dict_adapter.to_hf(archon_state)

    # Add PEFT prefix: base_model.model.{key}
    peft_state = {f"base_model.model.{k}": v for k, v in hf_state.items()}

    # Save weights (only rank 0)
    if rank == 0:
        weights_path = os.path.join(path, "adapter_model.safetensors")
        save_file(peft_state, weights_path)
        logger.info(f"Saved {len(peft_state)} adapter tensors to {weights_path}")

        # Determine target modules from actual adapter parameters
        target_modules = set()
        for key in adapter_params:
            parts = key.split(".")
            for i, part in enumerate(parts):
                if part in ("lora_a", "lora_b") and i > 0:
                    module_name = parts[i - 1]
                    target_modules.add(module_name)
                    break

        # Create config copy with actual target modules
        from dataclasses import replace

        lora_config_for_save = replace(
            engine.lora_config, target_modules=sorted(target_modules)
        )

        # Generate adapter config using model-specific state dict adapter
        adapter_config = engine.state_dict_adapter.create_peft_adapter_config(
            lora_config=lora_config_for_save,
            base_model_path=base_model_path,
        )

        config_path = os.path.join(path, "adapter_config.json")
        with open(config_path, "w") as f:
            json.dump(adapter_config, f, indent=2)
        logger.info(f"Saved adapter config to {config_path}")

    # Synchronize all ranks
    if dist.is_initialized():
        dist.barrier()


def load_lora_adapter(
    engine: "ArchonEngine",
    path: str,
    strict: bool = True,
) -> None:
    """Load LoRA adapter from PEFT format checkpoint.

    Args:
        engine: ArchonEngine instance with LoRA-enabled model
        path: Directory path containing adapter checkpoint
        strict: If True, raise error on missing/unexpected keys

    Raises:
        RuntimeError: If LoRA is not enabled on engine
        FileNotFoundError: If adapter checkpoint files not found
        ValueError: If strict=True and keys don't match
    """
    if engine.lora_config is None:
        raise RuntimeError("Cannot load LoRA adapter: LoRA not enabled on engine")

    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    if rank == 0:
        logger.info(f"Loading LoRA adapter from {path}")

    # Load adapter weights
    weights_path = os.path.join(path, "adapter_model.safetensors")
    if not os.path.exists(weights_path):
        # Fallback to .bin format
        weights_path = os.path.join(path, "adapter_model.bin")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"Adapter weights not found at {path}. "
                "Expected adapter_model.safetensors or adapter_model.bin"
            )
        peft_state = torch.load(weights_path, map_location="cpu", weights_only=True)
    else:
        peft_state = load_file(weights_path)

    if rank == 0:
        logger.info(f"Loaded {len(peft_state)} adapter tensors from {weights_path}")

    # Strip PEFT prefix: base_model.model.{key} -> {key}
    hf_state = {}
    for key, value in peft_state.items():
        if key.startswith("base_model.model."):
            hf_key = key.replace("base_model.model.", "", 1)
            hf_state[hf_key] = value
        else:
            hf_state[key] = value

    # Convert from HF format to Archon format
    archon_state = engine.state_dict_adapter.from_hf(hf_state)

    # Get expected adapter keys from model
    expected_adapter_params = get_adapter_params(engine.model)
    expected_keys = set(expected_adapter_params.keys())
    loaded_keys = set(archon_state.keys())

    missing_keys = expected_keys - loaded_keys
    unexpected_keys = loaded_keys - expected_keys

    if missing_keys or unexpected_keys:
        if strict:
            error_msg = []
            if missing_keys:
                error_msg.append(f"Missing keys: {sorted(missing_keys)[:5]}...")
            if unexpected_keys:
                error_msg.append(f"Unexpected keys: {sorted(unexpected_keys)[:5]}...")
            raise ValueError(
                "Adapter checkpoint keys don't match model. " + " ".join(error_msg)
            )
        else:
            if missing_keys and rank == 0:
                logger.warning(
                    f"Missing {len(missing_keys)} adapter keys: "
                    f"{sorted(missing_keys)[:5]}..."
                )
            if unexpected_keys and rank == 0:
                logger.warning(
                    f"Unexpected {len(unexpected_keys)} adapter keys: "
                    f"{sorted(unexpected_keys)[:5]}..."
                )

    # Load adapter weights into model
    loaded_count = 0
    for key, value in archon_state.items():
        if key in expected_adapter_params:
            param = expected_adapter_params[key]
            value = value.to(device=param.device, dtype=param.dtype)
            param.data.copy_(value)
            loaded_count += 1

    if rank == 0:
        logger.info(f"Loaded {loaded_count} adapter parameters into model")

    if dist.is_initialized():
        dist.barrier()


def is_lora_adapter_checkpoint(path: str) -> bool:
    """Check if path contains a PEFT LoRA adapter checkpoint.

    Args:
        path: Directory path to check

    Returns:
        True if path contains adapter_config.json with peft_type="LORA"
    """
    config_path = Path(path) / "adapter_config.json"

    if not config_path.exists():
        return False

    try:
        with open(config_path) as f:
            config = json.load(f)
        return config.get("peft_type") == "LORA"
    except (OSError, json.JSONDecodeError):
        return False
