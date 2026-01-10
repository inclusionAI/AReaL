from typing import Protocol

import torch
import torch.nn as nn

from areal.utils import logging

logger = logging.getLogger("ArchonCompile")


class Compilable(Protocol):
    """Protocol for models that can be compiled with apply_compile."""

    layers: nn.ModuleDict


def apply_compile(model: Compilable) -> None:
    """Apply torch.compile to each TransformerBlock.

    Compiling per-block is more efficient than whole model due to:
    1. Repeated structure allows compilation reuse
    2. Avoids graph breaks from embedding/output layers

    Must be called AFTER TP and AC, BEFORE FSDP.

    Args:
        model: The model to compile. Must have a `layers` attribute (ModuleDict).
    """
    for name, block in model.layers.items():
        model.layers[name] = torch.compile(
            block,
            backend="inductor",
            fullgraph=False,  # Allow graph breaks for dynamic shapes (packed sequences)
        )

    logger.info(f"Compiled {len(model.layers)} TransformerBlocks with torch.compile")
