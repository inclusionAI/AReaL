# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""
ThinkMorph Inference Package

Manual loading inference wrapper for ThinkMorph (like original implementation).
Provides interleaved image-text generation capabilities without vLLM dependency.

Supports:
- Single sample inference: ThinkMorphInference
- Multi-GPU Data Parallel: ThinkMorphDP (supports multiple models per GPU)
"""

from .inference import ThinkMorphInference, VLLMInterleavedInference
from .inference_dp import ThinkMorphDP
from .configs import (
    DEFAULT_CONFIG,
    FAST_CONFIG,
    HIGH_QUALITY_CONFIG,
    REASONING_CONFIG,
    EDITING_CONFIG,
)

__version__ = "0.2.0"
__all__ = [
    # Single inference
    "ThinkMorphInference",
    "VLLMInterleavedInference",
    # Multi-GPU DP inference
    "ThinkMorphDP",
    # Configs
    "DEFAULT_CONFIG",
    "FAST_CONFIG",
    "HIGH_QUALITY_CONFIG",
    "REASONING_CONFIG",
    "EDITING_CONFIG",
]
