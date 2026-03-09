# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""
ThinkMorph Inference Package

Manual loading inference wrapper for ThinkMorph (like original implementation).
Provides interleaved image-text generation capabilities without vLLM dependency.
"""

from .inference import ThinkMorphInference, VLLMInterleavedInference
from .configs import (
    DEFAULT_CONFIG,
    FAST_CONFIG,
    HIGH_QUALITY_CONFIG,
    REASONING_CONFIG,
    EDITING_CONFIG,
)

__version__ = "0.1.0"
__all__ = [
    "ThinkMorphInference",
    "VLLMInterleavedInference",  # Alias for backward compatibility
    "DEFAULT_CONFIG",
    "FAST_CONFIG",
    "HIGH_QUALITY_CONFIG",
    "REASONING_CONFIG",
    "EDITING_CONFIG",
]
