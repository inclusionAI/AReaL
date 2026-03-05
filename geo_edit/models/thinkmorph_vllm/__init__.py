# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""
ThinkMorph vLLM Inference Package

Simplified inference wrapper for ThinkMorph using vLLM backend.
Provides efficient interleaved image-text generation capabilities.
"""

from .inference import VLLMInterleavedInference
from .configs import (
    DEFAULT_CONFIG,
    FAST_CONFIG,
    HIGH_QUALITY_CONFIG,
    REASONING_CONFIG,
    EDITING_CONFIG,
)

__version__ = "0.1.0"
__all__ = [
    "VLLMInterleavedInference",
    "DEFAULT_CONFIG",
    "FAST_CONFIG",
    "HIGH_QUALITY_CONFIG",
    "REASONING_CONFIG",
    "EDITING_CONFIG",
]
