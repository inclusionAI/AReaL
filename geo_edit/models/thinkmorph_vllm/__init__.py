# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""
ThinkMorph Inference Package

Manual loading inference wrapper for ThinkMorph (like original implementation).
Provides interleaved image-text generation capabilities without vLLM dependency.
"""

from .inference import ThinkMorphInference, VLLMInterleavedInference
from .inference_batch import ThinkMorphBatchInference, BatchInferenceConfig, run_batch_evaluation
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
    "VLLMInterleavedInference",
    "ThinkMorphBatchInference",
    "BatchInferenceConfig",
    "run_batch_evaluation",
    "DEFAULT_CONFIG",
    "FAST_CONFIG",
    "HIGH_QUALITY_CONFIG",
    "REASONING_CONFIG",
    "EDITING_CONFIG",
]
