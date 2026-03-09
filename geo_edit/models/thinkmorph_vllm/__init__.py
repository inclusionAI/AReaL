# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""
ThinkMorph Inference Package

Manual loading inference wrapper for ThinkMorph (like original implementation).
Provides interleaved image-text generation capabilities without vLLM dependency.

Supports:
- Single sample inference: ThinkMorphInference
- Batch inference: ThinkMorphBatchInference
- Multi-GPU Data Parallel: ThinkMorphDP
"""

from .inference import ThinkMorphInference, VLLMInterleavedInference
from .inference_batch import ThinkMorphBatchInference, BatchInferenceConfig, run_batch_evaluation
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
    # Batch inference
    "ThinkMorphBatchInference",
    "BatchInferenceConfig",
    "run_batch_evaluation",
    # Multi-GPU DP inference
    "ThinkMorphDP",
    # Configs
    "DEFAULT_CONFIG",
    "FAST_CONFIG",
    "HIGH_QUALITY_CONFIG",
    "REASONING_CONFIG",
    "EDITING_CONFIG",
]
