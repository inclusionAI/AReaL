# SPDX-License-Identifier: Apache-2.0
"""
Triton-based fused linear-cross-entropy kernels for AReaL.

The kernel implementations under :mod:`areal.utils.kernel.kernels` fuse
the matmul with cross-entropy reduction, preserving numerical semantics
while avoiding materialization of the ``[num_tokens, vocab_size]`` logits
tensor. The :class:`LinearCrossEntropy` autograd function exposed below
provides a memory-efficient drop-in replacement for the materialized
``logits = hidden @ weight.T`` followed by softmax / log-softmax /
entropy computation.
"""

from areal.utils.kernel.linear_cross_entropy import (
    LinearCrossEntropy,
    linear_cross_entropy,
)

__all__ = [
    "LinearCrossEntropy",
    "linear_cross_entropy",
]
