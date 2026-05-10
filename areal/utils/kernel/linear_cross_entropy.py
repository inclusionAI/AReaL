# SPDX-License-Identifier: Apache-2.0
"""
Fused linear + cross-entropy autograd Function.

Dispatches to a Triton kernel that fuses the matmul with cross-entropy
so that the ``[num_tokens, vocab_size]`` logits tensor is never materialised.
"""

from __future__ import annotations

import torch
import torch.distributed as dist


class LinearCrossEntropy(torch.autograd.Function):
    """Fused linear + cross-entropy autograd Function.

    Args:
        hidden: ``(num_tokens, hidden_size)`` contiguous CUDA tensor.
        weight: ``(vocab_size, hidden_size)`` lm-head weight, contiguous CUDA.
        labels: ``(num_tokens,)`` integer label ids on CUDA.
        temperature: softmax temperature; defaults to ``1.0``.
        reduction: only ``"none"`` is supported.
        dist_process_group: optional TP group for vocab-sharded ``weight``.
            ``labels`` must contain *global* vocab ids on every rank.

    Returns:
        ``(logprobs, entropy)`` both shaped ``(num_tokens,)``.
    """

    @staticmethod
    def forward(
        ctx,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        labels: torch.Tensor,
        temperature: float | None = 1.0,
        reduction: str | None = "none",
        dist_process_group: dist.ProcessGroup | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(temperature, float):
            temperature = float(temperature)
        if not isinstance(reduction, str):
            raise TypeError(f"reduction must be str, got {type(reduction)}")

        from areal.utils.kernel import kernels

        REDUCTION = kernels.get_entropy_reduction_enum_number(reduction.lower())

        original_hidden_shape = hidden.shape
        if hidden.dim() != 2:
            hidden = hidden.reshape(-1, hidden.shape[-1])
        if labels.dim() != 1:
            labels = labels.reshape(-1)

        assert hidden.is_cuda and weight.is_cuda and labels.is_cuda, (
            "LinearCrossEntropy requires CUDA inputs"
        )
        assert hidden.is_contiguous() and weight.is_contiguous() and labels.is_contiguous(), (
            "LinearCrossEntropy requires contiguous tensors"
        )

        (
            logprobs,
            entropy,
            _maximum,
            _accumulate,
            _entropy_b,
        ) = kernels.efficient_entropy_forward(
            hidden,
            weight,
            labels,
            REDUCTION,
            temperature,
            dist_process_group,
        )

        ctx.save_for_backward(
            hidden, weight, labels, _maximum, _accumulate, _entropy_b
        )
        ctx.original_hidden_shape = original_hidden_shape
        ctx.REDUCTION = REDUCTION
        ctx.dist_process_group = dist_process_group
        ctx.should_return_fp32_grad = False
        ctx.temperature = temperature

        return logprobs, entropy

    @staticmethod
    def backward(
        ctx,
        dlogprobs: torch.Tensor,
        dentropy: torch.Tensor,
    ) -> tuple:
        from areal.utils.kernel import kernels

        (
            hidden,
            weight,
            labels,
            _maximum,
            _accumulate,
            _entropy_b,
        ) = ctx.saved_tensors

        dlogprobs = dlogprobs.contiguous()
        dentropy = dentropy.contiguous()

        d_hidden, d_weight = kernels.efficient_entropy_backward(
            dlogprobs,
            dentropy,
            hidden,
            weight,
            labels,
            _maximum,
            _accumulate,
            _entropy_b,
            ctx.REDUCTION,
            ctx.should_return_fp32_grad,
            ctx.temperature,
            ctx.dist_process_group,
        )

        # TP all-reduce on d_hidden: the fused path bypasses mcore's
        # ColumnParallelLinear which normally inserts this reduction.
        # d_weight does NOT need all-reduce (each rank owns its vocab shard).
        if (
            ctx.dist_process_group is not None
            and dist.get_world_size(ctx.dist_process_group) > 1
        ):
            dist.all_reduce(
                d_hidden,
                op=dist.ReduceOp.SUM,
                group=ctx.dist_process_group,
            )

        d_hidden = d_hidden.view(ctx.original_hidden_shape)

        return d_hidden, d_weight, None, None, None, None


def linear_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
    reduction: str = "none",
    dist_process_group: dist.ProcessGroup | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Functional wrapper around :class:`LinearCrossEntropy`."""
    return LinearCrossEntropy.apply(
        hidden, weight, labels, temperature, reduction, dist_process_group
    )
