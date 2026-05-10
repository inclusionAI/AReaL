# SPDX-License-Identifier: Apache-2.0
"""
``LinearCrossEntropy`` autograd Function for AReaL.

This module exposes a drop-in replacement for the
``logits = hidden @ weight.T`` -> ``log_softmax`` -> per-token
log-probability and entropy pipeline. Internally it dispatches to a Triton
kernel that fuses the matmul with the cross-entropy
reduction so that the ``[num_tokens, vocab_size]`` logits tensor is never
materialized.
"""

from __future__ import annotations

import torch
import torch.distributed as dist


class LinearCrossEntropy(torch.autograd.Function):
    """Fused linear + cross-entropy / token-entropy autograd Function.

    Forward signature:

    Args:
        hidden: ``(num_tokens, hidden_size)`` or
            ``(batch_size, seq_len, hidden_size)``. Must be contiguous on
            CUDA.
        weight: ``(vocab_size, hidden_size)`` lm-head weight. Must be
            contiguous on CUDA.
        labels: integer label ids; either ``(num_tokens,)`` or
            ``(batch_size, seq_len)``.
        temperature: softmax temperature; defaults to ``1.0``.
        reduction: only ``"none"`` is supported and returns per-token
            negative log-likelihood.
        dist_process_group: optional tensor-parallel group for vocab-sharded
            ``weight``. ``labels`` must contain *global* vocab ids on every
            rank; the kernel handles the per-rank slice internally.

    Returns:
        ``(logprobs, entropy)`` where both tensors have shape
        ``(num_tokens,)``.
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

        # Local import keeps Triton dependency lazy: tests can still
        # import this module on machines without Triton.
        from areal.utils.kernel import kernels

        REDUCTION = kernels.get_entropy_reduction_enum_number(reduction.lower())

        original_hidden_shape = hidden.shape
        if hidden.dim() != 2:
            hidden = hidden.reshape(-1, hidden.shape[-1])
        if labels.dim() != 1:
            labels = labels.reshape(-1)

        # Triton kernels demand contiguous CUDA tensors; bail out loudly
        # on misuse rather than silently materialising copies on a hot
        # path.
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

        # PyTorch autograd may produce non-contiguous gradient tensors
        # (e.g. expanded views from broadcast). Triton kernels require
        # contiguous inputs, so ensure contiguity before dispatching.
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

        # TP all-reduce on d_hidden.
        #
        # Why this is required:
        # ``efficient_entropy_backward`` computes a *local* contribution
        # ``d_hidden_local = d_logits_local @ weight_local`` where each TP
        # rank holds only a vocab-shard of ``weight``. The mathematically
        # correct gradient is the sum across the TP group:
        #     d_hidden = sum_over_tp_ranks(d_logits_local @ weight_local).
        # In Megatron's normal forward, the surrounding
        # ``ColumnParallelLinear`` (output_layer) inserts this all-reduce
        # via ``linear_with_grad_accumulation_and_async_allreduce``. The
        # fused-LCE fast path monkey-patches ``output_layer.forward`` to
        # return ``(hidden, None)`` (an autograd identity), which bypasses
        # mcore's machinery — so the all-reduce vanishes unless we
        # reproduce it here.
        #
        # Without this reduction, TP > 1 silently produces gradients that
        # equal each rank's local partial, leading to incorrect training
        # that is *not* caught by any forward-only invariant since the
        # forward kernel already all-reduces (max / logsumexp / entropy
        # auxiliaries) inside ``efficient_entropy_forward``.
        #
        # ``d_weight`` does NOT need an all-reduce: each rank legitimately
        # owns its vocab slice's weights, so the gradient on the local
        # weight shard is correctly local-only — exactly mirroring how
        # mcore handles ColumnParallel weight grads.
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

        # Order matches forward: hidden, weight, labels, temperature, reduction, group
        return d_hidden, d_weight, None, None, None, None


def linear_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
    reduction: str = "none",
    dist_process_group: dist.ProcessGroup | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Functional wrapper around :class:`LinearCrossEntropy`.

    Returns per-token ``(logprobs, entropy)``.
    """
    return LinearCrossEntropy.apply(
        hidden, weight, labels, temperature, reduction, dist_process_group
    )
