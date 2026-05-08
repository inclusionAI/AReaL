"""Fused MoE autograd functions adapted for AReaL FSDP backend.

Forward reuses SGLang's Triton kernels. Backward uses a Triton kernel written in
``fused_moe_triton_backward_kernels.py`` that computes ``grad_input``, ``grad_weight``
and (optionally) ``grad_topk_weights`` with ``tl.atomic_add``.

Debug logging can be enabled by setting the environment variable
``AREAL_FUSED_MOE_DEBUG=1``. When enabled, each intermediate tensor of the pipeline
is printed (shape + mean/std + first few values) so that the fused path can be
compared against a reference implementation by ``diff``-ing two runs.
"""

from __future__ import annotations

import os

import torch
import triton.language as tl
from sglang.srt.layers.moe.fused_moe_triton.fused_moe import (
    invoke_fused_moe_kernel,
    moe_align_block_size,
    moe_sum_reduce,
    silu_and_mul,
)

from .fused_moe_triton_backward_kernels import invoke_fused_moe_backward_kernel

_DEBUG = os.environ.get("AREAL_FUSED_MOE_DEBUG", "0") == "1"


def _dbg(tag: str, t: torch.Tensor | None) -> None:
    """Emit a compact summary of a tensor when debug mode is on.

    Only rank-0 logging is fine because callers run the same op on every rank and
    we just want to sanity-check numerical content during a single-process test.
    """
    if not _DEBUG or t is None:
        return
    try:
        flat = t.detach().float().reshape(-1)
        head = flat[:4].tolist()
        print(
            f"[fused_moe] {tag}: shape={tuple(t.shape)} dtype={t.dtype} "
            f"mean={flat.mean().item():.6e} std={flat.std().item():.6e} head={head}"
        )
    except Exception as e:  # pragma: no cover
        print(f"[fused_moe] {tag}: <failed to summarise: {e}>")


class GateUpProjFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ):
        num_tokens, _ = hidden_states.shape
        E, N, _ = w1.shape
        # Match slime / vLLM convention: chunked launch to avoid the bug
        # https://github.com/vllm-project/vllm/issues/5938
        CHUNK_SIZE = 64 * 1024

        # Default deterministic config. Tuned for H800 / A100 bf16 MoE.
        config = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
        }

        topk = topk_ids.shape[1]

        intermediate_cache1 = torch.empty(
            (num_tokens * topk, N),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        _dbg("gate_up.fwd.hidden_states", hidden_states)
        _dbg("gate_up.fwd.w1", w1)
        _dbg("gate_up.fwd.topk_weights", topk_weights)
        _dbg("gate_up.fwd.topk_ids", topk_ids)

        for chunk in range((num_tokens // CHUNK_SIZE) + 1):
            begin_chunk_idx, end_chunk_idx = (
                chunk * CHUNK_SIZE,
                min((chunk + 1) * CHUNK_SIZE, num_tokens),
            )
            curr_hidden_states = hidden_states[begin_chunk_idx:end_chunk_idx]
            cur_intermediate_cache1 = intermediate_cache1[
                begin_chunk_idx * topk : end_chunk_idx * topk
            ]

            curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
            curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]

            sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
                curr_topk_ids, config["BLOCK_SIZE_M"], E
            )

            invoke_fused_moe_kernel(
                curr_hidden_states,
                w1,
                None,
                cur_intermediate_cache1,
                None,
                None,
                None,
                curr_topk_weights,
                curr_topk_ids,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                False,
                topk_ids.shape[1],
                config,
                compute_type=tl.bfloat16,
                use_fp8_w8a8=False,
                use_int8_w8a8=False,
                use_int8_w8a16=False,
                use_int4_w4a16=False,
                per_channel_quant=False,
                block_shape=None,
                c_sorted=False,
                filter_expert=True,
            )

        _dbg("gate_up.fwd.intermediate_cache1", intermediate_cache1)

        ctx.save_for_backward(hidden_states, w1, topk_weights, topk_ids)
        ctx.config = config
        ctx.num_tokens = num_tokens
        ctx.topk = topk

        return intermediate_cache1

    @staticmethod
    def backward(ctx, grad_output):
        """Backward for GateUpProj using Triton kernels.

        ``grad_output`` has shape ``(num_tokens * topk, N)``. We return
        ``(grad_hidden_states, grad_w1, None, None)`` because ``topk_weights`` and
        ``topk_ids`` are not multiplied in the forward kernel for this stage.
        """
        hidden_states, w1, topk_weights, topk_ids = ctx.saved_tensors
        config = ctx.config
        num_tokens = ctx.num_tokens
        topk = ctx.topk

        E, N, D_in = w1.shape
        CHUNK_SIZE = 64 * 1024

        grad_hidden_states = torch.zeros_like(hidden_states)
        grad_w1 = torch.zeros_like(w1)
        grad_topk_weights = torch.zeros_like(topk_weights)

        _dbg("gate_up.bwd.grad_output", grad_output)

        for chunk in range((num_tokens // CHUNK_SIZE) + 1):
            begin_chunk_idx, end_chunk_idx = (
                chunk * CHUNK_SIZE,
                min((chunk + 1) * CHUNK_SIZE, num_tokens),
            )

            curr_num_tokens = end_chunk_idx - begin_chunk_idx
            if curr_num_tokens == 0:
                continue

            curr_hidden_states = hidden_states[begin_chunk_idx:end_chunk_idx]
            curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
            curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]
            curr_grad_output = grad_output[begin_chunk_idx * topk : end_chunk_idx * topk]

            sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
                curr_topk_ids, config["BLOCK_SIZE_M"], E
            )

            curr_grad_hidden_states = torch.zeros_like(curr_hidden_states)
            curr_grad_w1 = torch.zeros_like(w1)

            invoke_fused_moe_backward_kernel(
                grad_output=curr_grad_output,
                input=curr_hidden_states,
                weight=w1,
                grad_input=curr_grad_hidden_states,
                grad_weight=curr_grad_w1,
                grad_topk_weights=None,
                topk_weights=curr_topk_weights,
                topk_ids=curr_topk_ids,
                sorted_token_ids=sorted_token_ids,
                expert_ids=expert_ids,
                num_tokens_post_padded=num_tokens_post_padded,
                mul_routed_weight=False,
                top_k=topk,
                config=config,
                compute_type=tl.bfloat16,
            )

            grad_hidden_states[begin_chunk_idx:end_chunk_idx] += curr_grad_hidden_states
            grad_w1 += curr_grad_w1

        _dbg("gate_up.bwd.grad_hidden_states", grad_hidden_states)
        _dbg("gate_up.bwd.grad_w1", grad_w1)

        return grad_hidden_states, grad_w1, grad_topk_weights, None


class SiluAndMulFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, intermediate_cache1: torch.Tensor):
        num_tokens, N = intermediate_cache1.shape
        intermediate_cache2 = torch.empty(
            (num_tokens, N // 2),
            device=intermediate_cache1.device,
            dtype=intermediate_cache1.dtype,
        )
        silu_and_mul(intermediate_cache1.view(-1, N), intermediate_cache2)
        _dbg("silu.fwd.intermediate_cache2", intermediate_cache2)

        ctx.save_for_backward(intermediate_cache1)
        return intermediate_cache2

    @staticmethod
    def backward(ctx, grad_output):
        (intermediate_cache1,) = ctx.saved_tensors
        N = intermediate_cache1.shape[-1]
        x1, x2 = intermediate_cache1.view(-1, N).chunk(2, dim=-1)
        silu_x1 = torch.nn.functional.silu(x1)

        sig = torch.sigmoid(x1)
        dsilu_dx1 = sig + x1 * sig * (1 - sig)
        grad_x1 = grad_output * x2 * dsilu_dx1
        grad_x2 = grad_output * silu_x1
        grad_input = torch.cat([grad_x1, grad_x2], dim=-1)
        _dbg("silu.bwd.grad_input", grad_input)

        return grad_input.view_as(intermediate_cache1)


class DownProjFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        intermediate_cache2: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ):
        num_tokens, _ = intermediate_cache2.shape
        topk = topk_ids.shape[1]
        num_tokens //= topk
        E, _, _ = w2.shape
        CHUNK_SIZE = 64 * 1024

        config = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
        }

        intermediate_cache3 = torch.empty(
            (num_tokens, topk, w2.shape[1]),
            device=intermediate_cache2.device,
            dtype=intermediate_cache2.dtype,
        )

        _dbg("down.fwd.intermediate_cache2", intermediate_cache2)
        _dbg("down.fwd.w2", w2)

        for chunk in range((num_tokens // CHUNK_SIZE) + 1):
            begin_chunk_idx, end_chunk_idx = (
                chunk * CHUNK_SIZE,
                min((chunk + 1) * CHUNK_SIZE, num_tokens),
            )
            cur_intermediate_cache2 = intermediate_cache2[
                begin_chunk_idx * topk : end_chunk_idx * topk
            ]
            cur_intermediate_cache3 = intermediate_cache3[begin_chunk_idx:end_chunk_idx]

            curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
            curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]

            sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
                curr_topk_ids, config["BLOCK_SIZE_M"], E
            )
            invoke_fused_moe_kernel(
                cur_intermediate_cache2,
                w2,
                None,
                cur_intermediate_cache3,
                None,
                None,
                None,
                curr_topk_weights,
                curr_topk_ids,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                True,
                1,
                config,
                compute_type=tl.bfloat16,
                use_fp8_w8a8=False,
                use_int8_w8a8=False,
                use_int8_w8a16=False,
                use_int4_w4a16=False,
                per_channel_quant=False,
                block_shape=None,
                a_use_tma=False,
                b_use_tma=False,
            )

        _dbg("down.fwd.intermediate_cache3", intermediate_cache3)

        ctx.save_for_backward(intermediate_cache2, w2, topk_weights, topk_ids)
        ctx.config = config
        ctx.num_tokens = num_tokens
        ctx.topk = topk

        return intermediate_cache3

    @staticmethod
    def backward(ctx, grad_output):
        """Backward for DownProj.

        ``grad_output`` has shape ``(num_tokens, topk, hidden_size)``.
        Returns ``(grad_intermediate_cache2, grad_w2, grad_topk_weights, None)``.
        """
        intermediate_cache2, w2, topk_weights, topk_ids = ctx.saved_tensors
        config = ctx.config
        num_tokens = ctx.num_tokens
        topk = ctx.topk

        E, hidden_size, intermediate_size = w2.shape
        CHUNK_SIZE = 64 * 1024

        grad_intermediate_cache2 = torch.zeros_like(intermediate_cache2)
        grad_w2 = torch.zeros_like(w2)
        grad_topk_weights = torch.zeros_like(topk_weights)

        _dbg("down.bwd.grad_output", grad_output)

        for chunk in range((num_tokens // CHUNK_SIZE) + 1):
            begin_chunk_idx, end_chunk_idx = (
                chunk * CHUNK_SIZE,
                min((chunk + 1) * CHUNK_SIZE, num_tokens),
            )

            curr_num_tokens = end_chunk_idx - begin_chunk_idx
            if curr_num_tokens == 0:
                continue

            curr_intermediate_cache2 = intermediate_cache2[
                begin_chunk_idx * topk : end_chunk_idx * topk
            ]
            curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
            curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]
            curr_grad_output = grad_output[begin_chunk_idx:end_chunk_idx]

            sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
                curr_topk_ids, config["BLOCK_SIZE_M"], E
            )

            curr_grad_intermediate_cache2 = torch.zeros_like(curr_intermediate_cache2)
            curr_grad_w2 = torch.zeros_like(w2)
            curr_grad_topk_weights = torch.zeros_like(curr_topk_weights)

            # Note: Use top_k=1 to match forward pass indexing convention of
            # DownProj (each routed copy is its own "token").
            invoke_fused_moe_backward_kernel(
                grad_output=curr_grad_output,
                input=curr_intermediate_cache2,
                weight=w2,
                grad_input=curr_grad_intermediate_cache2,
                grad_weight=curr_grad_w2,
                grad_topk_weights=curr_grad_topk_weights,
                topk_weights=curr_topk_weights,
                topk_ids=curr_topk_ids,
                sorted_token_ids=sorted_token_ids,
                expert_ids=expert_ids,
                num_tokens_post_padded=num_tokens_post_padded,
                mul_routed_weight=True,
                top_k=1,
                config=config,
                compute_type=tl.bfloat16,
            )

            grad_intermediate_cache2[
                begin_chunk_idx * topk : end_chunk_idx * topk
            ] = curr_grad_intermediate_cache2
            grad_w2 += curr_grad_w2
            grad_topk_weights[begin_chunk_idx:end_chunk_idx] = curr_grad_topk_weights

        _dbg("down.bwd.grad_intermediate_cache2", grad_intermediate_cache2)
        _dbg("down.bwd.grad_w2", grad_w2)
        _dbg("down.bwd.grad_topk_weights", grad_topk_weights)

        return grad_intermediate_cache2, grad_w2, grad_topk_weights, None


class MoeSumReduceFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        intermediate_cache3: torch.Tensor,
        hidden_states_shape,
    ):
        out_hidden_states = torch.empty(
            hidden_states_shape,
            device=intermediate_cache3.device,
            dtype=intermediate_cache3.dtype,
        )
        moe_sum_reduce(
            intermediate_cache3,
            out_hidden_states,
            1.0,
        )
        _dbg("sum_reduce.fwd.out_hidden_states", out_hidden_states)
        ctx.save_for_backward(intermediate_cache3)
        return out_hidden_states

    @staticmethod
    def backward(ctx, grad_output):
        (intermediate_cache3,) = ctx.saved_tensors
        grad = grad_output.unsqueeze(1).expand_as(intermediate_cache3)
        _dbg("sum_reduce.bwd.grad_input", grad)
        return grad, None
