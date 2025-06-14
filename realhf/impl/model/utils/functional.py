# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

from typing import Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import transformers

from realhf.base import constants, logging
from realhf.impl.model.utils.padding import pad_input, unpad_input

logger = logging.getLogger("Modeling Functional Utils")


@torch.jit.script
def upcast_masked_softmax(
    x: torch.Tensor,
    mask: torch.Tensor,
    mask_value: torch.Tensor,
    scale: float,
    softmax_dtype: torch.dtype,
):
    input_dtype = x.dtype
    x = x.to(softmax_dtype) * scale
    x = torch.where(mask, x, mask_value)
    x = torch.nn.functional.softmax(x, dim=-1).to(input_dtype)
    return x


@torch.jit.script
def upcast_softmax(x: torch.Tensor, scale: float, softmax_dtype: torch.dtype):
    input_dtype = x.dtype
    x = x.to(softmax_dtype) * scale
    x = torch.nn.functional.softmax(x, dim=-1).to(input_dtype)
    return x


@torch.jit.script
def masked_softmax(x: torch.Tensor, mask: torch.Tensor, mask_value: torch.Tensor):
    x = torch.where(mask, x, mask_value)
    x = torch.nn.functional.softmax(x, dim=-1)
    return x


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    total_seqlen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :]
        .expand(total_seqlen, n_kv_heads, n_rep, head_dim)
        .reshape(total_seqlen, n_kv_heads * n_rep, head_dim)
    )


def mask_eos_token(
    logits: torch.Tensor,
    eos_token_id: Optional[int] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    # for min_new_tokens
    if eos_token_id is not None:
        logits[..., eos_token_id] = torch.finfo(logits.dtype).min
    return logits


def gather_shifted_log_probs(
    logits: torch.FloatTensor, labels: torch.LongTensor
) -> torch.FloatTensor:
    """Gather log probs of shifted labels from logits.

    Args:
        logits (torch.FloatTensor): Non-shifted logits with shape [bs, seqlen].
            The final value at [:, seqlen -1] is not used.
        labels (torch.LongTensor): Non-shifted labels/input_ids with shape [bs, seqlen].
            The first value at [:, 0] has no corresponding log prob.

    Returns:
        torch.FloatTensor: Shifted log probability with shape [bs, seqlen -1].
    """
    logits = logits[:, :-1]
    labels = labels[:, 1:]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)


def build_shift_one_indices(
    x: torch.HalfTensor, cu_seqlens: torch.IntTensor
) -> torch.IntTensor:
    """Build indices for shifting labels/input_ids one step to the left.

    Equivalent to:
    ```
    shift_one_indices = torch.cat([
        torch.arange(cu_seqlens[i] + 1, cu_seqlens[i + 1], dtype=torch.long, device=cu_seqlens.device)
        for i in range(cu_seqlens.shape[0] - 1)
    ])
    ```
    but the above implementaion will implicitly convert a tensor (cu_seqlens[i]) to an integer,
    which will cause a cuda device sync and slow down performance.

    Args:
        x (torch.HalfTensor): Shape [total_seqlen]. This tensor is required to get
            total_seqlen from its shape. Computing total_seqlen from cu_seqlens will implicitly cause
            a cuda device sync.
        cu_seqlens (torch.IntTensor): Shape [bs + 1]. Indices marking the start
            and end of each sequences.

    Returns:
        torch.IntTensor: Shape [tot_seqlen - bs]. Indices for shifting labels/input_ids
            one step to the left.
    """
    total_seqlen = x.shape[0]
    bs = cu_seqlens.shape[0] - 1
    short1lens = cu_seqlens[1:] - cu_seqlens[:-1] - 1
    short1cu_seqlens = torch.nn.functional.pad(short1lens.cumsum(0), (1, 0), value=0)
    indexing_t = torch.arange(
        total_seqlen - bs, dtype=torch.long, device=cu_seqlens.device
    )
    return indexing_t + (
        indexing_t.unsqueeze(0) >= short1cu_seqlens[:-1].unsqueeze(1)
    ).sum(0)


def build_leave_one_indices(
    x: torch.HalfTensor, cu_seqlens: torch.IntTensor
) -> torch.IntTensor:
    """Build indices for leaving one token out at the end of each sequence.

    Equivalent to:
    ```
    leave_one_indices = torch.cat([
        torch.arange(cu_seqlens[i], cu_seqlens[i + 1] - 1, dtype=torch.long, device=cu_seqlens.device)
        for i in range(cu_seqlens.shape[0] - 1)
    ])
    ```
    but the above implementaion will implicitly convert a tensor (cu_seqlens[i]) to an integer,
    which will cause a cuda device sync and slow down performance.

    Args:
        x (torch.HalfTensor): Shape [total_seqlen]. This tensor is required to get
            total_seqlen from its shape. Computing total_seqlen from cu_seqlens will implicitly cause
            a cuda device sync.
        cu_seqlens (torch.IntTensor): Shape [bs + 1]. Indices marking the start
            and end of each sequences.

    Returns:
        torch.IntTensor: Shape [tot_seqlen - bs]. Indices for shifting labels/input_ids
            one step to the left.
    """
    total_seqlen = x.shape[0]
    bs = cu_seqlens.shape[0] - 1
    short1lens = cu_seqlens[1:] - cu_seqlens[:-1] - 1
    short1cu_seqlens = torch.nn.functional.pad(short1lens.cumsum(0), (1, 0), value=0)
    indexing_t = torch.arange(
        total_seqlen - bs, dtype=torch.long, device=cu_seqlens.device
    )
    return (
        indexing_t
        + (indexing_t.unsqueeze(0) >= short1cu_seqlens[:-1].unsqueeze(1)).sum(0)
        - 1
    )


def gather_logprobs(
    logits: torch.Tensor,
    labels: torch.Tensor,
):
    """Gather log probs from logits and labels.

    Args:
        logits (torch.FloatTensor): Shape [tot_seqlen]. The final value at the end of
            each sequence is not used.
        labels (torch.LongTensor): Labels or input_ids with shape [tot_seqlen].
            The first value at the beginning of each sequence has no corresponding log prob.

    Returns:
        torch.FloatTensor: Log probability with shape [tot_seqlen - #seqs].
    """
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return log_probs_labels


def gather_packed_shifted_log_probs(
    logits: torch.FloatTensor,
    cu_seqlens: torch.Tensor,
    labels: torch.LongTensor,
) -> torch.FloatTensor:
    """Gather log probs from packed input_ids and logits.

    Args:
        logits (torch.FloatTensor): Shape [tot_seqlen]. The final value at the end of
            each sequence is not used.
        cu_seqlens (torch.Tensor): Shape [#seqs + 1]. Indices marking the start
            and end of each sequence.
        labels (torch.LongTensor): Labels or input_ids with shape [tot_seqlen].
            The first value at the beginning of each sequence has no corresponding log prob.

    Returns:
        torch.FloatTensor: Log probability with shape [tot_seqlen - #seqs].
    """
    labels = torch.nn.functional.pad(labels[1:], (0, 1), value=0)
    leave_one_indices = build_leave_one_indices(logits, cu_seqlens)
    if constants.tensor_parallel_world_size() > 1:
        # NOTE: logprobs is freaking sensitive to input_ids. If the input sequence is a natural sequence, everything will be fine.
        # However, if we input random token IDs, parallel cross entropy can produce VERY different results than the normal
        # torch.gather based version (e.g., the maximum absolute different can reach ~50).
        from realhf.impl.model.parallelism.tensor_parallel.modules import (
            vocab_parallel_cross_entropy,
        )

        logprobs = -vocab_parallel_cross_entropy(logits, labels)[leave_one_indices]
        return logprobs
    logits_shape = logits.shape
    # shift_one_indices = torch.cat([
    #     torch.arange(cu_seqlens[i] + 1 , cu_seqlens[i + 1], dtype=torch.long, device=cu_seqlens.device)
    #     for i in range(cu_seqlens.shape[0] - 1)
    # ])
    # shift labels one step to the left and pad it to match the shape of logits
    log_probs_labels = gather_logprobs(logits, labels)
    log_probs_labels = log_probs_labels[leave_one_indices]
    assert log_probs_labels.shape[0] == logits_shape[0] - cu_seqlens.shape[0] + 1, (
        log_probs_labels.shape,
        logits_shape,
        cu_seqlens.shape,
        cu_seqlens,
        # shift_one_indices,
    )
    return log_probs_labels


def apply_logits_mask(logits: torch.HalfTensor, mask: torch.BoolTensor):
    assert (
        mask.shape[-1] == logits.shape[-1] * constants.tensor_parallel_world_size()
    ), (
        constants.tensor_parallel_world_size(),
        logits.shape,
        mask.shape,
    )
    parallel_vocab_size = logits.shape[-1]
    tp_rank = constants.tensor_parallel_rank()
    mask = mask[:, tp_rank * parallel_vocab_size : (tp_rank + 1) * parallel_vocab_size]
    logits.masked_fill_(mask, torch.finfo(logits.dtype).min)


@torch.no_grad()
def masked_normalization(
    x: torch.Tensor,
    mask: Optional[torch.BoolTensor] = None,
    dim=None,
    inplace=False,
    unbiased=False,
    eps=1e-5,
    high_precision=True,
    all_reduce=True,
):
    """Normalize x with a mask. Typically used in advantage normalization.

    Args:
        x (torch.Tensor):
            Tensor to be normalized.
        mask (torch.Tensor, optional):
            A mask with the same shape as x. Defaults to None.
        dim (int or tuple of ints, optional):
            Dimensions to be normalized. Defaults to None.
        inplace (bool, optional):
            Whether to perform in-place operation. Defaults to False.
        eps (torch.Tensor, optional):
            Minimal denominator. Defaults to 1e-5.

    Returns:
        torch.Tensor:
            Normalized x, with the same shape as x.
    """
    dtype = torch.float64 if high_precision else torch.float32
    x = x.to(dtype)
    if not inplace:
        x = x.clone()
    if dim is None:
        dim = tuple(range(len(x.shape)))
    if mask is None:
        factor = torch.tensor(
            np.prod([x.shape[d] for d in dim]), dtype=dtype, device=x.device
        )
    else:
        mask = mask.to(dtype)
        assert len(mask.shape) == len(x.shape), (mask.shape, x.shape, dim)
        for i in range(len(x.shape)):
            if i in dim:
                assert mask.shape[i] == x.shape[i], (mask.shape, x.shape, dim)
            else:
                assert mask.shape[i] == 1, (mask.shape, x.shape, dim)
        x = x * mask
        factor = mask.sum(dim, keepdim=True)
    x_sum = x.sum(dim=dim, keepdim=True)
    x_sum_sq = x.square().sum(dim=dim, keepdim=True)
    if dist.is_initialized() and all_reduce:
        dist.all_reduce(
            factor, op=dist.ReduceOp.SUM, group=constants.data_parallel_group()
        )
        dist.all_reduce(
            x_sum, op=dist.ReduceOp.SUM, group=constants.data_parallel_group()
        )
        dist.all_reduce(
            x_sum_sq,
            op=dist.ReduceOp.SUM,
            group=constants.data_parallel_group(),
        )
    mean = x_sum / factor
    meansq = x_sum_sq / factor
    var = meansq - mean**2
    if unbiased:
        var *= factor / (factor - 1)
    return ((x - mean) / (var.sqrt() + eps)).float()


def get_eos_indices(
    input_ids: torch.LongTensor,
    tokenizer: transformers.PreTrainedTokenizerFast,
) -> Tuple[torch.LongTensor, torch.FloatTensor]:
    if torch.any(input_ids[:, 0] == tokenizer.eos_token_id):
        indices = (input_ids[:, 0] == tokenizer.eos_token_id).nonzero().flatten()
        bad_input_ids = input_ids[indices]
        bad_strs = tokenizer.batch_decode(
            bad_input_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        raise RuntimeError(
            f"Generated sequence terminates unexpectedly early: {bad_strs}"
        )
    seq_len = input_ids.shape[1]
    eos_mask = (input_ids == tokenizer.eos_token_id).float()
    seq_no_eos_mask = (eos_mask.sum(1) == 0).float()
    eos_indices = eos_mask.argmax(1)
    eos_indices = (
        eos_indices * (1 - seq_no_eos_mask) + seq_no_eos_mask * (seq_len - 1)
    ).long()
    return eos_indices, seq_no_eos_mask


def torch_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    cu_seqlens_q: torch.IntTensor,
    max_seqlen_q: int,
    cu_seqlens_k: torch.IntTensor,
    max_seqlen_k: int,
    dropout_p: float,
    softmax_scale: float,
    upcast_unscale: float = 1.0,
) -> torch.Tensor:
    """PyTorch implementation of the attention function with a flash-attn-like
    realhf.api.

    We use this function to compare the output of our model and huggingface models.
    Flash-attn/float16/CUDAkernels will all more or less suffer from float point errors.
    We call this function with float32 and CPU to get the "ground truth" output.

    Args:
        q (torch.Tensor): Shape [total_seqlen, #q, head_dim].
        k (torch.Tensor): Shape [total_seqlen, #kv, head_dim].
        v (torch.Tensor): Shape [total_seqlen, #kv, head_dim].
        causal (bool): .
        dropout_p (float): .
        softmax_scale (float): .
        upcast_unscale (float, optional): Scale factor when upcastin attention scores.
            Defaults to 1.0.

    Returns:
        torch.Tensor: Attention score. Shape [bs, seqlen, #q, head_dim].
    """
    nq = q.shape[-2]
    nkv = k.shape[-2]
    n_rep = q.shape[-2] // k.shape[-2]
    bsz = cu_seqlens_q.shape[0] - 1
    # repeat k/v heads if n_kv_heads < n_heads
    k = repeat_kv(k, n_rep)  # (total_seqlen, nq, head_dim)
    v = repeat_kv(v, n_rep)  # (total_seqlen, nq, head_dim)

    input_lens_k = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
    attention_mask_k = torch.arange(
        max_seqlen_k, dtype=torch.long, device="cpu"
    ).unsqueeze(0) < input_lens_k.unsqueeze(1)
    _, _pad_indices_k, _, _ = unpad_input(attention_mask_k, attention_mask_k)

    input_lens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    attention_mask_q = torch.arange(
        max_seqlen_q, dtype=torch.long, device="cpu"
    ).unsqueeze(0) < input_lens_q.unsqueeze(1)
    _, _pad_indices_q, _, _ = unpad_input(attention_mask_q, attention_mask_q)

    q = pad_input(q, _pad_indices_q, bsz, max_seqlen_q)
    k = pad_input(k, _pad_indices_k, bsz, max_seqlen_k)
    v = pad_input(v, _pad_indices_k, bsz, max_seqlen_k)

    q = q.transpose(1, 2)  # (bs, nq, seqlen, head_dim)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    scores = torch.matmul(q, k.transpose(2, 3)) * softmax_scale

    mask = (
        attention_mask_k.unsqueeze(1).unsqueeze(1).repeat(1, nq, max_seqlen_q, 1)
    )  # [bs, nq, seqlen, seqlen]
    if causal:
        _ms = max(max_seqlen_q, max_seqlen_k)
        causal_mask = torch.tril(
            torch.ones(_ms, _ms, device=q.device, dtype=torch.bool)
        )[-max_seqlen_q:, -max_seqlen_k:]
        mask = mask & causal_mask

    # if mask_softmax:
    scores = upcast_masked_softmax(
        scores,
        mask,
        mask_value=torch.full(
            [],
            torch.finfo(torch.float32).min,
            device=scores.device,
            dtype=torch.float32,
        ),
        scale=upcast_unscale,
        softmax_dtype=torch.float32,
    )
    # else:
    #     scores = upcast_softmax(scores, scale=upcast_unscale, softmax_dtype=torch.float32)
    scores = torch.nn.functional.dropout(scores, p=dropout_p)
    scores = scores.to(q.dtype)
    output = torch.matmul(scores, v)  # (bs, nq, seqlen, head_dim)
    output = output.transpose(1, 2).contiguous()

    output = unpad_input(output, attention_mask_q)[0]
    return output


def rotate_half(x: torch.HalfTensor, interleaved: bool = False):
    if not interleaved:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    else:
        x1, x2 = x[..., ::2], x[..., 1::2]
        # return rearrange(torch.stack((-x2, x1), dim=-1), "... d two -> ... (d two)", two=2)
        return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


@torch.no_grad()
@torch.jit.script
def compute_varlen_position_indices(
    total_seqlen: int,
    cu_seqlens: torch.IntTensor,
    seqlen_offsets: Optional[torch.IntTensor] = None,
) -> torch.IntTensor:
    indexing_t = torch.arange(
        total_seqlen, dtype=torch.long, device=cu_seqlens.device
    ).unsqueeze_(0)
    indexing_t = (cu_seqlens[:-1].unsqueeze(1) <= indexing_t) & (
        indexing_t < cu_seqlens[1:].unsqueeze(1)
    )
    indices = indexing_t.cumsum(1) - 1
    if seqlen_offsets is not None:
        indices += seqlen_offsets.unsqueeze(1)
    return torch.where(indexing_t, indices, 0).sum(0)


# @torch.jit.script
def apply_rotary_varlen(
    x: torch.HalfTensor,
    cos: torch.HalfTensor,
    sin: torch.HalfTensor,
    cu_seqlens: torch.IntTensor,
    interleaved: bool,
    seqlen_offsets: Optional[torch.IntTensor] = None,
    rotary_indices: Optional[torch.LongTensor] = None,
    special_impl: Optional[str] = None,
) -> Tuple[torch.HalfTensor, torch.LongTensor]:
    if rotary_indices is None:
        rotary_indices = compute_varlen_position_indices(
            x.shape[0], cu_seqlens, seqlen_offsets
        )

    cos = cos[rotary_indices]
    sin = sin[rotary_indices]
    if special_impl == "bailing":
        return x * cos[:, None, :] + rotate_half(x, interleaved) * sin[:, None, :]

    assert special_impl is None
    ro_dim = cos.shape[-1] * 2
    assert ro_dim <= x.shape[-1], (x.shape, cos.shape)
    if not interleaved:
        cos = cos[:, None, None, :].repeat(1, 1, 2, 1).flatten(start_dim=-2)
        sin = sin[:, None, None, :].repeat(1, 1, 2, 1).flatten(start_dim=-2)
    else:
        cos = cos[:, None, :, None].repeat(1, 1, 1, 2).flatten(start_dim=-2)
        sin = sin[:, None, :, None].repeat(1, 1, 1, 2).flatten(start_dim=-2)

    # cos = repeat(cos, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)")
    # sin = repeat(sin, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)")
    return torch.cat(
        [
            x[..., :ro_dim] * cos + rotate_half(x[..., :ro_dim], interleaved) * sin,
            x[..., ro_dim:],
        ],
        dim=-1,
    )


def apply_rotary(
    x: torch.HalfTensor,
    cos: torch.HalfTensor,
    sin: torch.HalfTensor,
    interleaved: bool = False,
):
    """
    x: (batch_size, seqlen, nheads, headdim)
    cos, sin: (seqlen, rotary_dim / 2) or (batch_size, seqlen, rotary_dim / 2)
    """
    ro_dim = cos.shape[-1] * 2
    assert ro_dim <= x.shape[-1]
    if not interleaved:
        cos = cos[:, None, None, :].repeat(1, 1, 2, 1).flatten(start_dim=-2)
        sin = sin[:, None, None, :].repeat(1, 1, 2, 1).flatten(start_dim=-2)
    else:
        cos = cos[:, None, :, None].repeat(1, 1, 1, 2).flatten(start_dim=-2)
        sin = sin[:, None, :, None].repeat(1, 1, 1, 2).flatten(start_dim=-2)
    return torch.cat(
        [
            x[..., :ro_dim] * cos + rotate_half(x[..., :ro_dim], interleaved) * sin,
            x[..., ro_dim:],
        ],
        dim=-1,
    )
