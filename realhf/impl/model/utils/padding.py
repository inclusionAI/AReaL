# Modified from flash-attention under BSD-3 license.
# Copyright (c) 2023, Tri Dao.
import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from realhf.base import constants


class IndexFirstAxis(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, indices):
        ctx.save_for_backward(indices)
        assert input.ndim >= 2
        ctx.first_axis_dim, other_shape = input.shape[0], input.shape[1:]
        second_dim = other_shape.numel()
        # TD [2022-03-04] For some reason torch.gather is a bit faster than indexing.
        # return input[indices]
        return torch.gather(
            rearrange(input, "b ... -> b (...)"),
            0,
            repeat(indices, "z -> z d", d=second_dim),
        ).reshape(-1, *other_shape)

    @staticmethod
    def backward(ctx, grad_output):
        (indices,) = ctx.saved_tensors
        assert grad_output.ndim >= 2
        other_shape = grad_output.shape[1:]
        grad_output = rearrange(grad_output, "b ... -> b (...)")
        grad_input = torch.zeros(
            [ctx.first_axis_dim, grad_output.shape[1]],
            device=grad_output.device,
            dtype=grad_output.dtype,
        )
        # TD [2022-03-04] For some reason torch.scatter is a bit faster than indexing.
        # grad_input[indices] = grad_output
        grad_input.scatter_(
            0, repeat(indices, "z -> z d", d=grad_output.shape[1]), grad_output
        )
        return grad_input.reshape(ctx.first_axis_dim, *other_shape), None


def index_first_axis(x: torch.Tensor, indices: torch.LongTensor):
    if len(x.shape) == 1:
        return x[indices]
    else:
        return IndexFirstAxis.apply(x, indices)


class IndexPutFirstAxis(torch.autograd.Function):

    @staticmethod
    def forward(ctx, values, indices, first_axis_dim):
        ctx.save_for_backward(indices)
        assert indices.ndim == 1
        assert values.ndim >= 2
        output = torch.zeros(
            first_axis_dim,
            *values.shape[1:],
            device=values.device,
            dtype=values.dtype,
        )
        # TD [2022-03-04] For some reason torch.scatter is a bit faster than indexing.
        output[indices] = values
        # output.scatter_(0, repeat(indices, "z -> z d", d=values.shape[1]), values)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (indices,) = ctx.saved_tensors
        # TD [2022-03-04] For some reason torch.gather is a bit faster than indexing.
        grad_values = grad_output[indices]
        # grad_values = torch.gather(grad_output, 0, repeat(indices, "z -> z d", d=grad_output.shape[1]))
        return grad_values, None, None


def index_put_first_axis(
    values: torch.Tensor, indices: torch.LongTensor, first_axis_dim: int
):
    if len(values.shape) == 1:
        output = torch.zeros(first_axis_dim, device=values.device, dtype=values.dtype)
        output[indices] = values
        return output
    else:
        return IndexPutFirstAxis.apply(values, indices, first_axis_dim)


class IndexFirstAxisResidual(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, indices):
        ctx.save_for_backward(indices)
        assert input.ndim >= 2
        ctx.first_axis_dim, other_shape = input.shape[0], input.shape[1:]
        second_dim = other_shape.numel()
        # TD [2022-03-04] For some reason torch.gather is a bit faster than indexing.
        output = input[indices]
        # We don't want to reshape input (b ... -> b (...)) since it could change the channel_last
        # memory format to channel_first. In other words, input might not be contiguous.
        # If we don't detach, Pytorch complains about output being a view and is being modified inplace
        return output, input.detach()

    @staticmethod
    def backward(ctx, grad_output, grad_residual):
        (indices,) = ctx.saved_tensors
        assert grad_output.ndim >= 2
        other_shape = grad_output.shape[1:]
        assert grad_residual.shape[1:] == other_shape
        grad_input = grad_residual
        # grad_input[indices] += grad_output
        indices = indices.reshape(indices.shape[0], *((1,) * (grad_output.ndim - 1)))
        indices = indices.expand_as(grad_output)
        grad_input.scatter_add_(0, indices, grad_output)
        return grad_input.reshape(ctx.first_axis_dim, *other_shape), None


index_first_axis_residual = IndexFirstAxisResidual.apply


def unpad_input(hidden_states, attention_mask):
    """
    Arguments:
        hidden_states: (batch, seqlen, ...)
        attention_mask: (batch, seqlen), bool / int, 1 means valid and 0 means not valid.
    Return:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
        cu_seqlens: (batch + 1), the cumulative sequence lengths, used to index into hidden_states.
        max_seqlen_in_batch: int
    """
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(
        torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0)
    )
    # TD [2022-03-04] We don't want to index with a bool mask, because Pytorch will expand the
    # bool mask, then call nonzero to get the indices, then index with those. The indices is @dim
    # times larger than it needs to be, wasting memory. It's faster and more memory-efficient to
    # index with integer indices. Moreover, torch's index is a bit slower than it needs to be,
    # so we write custom forward and backward to make it a bit faster.
    return (
        index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices),
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def unpad_input_for_concatenated_sequences(hidden_states, attention_mask_in_length):
    """
    Supports concatenating short samples in one sequence.
    The attention_mask_in_length is utilized to mask other short samples.
    It helps efficient training of variant lengths-based samples
    (e.g., the supervised fine-tuning task in large language model).
    The motivation for this function is explained
    [here](https://github.com/Dao-AILab/flash-attention/issues/432#issuecomment-1668822286).

    For example, if batch = 3 and seqlen = 6, the attention_mask_in_length is:
        ```
        [
          [2, 3, 0, 0, 0, 0],
          [3, 2, 0, 0, 0, 0],
          [6, 0, 0, 0, 0, 0]
        ]
        ```
    , which refers to the 3D-attention mask:
        ```
        [
          [
            [1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 1]
          ],
          [
            [1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 1]
          ],
          [
            [1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1]
          ]
        ]
        ```.

    Arguments:
        hidden_states: (batch, seqlen, ...)
        attention_mask_in_length: (batch, seqlen), int, a nonzero number (e.g., 1, 2, 3, etc.) means length of concatenated sequence in b-th batch, and 0 means none.
    Return:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
        cu_seqlens: (batch + 1), the cumulative sequence lengths, used to index into hidden_states.
        max_seqlen_in_batch: int
    """
    length = attention_mask_in_length.sum(dim=-1)
    seqlen = attention_mask_in_length.size(-1)
    attention_mask_2d = torch.arange(
        seqlen, device=length.device, dtype=length.dtype
    ).expand(len(length), seqlen) < length.unsqueeze(1)
    real_indices_idx = torch.nonzero(
        attention_mask_in_length.flatten(), as_tuple=False
    ).flatten()
    seqlens_in_batch = attention_mask_in_length.flatten()[real_indices_idx]
    indices = torch.nonzero(attention_mask_2d.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(
        torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0)
    )
    # TD [2022-03-04] We don't want to index with a bool mask, because Pytorch will expand the
    # bool mask, then call nonzero to get the indices, then index with those. The indices is @dim
    # times larger than it needs to be, wasting memory. It's faster and more memory-efficient to
    # index with integer indices. Moreover, torch's index is a bit slower than it needs to be,
    # so we write custom forward and backward to make it a bit faster.
    return (
        index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices),
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def pad_input(hidden_states, indices, batch, seqlen):
    """
    Arguments:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
        indices: (total_nnz)
    Return:
        hidden_states: (batch, seqlen, ...)
    """
    dim = hidden_states.shape[-1]
    # output = torch.zeros((batch * seqlen), dim, device=hidden_states.device, dtype=hidden_states.dtype)
    # output[indices] = hidden_states
    output = index_put_first_axis(hidden_states, indices, batch * seqlen)
    return rearrange(output, "(b s) ... -> b s ...", b=batch)


def pad_sequence_parallel_input(
    packed_input_ids: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int
):
    """Sequence parallel requires packed_input_ids has a shape of 1 dimension
    [total_seq_len], and total_seq_len should be divisible by
    tensor_parallel_world_size. This function is used to pad packed_input_ids to
    suitable length with an empty sequence, and return new packed_input_ids,
    cu_seqlens and max_seqlen.

    Args:
        packed_input_ids (torch.Tensor): unpadded packed_input_ids
        cu_seqlens (torch.Tensor): unpadded cu_seqlens
        max_seqlen (int): unpadded max_seqlen

    Returns:
        (torch.Tensor, torch.Tensor, int, int): padded (packed_input_ids, cu_seqlens, max_seqlen, pad_size)
    """
    tp_world_size = constants.tensor_parallel_world_size()
    pad_size = 0
    if len(packed_input_ids) % tp_world_size != 0:
        pad_size = tp_world_size - len(packed_input_ids) % tp_world_size
        packed_input_ids = torch.nn.functional.pad(
            packed_input_ids, (0, pad_size), value=1
        )
        cu_seqlens = torch.nn.functional.pad(
            cu_seqlens, (0, 1), value=len(packed_input_ids)
        )
        max_seqlen = max_seqlen if pad_size < max_seqlen else pad_size
    return packed_input_ids, cu_seqlens, max_seqlen, pad_size


def pad_sequence_parallel_generate_input(
    packed_input_ids: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int
):
    """Only for pipeline generate input when model+seq parallel is enabled. To
    make sure inputs for seq parallel model have a shape with first dimension
    divisible by tensor_parallel_world_size, the packed_input_ids should have
    length divisible by tensor_parallel_world_size, and contains number of
    sequences divisible by tensor_parallel_world_size.

    Args:
        packed_input_ids (torch.Tensor): unpadded packed_input_ids
        cu_seqlens (torch.Tensor): unpadded cu_seqlens
        max_seqlen (int): unpadded max_seqlen

    Returns:
        (torch.Tensor, torch.Tensor, int, int, int): padded (packed_input_ids, cu_seqlens, max_seqlen, pad_size, pad_seq_size)
    """
    tp_world_size = constants.tensor_parallel_world_size()
    pad_size, pad_seq_size = 0, 0
    if (
        len(packed_input_ids) % tp_world_size != 0
        or (len(cu_seqlens) - 1) % tp_world_size != 0
    ):
        pad_size = tp_world_size - len(packed_input_ids) % tp_world_size
        pad_seq_size = tp_world_size - (len(cu_seqlens) - 1) % tp_world_size
        if pad_size < pad_seq_size:
            pad_size += tp_world_size
        pad_cu_seqlens = torch.tensor(list(range(1, pad_seq_size)) + [pad_size]) + len(
            packed_input_ids
        )
        pad_cu_seqlens = pad_cu_seqlens.to(
            dtype=cu_seqlens.dtype, device=cu_seqlens.device
        )
        packed_input_ids = torch.nn.functional.pad(
            packed_input_ids, (0, pad_size), value=1
        )
        cu_seqlens = torch.cat([cu_seqlens, pad_cu_seqlens], dim=0)
        max_seqlen = (
            max_seqlen
            if (pad_size - pad_seq_size + 1) < max_seqlen
            else (pad_size - pad_seq_size + 1)
        )
    return packed_input_ids, cu_seqlens, max_seqlen, pad_size, pad_seq_size
