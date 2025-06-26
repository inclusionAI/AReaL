# Copyright 2025 Ant Group Inc.
# Licensed under the Apache License, Version 2.0

# Pad/unpad operations are modified from flash-attention under BSD-3 license.
# Copyright (c) 2023, Tri Dao.

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from einops import rearrange, repeat
from tensorboardX import SummaryWriter

from arealite.api.cli_args import MicroBatchSpec, TrainingArgs
from realhf.base import constants, datapack

############### Dict and list operations begin ###############


def recorder_list(xs: List, indices: List[int]) -> List:
    assert len(set(indices)) == len(xs)
    return [xs[i] for i in indices]


def dict_map(x: Dict, fn: Callable) -> Dict:
    """Apply a function to each value in a dictionary."""
    return {k: fn(v) for k, v in x.items()}


def dict_of_list2list_of_dict(
    dict_of_lists: Dict[str, List[Any]],
) -> List[Dict[str, Any]]:
    """
    Convert a dictionary of lists into a list of dictionaries.

    Args:
        dict_of_lists: Dictionary where each value is a list

    Returns:
        List of dictionaries where each dictionary contains one item from each list

    Example:
        >>> data = {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}
        >>> result = dict_of_list2list_of_dict(data)
        >>> result
        [{"a": 1, "b": 4, "c": 7}, {"a": 2, "b": 5, "c": 8}, {"a": 3, "b": 6, "c": 9}]
    """
    if not dict_of_lists:
        return []

    # Get the length from the first key's list
    keys = list(dict_of_lists.keys())
    length = len(dict_of_lists[keys[0]])

    # Verify all lists have the same length
    for key, value_list in dict_of_lists.items():
        if len(value_list) != length:
            raise ValueError(
                f"All lists must have the same length. Key '{key}' has length {len(value_list)}, expected {length}"
            )

    # Convert to list of dicts
    return [{key: dict_of_lists[key][i] for key in keys} for i in range(length)]


def list_of_dict2dict_of_list(
    list_of_dicts: List[Dict[str, Any]],
) -> Dict[str, List[Any]]:
    """
    Convert a list of dictionaries into a dictionary of lists.

    Args:
        list_of_dicts: List where each element is a dictionary

    Returns:
        Dictionary where each key maps to a list of values from all dictionaries

    Example:
        >>> data = [{"a": 1, "b": 4, "c": 7}, {"a": 2, "b": 5, "c": 8}, {"a": 3, "b": 6, "c": 9}]
        >>> result = lod2dol(data)
        >>> result
        {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}
    """
    if not list_of_dicts:
        return {}

    # Get all keys from the first dictionary
    keys = list(list_of_dicts[0].keys())

    # Verify all dictionaries have the same keys
    for i, dict_item in enumerate(list_of_dicts):
        if set(dict_item.keys()) != set(keys):
            raise ValueError(
                f"All dictionaries must have the same keys. Dictionary at index {i} has keys {set(dict_item.keys())}, expected {set(keys)}"
            )

    # Convert to dict of lists
    return {key: [dict_item[key] for dict_item in list_of_dicts] for key in keys}


############### Dict and list operations end ###############

############### Pad and unpad operations begin ###############


def pad_sequences_to_tensors(
    sequence_list: List[Dict[str, torch.Tensor]], pad_value: float = 0.0
) -> Dict[str, torch.Tensor]:
    if not sequence_list:
        return {}

    # Find max length across all sequences
    max_length = max(len(seq) for item in sequence_list for seq in item.values())

    result = {}

    # Create padded tensors for each key
    for key in sequence_list[0].keys():
        padded = [
            torch.nn.functional.pad(
                item[key], (0, max_length - len(item[key])), value=pad_value
            )
            for item in sequence_list
        ]
        result[key] = torch.stack(padded)

    # Create attention mask
    attention_mask = [
        [1] * len(next(iter(item.values())))
        + [0] * (max_length - len(next(iter(item.values()))))
        for item in sequence_list
    ]

    result["attention_mask"] = torch.tensor(attention_mask, dtype=torch.long)
    return result


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
        other_shape.numel()
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


def unpad_input(
    hidden_states, attention_mask
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
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


def pad_input(hidden_states, indices, batch, seqlen):
    """
    Arguments:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
        indices: (total_nnz)
    Return:
        hidden_states: (batch, seqlen, ...)
    """
    hidden_states.shape[-1]
    # output = torch.zeros((batch * seqlen), dim, device=hidden_states.device, dtype=hidden_states.dtype)
    # output[indices] = hidden_states
    output = index_put_first_axis(hidden_states, indices, batch * seqlen)
    return rearrange(output, "(b s) ... -> b s ...", b=batch)


def concat_padded_tensors(
    tensor_dicts: List[Dict[str, torch.Tensor]], pad_value: float = 0.0
) -> Dict[str, torch.Tensor]:
    """Concatenate and pad tensors from multiple padded tensor dictionaries."""
    if not tensor_dicts:
        return {}

    # Find max sequence length across all dictionaries
    lens = []
    for tensor_dict in tensor_dicts:
        for key, tensor in tensor_dict.items():
            if key != "attention_mask" and len(tensor.shape) == 2:
                lens.append(tensor.shape[1])
                break
    max_length = max(lens)
    attn_mask = torch.arange(max_length).unsqueeze(0) < torch.tensor(lens).unsqueeze(1)

    result = {}
    # Process each key
    for key in tensor_dicts[0].keys():
        tensors_to_concat = []
        for tensor_dict in tensor_dicts:
            tensor = tensor_dict[key]
            # Skip 1D tensors like rewards
            if len(tensor.shape) == 1:
                tensors_to_concat.append(tensor)
                continue
            current_length = tensor.shape[1]
            if current_length < max_length:
                # Pad tensor to max_length
                pad_width = max_length - current_length
                if key == "attention_mask":
                    # Pad attention mask with 0s
                    padding = torch.zeros(
                        (tensor.shape[0], pad_width), dtype=tensor.dtype
                    )
                else:
                    # Pad feature tensors with pad_value
                    padding = torch.full(
                        (tensor.shape[0], pad_width), pad_value, dtype=tensor.dtype
                    )
                tensor = torch.cat([tensor, padding], dim=1)
            tensors_to_concat.append(tensor)

        result[key] = torch.cat(tensors_to_concat, dim=0)
    if "attention_mask" not in result:
        result["attention_mask"] = attn_mask
    return result


############### Pad and unpad operations end ###############

############### Tensor transformations begin ###############


def to_device(data: Dict[str, torch.Tensor | Any], device) -> Dict[str, torch.Tensor]:
    """Move tensors in a dictionary to the specified device."""
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in data.items()
    }


def unpack_sequence(
    x: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor] = None,
    lens: Optional[List[int]] = None,
    dim: int = 0,
):
    """Unpack a sequence tensor into a list of tensors based on cumulative sequence lengths."""
    if lens is not None:
        return torch.split(x, lens, dim=dim)
    if cu_seqlens is not None:
        return torch.split(
            x, (cu_seqlens[1:] - cu_seqlens[:-1]).cpu().numpy().tolist(), dim=dim
        )
    raise ValueError("Either cu_seqlens or input_lens must be provided.")


def allocate_balanced_mbs(mb_spec: MicroBatchSpec, lens: List[int]) -> List[List[int]]:
    group_indices = datapack.ffd_allocate(
        lens, mb_spec.max_tokens_per_mb, min_groups=mb_spec.n_mbs
    )
    group_indices = sorted([sorted(g) for g in group_indices])
    return group_indices


def allocate_balanced_mbs_synced(
    mb_spec: MicroBatchSpec,
    lens: List[int],
    group: Optional[dist.ProcessGroup] = None,
) -> List[List[int]]:
    group_indices = allocate_balanced_mbs(mb_spec, lens)
    if not dist.is_initialized():
        return group_indices

    all_n_mbs = [None for _ in range(dist.get_world_size(group))]
    dist.all_gather_object(all_n_mbs, len(group_indices), group=group)
    if all(mbs == len(group_indices) for mbs in all_n_mbs):
        return group_indices
    return allocate_balanced_mbs_synced(
        MicroBatchSpec.new(mb_spec, n_mbs=max(all_n_mbs)), lens
    )


@dataclass
class MicroBatchSplitResult:
    data: Dict[str, Any]
    mb_spec: MicroBatchSpec
    mbs: List[Dict[str, Any]]
    forward_indices: List[int]
    backward_indices: List[int]


def split_dict_tensor_with_cu_seqlens(
    data: Dict[str, torch.Tensor],
    mb_spec: MicroBatchSpec,
    group: Optional[dist.ProcessGroup] = None,
) -> MicroBatchSplitResult:
    assert "cu_seqlens" in data
    cu_seqlens = data["cu_seqlens"]
    bs = cu_seqlens.shape[0] - 1
    total_lens = int(cu_seqlens[-1])
    input_lens = (cu_seqlens[1:] - cu_seqlens[:-1]).cpu().numpy()

    # check tensor shape, split only 1d tensors with length "total_lens"
    to_split = {}
    not_to_split = {}
    keys_to_unsqueeze = set()
    for key, value in data.items():
        if key == "cu_seqlens" or key == "max_seqlen":
            continue
        if not torch.is_tensor(value):
            not_to_split[key] = value
        else:
            assert value.numel() == total_lens, (key, value.shape)
            if value.shape[0] == 1:
                keys_to_unsqueeze.add(key)
                to_split[key] = value.squeeze()
            else:
                to_split[key] = value

    # split
    group_indices = allocate_balanced_mbs_synced(mb_spec, input_lens, group=group)
    splitted_lens = [
        [input_lens[i] for i in group_index] for group_index in group_indices
    ]
    group_lens = [sum(x) for x in splitted_lens]

    forward_indices = datapack.flat2d(group_indices)
    backward_indices = np.zeros(bs, dtype=np.int64)
    backward_indices[forward_indices] = np.arange(bs)

    to_split = dict_map(to_split, lambda x: unpack_sequence(x, cu_seqlens=cu_seqlens))
    to_split = dict_map(to_split, lambda x: recorder_list(x, forward_indices))
    to_split = dict_map(to_split, lambda x: torch.cat(x))
    to_split = dict_map(to_split, lambda x: unpack_sequence(x, lens=group_lens))
    mbs = dict_of_list2list_of_dict(to_split)

    results = []
    # organize splitted micro batches
    assert len(mbs) == len(splitted_lens), (len(mbs), len(splitted_lens))
    for i, (mb, lens) in enumerate(zip(mbs, splitted_lens)):
        mb = {
            k: v if k not in keys_to_unsqueeze else v.unsqueeze(0)
            for k, v in mb.items()
        }
        max_seqlen = max(lens)
        lens = torch.tensor(lens, device="cuda")
        batch_cu_seqlens = torch.nn.functional.pad(
            lens.cumsum(0, dtype=torch.int), (1, 0)
        )
        results.append(
            {
                **mb,
                **not_to_split,
                "max_seqlen": max_seqlen,
                "cu_seqlens": batch_cu_seqlens,
            }
        )
    return MicroBatchSplitResult(
        data=data,
        mbs=results,
        mb_spec=mb_spec,
        forward_indices=forward_indices,
        backward_indices=backward_indices,
    )


############### Tensor transformations end ###############


############### Tensor computations begin ###############


@torch.no_grad()
def compute_varlen_position_indices(
    total_seqlen: int,
    cu_seqlens: torch.Tensor,
    seqlen_offsets: Optional[torch.Tensor] = None,
) -> torch.LongTensor:
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


@torch.compile
@torch.no_grad()
def calc_entropy(logits, cu_seqlens):
    probs = torch.nn.functional.softmax(logits.detach().float(), dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-7), dim=-1)
    return entropy


@torch.no_grad()
def masked_normalization(
    x: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dim=None,
    inplace=False,
    unbiased=False,
    eps=1e-5,
    high_precision=True,
    all_reduce=True,
    reduce_group=None,
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
        dist.all_reduce(factor, op=dist.ReduceOp.SUM, group=reduce_group)
        dist.all_reduce(x_sum, op=dist.ReduceOp.SUM, group=reduce_group)
        dist.all_reduce(
            x_sum_sq,
            op=dist.ReduceOp.SUM,
            group=reduce_group,
        )
    mean = x_sum / factor
    meansq = x_sum_sq / factor
    var = meansq - mean**2
    if unbiased:
        var *= factor / (factor - 1)
    return ((x - mean) / (var.sqrt() + eps)).float()


def gather_logprobs(
    logits: torch.Tensor,
    labels: torch.Tensor,
):
    """Gather log probs from logits and labels.

    Args:
        logits (torch.Tensor): Shape [tot_seqlen]. The final value at the end of
            each sequence is not used.
        labels (torch.LongTensor): Labels or input_ids with shape [tot_seqlen].
            The first value at the beginning of each sequence has no corresponding log prob.

    Returns:
        torch.Tensor: Log probability with shape [tot_seqlen - #seqs].
    """
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return log_probs_labels


############### Tensor computations end ###############


def init_stats_logging(args: TrainingArgs):
    """
    Initialize wandb and/or tensorboard according to config.
    If torch.distributed is initialized

    Return:
        tensorboard SummaryWriter if args.tensorboard.path is not None
    """
    if dist.is_initialized() and dist.get_rank() != 0:
        return

    # wandb init, connect to remote wandb host
    if args.wandb.mode != "disabled":
        wandb.login()
    wandb.init(
        mode=args.wandb.mode,
        entity=args.wandb.entity,
        project=args.wandb.project or args.experiment_name,
        name=args.wandb.name or args.trial_name,
        job_type=args.wandb.job_type,
        group=args.wandb.group or f"{args.experiment_name}_{args.trial_name}",
        notes=args.wandb.notes,
        tags=args.wandb.tags,
        config=args.wandb.config,
        dir=constants.get_log_path(args),
        force=True,
        id=f"{args.experiment_name}_{args.trial_name}_train",
        resume="allow",
        settings=wandb.Settings(start_method="fork"),
    )
    # tensorboard logging
    summary_writer = None
    if args.tensorboard.path is not None:
        summary_writer = SummaryWriter(log_dir=args.tensorboard.path)

    return summary_writer


def log_wandb_tensorboard(step, data, summary_writer=None):
    if dist.is_initialized() and dist.get_rank() != 0:
        return

    wandb.log(data, step=step)
    if summary_writer is not None:
        for key, val in data.items():
            summary_writer.add_scalar(f"{key}", val, step)


def close_wandb_tensorboard(summary_writer=None):
    if dist.is_initialized() and dist.get_rank() != 0:
        return

    wandb.finish()
    if summary_writer is not None:
        summary_writer.close()


@contextmanager
def record_timing(name, timing_stats):
    start_time = time.perf_counter()
    yield
    timing_stats[name] = time.perf_counter() - start_time
