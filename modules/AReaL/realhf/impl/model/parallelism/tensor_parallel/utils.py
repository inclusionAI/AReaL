# Modified from Megatron-LM.
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from typing import List, Sequence

import numpy as np
import torch

import realhf.base.constants as constants

_MODEL_PARALLEL_ATTRIBUTE_DEFAULTS = {
    "tensor_model_parallel": False,
    "partition_dim": -1,
    "partition_stride": 1,
}


def param_is_not_model_parallel_duplicate(param):
    return (
        hasattr(param, "tensor_model_parallel") and param.tensor_model_parallel
    ) or (constants.tensor_parallel_rank() == 0)


def set_tensor_model_parallel_attributes(tensor, is_parallel, dim, stride):
    # Make sure the attributes are not set.
    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        assert not hasattr(tensor, attribute)
    # Set the attributes.
    setattr(tensor, "tensor_model_parallel", is_parallel)
    setattr(tensor, "partition_dim", dim)
    setattr(tensor, "partition_stride", stride)


def set_defaults_if_not_set_tensor_model_parallel_attributes(tensor):

    def maybe_set(attribute, value):
        if not hasattr(tensor, attribute):
            setattr(tensor, attribute, value)

    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        maybe_set(attribute, _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS[attribute])


def copy_tensor_model_parallel_attributes(destination_tensor, source_tensor):

    def maybe_copy(attribute):
        if hasattr(source_tensor, attribute):
            setattr(destination_tensor, attribute, getattr(source_tensor, attribute))

    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        maybe_copy(attribute)


def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, "{} is not divisible by {}".format(
        numerator, denominator
    )


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return the
    division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


def split_tensor_along_last_dim(
    tensor: torch.Tensor,
    num_partitions: int,
    contiguous_split_chunks: bool = False,
) -> List[torch.Tensor]:
    """Split a tensor along its last dimension.

    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.

    Returns:
        A list of Tensors
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = divide(tensor.size()[last_dim], num_partitions)
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list


def split_tensor_into_1d_equal_chunks(tensor, new_buffer=False):
    """Break a tensor into equal 1D chunks across tensor parallel ranks.

    Returns a Tensor or View with this rank's portion of the data.

    Arguments:
        tensor: The tensor to split

    Keyword Arguments:
        new_buffer (bool): If True, returns a new Tensor.
                           If False, returns a view into the existing Tensor.
                           Default is False
    """
    partition_size = torch.numel(tensor) // constants.tensor_parallel_world_size()
    start_index = partition_size * constants.tensor_parallel_rank()
    end_index = start_index + partition_size
    if new_buffer:
        data = torch.empty(
            partition_size,
            dtype=tensor.dtype,
            device=constants.current_device(),
            requires_grad=False,
        )
        data.copy_(tensor.view(-1)[start_index:end_index])
    else:
        data = tensor.view(-1)[start_index:end_index]
    return data


def gather_split_1d_tensor(tensor):
    """Opposite of split_tensor_into_1d_equal_chunks. Gather values from tensor
    model parallel ranks.

    Returns a new Tensor with the gathered data.

    Arguments:
        tensor: A Tensor or view of this rank's portion of the data.
    """
    numel_gathered = torch.numel(tensor) * constants.tensor_parallel_world_size()
    gathered = torch.empty(
        numel_gathered,
        dtype=tensor.dtype,
        device=constants.current_device(),
        requires_grad=False,
    )
    # NOTE: This API is experimental in pytorch (as of Feb 2022) and
    # this might break in future pytorch releases. We chose this API
    # as opposed to torch.distributed.all_gather for efficiency reasons.
    # This API calls directly NCCL all-gather versus the former does
    # internal copies and can potentially cause slow down.
    torch.distributed._all_gather_base(
        gathered, tensor, group=constants.tensor_parallel_group()
    )
    return gathered


class VocabUtility:
    """Split the vocabulary into `world_size` chunks and return the first
    and last index of the vocabulary belonging to the `rank`
    partition: Note that indices in [fist, last)

    """

    @staticmethod
    def vocab_range_from_per_partition_vocab_size(
        per_partition_vocab_size: int, rank, world_size: int
    ) -> Sequence[int]:
        index_f = rank * per_partition_vocab_size
        index_l = index_f + per_partition_vocab_size
        return index_f, index_l

    @staticmethod
    def vocab_range_from_global_vocab_size(
        global_vocab_size: int, rank: int, world_size: int
    ) -> Sequence[int]:
        per_partition_vocab_size = divide(global_vocab_size, world_size)
        return VocabUtility.vocab_range_from_per_partition_vocab_size(
            per_partition_vocab_size, rank, world_size
        )


def assert_viewless_tensor(tensor, extra_msg=None):
    """Assert that a tensor is not a view (i.e., its '._base' field is not
    set)."""
    if isinstance(tensor, list):
        [assert_viewless_tensor(t) for t in tensor]
        return tensor
    if not isinstance(tensor, torch.Tensor):
        return tensor
    assert tensor._base is None, (
        "Ensure tensor._base is None before setting tensor.data or storing "
        "tensor to memory buffer. Otherwise, a memory leak will occur (and "
        "likely accumulate over iterations). %s"
    ) % extra_msg
    return tensor


def safely_set_viewless_tensor_data(tensor, new_data_tensor):
    """Safely set tensor's '.data' field.

    Check first that the tensor is viewless (i.e., '._base' not set). If
    not, raise an exception.
    """
    assert_viewless_tensor(
        tensor,
        extra_msg="FYI, tensor._base has shape %s, and new_data_tensor has shape %s."
        % (
            "--" if tensor._base is None else tensor._base.shape,
            new_data_tensor.shape,
        ),
    )
    tensor.data = new_data_tensor
