# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import contextlib
import dataclasses
import math
import os
import subprocess
from typing import *

import numpy as np
import torch
import torch.utils.cpp_extension as torch_cpp_ext
from packaging.version import Version, parse

import realhf
from realhf.api.core import model_api
from realhf.api.core.config import ModelName
from realhf.base import constants, logging

from .real_llm_base import ReaLModelParamKeys
from .real_llm_parallel import (
    get_real_model_param_shape,
    intervals_partition_fn,
    shape_partition_fn,
    tp_partition_key,
)

try:
    from realhf._C.interval_op import merge_intervals
except ImportError:
    merge_intervals = None
logger = logging.getLogger("FlattenParam")

_FLAT_PARAM_INDICES_CACHE = {}


@dataclasses.dataclass
class ContiguousParamSpec:
    start_idx: int
    end_idx: int
    shape: torch.Size


def _is_integer_list_contiguous(l: List[int]) -> bool:
    return np.all(np.array(l) == np.arange(len(l)) + l[0])


def _are_intervals_contiguous(l: List[Tuple[int, int]]) -> bool:
    l = sorted(l, key=lambda x: x[0])
    res = True
    for i in range(len(l) - 1):
        res &= l[i][1] == l[i + 1][0]
    return res


def recursive_getattr(obj, attr_string):
    attrs = attr_string.split(".")
    for attr in attrs:
        obj = getattr(obj, attr)
    return obj


def _slice_intervals_py(src: torch.Tensor, intervals: List[Tuple[int, int]]):
    # Drop-in replacement for the C++ implementation.
    assert len(src.shape) == 1
    assert all([x[0] >= 0 for x in intervals])
    assert all([x[1] <= src.shape[0] for x in intervals])
    N = len(intervals)
    slices = []
    for i, j in intervals:
        slices.append(src[i:j])
    return torch.cat(slices, dim=0)


def _set_intervals_py(
    src: torch.Tensor,
    dst: torch.Tensor,
    intervals: List[Tuple[int, int]],
):
    # Drop-in replacement for the C++ implementation.
    assert len(dst.shape) == len(src.shape) == 1
    offset = 0
    for i, j in intervals:
        assert i >= 0
        assert j <= dst.shape[0], (j, dst.shape[0])
        dst[i:j] = src[offset : offset + j - i]
        offset += j - i
    assert offset == src.shape[0]


_SLICE_INTERVAL_EXT_WARNED = False
_SET_INTERVAL_EXT_WARNED = False


def slice_intervals(
    src: torch.Tensor,
    intervals_cpu: List[Tuple[int, int]] = None,
    intervals_cuda: torch.Tensor = None,
    output_size: int = None,
    max_interval_size: Optional[int] = None,
):
    if intervals_cpu is not None:
        N = len(intervals_cpu)
    else:
        N = intervals_cuda.size(0)
    if not constants.use_cuda() or N < 1024:
        # NOTE: The CUDA implementation will launch a thread for each interval,
        # which has a negative effect when the number of intervals is small.
        return _slice_intervals_py(src, intervals_cpu)
    try:
        from realhf._C.interval_op_cuda import (
            slice_intervals_bf16,
            slice_intervals_fp16,
            slice_intervals_fp32,
        )

        if src.dtype == torch.float32:
            return slice_intervals_fp32(
                src, intervals_cuda, output_size, max_interval_size
            )
        elif src.dtype == torch.float16:
            return slice_intervals_fp16(
                src, intervals_cuda, output_size, max_interval_size
            )
        elif src.dtype == torch.bfloat16:
            return slice_intervals_bf16(
                src, intervals_cuda, output_size, max_interval_size
            )
        else:
            raise NotImplementedError(src.dtype)
    except ImportError:
        global _SLICE_INTERVAL_EXT_WARNED
        if not _SLICE_INTERVAL_EXT_WARNED:
            _SLICE_INTERVAL_EXT_WARNED = True
            logger.warning(
                f"The `slice_interval` extension not found. "
                "Fallback to python, which can be very slow. "
                "You should re-install the package with REAL_CUDA=1 or "
                "set REAL_PARAM_REALLOC_OPT_LEVEL=1."
            )
        return _slice_intervals_py(src, intervals_cpu)


def set_intervals(
    src: torch.Tensor,
    dst: torch.Tensor,
    intervals_cpu: List[Tuple[int, int]] = None,
    intervals_cuda: torch.Tensor = None,
    max_interval_size: Optional[int] = None,
):
    if intervals_cpu is not None:
        N = len(intervals_cpu)
    else:
        N = intervals_cuda.size(0)
    if not constants.use_cuda() or N < 512 or not (src.is_cuda and dst.is_cuda):
        # NOTE: The CUDA implementation will launch a thread for each interval,
        # which has a negative effect when the number of intervals is small.
        return _set_intervals_py(src, dst, intervals_cpu)
    try:
        from realhf._C.interval_op_cuda import (
            set_intervals_bf16,
            set_intervals_fp16,
            set_intervals_fp32,
        )

        if src.dtype == torch.float32:
            return set_intervals_fp32(src, dst, intervals_cuda, max_interval_size)
        elif src.dtype == torch.float16:
            return set_intervals_fp16(src, dst, intervals_cuda, max_interval_size)
        elif src.dtype == torch.bfloat16:
            return set_intervals_bf16(src, dst, intervals_cuda, max_interval_size)
        else:
            raise NotImplementedError(src.dtype)
    except ImportError:
        global _SET_INTERVAL_EXT_WARNED
        if not _SET_INTERVAL_EXT_WARNED:
            _SET_INTERVAL_EXT_WARNED = True
            logger.warning(
                f"The `set_interval` extension not found. "
                "Fallback to python, which can be very slow. "
                "You should re-install the package with REAL_CUDA=1 or "
                "set REAL_PARAM_REALLOC_OPT_LEVEL=1."
            )
        return _set_intervals_py(src, dst, intervals_cpu)


def param_size_from_keys(
    config: model_api.ReaLModelConfig,
    src_tp_size: int,
    sd_keys: List[str],
    src2dst_tp_size: int,
    src2dst_tp_rank: int,
    head_param_point_to_embedding: bool,
) -> Tuple[List[int], int]:
    param_size = 0
    for k in sd_keys:
        if (
            head_param_point_to_embedding
            and k == f"{config.n_layers + 1}.weight"
            and "0.wte.weight" in sd_keys
        ):
            continue
        new_shape = tp_partition_key(
            k,
            get_real_model_param_shape(k, config, src_tp_size),
            src2dst_tp_rank,
            src2dst_tp_size,
            config,
            partition_fn=shape_partition_fn,
        )
        param_size += int(np.prod(new_shape))
    return param_size


def build_param_spec(
    layer_indices: List[int],
    config: model_api.ReaLModelConfig,
    dp_size: int,
    tp_size: int,
    pp_size: int,
    head_param_point_to_embedding: bool,
    bucket_size: int = 40000000,
) -> Tuple[Dict[str, ContiguousParamSpec], int]:
    # TODO: omit parameters that do not require gradient?
    # TODO: allow different dtypes for different buckets
    if len(layer_indices) == 0:
        return {}, 0

    disable_bucketing = 0 not in layer_indices

    sd_keys = []
    for layer_idx in sorted(layer_indices):
        if layer_idx == 0:
            sd_keys += ReaLModelParamKeys.embed(config)
        elif layer_idx == config.n_layers + 1:
            sd_keys += ReaLModelParamKeys.head(config)
        else:
            sd_keys += ReaLModelParamKeys.tblock(config, layer_idx - 1)

    # In the reverse order as backpropagation, consistent with Megatron.
    sd_keys = list(reversed(sd_keys))

    data_start_index = 0
    bucket_data_start_index = data_start_index
    bucket_params = set()

    def _requires_new_allreduce_bucket(k):
        if pp_size == 1:
            return False
        if config.is_critic:
            return False
        if not config.tied_embedding:
            return False
        return k == f"{config.n_layers + 1}.weight" or k == "0.wte.weight"

    def _pad_to_multiple(x, m):
        return int(math.ceil(x / m)) * m

    def _create_fake_bucket(data_end_index) -> int:
        nonlocal bucket_data_start_index, bucket_params
        data_end_index = _pad_to_multiple(data_end_index, dp_size)
        # Update bucket metadata.
        bucket_data_start_index = data_end_index
        # Re-set bucket_params and increment bucket_id for next bucket.
        bucket_params = set()
        # Return the potentially padded data_end_index.
        return data_end_index

    param_spec = {}
    for k in sd_keys:
        if head_param_point_to_embedding and k == f"{config.n_layers + 1}.weight":
            continue

        shape = get_real_model_param_shape(k, config, tp_size)
        numel = int(np.prod(shape))
        data_end_index = data_start_index + numel

        if _requires_new_allreduce_bucket(k) and len(bucket_params) > 0:
            _create_fake_bucket(data_start_index)

        param_spec[k] = ContiguousParamSpec(
            data_start_index,
            data_end_index,
            shape,
        )
        bucket_params.add(k)
        if (
            not disable_bucketing
            and (data_end_index - bucket_data_start_index) >= bucket_size
        ) or _requires_new_allreduce_bucket(k):
            data_end_index = _create_fake_bucket(data_end_index)

        data_start_index = data_end_index

    if len(bucket_params) > 0:
        data_end_index = _create_fake_bucket(data_end_index)

    if head_param_point_to_embedding and f"{config.n_layers + 1}.weight" in sd_keys:
        param_spec[f"{config.n_layers + 1}.weight"] = param_spec["0.wte.weight"]
    return param_spec, data_end_index


def param_intervals_from_keys(
    model_name: ModelName,
    config: model_api.ReaLModelConfig,
    head_param_point_to_embedding: bool,
    param_spec: Dict[str, ContiguousParamSpec],
    tp_size: int,
    sd_keys: List[str],
    portion_size: int,
    portion_rank: int,
) -> List[int]:
    param_size = param_size_from_keys(
        config=config,
        src_tp_size=tp_size,
        sd_keys=sd_keys,
        src2dst_tp_size=portion_size,
        src2dst_tp_rank=portion_rank,
        head_param_point_to_embedding=head_param_point_to_embedding,
    )

    interval_size = 0
    intervals = []
    for k in sd_keys:
        if (
            head_param_point_to_embedding
            and k == f"{config.n_layers + 1}.weight"
            and "0.wte.weight" in sd_keys
        ):
            continue
        if (
            model_name,
            k.split(".", 1)[1],
            tp_size,
            portion_rank,
            portion_size,
        ) not in _FLAT_PARAM_INDICES_CACHE:
            zero_start_intervals = tp_partition_key(
                k,
                get_real_model_param_shape(k, config, tp_size),
                portion_rank,
                portion_size,
                config,
                partition_fn=intervals_partition_fn,
            )
            _FLAT_PARAM_INDICES_CACHE[
                (
                    model_name,
                    k.split(".", 1)[1],
                    tp_size,
                    portion_rank,
                    portion_size,
                )
            ] = zero_start_intervals
        else:
            zero_start_intervals = _FLAT_PARAM_INDICES_CACHE[
                (
                    model_name,
                    k.split(".", 1)[1],
                    tp_size,
                    portion_rank,
                    portion_size,
                )
            ]
        intervals += (zero_start_intervals + param_spec[k].start_idx).tolist()
        interval_size += sum(zero_start_intervals[:, 1] - zero_start_intervals[:, 0])
    # assert len(set([x[0] for x in intervals])) == len(intervals)
    assert interval_size == param_size, (interval_size, param_size)
    if merge_intervals is not None:
        intervals = merge_intervals(intervals)
    return intervals


def map_param_to_contigous_memory(
    layers: torch.nn.ModuleList,
    config: model_api.ReaLModelConfig,
    head_param_point_to_embedding: bool,
    param_spec: Dict[str, ContiguousParamSpec],
    contiguous_param: torch.Tensor,
    layer_idx_offset: int,
    allocate_only: bool,
):
    for local_layer_idx, l in enumerate(layers):
        layer_idx = local_layer_idx + layer_idx_offset
        for k, v in l.named_parameters():
            spec = param_spec[f"{layer_idx}.{k}"]
            old_param_data = v.data
            target = contiguous_param[spec.start_idx : spec.end_idx].view(spec.shape)
            if not allocate_only:
                target.copy_(old_param_data)
            else:
                if not (
                    head_param_point_to_embedding and layer_idx == config.n_layers + 1
                ):
                    assert old_param_data.shape == torch.Size([0]), (
                        old_param_data.shape,
                        spec.shape,
                        f"{layer_idx}.{k}",
                    )
            recursive_getattr(l, k).data = target
