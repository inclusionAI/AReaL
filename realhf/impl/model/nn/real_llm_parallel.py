# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

from typing import *

import numpy as np
import torch

from realhf.api.core import model_api
from realhf.base import constants, datapack, logging
from realhf.impl.model.nn.real_llm_base import ReaLModelParamKeys

logger = logging.getLogger("ReaL parallel")

# keys used to identify modules
EMBEDDING_KEYS = [".wte", ".wpe"]  # dim=0 no bias
COLUMN_LINEAR_KEYS = [
    ".attn.c_attn.q_attn",
    ".attn.c_attn.k_attn",
    ".attn.c_attn.v_attn",
    ".mlp.c_fc",
    ".gate_proj",
    ".up_proj",
]  # dim=0 + partition bias
ROW_LINEAR_KEYS = [
    ".attn.c_proj",
    ".down_proj",
    ".mlp.c_proj",
]  # dim=1 + no partition bias

if constants.use_te_impl():
    COLUMN_LINEAR_KEYS = [
        ".attn.c_attn.q_attn",
        ".attn.c_attn.k_attn",
        ".attn.c_attn.v_attn",
        ".mlp.c_fc",
        ".mlp.fc1_weight",
    ]  # dim=0 + partition bias
    ROW_LINEAR_KEYS = [".attn.c_proj", ".mlp.fc2_weight"]


def tensor_slice_partition_fn(
    tensor: torch.Tensor,
    tp_rank: Optional[int],
    tp_world_size: int,
    dim: Optional[int],
) -> Union[List[torch.Tensor], torch.Tensor]:
    """Partition a tensor by slicing along a dimension for tensor-model
    parallelism."""
    if dim is None:
        splits = [tensor for _ in range(tp_world_size)]
    else:
        assert tensor.shape[dim] % tp_world_size == 0
        splits = torch.split(tensor, tensor.shape[dim] // tp_world_size, dim=dim)
    if tp_rank is None:
        return [s.contiguous() for s in splits]
    else:
        return splits[tp_rank].contiguous()


def intervals_partition_fn(
    shape: torch.Size,
    tp_rank: Optional[int],
    tp_world_size: int,
    dim: Optional[int],
) -> Union[List[torch.Tensor], torch.Tensor]:
    """Get the intervals of a MP-partitioned tensor in the flatten view.

    For example, if a tensor of shape (2, 4) is partitioned along the
    second dimension into 2 parts, then the intervals of the first part
    are [(0, 2), (2, 4)].

    Used by parameter reallocation. Return a numpy array of shape [N,
    2], where N is the number of intervals.
    """
    assert tp_rank is not None
    param_size = int(np.prod(shape))
    if dim is None:
        return np.array([(0, param_size)], dtype=np.int64)

    if dim < 0:
        dim = len(shape) + dim
    assert shape[dim] % tp_world_size == 0

    if len(shape) == 1:
        assert dim == 0
        partition_size = shape[0] // tp_world_size
        return np.array(
            [(partition_size * tp_rank, partition_size * (tp_rank + 1))],
            dtype=np.int64,
        )
    else:
        assert len(shape) == 2, shape
        if dim == 0:
            row_start = tp_rank * shape[0] // tp_world_size
            row_end = (tp_rank + 1) * shape[0] // tp_world_size
            return np.array(
                [(row_start * shape[1], row_end * shape[1])], dtype=np.int64
            )
        else:
            assert dim == 1
            col_start = tp_rank * shape[1] // tp_world_size
            col_end = (tp_rank + 1) * shape[1] // tp_world_size
            return np.arange(shape[0], dtype=np.int64)[:, None] * shape[1] + np.array(
                [(col_start, col_end)], dtype=np.int64
            )


def shape_partition_fn(
    shape: torch.Size,
    tp_rank: Optional[int],
    tp_world_size: int,
    dim: Optional[int],
):
    """Get the partitioned shape of a tensor for tensor-model parallelism."""
    if dim is None:
        splits = [shape for _ in range(tp_world_size)]
    else:
        if dim < 0:
            dim = len(shape) + dim
        assert shape[dim] % tp_world_size == 0
        splits = [
            (*shape[:dim], shape[dim] // tp_world_size, *shape[dim + 1 :])
            for _ in range(tp_world_size)
        ]
    if tp_rank is None:
        return [s for s in splits]
    else:
        return splits[tp_rank]


def tp_partition_key(
    key: str,
    tensor_or_shape: torch.Tensor | torch.Size,
    tp_rank: Optional[int],
    tp_size: Optional[int],
    config: model_api.ReaLModelConfig,
    partition_fn: Callable[
        [torch.Tensor, Optional[int], int, Optional[int]],
        Union[List[torch.Tensor], torch.Tensor],
    ] = tensor_slice_partition_fn,
) -> torch.Tensor:
    """Run the partition functor on the tensor or shape based on the key.

    The key determines the partitioning strategy, e.g., whether to
    perform partition and along which dimension.
    """

    if any([ek in key for ek in EMBEDDING_KEYS]):
        assert "weight" in key
        return partition_fn(tensor_or_shape, tp_rank, tp_size, dim=0)
    elif key == f"{config.n_layers + 1}.weight":  # output head
        if (
            isinstance(tensor_or_shape, torch.Tensor) and tensor_or_shape.shape[0] == 1
        ) or (
            not isinstance(tensor_or_shape, torch.Tensor) and tensor_or_shape[0] == 1
        ):
            assert config.is_critic
            return partition_fn(tensor_or_shape, tp_rank, tp_size, dim=None)
        else:
            return partition_fn(tensor_or_shape, tp_rank, tp_size, dim=0)
    elif any([ck in key for ck in COLUMN_LINEAR_KEYS]):
        if (
            ("k_attn" in key) or ("v_attn" in key)
        ) and config.n_kv_heads % tp_size != 0:
            return partition_fn(tensor_or_shape, tp_rank, tp_size, dim=None)
        # partition both weight and bias
        return partition_fn(tensor_or_shape, tp_rank, tp_size, dim=0)
    elif any([rk in key for rk in ROW_LINEAR_KEYS]):
        # only partition weight
        if "weight" in key:
            return partition_fn(tensor_or_shape, tp_rank, tp_size, dim=1)
        else:
            assert "bias" in key, key
            return partition_fn(tensor_or_shape, tp_rank, tp_size, dim=None)
    else:
        return partition_fn(tensor_or_shape, tp_rank, tp_size, dim=None)


def tp_partition_real_model_state_dict(
    state_dict: Dict[str, torch.Tensor],
    config: model_api.ReaLModelConfig,
    tp_size: int,
    tp_rank: Optional[int] = None,
) -> Union[Dict, List[Dict]]:
    """A helper function to partition a state dict using `tp_partition_key`."""
    if tp_size == 1:
        if tp_rank is None:
            return [state_dict]
        else:
            return state_dict

    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k] = tp_partition_key(k, v, tp_rank, tp_size, config)

    if tp_rank is None:
        return [
            {k: v[tp_rank] for k, v in new_state_dict.items()}
            for tp_rank in range(tp_size)
        ]
    else:
        return new_state_dict


def get_real_model_param_shape(
    k: str, config: model_api.ReaLModelConfig, tp_size: int
) -> Tuple:
    if "wte.weight" in k:
        assert config.vocab_size % tp_size == 0
        return (config.vocab_size // tp_size, config.hidden_dim)
    elif "wpe.weight" in k:
        assert config.n_positions % tp_size == 0
        if (config.n_positions + config.abs_position_embedding_offset) % tp_size != 0:
            raise ValueError(
                f"The dimenstion of position embedding "
                f"({config.n_positions} + offset {config.abs_position_embedding_offset}) "
                f"is not divisible by tp_size ({tp_size}). "
                "Models like this (e.g. OPT-350m) inherently do not support tensor parallelism."
            )
        return (
            (config.n_positions + config.abs_position_embedding_offset) // tp_size,
            config.hidden_dim,
        )
    elif ".ln." in k or ".ln_f." in k:
        return (config.hidden_dim,)
    elif ".q_ln." in k or ".k_ln." in k:
        return (config.head_dim,)
    elif k == f"{config.n_layers + 1}.weight":  # output head
        if config.is_critic:
            return (1, config.hidden_dim)
        elif tp_size > 1:
            assert config.vocab_size % tp_size == 0
            return (config.vocab_size // tp_size, config.hidden_dim)
        else:
            return (config.vocab_size, config.hidden_dim)
    elif any([ck in k for ck in COLUMN_LINEAR_KEYS]):
        if "k_attn" in k or "v_attn" in k:
            if "weight" in k:
                if config.n_kv_heads % tp_size == 0:
                    return (
                        config.head_dim * config.n_kv_heads // tp_size,
                        config.hidden_dim,
                    )
                else:
                    return (
                        config.head_dim * config.n_kv_heads,
                        config.hidden_dim,
                    )
            else:
                assert "bias" in k
                if config.n_kv_heads % tp_size == 0:
                    return (config.head_dim * config.n_kv_heads // tp_size,)
                else:
                    return (config.head_dim * config.n_kv_heads,)
        if "mlp" in k:
            if "weight" in k:
                return (config.intermediate_dim // tp_size, config.hidden_dim)
            else:
                assert "bias" in k
                return (config.intermediate_dim // tp_size,)
        if "weight" in k:
            assert config.n_q_heads % tp_size == 0
            return (config.n_q_heads * config.head_dim // tp_size, config.hidden_dim)
        else:
            assert "bias" in k
            return (config.n_q_heads * config.head_dim // tp_size,)
    elif any([rk in k for rk in ROW_LINEAR_KEYS]):
        if "mlp" in k and "weight" in k:
            return (config.hidden_dim, config.intermediate_dim // tp_size)
        elif "attn" in k and "weight" in k:
            return (config.hidden_dim, config.n_q_heads * config.head_dim // tp_size)
        elif "bias" in k:
            return (config.hidden_dim,)
        else:
            raise NotImplementedError(f"unkown shape of key {k}.")
    elif ".mlp.router" in k:
        # mp does not partition router weights
        return (config.moe.num_experts, config.hidden_dim)
    else:
        raise NotImplementedError(f"unkown shape of key {k}.")


def tp_merge_key(
    k: str,
    tensors: List[torch.Tensor],
    config: model_api.ReaLModelConfig,
) -> torch.Tensor:
    if any([ek in k for ek in EMBEDDING_KEYS]) and "weight" in k:
        return torch.cat(tensors, dim=0)
    elif k == f"{config.n_layers + 1}.weight" and not config.is_critic:
        return torch.cat(tensors, dim=0)
    elif any([ck in k for ck in COLUMN_LINEAR_KEYS]):
        return torch.cat(tensors, dim=0)
    elif any([rk in k for rk in ROW_LINEAR_KEYS]) and "weight" in k:
        return torch.cat(tensors, dim=1)
    else:
        return tensors[0]


def tp_merge_real_model_state_dict(
    state_dicts: List[Dict[str, torch.Tensor]],
    config: model_api.ReaLModelConfig,
) -> Dict:
    tp_size = len(state_dicts)
    if tp_size == 1:
        return state_dicts[0]

    new_state_dict = {}
    for k in state_dicts[0].keys():
        new_state_dict[k] = tp_merge_key(k, [sd[k] for sd in state_dicts], config)

    return new_state_dict


class ReaLModelParamCount:
    """Paramter count, used for partitioning pipeline stages."""

    @staticmethod
    def _derive_count_from_keys(
        keys: List[str], config: model_api.ReaLModelConfig, tp_size: int
    ) -> int:
        count = 0
        for k in keys:
            count += np.prod(get_real_model_param_shape(k, config, tp_size))
        return int(count)

    @staticmethod
    def embed(config: model_api.ReaLModelConfig, tp_size: int) -> int:
        return ReaLModelParamCount._derive_count_from_keys(
            ReaLModelParamKeys.embed(config), config, tp_size
        )

    @staticmethod
    def tblock(config: model_api.ReaLModelConfig, idx: int, tp_size: int) -> int:
        return ReaLModelParamCount._derive_count_from_keys(
            ReaLModelParamKeys.tblock(config, idx), config, tp_size
        )

    @staticmethod
    def head(config: model_api.ReaLModelConfig, tp_size: int) -> int:
        return ReaLModelParamCount._derive_count_from_keys(
            ReaLModelParamKeys.head(config), config, tp_size
        )

    @staticmethod
    def total(config: model_api.ReaLModelConfig, idx: int, tp_size: int) -> int:
        return (
            config.n_layers * ReaLModelParamCount.tblock(config, idx, tp_size)
            + ReaLModelParamCount.head(config, tp_size)
            + ReaLModelParamCount.embed(config, tp_size)
        )


def partition_pipeline_layers(
    config: model_api.ReaLModelConfig,
    num_stages: int,
    method: str = "parameters",
) -> Dict[int, Tuple[int, int]]:
    # Ignoring tp_size in param count because tensor parallel equally partitions parameters.
    # It is irrelevant to how we partition pipeline stages.
    param_counts = (
        [ReaLModelParamCount.embed(config, 1)]
        + [ReaLModelParamCount.tblock(config, i, 1) for i in range(config.n_layers)]
        + [ReaLModelParamCount.head(config, 1)]
    )

    parts = None
    if method == "uniform":
        # Each stage gets a simple uniform number of layers.
        from deepspeed.runtime import utils as ds_utils

        parts = ds_utils.partition_uniform(
            num_items=config.n_layers + 2, num_parts=num_stages
        )
    elif method == "parameters":
        # Partition according to the parameter count.
        param_counts = np.array(param_counts)
        parts = datapack.partition_balanced(param_counts, k=num_stages)
    else:
        raise NotImplementedError(f"Partitioning method {method} not implemented.")

    stage_to_layer_idx = {}
    for stage in range(num_stages):
        start = parts[stage]
        stop = parts[stage + 1]
        stage_to_layer_idx[stage] = (start, stop)
    return stage_to_layer_idx


def pipeline_repartition_strategy(
    layer_mapping1: Dict[int, List[int]],
    layer_mapping2: Dict[int, List[int]],
):
    assert set(sum(layer_mapping1.values(), [])) == set(
        sum(layer_mapping2.values(), [])
    )
    assert all(isinstance(i, int) for i in layer_mapping1)
    assert all(isinstance(i, int) for i in layer_mapping2)

    layer_mapping1 = dict(sorted(layer_mapping1.items()))
    layer_mapping2 = dict(sorted(layer_mapping2.items()))

    layer_map: Dict[Tuple[int, int], List[int]] = {}
    for pp_rank2, layer_indices2 in layer_mapping2.items():
        for pp_rank1, layer_indices1 in layer_mapping1.items():
            layer_map[(pp_rank1, pp_rank2)] = sorted(
                list(set(layer_indices1).intersection(set(layer_indices2)))
            )

    return layer_map
