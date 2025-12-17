import json
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from glob import glob

import torch
import torch.distributed as dist
from mbridge.core.bridge import Bridge
from megatron.core import parallel_state as mpu
from megatron.core.fp8_utils import is_float8tensor
from safetensors import safe_open
from transformer_engine.pytorch.constants import TE_DType_To_Torch

from areal.platforms import current_platform
from areal.utils import logging
from areal.utils.fp8_utils import dequantize_params

logger = logging.getLogger("HF WeightsLoader")


def _get_tp_slice(shape, dim, tp_rank, tp_size) -> tuple:
    size_per_tp = shape[dim] // tp_size
    res = [slice(None) for _ in range(dim)]
    res.append(slice(tp_rank * size_per_tp, (tp_rank + 1) * size_per_tp))
    return tuple(res)


def _get_shape(obj) -> list:
    """Get shape from either a tensor or PySafeSlice object."""
    if isinstance(obj, torch.Tensor):
        return list(obj.shape)
    else:
        # PySafeSlice object
        return obj.get_shape()


def _pytorch_fp8_to_te_fp8(
    pytorch_fp8_tensor: torch.Tensor,
    scale_inv: torch.Tensor,
    target_te_tensor: torch.Tensor,
) -> None:
    """Convert PyTorch float8 tensor to Transformer Engine Float8BlockwiseQTensor format inplace.

    This function copies the data and scale_inv from a PyTorch float8 tensor
    to an existing TE Float8BlockwiseQTensor

    Args:
        pytorch_fp8_tensor: PyTorch float8 tensor (like torch.float8_e4m3fn)
        scale_inv: Inverse scale tensor (1/scale) with blockwise shape
        target_te_tensor: Target TE Float8BlockwiseQTensor to copy into
    """
    if not is_float8tensor(target_te_tensor):
        raise ValueError("target_te_tensor must be a Transformer Engine Float8Tensor")

    # For Float8BlockwiseQTensor, copy rowwise_data and rowwise_scale_inv
    if hasattr(target_te_tensor, "_rowwise_data") and hasattr(
        target_te_tensor, "_rowwise_scale_inv"
    ):
        assert pytorch_fp8_tensor.shape == target_te_tensor._rowwise_data.shape
        # rowwise_data is stored in uint8 format
        target_te_tensor._rowwise_data.copy_(
            pytorch_fp8_tensor.view(torch.uint8), non_blocking=True
        )
        target_te_tensor._columnwise_data.copy_(
            pytorch_fp8_tensor.t().contiguous().view(torch.uint8), non_blocking=True
        )
        scale_inv_shape = scale_inv.shape
        assert len(scale_inv_shape) == 2
        target_te_tensor._rowwise_scale_inv[
            : scale_inv_shape[0], : scale_inv_shape[1]
        ].copy_(scale_inv, non_blocking=True)
        target_te_tensor._columnwise_scale_inv[
            : scale_inv_shape[1], : scale_inv_shape[0]
        ].copy_(scale_inv.t().contiguous(), non_blocking=True)
        # target_te_tensor._create_columnwise()

    else:
        # Fallback for non-blockwise tensors
        target_te_tensor._data.copy_(pytorch_fp8_tensor.view(torch.uint8))
        if scale_inv.numel() == 1:
            target_te_tensor._scale_inv.fill_(scale_inv.item())
        else:
            target_te_tensor._scale_inv.copy_(scale_inv)


def _get_tp_slice_for_scale_inv(
    scale_inv_shape: list,
    weight_shape: list,
    partition_dim: int,
    tp_rank: int,
    tp_size: int,
    weight_block_size: list[int, int],
) -> tuple:
    """Get TP slice for scale_inv tensor.

    Args:
        scale_inv_shape: Shape of scale_inv tensor [M/block_size, N/block_size]
        weight_shape: Shape of weight tensor [M, N]
        partition_dim: Dimension along which weight is partitioned
        tp_rank: TP rank
        tp_size: TP size
        weight_block_size: Block size [block_m, block_n]

    Returns:
        Tuple of slices for scale_inv
    """
    # scale_inv shape is [M/block_m, N/block_n] for weight shape [M, N]
    # When weight is partitioned along partition_dim, scale_inv should be partitioned accordingly
    slices = [slice(None)] * len(scale_inv_shape)
    block_size = weight_block_size[partition_dim]
    size_per_tp = weight_shape[partition_dim] // tp_size
    assert size_per_tp % block_size == 0, (
        f"TP split size {size_per_tp} must be divisible by block_size {block_size}"
    )
    scale_inv_size_per_tp = size_per_tp // block_size
    slices[partition_dim] = slice(
        tp_rank * scale_inv_size_per_tp, (tp_rank + 1) * scale_inv_size_per_tp
    )

    return tuple(slices)


def _weight_to_mcore_tp(
    hf_config,
    mcore_weights_name: str,
    mcore_param_shape: list,
    hf_weights_safe_slice: list,
    tp_rank: int,
    tp_size: int,
    dtype: torch.dtype | None = None,
    hf_scale_invs: list | None = None,
    weight_block_size: list[int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if (
        "self_attention.linear_qkv." in mcore_weights_name
        and "layer_norm" not in mcore_weights_name
    ):
        # merge qkv
        assert len(hf_weights_safe_slice) == 3
        num_key_value_heads = hf_config.num_key_value_heads
        hidden_dim = hf_config.hidden_size
        num_attention_heads = hf_config.num_attention_heads
        head_dim = getattr(hf_config, "head_dim", hidden_dim // num_attention_heads)
        group_dim = head_dim * num_attention_heads // num_key_value_heads
        q, k, v = hf_weights_safe_slice
        # q k v might be tp split
        real_num_key_value_heads = _get_shape(q)[0] // group_dim
        s = _get_tp_slice((real_num_key_value_heads * group_dim,), 0, tp_rank, tp_size)
        q = q[s].reshape(
            real_num_key_value_heads // tp_size,
            group_dim,
            -1,
        )
        s = _get_tp_slice((real_num_key_value_heads * head_dim,), 0, tp_rank, tp_size)
        k = k[s].reshape(real_num_key_value_heads // tp_size, head_dim, -1)
        v = v[s].reshape(real_num_key_value_heads // tp_size, head_dim, -1)
        out_shape = [-1, hidden_dim] if ".bias" not in mcore_weights_name else [-1]
        res = torch.cat([q, k, v], dim=1).view(*out_shape).contiguous()

        # Merge scale_inv for FP8: merge along dim 1 (q/k/v -> qkv)
        scale_inv = None
        if hf_scale_invs is not None and len(hf_scale_invs) == 3:
            q_scale_inv, k_scale_inv, v_scale_inv = hf_scale_invs
            if (
                q_scale_inv is not None
                and k_scale_inv is not None
                and v_scale_inv is not None
            ):
                if weight_block_size is not None:
                    # q, k, v weights are split along dim=0, so scale_inv should be split along dim=0 first
                    # Get original weight shapes for q (assuming they have same shape)
                    # q_shape = _get_shape(hf_weights_safe_slice[0])

                    scale_inv_shape = _get_shape(q_scale_inv)
                    # TP split scale_inv along dim=0
                    slices = _get_tp_slice(scale_inv_shape, 0, tp_rank, tp_size)
                    # slices = _get_tp_slice_for_scale_inv(
                    #     q_scale_inv_shape, q_shape, 0, tp_rank, tp_size, weight_block_size
                    # )
                    q_scale_inv = q_scale_inv[slices]
                    scale_inv_shape = _get_shape(k_scale_inv)
                    slices = _get_tp_slice(scale_inv_shape, 0, tp_rank, tp_size)
                    k_scale_inv = k_scale_inv[slices]
                    v_scale_inv = v_scale_inv[slices]
                    # Then merge along dim=1
                    scale_inv = torch.cat(
                        [q_scale_inv, k_scale_inv, v_scale_inv], dim=0
                    )
                else:
                    # Per-tensor quantization: take max
                    raise NotImplementedError(
                        "Per-tensor quantization is not supported for FP8"
                    )
                    # scale_inv = torch.maximum(q_scale_inv, k_scale_inv, v_scale_inv)
    elif (
        "linear_fc1.weight" in mcore_weights_name
        or "linear_fc1.bias" in mcore_weights_name
    ):
        # merge gate_proj and up_proj
        assert len(hf_weights_safe_slice) == 2, len(hf_weights_safe_slice)
        gate, up = hf_weights_safe_slice
        # chunk 0 for TP split
        gate = gate[
            _get_tp_slice(_get_shape(gate), dim=0, tp_rank=tp_rank, tp_size=tp_size)
        ]
        up = up[_get_tp_slice(_get_shape(up), dim=0, tp_rank=tp_rank, tp_size=tp_size)]
        res = torch.cat([gate, up], dim=0)

        # Merge scale_inv for FP8: merge along dim 0 (gate/up -> fc1)
        scale_inv = None
        if hf_scale_invs is not None and len(hf_scale_invs) == 2:
            gate_scale_inv, up_scale_inv = hf_scale_invs
            if gate_scale_inv is not None and up_scale_inv is not None:
                if weight_block_size is not None:
                    # gate, up weights are split along dim=0, so scale_inv should be split along dim=0 first
                    # gate_shape = _get_shape(hf_weights_safe_slice[0])
                    # gate_scale_inv_shape = _get_shape(gate_scale_inv)
                    # TP split scale_inv along dim=0
                    # slices = _get_tp_slice_for_scale_inv(
                    #     gate_scale_inv_shape, gate_shape, 0, tp_rank, tp_size, weight_block_size
                    # )
                    slices = _get_tp_slice(
                        _get_shape(gate_scale_inv), 0, tp_rank, tp_size
                    )
                    gate_scale_inv = gate_scale_inv[slices]
                    slices = _get_tp_slice(
                        _get_shape(up_scale_inv), 0, tp_rank, tp_size
                    )
                    up_scale_inv = up_scale_inv[slices]
                    scale_inv = torch.cat([gate_scale_inv, up_scale_inv], dim=0)
                else:
                    # Per-tensor quantization: take max
                    raise NotImplementedError(
                        "Per-tensor quantization is not supported for FP8"
                    )
                    # scale_inv = torch.maximum(gate_scale_inv, up_scale_inv)
    elif "mlp.experts.linear_fc2.weight" in mcore_weights_name:  # moe
        assert len(hf_weights_safe_slice) == 1
        x = hf_weights_safe_slice[0]
        shape = _get_shape(x)
        # dim 1 chunk
        partition_dim = 1
        res = x[
            _get_tp_slice(shape, dim=partition_dim, tp_rank=tp_rank, tp_size=tp_size)
        ]

        # Handle TP split for scale_inv
        scale_inv = None
        if (
            hf_scale_invs is not None
            and len(hf_scale_invs) == 1
            and hf_scale_invs[0] is not None
        ):
            scale_inv = hf_scale_invs[0]
            if weight_block_size is not None:
                scale_inv_shape = _get_shape(scale_inv)
                # slices = _get_tp_slice_for_scale_inv(
                #     scale_inv_shape, shape, partition_dim, tp_rank, tp_size, weight_block_size
                # )
                slices = _get_tp_slice(scale_inv_shape, partition_dim, tp_rank, tp_size)
                scale_inv = scale_inv[slices]
    else:
        assert len(hf_weights_safe_slice) == 1
        x = hf_weights_safe_slice[0]
        x_shape = _get_shape(x)
        partition_dim = None
        if mcore_param_shape == x_shape:
            res = x[:] if not isinstance(x, torch.Tensor) else x
        else:
            assert len(x_shape) == len(mcore_param_shape)
            for dim, (s1, s2) in enumerate(zip(x_shape, mcore_param_shape)):
                if s1 != s2:
                    partition_dim = dim
                    break
            # chunk on `partition_dim`
            res = x[
                _get_tp_slice(
                    x_shape, dim=partition_dim, tp_rank=tp_rank, tp_size=tp_size
                )
            ]

        scale_inv = None
        if (
            hf_scale_invs is not None
            and len(hf_scale_invs) == 1
            and hf_scale_invs[0] is not None
        ):
            scale_inv = hf_scale_invs[0]
            if weight_block_size is not None:
                if partition_dim is not None:
                    scale_inv_shape = _get_shape(scale_inv)
                    # slices = _get_tp_slice_for_scale_inv(
                    #     scale_inv_shape, x_shape, partition_dim, tp_rank, tp_size, weight_block_size
                    # )
                    slices = _get_tp_slice(
                        scale_inv_shape, partition_dim, tp_rank, tp_size
                    )
                    scale_inv = scale_inv[slices]
                else:
                    scale_inv = scale_inv[:]
    if dtype is not None:
        res = res.to(dtype)
    return res, scale_inv


def _load_weight_with_bridge_worker(
    bridge: Bridge,
    state_dict: dict[str, torch.Tensor],
    local_names: list[str],
    filenames: list[str],
    local_to_hf_map: dict[str, list[str]],
    weights_path: str,
    torch_fp8_to_te_fp8: bool = False,
):
    all_slices = {}
    for filename in filenames:
        safetensor_file = os.path.join(weights_path, filename)
        with safe_open(safetensor_file, framework="pt", device="cpu") as f:
            for name in f.keys():
                all_slices[name] = f.get_slice(name)

    quantization_config = getattr(bridge.hf_config, "quantization_config", None)
    enable_fp8_param = (
        bridge.config.fp8 is not None
        and bridge.config.fp8_param
        and torch_fp8_to_te_fp8
    )

    for local_name in local_names:
        hf_names = local_to_hf_map[local_name]
        param = state_dict[local_name]

        if "experts" in local_name and "shared_experts" not in local_name:
            tp_size = mpu.get_expert_tensor_parallel_world_size()
            tp_rank = mpu.get_expert_tensor_parallel_rank()
        else:
            tp_size = mpu.get_tensor_model_parallel_world_size()
            tp_rank = mpu.get_tensor_model_parallel_rank()

        # Get weight_block_size from quantization_config
        weight_block_size = None
        if quantization_config is not None:
            weight_block_size = quantization_config.get("weight_block_size", None)
            assert (
                isinstance(weight_block_size, (list, tuple))
                and len(weight_block_size) == 2
            )

        is_te_fp8_param = is_float8tensor(param)
        # Check if any HF weight is FP8 (has _scale_inv suffix)
        # If fp8 mode is not enabled in megatron,
        # we need to dequantize FP8 weights before converting to mcore format
        # Now only support FP8 dequantization
        hf_weights_safe_slice = []
        hf_scale_invs = []
        hf_has_fp8 = False
        hf_all_fp8 = True  # Track if all inputs are FP8

        for hf_name in hf_names:
            if "_scale_inv" in hf_name:
                continue
            hf_slice = all_slices[hf_name]
            scale_inv_name = f"{hf_name}_scale_inv"
            if scale_inv_name in all_slices:
                # HF weight is FP8
                hf_has_fp8 = True
                scale_inv_slice = all_slices[scale_inv_name]

                if is_te_fp8_param and enable_fp8_param:
                    hf_weights_safe_slice.append(hf_slice)
                    hf_scale_invs.append(scale_inv_slice)
                else:
                    # Dequantize to higher precision
                    device = torch.device(current_platform.device_type)
                    weight = hf_slice[:].to(device)
                    scale_inv = scale_inv_slice[:].to(device)
                    dequantized_weight = dequantize_params(
                        weight,
                        scale_inv,
                        dst_dtype=bridge.dtype,
                        quantization_config=quantization_config,
                    )
                    if param.device.type == "cpu":
                        dequantized_weight = dequantized_weight.cpu()
                    hf_weights_safe_slice.append(dequantized_weight)
                    hf_all_fp8 = False
            else:
                hf_weights_safe_slice.append(hf_slice)
                hf_all_fp8 = False

        # If target is TE FP8 but not all inputs are FP8, we can't merge FP8 and non-FP8 tensors
        if is_te_fp8_param and enable_fp8_param and hf_has_fp8 and not hf_all_fp8:
            raise RuntimeError("Expected all inputs to be FP8 for TE FP8 parameter")

        # TODO: check fp type is matched between pytorch and te

        param_to_load, merged_scale_inv = _weight_to_mcore_tp(
            hf_config=bridge.hf_config,
            mcore_weights_name=local_name,
            mcore_param_shape=list(param.shape),
            hf_weights_safe_slice=hf_weights_safe_slice,
            tp_rank=tp_rank,
            tp_size=tp_size,
            dtype=bridge.dtype
            if not (is_te_fp8_param and hf_has_fp8 and hf_all_fp8)
            else None,
            hf_scale_invs=hf_scale_invs
            if (is_te_fp8_param and hf_has_fp8 and hf_all_fp8)
            else None,
            weight_block_size=weight_block_size,
        )

        # Load the parameter
        if is_te_fp8_param and hf_has_fp8 and hf_all_fp8 and enable_fp8_param:
            # Direct FP8 to FP8 conversion
            if TE_DType_To_Torch[param._fp8_dtype] is not param_to_load.dtype:
                raise ValueError(
                    f"Expected {TE_DType_To_Torch[param._fp8_dtype]} tensor for TE FP8 param, got {param_to_load.dtype}"
                )
            if merged_scale_inv is None:
                raise ValueError(
                    f"Expected scale_inv for FP8 parameter, got {merged_scale_inv}"
                )
            _pytorch_fp8_to_te_fp8(param_to_load, merged_scale_inv, param)
        else:
            # Standard copy (dequantized or non-FP8)
            param.copy_(param_to_load, non_blocking=True)


def make_filename_bins(
    local_to_file_map: dict[str, list[str]],
) -> tuple[list[list[str]], list[list[str]]]:
    # Allocate local weight name into bins, where each bin access independent files
    # Then we can use multiple threads to concurrently load each bin's parameters.
    # This function has a complexity of O(F + L²)
    # where F = total number of files, L = number of local names
    if not local_to_file_map:
        return [], []

    local_names = list(local_to_file_map.keys())
    n = len(local_names)

    # Convert file lists to sets for O(1) lookups and create file-to-locals mapping
    local_to_files = {name: set(local_to_file_map[name]) for name in local_names}
    file_to_locals = defaultdict(set)
    for local_name, files in local_to_files.items():
        for file in files:
            file_to_locals[file].add(local_name)

    # Union-Find with path compression and union by rank
    parent = list(range(n))
    rank = [0] * n

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])  # Path compression
        return parent[x]

    def union(x, y):
        root_x, root_y = find(x), find(y)
        if root_x == root_y:
            return

        # Union by rank
        if rank[root_x] < rank[root_y]:
            root_x, root_y = root_y, root_x
        parent[root_y] = root_x
        if rank[root_x] == rank[root_y]:
            rank[root_x] += 1

    # Create name-to-index mapping for O(1) lookups
    name_to_idx = {name: i for i, name in enumerate(local_names)}

    # Union locals that share files - O(F) where F is total number of files
    for locals_sharing_file in file_to_locals.values():
        if len(locals_sharing_file) > 1:
            locals_list = list(locals_sharing_file)
            first_idx = name_to_idx[locals_list[0]]
            for local_name in locals_list[1:]:
                union(first_idx, name_to_idx[local_name])

    # Group by root - O(L)
    root_to_group = defaultdict(list)
    for i, name in enumerate(local_names):
        root_to_group[find(i)].append(name)

    # Build result groups - O(L + F)
    grouped_local_names = []
    grouped_filenames = []

    for group in root_to_group.values():
        grouped_local_names.append(group)
        # Use set union to merge files from all locals in group
        all_files = set()
        for local_name in group:
            all_files.update(local_to_files[local_name])
        grouped_filenames.append(list(all_files))

    return grouped_local_names, grouped_filenames


def load_weights_from_hf_with_mbridge_fast(
    bridge: Bridge,
    models: list[torch.nn.Module],
    weights_path: str,
    max_workers: int | None = None,
) -> None:
    weights_path = bridge._get_actual_hf_path(weights_path)
    index_file = os.path.join(weights_path, "model.safetensors.index.json")
    manual_tie_word_embedding = False
    index = {}
    if os.path.exists(index_file):
        with open(index_file, encoding="utf-8") as f:
            index = json.load(f)["weight_map"]
    else:
        # Search all safetensors files
        safetensor_files = glob(os.path.join(weights_path, "*.safetensors"))
        # If there are safetensors files
        if safetensor_files:
            # Iterate through each safetensors file
            for safetensor_file in safetensor_files:
                with safe_open(safetensor_file, framework="pt", device="cpu") as f:
                    for k in f.keys():
                        index[k] = safetensor_file
                if (
                    "model.embed_tokens.weight" in index
                    and "lm_head.weight" not in index
                ):
                    manual_tie_word_embedding = True
                    index["lm_head.weight"] = index["model.embed_tokens.weight"]
        else:
            raise FileNotFoundError("No safetensors found in the model path to load.")

    # Calling model.state_dict() is very expensive
    # We call it in advance
    state_dicts = [model.state_dict() for model in models]

    worker_args = []
    tik = time.perf_counter()
    for model_index, model in enumerate(models):
        # map local weight names to global weight names
        local_to_global_map = bridge._weight_name_mapping_mcore_local_to_global(model)
        # map local weight names to huggingface weight names
        local_to_hf_map = {
            k: bridge._weight_name_mapping_mcore_to_hf(v)
            for k, v in local_to_global_map.items()
            if "_extra_state" not in k
        }
        if manual_tie_word_embedding:
            for k, v in local_to_hf_map.items():
                if "lm_head.weight" in v:
                    v.remove("lm_head.weight")
                    if "model.embed_tokens.weight" not in v:
                        v.append("model.embed_tokens.weight")

        local_to_file_map = defaultdict(list)
        for local_name, hf_names in local_to_hf_map.items():
            for name in hf_names:
                if "_scale_inv" in name:
                    continue
                filename = index[name]
                if filename not in local_to_file_map[local_name]:
                    local_to_file_map[local_name].append(filename)
                # Also include the scale_inv file if it exists
                scale_inv_name = f"{name}_scale_inv"
                if scale_inv_name in index:
                    scale_inv_filename = index[scale_inv_name]
                    if scale_inv_filename not in local_to_file_map[local_name]:
                        local_to_file_map[local_name].append(scale_inv_filename)

        grouped_local_names, grouped_filenames = make_filename_bins(local_to_file_map)

        for local_names, filenames in zip(grouped_local_names, grouped_filenames):
            worker_args.append(
                dict(
                    bridge=bridge,
                    state_dict=state_dicts[model_index],
                    local_names=local_names,
                    filenames=filenames,
                    local_to_hf_map=local_to_hf_map,
                    weights_path=weights_path,
                )
            )

    logger.debug(
        f"Loading mcore weights from HF preparation time: {time.perf_counter() - tik}"
    )
    if max_workers is None:
        max_workers = min(8, max(1, os.cpu_count() // dist.get_world_size()))
    max_workers = min(max_workers, len(worker_args))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(
            lambda kwargs: _load_weight_with_bridge_worker(**kwargs), worker_args
        )
        # Consume all results to make result all tasks complete
        for _ in results:
            pass
