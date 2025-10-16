import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
from mbridge.core import Bridge
from mbridge.core.bridge import Bridge
from mbridge.core.util import unwrap_model
from megatron.core import parallel_state as mpu
from safetensors.torch import save_file
from torch.distributed._functional_collectives import all_gather_into_tensor_coalesced

from areal.platforms import current_platform
from areal.utils import logging

logger = logging.getLogger("HF WeightsSaver")

HF_MODEL_CONFIG_FILES = [
    "generation_config.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
]


def copy_hf_configs(src_model_dir, dst_model_dir):
    for file in HF_MODEL_CONFIG_FILES:
        try:
            shutil.copy(
                os.path.join(src_model_dir, file),
                os.path.join(dst_model_dir, file),
            )
            logger.info(f"copied {file} from {src_model_dir} to {dst_model_dir}")
        except FileNotFoundError:
            logger.info(f"{file} not exist in {src_model_dir} skipping.")
    # Copy remote codes
    for file in os.listdir(src_model_dir):
        for prefix in ["chat_format", "configuration_", "modeling_", "tokenization_"]:
            if file.startswith(prefix) and file.endswith(".py"):
                shutil.copy(
                    os.path.join(src_model_dir, file),
                    os.path.join(dst_model_dir, file),
                )
                logger.info(f"copied {file} from {src_model_dir} to {dst_model_dir}")


def split_state_dict_into_shards(state_dict: Dict, n_shards: int) -> List[Dict]:
    if n_shards == 1:
        return [state_dict]

    keys = list(state_dict.keys())
    if len(keys) < n_shards:
        raise ValueError(f"state_dict has {len(keys)} keys, but n_shards={n_shards}")

    shard_size = len(keys) // n_shards
    extra = len(keys) % n_shards
    shard_size_list = [shard_size for _ in range(n_shards)]
    shard_size_list[-1] = shard_size + extra
    start, shards = 0, []
    for i, size in enumerate(shard_size_list):
        shard = {}
        for j in range(start, start + size):
            shard[keys[j]] = state_dict[keys[j]]
        start += size
        shards.append(shard)
    return shards


@dataclass
class McoreDistributedWeightSpec:
    param: torch.Tensor
    local_name: str
    global_name: str
    local_shape: List[int]
    dtype: str
    tensor_model_parallel: bool
    pp_rank: int
    vpp_rank: int

    def full_param_size_byte(self) -> int:
        s = np.prod(self.local_shape) * getattr(torch, self.dtype).itemsize
        if self.tensor_model_parallel:
            if ".mlp.experts.linear_fc" in self.global_name:
                s *= mpu.get_expert_tensor_parallel_world_size()
            else:
                s *= mpu.get_tensor_model_parallel_world_size()
        return int(s)


def save_weights_to_hf_with_mbridge_fast(
    bridge: Bridge,
    models: list,
    weights_path: str,
    base_model_path: Optional[str] = None,
    max_shard_size_byte: int = int(3e9),
    max_workers: Optional[int] = None,
):
    # 1. Prepare some global metadata required for saving the model.
    models = [unwrap_model(model) for model in models]
    pp_size = mpu.get_pipeline_model_parallel_world_size()
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    pp_group = mpu.get_pipeline_model_parallel_group()

    local_to_global_maps = [
        bridge._weight_name_mapping_mcore_local_to_global(model, consider_ep=False)
        for model in models
    ]
    state_dicts = [m.state_dict() for m in models]

    # 2. Get all weights specification used to decide how we should save the model.
    device = None
    weight_specs = []
    for vpp_rank, model in enumerate(models):
        existing_keys = set()
        for name, param in model.named_parameters():
            device = param.device
            existing_keys.add(name)
            global_name = local_to_global_maps[vpp_rank][name]
            weight_specs.append(
                McoreDistributedWeightSpec(
                    param=param,
                    local_name=name,
                    global_name=global_name,
                    pp_rank=mpu.get_pipeline_model_parallel_rank(),
                    vpp_rank=vpp_rank,
                    local_shape=list(param.shape),
                    dtype=str(param.dtype).lstrip("torch."),
                    tensor_model_parallel=(
                        hasattr(param, "tensor_model_parallel")
                        and param.tensor_model_parallel
                    ),
                )
            )
        # note
        # there is a bug in megatron GPTModel
        # decoder.layers[n].mlp.router.expert_bias" in GPTModel is not registered in named_parameter, but in state_dict().
        # for now we patch it by adding those keys to extra_keys.
        extra_keys = [
            x
            for x in state_dicts[vpp_rank].keys()
            if "_extra_state" not in x and "expert_bias" in x and x not in existing_keys
        ]
        for name in extra_keys:
            param = state_dicts[vpp_rank][name].to(current_platform.current_device())
            global_name = local_to_global_maps[vpp_rank][name]
            weight_specs.append(
                McoreDistributedWeightSpec(
                    param=param,
                    local_name=name,
                    global_name=global_name,
                    pp_rank=pp_rank,
                    vpp_rank=vpp_rank,
                    local_shape=list(param.shape),
                    dtype=str(param.dtype).lstrip("torch."),
                    tensor_model_parallel=(
                        hasattr(param, "tensor_model_parallel")
                        and param.tensor_model_parallel
                    ),
                )
            )

    # 3. Separate parameters to be saved into expert/non-expert groups.
    # Non-expert parameters can be collectively saved by the (dp, cp, tp) group
    # while different pp ranks save their own disjoint parameters independently.
    non_expert_specs = list(
        filter(lambda s: ".mlp.experts.linear_fc" not in s.global_name, weight_specs)
    )
    non_expert_param_size = [s.full_param_size_byte() for s in non_expert_specs]
    non_expert_shards_this_stage = torch.tensor(
        min(
            len(non_expert_param_size),
            (sum(non_expert_param_size) + max_shard_size_byte - 1)
            // max_shard_size_byte,
        ),
        dtype=torch.int32,
        device=device,
    )
    pp_stage_non_expert_shards = [
        torch.zeros_like(non_expert_shards_this_stage) for _ in range(pp_size)
    ]
    dist.all_gather(
        pp_stage_non_expert_shards,
        non_expert_shards_this_stage,
        group=pp_group,
    )
    pp_stage_non_expert_shards = [int(x) for x in pp_stage_non_expert_shards]
    assert all(x >= 1 for x in pp_stage_non_expert_shards)

    # Expert parameters can be collectively saved by the the (edp, etp) group
    # while different (pp, ep) ranks save their own disjoint parameter independently.
    expert_specs = list(
        filter(lambda s: ".mlp.experts.linear_fc" in s.global_name, weight_specs)
    )
    expert_param_size = [s.full_param_size_byte() for s in expert_specs]
    expert_shards_this_stage = torch.tensor(
        min(
            len(expert_param_size),
            (sum(expert_param_size) + max_shard_size_byte - 1) // max_shard_size_byte,
        ),
        dtype=torch.int32,
        device=device,
    )
    pp_stage_expert_shards = [
        torch.zeros_like(expert_shards_this_stage) for _ in range(pp_size)
    ]
    dist.all_gather(
        pp_stage_expert_shards,
        expert_shards_this_stage,
        group=pp_group,
    )
    ep_size = mpu.get_expert_model_parallel_world_size()
    # EP equally partition weights, so we simply multipy ep_size
    pp_stage_expert_shards = [int(x) for x in pp_stage_expert_shards] * ep_size
    if len(expert_param_size) > 0:
        assert all(x >= 1 for x in pp_stage_expert_shards)

    # 4. Compute the number of totoal required model shards.
    total_n_shards = sum(pp_stage_non_expert_shards) + sum(pp_stage_expert_shards)
    output_filename = "model" + "-{shard:05d}" + f"-of-{total_n_shards:05d}.safetensors"
    bin_index = {}
    bin_index["metadata"] = dict(
        total_size=sum(non_expert_param_size) + sum(expert_param_size)
    )
    bin_index["weight_map"] = {}
    weight_map = {}

    # 5. Save non-expert weights.
    # Each process independently saves its own portion. The following logic computes
    # which portion should this process save.
    shard_offset = sum(pp_stage_non_expert_shards[:pp_rank])
    g = mpu.get_tensor_and_data_parallel_group(with_context_parallel=True)
    mesh_size = dist.get_world_size(g)
    mesh_idx = dist.get_rank(group=g)
    n_shards = int(non_expert_shards_this_stage)
    n_shards_per_gpu = (n_shards + mesh_size - 1) // mesh_size
    if mesh_idx < len(range(0, n_shards, n_shards_per_gpu)):
        local_start = list(range(0, n_shards, n_shards_per_gpu))[mesh_idx]
    else:
        local_start = n_shards
    # all-gather weights across the TP group and converts to HF format
    # Optimized via a single `all_gather_into_tensor_coalesced` call, which should be
    # faster than plain all_gather.
    non_expert_sd = {}
    _all_gather_specs = []
    all_gather_outputs = {}
    for s in non_expert_specs:
        if s.tensor_model_parallel and mpu.get_tensor_model_parallel_world_size() > 1:
            _all_gather_specs.append(s)
    if _all_gather_specs:
        _all_gather_outputs = all_gather_into_tensor_coalesced(
            [s.param for s in _all_gather_specs],
            group=mpu.get_tensor_model_parallel_group(),
        )
        for s, gathered_param in zip(_all_gather_specs, _all_gather_outputs):
            all_gather_outputs[s.global_name] = gathered_param
    for s in non_expert_specs:
        param = s.param
        if s.tensor_model_parallel:
            # allocate a new tensor with proper size
            if mpu.get_tensor_model_parallel_world_size() <= 1:
                infer_params = [param]
            else:
                infer_params = all_gather_outputs[s.global_name].chunk(
                    mpu.get_tensor_model_parallel_world_size(), dim=0
                )
            infer_params = bridge._weight_merge_across_tp(
                s.global_name, infer_params, param
            )
        else:
            infer_params = param
        converted_names, converted_params = bridge._weight_to_hf_format(
            s.global_name, infer_params
        )
        for n, p in zip(converted_names, converted_params):
            assert n not in non_expert_sd, n
            non_expert_sd[n] = p
    # Split the state dict into shards and save the process's own shard.
    shards = split_state_dict_into_shards(non_expert_sd, n_shards)

    def _save_one_shard(x):
        i, shard = x
        shard_idx = shard_offset + i + local_start
        save_file(
            shard,
            os.path.join(weights_path, output_filename.format(shard=shard_idx + 1)),
        )

    # Multi-threaded save.
    _max_workers = max_workers
    if _max_workers is None:
        _max_workers = min(8, max(1, os.cpu_count() // dist.get_world_size()))
    _max_workers = min(_max_workers, n_shards_per_gpu)
    with ThreadPoolExecutor(max_workers=_max_workers) as executor:
        results = executor.map(
            _save_one_shard,
            list(enumerate(shards[local_start : local_start + n_shards_per_gpu])),
        )
        # consume the result
        for _ in results:
            pass
    # organize metadata
    for i, shard in enumerate(shards):
        shard_idx = shard_offset + i
        for k in shard:
            weight_map[k] = output_filename.format(shard=shard_idx + 1)
    weight_map_list = [None for _ in range(pp_size)]
    dist.all_gather_object(
        weight_map_list,
        weight_map,
        group=pp_group,
    )
    for wm in weight_map_list:
        bin_index["weight_map"].update(wm)

    # 6. Save expert weights.
    if len(expert_param_size) > 0:
        ep_size = mpu.get_expert_model_parallel_world_size()
        ep_rank = mpu.get_expert_model_parallel_rank()
        edp_size = dist.get_world_size(mpu.get_expert_data_parallel_group())
        edp_rank = mpu.get_expert_data_parallel_rank()
        etp_size: int = mpu.get_expert_tensor_parallel_world_size()
        etp_rank: int = mpu.get_expert_tensor_parallel_rank()
        etp_group = mpu.get_expert_tensor_parallel_group()

        shard_offset = sum(pp_stage_non_expert_shards) + sum(
            pp_stage_expert_shards[: ep_rank * pp_size + pp_rank]
        )
        mesh_size = edp_size * etp_size
        mesh_idx = edp_rank * etp_size + etp_rank
        n_shards = int(expert_shards_this_stage)
        n_shards_per_gpu = (n_shards + mesh_size - 1) // mesh_size
        if mesh_idx < len(range(0, n_shards, n_shards_per_gpu)):
            local_start = list(range(0, n_shards, n_shards_per_gpu))[mesh_idx]
        else:
            local_start = n_shards
        # map local expert name to global name if using expert parallel
        for s in expert_specs:
            if ep_size == 1:
                break
            num_experts = bridge.config.num_moe_experts
            num_experts_per_rank = num_experts // ep_size
            name_prefix, local_expert_id = s.global_name.split(".weight")
            local_expert_id = int(local_expert_id)
            global_expert_id = num_experts_per_rank * ep_rank + local_expert_id
            s.global_name = f"{name_prefix}.weight{global_expert_id}"
        # all-gather weights across the TP group and converts to HF format
        expert_sd = {}
        _all_gather_specs = []
        all_gather_outputs = {}
        for s in expert_specs:
            if etp_size > 1:
                _all_gather_specs.append(s)
        if _all_gather_specs:
            _all_gather_outputs = all_gather_into_tensor_coalesced(
                [s.param for s in _all_gather_specs],
                group=etp_group,
            )
            for s, gathered_param in zip(_all_gather_specs, _all_gather_outputs):
                all_gather_outputs[s.global_name] = gathered_param
        for s in expert_specs:
            param = s.param
            if etp_size > 1:
                params = all_gather_outputs[s.global_name].chunk(etp_size, dim=0)
            else:
                params = [param]
            merge_params = bridge._weight_merge_across_tp(s.global_name, params, param)
            converted_names, converted_params = bridge._weight_to_hf_format(
                s.global_name, merge_params
            )
            for n, p in zip(converted_names, converted_params):
                assert n not in expert_sd, n
                expert_sd[n] = p
        # Split the state dict into shards and save the process's own shard.
        shards = split_state_dict_into_shards(expert_sd, n_shards)

        def _save_one_shard(x):
            i, shard = x
            shard_idx = shard_offset + i + local_start
            save_file(
                shard,
                os.path.join(weights_path, output_filename.format(shard=shard_idx + 1)),
            )

        _max_workers = max_workers
        if _max_workers is None:
            _max_workers = min(8, max(1, os.cpu_count() // dist.get_world_size()))
        _max_workers = min(_max_workers, n_shards_per_gpu)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = executor.map(
                _save_one_shard,
                list(enumerate(shards[local_start : local_start + n_shards_per_gpu])),
            )
            # consume the result
            for _ in results:
                pass
        # organize metadata
        for i, shard in enumerate(shards):
            shard_idx = shard_offset + i
            for k in shard:
                weight_map[k] = output_filename.format(shard=shard_idx + 1)
        ep_pp_group = mpu.get_expert_tensor_model_pipeline_parallel_group()
        weight_map_list = [None for _ in range(dist.get_world_size(ep_pp_group))]
        dist.all_gather_object(
            weight_map_list,
            weight_map,
            group=ep_pp_group,
        )
        for wm in weight_map_list:
            bin_index["weight_map"].update(wm)

    # 7. save metadata
    if dist.get_rank() == 0:
        bridge.hf_config.save_pretrained(weights_path)
        with open(os.path.join(weights_path, "model.safetensors.index.json"), "w") as f:
            json.dump(bin_index, f, indent=4)
        if base_model_path is not None:
            copy_hf_configs(base_model_path, weights_path)
