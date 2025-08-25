import dataclasses
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from mbridge.core.bridge import Bridge
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed import DistributedDataParallelConfig as MCoreDDPConfig
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer import TransformerConfig
from safetensors import safe_open
from transformers import AutoConfig, PretrainedConfig

from areal.experimental.api.cli_args import MegatronEngineConfig
from areal.experimental.model.qwen3 import (
    hf_to_mcore_config_qwen3_dense,
    make_mcore_layer_specs_qwen3_dense,
)
from areal.utils import logging

logger = logging.getLogger("areal.experimental.model.registry")


# Model registry for different architectures
def make_hf_and_mcore_config(
    hf_path: str, dtype: torch.dtype, bridge=None
) -> Tuple[PretrainedConfig, TransformerConfig]:
    if bridge is not None:
        hf_config = bridge.hf_config
        hf_config._name_or_path = hf_path
        return hf_config, bridge.config
    else:
        hf_config: PretrainedConfig = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=hf_path,
            trust_remote_code=True,
        )
        assert len(hf_config.architectures) == 1
        architecture = hf_config.architectures[0]
        if architecture == "Qwen3ForCausalLM":
            return hf_config, hf_to_mcore_config_qwen3_dense(hf_config, dtype)
        else:
            raise ValueError(
                f"Architecture not registered for config conversion: {architecture}."
            )


def make_mcore_layer_specs(hf_config: PretrainedConfig, tf_config: TransformerConfig):
    assert len(hf_config.architectures) == 1
    architecture = hf_config.architectures[0]
    if architecture == "Qwen3ForCausalLM":
        return make_mcore_layer_specs_qwen3_dense(tf_config, use_te=True)
    else:
        raise ValueError(
            f"Architecture not registered for config conversion: {architecture}."
        )


def make_mcore_model(
    hf_config: PretrainedConfig,
    tf_config: TransformerConfig,
    mcore_config: Optional[MegatronEngineConfig] = None,
    bridge=None,
) -> GPTModel:
    if bridge is not None:
        model = bridge.get_model(
            # TODO: Add DDP options when supporting training
            wrap_with_ddp=mcore_config.wrap_with_ddp,
            ddp_config=dataclasses.asdict(mcore_config.ddp),
            use_torch_fsdp2=mcore_config.use_torch_fsdp2,
            use_custom_fsdp=mcore_config.use_custom_fsdp,
            fp16=tf_config.fp16,
            bf16=tf_config.bf16,
            use_precision_aware_optimizer=mcore_config.use_precision_aware_optimizer,
            overlap_param_gather_with_optimizer_step=mcore_config.overlap_param_gather_with_optimizer_step,
        )[0]
        # TODO: when supporting VPP, model should be a list
        return model
    else:
        transformer_layer_spec = make_mcore_layer_specs(hf_config, tf_config)
        rope_scaling_args = {}
        if hf_config.rope_scaling is not None:
            assert (
                hf_config.rope_scaling["type"] == "linear"
            ), "only linear scaling is supported for now"
            rope_scaling_args["seq_len_interpolation_factor"] = hf_config.rope_scaling[
                "factor"
            ]

        model = GPTModel(
            config=tf_config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=hf_config.vocab_size,
            max_sequence_length=hf_config.max_position_embeddings,
            pre_process=True,  # TODO: pipeline parallel
            post_process=True,  # TODO: pipeline parallel
            share_embeddings_and_output_weights=False,  # TODO: implement share output weights
            position_embedding_type="rope",
            rotary_base=hf_config.rope_theta,
            **rope_scaling_args,
            # vp_stage=None TODO: virtual pipeline parallel
        )
        ddp_config = MCoreDDPConfig(**dataclasses.asdict(mcore_config.ddp))
        return DDP(
            config=tf_config,
            ddp_config=ddp_config,
            module=model,
            disable_bucketing=False,
        )


def _get_tp_slice(shape, dim, tp_rank, tp_size) -> Tuple:
    size_per_tp = shape[dim] // tp_size
    res = [slice(None) for _ in range(dim)]
    res.append(slice(tp_rank * size_per_tp, (tp_rank + 1) * size_per_tp))
    return tuple(res)


def _weight_to_mcore_tp(
    hf_config,
    mcore_weights_name: str,
    mcore_param_shape: List,
    hf_weights_safe_slice: List,
    tp_rank: int,
    tp_size: int,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
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
        real_num_key_value_heads = q.get_shape()[0] // group_dim
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
    elif (
        "linear_fc1.weight" in mcore_weights_name
        or "linear_fc1.bias" in mcore_weights_name
    ):
        # merge gate_proj and up_proj
        assert len(hf_weights_safe_slice) == 2, len(hf_weights_safe_slice)
        gate, up = hf_weights_safe_slice
        # chunk 0 for TP split
        gate = gate[
            _get_tp_slice(gate.get_shape(), dim=0, tp_rank=tp_rank, tp_size=tp_size)
        ]
        up = up[_get_tp_slice(up.get_shape(), dim=0, tp_rank=tp_rank, tp_size=tp_size)]
        res = torch.cat([gate, up], dim=0)
    elif "mlp.experts.linear_fc2.weight" in mcore_weights_name:  # moe
        assert len(hf_weights_safe_slice) == 1
        x = hf_weights_safe_slice[0]
        shape = x.get_shape()
        # dim 1 chunk
        res = x[_get_tp_slice(shape, dim=1, tp_rank=tp_rank, tp_size=tp_size)]
    else:
        assert len(hf_weights_safe_slice) == 1
        x = hf_weights_safe_slice[0]
        if mcore_param_shape == x.get_shape():
            res = x[:]
        else:
            assert len(x.get_shape()) == len(mcore_param_shape)
            for partition_dim, (s1, s2) in enumerate(
                zip(x.get_shape(), mcore_param_shape)
            ):
                if s1 != s2:
                    break
            # chunk on `partition_dim`
            res = x[
                _get_tp_slice(
                    x.get_shape(), dim=partition_dim, tp_rank=tp_rank, tp_size=tp_size
                )
            ]
    if dtype is not None:
        res = res.to(dtype)
    return res


def _load_weight_with_bridge_worker(
    bridge: Bridge,
    state_dict: Dict[str, torch.Tensor],
    local_names: List[str],
    filenames: List[str],
    local_to_hf_map: Dict[str, List[str]],
    weights_path: str,
):
    from megatron.core import parallel_state

    all_slices = {}
    for filename in filenames:
        safetensor_file = os.path.join(weights_path, filename)
        with safe_open(safetensor_file, framework="pt", device="cpu") as f:
            for name in f.keys():
                all_slices[name] = f.get_slice(name)

    for local_name in local_names:
        hf_names = local_to_hf_map[local_name]
        param = state_dict[local_name]

        if "experts" in local_name:
            tp_size = parallel_state.get_expert_tensor_parallel_world_size()
            tp_rank = parallel_state.get_expert_tensor_parallel_rank()
        else:
            tp_size = parallel_state.get_tensor_model_parallel_world_size()
            tp_rank = parallel_state.get_tensor_model_parallel_rank()

        param_to_load = _weight_to_mcore_tp(
            hf_config=bridge.hf_config,
            mcore_weights_name=local_name,
            mcore_param_shape=list(param.shape),
            hf_weights_safe_slice=[all_slices[hf_name] for hf_name in hf_names],
            tp_rank=tp_rank,
            tp_size=tp_size,
            dtype=bridge.dtype,
        )
        # load
        param.copy_(param_to_load, non_blocking=True)


def load_weights_with_mbridge(
    bridge: Bridge,
    models: list[torch.nn.Module],
    weights_path: str,
    max_workers: Optional[int] = None,
) -> None:
    weights_path = bridge._get_actual_hf_path(weights_path)
    index_file = os.path.join(weights_path, "model.safetensors.index.json")
    index = {}
    origin_index = {}
    assert os.path.exists(index_file)
    with open(index_file, "r") as f:
        origin_index = json.load(f)
        index = origin_index["weight_map"]
        origin_index = origin_index

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

        local_to_file_map = defaultdict(list)
        for local_name, hf_names in local_to_hf_map.items():
            for name in hf_names:
                filename = index[name]
                if filename not in local_to_file_map[local_name]:
                    local_to_file_map[local_name].append(filename)

        # Allocate local weight name into bins, where each bin access independent files
        # Then we can use multiple threads to concurrently load each bin's parameters
        weight_name_bins = {}
        file_name_to_bin_index = {}
        bin_index = 0
        for local_name, filenames in local_to_file_map.items():
            if not all(filename in file_name_to_bin_index for filename in filenames):
                # Some required filenames doesn't have existing bins
                # Allocate a new bin, and merge all previous bins into this new bin
                weight_name_bins[bin_index] = [local_name]
                for filename in filenames:
                    if filename in file_name_to_bin_index:
                        i = file_name_to_bin_index.pop(filename)
                        if i in weight_name_bins:
                            weight_name_bins[bin_index] += weight_name_bins.pop(i)
                    file_name_to_bin_index[filename] = bin_index
                bin_index += 1
            else:
                # All required filenames have existing bins
                # Use the head bin as the master bin, and merge all other bins into the master bin
                head_i = file_name_to_bin_index[filenames[0]]
                weight_name_bins[head_i].append(local_name)
                if not all(
                    file_name_to_bin_index[filename] == head_i for filename in filenames
                ):
                    # merge to head bin
                    for filename in filenames[1:]:
                        if file_name_to_bin_index[filename] != head_i:
                            i = file_name_to_bin_index.pop(filename)
                            if i in weight_name_bins:
                                weight_name_bins[head_i] += weight_name_bins.pop(i)
                            file_name_to_bin_index[filename] = head_i

        bin_index_to_file_names = defaultdict(list)
        for filename, bin_index in file_name_to_bin_index.items():
            bin_index_to_file_names[bin_index].append(filename)

        grouped_local_names = list(weight_name_bins.values())
        grouped_filenames = list(bin_index_to_file_names[i] for i in weight_name_bins)

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
    max_workers = max(max_workers, len(worker_args))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(
            lambda kwargs: _load_weight_with_bridge_worker(**kwargs), worker_args
        )
        # Consume all results to make result all tasks complete
        for _ in results:
            pass


def load_from_hf(
    hf_config: PretrainedConfig,
    load_path: str,
    model: torch.nn.Module,
    bridge=None,
):
    if bridge is not None:
        # TODO: when supporting VPP, model should be a list
        models = [model]
        load_weights_with_mbridge(bridge, models=models, weights_path=load_path)
    else:
        raise NotImplementedError(
            "Loading hf model into Megatron model requires mbridge."
        )


def save_to_hf(
    hf_config: PretrainedConfig,
    save_path: str,
    model: torch.nn.Module,
    bridge=None,
):
    if bridge is not None:
        # TODO: when supporting VPP, model should be a list
        if bridge.safetensor_io is None:
            assert hf_config._name_or_path
            bridge.safetensor_io = bridge._get_safetensor_io(hf_config._name_or_path)
        models = [model]
        return bridge.save_weights(
            models=models,
            weights_path=save_path,
        )
    else:
        raise NotImplementedError(
            "Saving Megatron model into hf model requires mbridge."
        )
