# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import dataclasses
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import *

import torch
import torch.distributed as dist
import transformers

from realhf.api.core import model_api
from realhf.base import constants, logging
from realhf.base.saveload_utils import (
    copy_hf_configs,
    load_safetensor,
    split_state_dict_into_shards,
)
from realhf.impl.model.nn.real_llm_api import ReaLModel
from realhf.impl.model.nn.real_llm_parallel import (
    tp_merge_key,
    tp_partition_real_model_state_dict,
)

logger = logging.getLogger("HF Registry")


@dataclasses.dataclass
class HFModelRegistry:
    name: str
    hf_cls_name: str
    config_from_hf_converter: Callable[
        [transformers.PretrainedConfig], model_api.ReaLModelConfig
    ]
    config_to_hf_converter: Callable[
        [model_api.ReaLModelConfig], transformers.PretrainedConfig
    ]
    sd_from_hf_converter: Callable[[Dict, model_api.ReaLModelConfig], Dict]
    sd_to_hf_converter: Callable[[Dict, model_api.ReaLModelConfig], Dict]
    embedding_param_names: Callable[[model_api.ReaLModelConfig], List[str]]
    tblock_param_names: Callable[[model_api.ReaLModelConfig, int], List[str]]
    head_param_names: Callable[[model_api.ReaLModelConfig], List[str]]
    real_config_maker: Optional[Callable] = None

    def config_from_hf(
        self,
        hf_config: Optional[transformers.PretrainedConfig] = None,
        model_path: Optional[str] = None,
        is_critic: bool = False,
    ) -> model_api.ReaLModelConfig:
        if hf_config is None:
            hf_config = transformers.AutoConfig.from_pretrained(
                model_path,
                trust_remote_code=True,
                force_download=True,
            )
        config = self.config_from_hf_converter(hf_config)
        config.base_model_path = model_path
        config.is_critic = is_critic
        if config.is_critic:
            config.tied_embedding = False
        return config

    def config_to_hf(
        self, real_config: model_api.ReaLModelConfig
    ) -> transformers.PretrainedConfig:
        return self.config_to_hf_converter(real_config)

    def load(
        self,
        model: ReaLModel,
        load_dir: str,
        init_critic_from_actor: bool = False,
    ):
        tik = time.perf_counter()
        with open(os.path.join(load_dir, "config.json"), "r") as f:
            hf_config = json.load(f)
        if "architectures" in hf_config:
            assert (
                self.hf_cls_name == hf_config["architectures"][0]
            ), f"{self.hf_cls_name} != {hf_config['architectures'][0]}"

        layer_indices = range(model.layer_idx_start, model.layer_idx_end)

        required_hf_sd_names = []
        for lidx in layer_indices:
            if lidx == 0:
                required_hf_sd_names += self.embedding_param_names(model.config)
            elif lidx == model.config.n_layers + 1:
                required_hf_sd_names += self.head_param_names(model.config)
            else:
                required_hf_sd_names += self.tblock_param_names(model.config, lidx - 1)

        # Load embedding weights as well if tied_embedding is True.
        required_hf_sd_names = set(required_hf_sd_names)
        if (
            model.config.tied_embedding
            and not model.config.is_critic
            and constants.is_last_pipe_stage()
        ):
            required_hf_sd_names.union(self.embedding_param_names(model.config))

        if os.path.exists(os.path.join(load_dir, "pytorch_model.bin.index.json")):
            with open(os.path.join(load_dir, "pytorch_model.bin.index.json"), "r") as f:
                hf_sd_mapping = json.load(f)["weight_map"]
            files_to_load = set()
            for name in required_hf_sd_names:
                if name in hf_sd_mapping:
                    files_to_load.add(hf_sd_mapping[name])
        elif os.path.exists(os.path.join(load_dir, "model.safetensors.index.json")):
            with open(os.path.join(load_dir, "model.safetensors.index.json"), "r") as f:
                hf_sd_mapping = json.load(f)["weight_map"]
            files_to_load = set()
            for name in required_hf_sd_names:
                if name in hf_sd_mapping:
                    files_to_load.add(hf_sd_mapping[name])
        elif os.path.exists(os.path.join(load_dir, "pytorch_model.bin")):
            files_to_load = ["pytorch_model.bin"]
        elif os.path.exists(os.path.join(load_dir, "model.safetensors")):
            files_to_load = ["model.safetensors"]
        else:
            raise ValueError(
                f"Could not find model file in {load_dir}. "
                "Make sure you have downloaded the model correctly."
            )
        setup_time = time.perf_counter() - tik

        def _load_ckpt(fn):
            load_tik = time.perf_counter()
            if fn.endswith(".safetensors"):
                sd = load_safetensor(os.path.join(load_dir, fn))
            else:
                # set map_location to be CPU is a little bit faster
                sd = torch.load(
                    os.path.join(load_dir, fn), map_location="cpu", weights_only=True
                )
            partition_tik = time.perf_counter()
            sd = {k: v for k, v in sd.items() if k in required_hf_sd_names}
            sd = self.sd_from_hf_converter(sd, model.config)
            psd = tp_partition_real_model_state_dict(
                sd,
                model.config,
                constants.tensor_parallel_world_size(),
                constants.tensor_parallel_rank(),
            )
            return psd, partition_tik - load_tik, time.perf_counter() - partition_tik

        load_times, partition_times = [], []
        state_dict = {}
        with ThreadPoolExecutor(
            max_workers=min(4, max(1, os.cpu_count() // 8))
        ) as executor:
            future_to_checkpoint = {
                executor.submit(_load_ckpt, path): path for path in files_to_load
            }

            for future in as_completed(future_to_checkpoint):
                path = future_to_checkpoint[future]
                try:
                    psd, loat_t, part_t = future.result()
                    state_dict.update(psd)
                    load_times.append(loat_t)
                    partition_times.append(part_t)
                except Exception as e:
                    raise RuntimeError(f"Error loading checkpoint from {path}: {e}")

        # Remap embedding weights to the last layer if tied_embedding is True.
        if (
            model.config.tied_embedding
            and not model.config.is_critic
            and constants.is_last_pipe_stage()
        ):
            state_dict[f"{model.config.n_layers + 1}.weight"] = state_dict[
                "0.wte.weight"
            ]
        if not constants.is_first_pipe_stage() and "0.wte.weight" in state_dict:
            state_dict.pop("0.wte.weight")

        copy_tik = time.perf_counter()
        if init_critic_from_actor and constants.is_last_pipe_stage():
            if f"{model.config.n_layers + 1}.weight" in state_dict:
                state_dict.pop(f"{model.config.n_layers + 1}.weight")
            assert len(state_dict) == len(model.state_dict()) - 1, (
                len(state_dict),
                len(model.state_dict()),
            )
            model.load_state_dict(state_dict, strict=False)
        else:
            try:
                model.load_state_dict(state_dict, strict=True)
            except Exception as e:
                logger.error(
                    f"Loading state dict with strict=True failed. "
                    f"Have you set init_critic_from_actor=True "
                    f"in the model config if you are initializing "
                    f"a critic model from a regular LLM? Err: {e}"
                )
                raise e

        # Some logging info
        copy_time = time.perf_counter() - copy_tik
        load_times = "[" + ", ".join(f"{t:.2f}" for t in load_times) + "]"
        partition_times = "[" + ", ".join(f"{t:.2f}" for t in partition_times) + "]"
        logger.debug(
            f"Loading from HuggingFace Model setup time cost={setup_time:.2f}s, load time cost={load_times}, "
            f"partition time cost={partition_times}, copy time cost={copy_time:.2f}s"
        )
        return model

    def save(
        self,
        model: ReaLModel,
        tokenizer: Optional[transformers.PreTrainedTokenizer],
        save_dir: str,
    ):
        tik = time.perf_counter()
        os.makedirs(save_dir, exist_ok=True)

        dp_rank = constants.data_parallel_rank()
        pp_rank = constants.pipe_parallel_rank()
        tp_rank = constants.tensor_parallel_rank()
        tp_size = constants.tensor_parallel_world_size()
        pp_size = constants.pipe_parallel_world_size()
        dp_size = constants.data_parallel_world_size()

        # We will gather parameters across the model parallel group,
        # and save parameters to separate shards across the pipeline parallel group.

        # To decrease the size of each saved file, we split the file
        # of each pipeline stage into smaller shards.
        approx_param_size = (
            sum(v.numel() * v.element_size() for v in model.state_dict().values())
            * tp_size
        )

        # By default a shard is at most 1GB. A small size enables parallel saving during training.
        max_shard_size_byte = int(os.getenv("REAL_SAVE_MAX_SHARD_SIZE_BYTE", int(1e10)))
        n_shards_this_stage = (
            approx_param_size + max_shard_size_byte - 1
        ) // max_shard_size_byte
        if approx_param_size <= 0 or n_shards_this_stage <= 0:
            raise ValueError(
                f"Invalid param_size={approx_param_size}, n_shards_this_stage={n_shards_this_stage}. "
                "Have you instantiated the model?"
            )

        n_shards_this_stage = torch.tensor(
            n_shards_this_stage, dtype=torch.int32, device=model.device
        )
        pp_stage_n_shards = [
            torch.zeros_like(n_shards_this_stage) for _ in range(pp_size)
        ]
        dist.all_gather(
            pp_stage_n_shards,
            n_shards_this_stage,
            group=constants.pipe_parallel_group(),
        )
        pp_stage_n_shards = [int(n.item()) for n in pp_stage_n_shards]
        assert all(x >= 1 for x in pp_stage_n_shards)

        t1 = time.perf_counter()

        # Gather parameters across the model parallel group.
        sd = model.state_dict()
        cpu_sd = {}
        for k, v in sd.items():
            if (
                model.config.tied_embedding
                and not model.config.is_critic
                and k == f"{model.config.n_layers + 1}.weight"
            ):
                continue
            gather_list = [torch.zeros_like(v) for _ in range(tp_size)]
            dist.all_gather(gather_list, v, group=constants.tensor_parallel_group())
            gathered = tp_merge_key(k, gather_list, model.config)
            cpu_sd[k] = gathered.cpu()

        t2 = time.perf_counter()

        hf_sd = self.sd_to_hf_converter(cpu_sd, model.config)
        hf_config = self.config_to_hf_converter(model.config)
        hf_config.architectures = [self.hf_cls_name]
        hf_config.name_or_path = str(save_dir)
        hf_config.torch_dtype = str(model.dtype).strip("torch.")

        param_size = sum(
            [value.numel() * value.element_size() for value in hf_sd.values()]
        )
        param_size = torch.tensor(param_size, dtype=torch.int64, device=model.device)
        dist.all_reduce(
            param_size,
            op=dist.ReduceOp.SUM,
            group=constants.pipe_parallel_group(),
        )
        param_size = param_size.item()

        # Save tokenizer and huggingface model config.
        if pp_rank == 0 and dp_rank == 0 and tp_rank == 0:
            hf_config.save_pretrained(save_dir)
            if tokenizer is not None:
                tokenizer.save_pretrained(save_dir)

        # Dump parameters to disk.
        if len(pp_stage_n_shards) == 1 and pp_stage_n_shards[0] == 1:
            fn = "pytorch_model.bin"
            if pp_rank == 0 and dp_rank == 0 and tp_rank == 0:
                torch.save(hf_sd, os.path.join(save_dir, fn))
        else:
            output_fn = (
                "pytorch_model"
                + "-{shard:05d}"
                + f"-of-{sum(pp_stage_n_shards):05d}.bin"
            )

            n_shards = pp_stage_n_shards[pp_rank]
            shard_offset = sum(pp_stage_n_shards[:pp_rank])

            shards = split_state_dict_into_shards(hf_sd, n_shards)

            bin_index = {}
            bin_index["metadata"] = dict(total_size=param_size)
            bin_index["weight_map"] = {}
            weight_map = {}

            mesh_size = dp_size * tp_size
            mesh_idx = dp_rank * tp_size + tp_rank
            n_shards_per_gpu = (n_shards + mesh_size - 1) // mesh_size
            if mesh_idx < len(range(0, n_shards, n_shards_per_gpu)):
                s = list(range(0, n_shards, n_shards_per_gpu))[mesh_idx]
            else:
                s = n_shards

            # Since torch.save requires pickling, which is CPU-bound,
            # parallelizing the saving process is not beneficial.
            for i, shard in enumerate(shards[s : s + n_shards_per_gpu]):
                shard_idx = shard_offset + i + s
                torch.save(
                    shard,
                    os.path.join(save_dir, output_fn.format(shard=shard_idx + 1)),
                )

            for i, shard in enumerate(shards):
                shard_idx = shard_offset + i
                for k in shard:
                    weight_map[k] = output_fn.format(shard=shard_idx + 1)

            weight_map_list = [None for _ in range(pp_size)]
            dist.all_gather_object(
                weight_map_list,
                weight_map,
                group=constants.pipe_parallel_group(),
            )
            for wm in weight_map_list:
                bin_index["weight_map"].update(wm)

            if pp_rank == 0 and dp_rank == 0 and tp_rank == 0:
                with open(
                    os.path.join(save_dir, "pytorch_model.bin.index.json"), "w"
                ) as f:
                    json.dump(bin_index, f, indent=4)

        # Copy other configs and remote codes.
        if (
            constants.parallelism_rank() == 0
            and model.config.base_model_path is not None
        ):
            copy_hf_configs(model.config.base_model_path, save_dir)

        t3 = time.perf_counter()

        metadata_t = t1 - tik
        gather_cpu_t = t2 - t1
        dump_t = t3 - t2
        logger.debug(
            f"Saving to HuggingFace Model metadata cost={metadata_t:.2f}s, "
            f"gather/cpu copy cost={gather_cpu_t:.2f}s, "
            f"dump cost={dump_t:.2f}s"
        )
