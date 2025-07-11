# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import collections
import dataclasses
import enum
import itertools
import re
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

import numpy as np
from omegaconf import DictConfig, OmegaConf

from realhf.api.cli_args import ModelTrainEvalConfig, ParallelismConfig
from realhf.api.core.config import (
    ModelAbstraction,
    ModelBackendAbstraction,
    ModelInterfaceType,
    ModelName,
)
from realhf.api.core.dfg import OffloadHook, ParamReallocHook
from realhf.api.quickstart.device_mesh import RPCAllocation
from realhf.base import logging
from realhf.base.topology import (
    DataPipeTensorParallelTopology,
    PipeDataTensorParallelTopology,
    ProcessTopology,
)

logger = logging.getLogger("Experiment Common Utils", "benchmark")


def get_real_model_config(
    model_path: str,
    hf_model_family: str,
    is_critic: bool,
    init_from_scratch: bool,
    init_critic_from_actor: bool,
    dtype: Optional[str] = None,
) -> ModelAbstraction:
    """Make a configuration to build model."""
    model = ModelAbstraction(
        "real_model",
        args=dict(
            model_path=model_path,
            is_critic=is_critic,
            init_critic_from_actor=init_critic_from_actor,
            dtype=dtype,
            hf_model_family=hf_model_family,
            init_from_scratch=init_from_scratch,
        ),
    )
    return model


def get_topo(
    parallel: ParallelismConfig,
    gradient_checkpointing: bool,
    gradient_accumulation_fusion: bool,
    is_train: bool,
    max_prompt_len: Optional[int] = None,
) -> ProcessTopology:
    if is_train:
        return PipeDataTensorParallelTopology(
            num_tp=parallel.tensor_parallel_size,
            num_pp=parallel.pipeline_parallel_size,
            num_dp=parallel.data_parallel_size,
            sequence_parallel=parallel.use_sequence_parallel,
            gradient_checkpointing=gradient_checkpointing,
            max_prompt_len=max_prompt_len,
            gradient_accumulation_fusion=gradient_accumulation_fusion,
        )
    return DataPipeTensorParallelTopology(
        num_tp=parallel.tensor_parallel_size,
        num_pp=parallel.pipeline_parallel_size,
        num_dp=parallel.data_parallel_size,
        sequence_parallel=parallel.use_sequence_parallel,
        max_prompt_len=max_prompt_len,
    )


def get_world_size(parallel: ParallelismConfig) -> int:
    return (
        parallel.tensor_parallel_size
        * parallel.pipeline_parallel_size
        * parallel.data_parallel_size
    )


def make_train_backend_config(
    model_cfg: ModelTrainEvalConfig, parallel_cfg: ParallelismConfig
):
    if model_cfg.backend == "megatron":
        megatron_args: Dict[str, Any] = asdict(model_cfg.megatron)
        return ModelBackendAbstraction(
            "megatron",
            args=dict(
                bf16=model_cfg.bf16,
                optimizer=model_cfg.optimizer,
                **megatron_args,
            ),
        )
    elif model_cfg.backend == "mock_train":
        return ModelBackendAbstraction(
            "mock_train",
            args=dict(
                optimizer_name="adam",
                optimizer_config=dict(
                    lr=model_cfg.optimizer.lr,
                    weight_decay=model_cfg.optimizer.weight_decay,
                    eps=model_cfg.optimizer.eps,
                    betas=(
                        model_cfg.optimizer.beta1,
                        model_cfg.optimizer.beta2,
                    ),
                ),
            ),
        )
    else:
        raise NotImplementedError(f"Backend {model_cfg.backend} is not supported.")


def make_inf_backend_config(
    model_cfg: ModelTrainEvalConfig, parallel_cfg: ParallelismConfig
):
    return ModelBackendAbstraction("inference")


def resolve_replica_ids(
    rpc_allocs: List[RPCAllocation], models: Dict[str, ModelTrainEvalConfig]
):
    role_rpcs = collections.defaultdict(list)
    for alloc in rpc_allocs:
        rpc = alloc.rpc
        role_rpcs[rpc.role].append(alloc)

    for role, allocs in role_rpcs.items():
        cnt = len(allocs)
        if cnt == 1:
            allocs[0].rpc.model_name = ModelName(role, 0)
            continue
        rpcs = [alloc.rpc for alloc in allocs]
        if any(rpc.is_train() for rpc in rpcs):
            main_alloc = next(alloc for alloc in allocs if alloc.rpc.is_train())
        elif any(rpc.is_inference() for rpc in rpcs):
            main_alloc = next(alloc for alloc in allocs if alloc.rpc.is_inference())
        else:
            main_alloc = allocs[0]
        main_alloc.rpc.model_name = ModelName(role, 0)
        i = 1
        for alloc in allocs:
            if alloc.rpc.name == main_alloc.rpc.name:
                continue
            same_alloc = (
                alloc.device_mesh == main_alloc.device_mesh
                and ParallelismConfig.parallelism_eq(
                    alloc.parallel, main_alloc.parallel
                )
            )
            if not same_alloc or (
                alloc.rpc.is_generate()
                and main_alloc.rpc.is_train()
                and (models[role].vllm.hybrid_train or models[role].sglang.hybrid_train)
            ):
                alloc.rpc.model_name = ModelName(role, i)
                i += 1
            else:
                alloc.rpc.model_name = ModelName(role, 0)


def resolve_rpc_hooks(
    rpc_allocs: List[RPCAllocation], model_configs: Dict[str, ModelTrainEvalConfig]
):
    role_interface_types = collections.defaultdict(set)
    for rpc_alloc in rpc_allocs:
        role_interface_types[rpc_alloc.rpc.role].add(rpc_alloc.rpc.interface_type)

    for i, rpc_alloc in enumerate(rpc_allocs):
        rpc = rpc_alloc.rpc
        parallel = rpc_alloc.parallel
        device_mesh = rpc_alloc.device_mesh
        # check param realloc hooks for train_step rpcs
        if rpc.interface_type == ModelInterfaceType.TRAIN_STEP:
            for j, other in enumerate(rpc_allocs):
                if rpc.name == other.rpc.name:
                    continue
                if rpc.role != other.rpc.role:
                    continue
                if (
                    ParallelismConfig.parallelism_eq(parallel, other.parallel)
                    and device_mesh == other.device_mesh
                    and not (
                        model_configs[rpc.role].vllm.hybrid_train
                        and other.rpc.is_generate()
                    )
                ):
                    continue
                self_config = model_configs[rpc.model_name.role]
                other_config = model_configs[other.rpc.model_name.role]
                if (
                    self_config.backend == "deepspeed"
                    or other_config.backend == "deepspeed"
                ):
                    raise ValueError(
                        "Param realloc hooks are not supported in DeepSpeed backend."
                    )
                other.rpc.add_pre_hook(ParamReallocHook(source=rpc.model_name))
                other.rpc.add_post_hook(ParamReallocHook(target=rpc.model_name))
                logger.info(
                    f"Add param sync hooks between "
                    f"{rpc.name} and {other.rpc.name} for role {rpc.role}"
                )

        # Add offload hooks for inference and generate rpcs.
        # Add the offload hook only if the role will not be trained (e.g., reward model)
        # and its allocation is overlapped with at least one other RPCs.
        # As a result, a single inference/generate RPC will not be offloaded.
        overlapped_with_other = False
        for other in rpc_allocs:
            if rpc.name == other.rpc.name:
                continue
            if np.any(np.logical_and(other.device_mesh.mapping, device_mesh.mapping)):
                overlapped_with_other = True
                break
        if (
            ModelInterfaceType.TRAIN_STEP not in role_interface_types[rpc.role]
            and overlapped_with_other
        ):
            rpc.add_post_hook(OffloadHook())
            logger.info(f"Add offload hook for rpc {rpc.name} for role {rpc.role}")


class AllocationType(enum.Enum):
    DECOUPLED_vLLM = 1
    GLOBAL_HYBRID = 2
    MANUAL = 3
    HEURISTIC = 4
    DECOUPLED_SGLANG = 5
    DECOUPLED_MOCK = 6


@dataclasses.dataclass
class AllocationMode:
    type_: AllocationType
    parallel_strat: None | Dict[str, Dict[str, int]]

    def is_decoupled(self):
        return self.type_ in [
            AllocationType.DECOUPLED_vLLM,
            AllocationType.DECOUPLED_SGLANG,
            AllocationType.DECOUPLED_MOCK,
        ]

    def is_decoupled_vllm(self):
        return self.type_ == AllocationType.DECOUPLED_vLLM

    def is_decoupled_sglang(self):
        return self.type_ == AllocationType.DECOUPLED_SGLANG

    def is_decoupled_mock(self):
        return self.type_ == AllocationType.DECOUPLED_MOCK

    def is_global_hybrid(self):
        return self.type_ == AllocationType.GLOBAL_HYBRID

    def get_gen_size(self):
        assert self.is_decoupled()
        paras = self.parallel_strat
        gdp, gpp, gmp = paras["gen"]["d"], paras["gen"]["p"], paras["gen"]["m"]
        return gdp * gpp * gmp

    def get_gen_tp_size(self):
        assert self.is_decoupled()
        paras = self.parallel_strat
        return paras["gen"]["m"]

    @classmethod
    def from_str(cls, allocation_mode: str):
        if allocation_mode == "manual":
            return cls(AllocationType.MANUAL, None)
        if allocation_mode == "heuristic":
            return cls(AllocationType.HEURISTIC, None)

        alloc_3d = AllocationMode.extract_3d_alloc(allocation_mode)
        alloc_hybrid = AllocationMode.extract_key_value_alloc(allocation_mode)
        alloc_decoupled = AllocationMode.extract_decoupled_alloc(allocation_mode)
        if alloc_decoupled:
            if "vllm" in allocation_mode:
                return cls(AllocationType.DECOUPLED_vLLM, alloc_decoupled)
            elif "sglang" in allocation_mode:
                return cls(AllocationType.DECOUPLED_SGLANG, alloc_decoupled)
            elif "mock" in allocation_mode:
                return cls(AllocationType.DECOUPLED_MOCK, alloc_decoupled)
        if alloc_3d:
            return cls(AllocationType.GLOBAL_HYBRID, alloc_3d)
        if alloc_hybrid:
            return cls(AllocationType.GLOBAL_HYBRID, alloc_hybrid)
        raise NotImplementedError(f"Failed to parse allocation: {allocation_mode}")

    @staticmethod
    def extract_3d_alloc(allocation_mode: str) -> Dict | None:
        for x, y, z in itertools.permutations(["d", "m", "p"]):
            pattern = rf"{x}(\d+){y}(\d+){z}(\d+)"
            m = re.match(pattern, allocation_mode)
            if not m:
                continue
            a, b, c = map(int, m.groups())
            # to be consistent with the key-value pattern
            return {
                "*": {
                    x: a,
                    y: b,
                    z: c,
                }
            }

    @staticmethod
    def extract_decoupled_alloc(allocation_mode: str) -> Dict | None:
        pattern = re.compile(
            r"(?:(?:vllm|sglang|mock)\.(.+?)\+(.+))|(?:(.+?)\+(?:vllm|sglang|mock)\.(.+))"
        )
        m = pattern.match(allocation_mode)
        if not m:
            return
        if m.group(1):
            gen_alloc = m.group(1)
            other_alloc = m.group(2)
        else:
            gen_alloc = m.group(4)
            other_alloc = m.group(3)
        gen_alloc = AllocationMode.extract_3d_alloc(gen_alloc)
        if not gen_alloc:
            return
        other_alloc = AllocationMode.extract_3d_alloc(
            other_alloc
        ) or AllocationMode.extract_key_value_alloc(other_alloc)
        if not other_alloc:
            return
        other_alloc.update({"gen": gen_alloc["*"]})
        return other_alloc

    @staticmethod
    def extract_key_value_alloc(
        allocation_mode: str,
    ) -> Dict[str, Dict[str, int]] | None:
        def parse_key_value_pairs(s: str):
            pattern = re.compile(r"([^:,]+):([^:,]+)")
            matches = pattern.findall(s)
            if not matches:
                return None
            return {key: value for key, value in matches}

        allocs = parse_key_value_pairs(allocation_mode)
        if not allocs:
            return
        for k, v in allocs.items():
            v = AllocationMode.extract_3d_alloc(v)
            if not v:
                return
            allocs[k] = v["*"]
        return allocs


def asdict(cfg):
    if isinstance(cfg, (OmegaConf, DictConfig)):
        return OmegaConf.to_container(cfg, resolve=True)
    return dataclasses.asdict(cfg)
