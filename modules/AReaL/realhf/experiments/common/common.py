# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import dataclasses
import itertools
import os
from collections import defaultdict
from typing import *

import numpy as np
import transformers
from omegaconf import MISSING, OmegaConf

import realhf.base.logging as logging
from realhf.api.cli_args import (
    BaseExperimentConfig,
    MFCConfig,
    ModelTrainEvalConfig,
    ParallelismConfig,
)
from realhf.api.core.config import (
    DatasetAbstraction,
    ModelAbstraction,
    ModelBackendAbstraction,
    ModelName,
    ModelShardID,
    StandaloneModelShardAbstraction,
)
from realhf.api.core.dfg import MFCDef, ModelInterfaceType
from realhf.api.core.model_api import HF_MODEL_FAMILY_REGISTRY
from realhf.api.core.system_api import (
    Experiment,
    ExperimentConfig,
    ExperimentScheduling,
    ModelWorker,
    Scheduling,
    TasksGroup,
)
from realhf.api.quickstart.device_mesh import (
    DeviceMesh,
    RPCAllocation,
    make_device_mesh_from_name,
)
from realhf.experiments.common.check import (
    check_valid_model_and_path,
    check_valid_optimizer,
    check_valid_parallel_batch_size,
    check_valid_sglang,
    check_valid_vllm,
)
from realhf.experiments.common.utils import (
    AllocationMode,
    asdict,
    get_real_model_config,
    get_topo,
    make_inf_backend_config,
    make_train_backend_config,
    resolve_replica_ids,
    resolve_rpc_hooks,
)

# Register all HF models
import realhf.api.from_hf  # isort:skip

logger = logging.getLogger("CommonExperimentConfig", "colored")

GEN_HYBRID_TRAIN_DECOUPLE_ALLOC_WARN = False


@dataclasses.dataclass
class CommonExperimentConfig(BaseExperimentConfig, Experiment):

    @property
    def models(self) -> Dict[str, ModelTrainEvalConfig]:
        """A dict mapping from model roles to model configurations.

        Should be implemented in all subclasses.
        """
        raise NotImplementedError(f"models is not implemented in {self.__class__}")

    @property
    def rpcs(self) -> Dict[str, MFCDef]:
        """A dict mapping from model function call names to the MFCDef objects.

        Note that model function call names are different from model
        names. Should be implemented in all subclasses.

        NOTE: in implementation of ReaL, term RPC also refers to MFC.
        """
        raise NotImplementedError(f"rpcs is not implemented in {self.__class__}")

    @property
    def allocations(self) -> Dict[str, MFCConfig]:
        """The allocation configuration for each model function call.

        A dictionary mapping MFC names to its allocation configuration.
        Must be implemented in the subclass.
        """
        raise NotImplementedError(
            f"allocations is not implemented in {self.__class__}."
        )

    @property
    def datasets(self) -> List[DatasetAbstraction]:
        """A list of dataset configurations used for training.

        Should be implemented in all subclasses.
        """
        raise NotImplementedError(f"datasets is not implemented in {self.__class__}")

    @property
    def eval_dataset(self) -> DatasetAbstraction | None:
        """The dataset configuration used for evaluation.

        Can be None if runtime evaluation is not needed.
        """
        return None

    @property
    def eval_bs(self) -> int:
        """The batch size for runtime evaluation."""
        return 128

    @property
    def tokenizer_name_or_path(self) -> str:
        """The tokenizer for tokenizing train/validation datasets.

        Required for all experiments.
        """
        raise NotImplementedError(
            f"tokenizer_name_or_path is not implemented in {self.__class__}."
        )

    @property
    def max_prompt_len(self) -> int:
        """The maximum prompt length for generation.

        Used by CUDAGraph-enabled generation. If no generation is used
        in the algorithm, this property can be None.
        """
        return None

    @property
    def global_device_mesh(self) -> DeviceMesh:
        return DeviceMesh(
            n_nodes=self.n_nodes,
            n_gpus_per_node=self.n_gpus_per_node,
            mapping=np.ones((self.n_nodes, self.n_gpus_per_node), dtype=np.int32),
        )

    def _heuristic_rpc_allocation(self) -> List[RPCAllocation]:
        raise NotImplementedError(
            f"_heuristic_rpc_allocation is not implemented in {self.__class__}"
        )

    def scheduling_setup(self) -> ExperimentScheduling:
        """The resourced occupied by each worker.

        The resource requirements will be sent to SLURM or Ray, while
        being ignored in the local mode.
        """
        return ExperimentScheduling(
            master_worker=TasksGroup(
                count=1,
                scheduling=Scheduling(
                    cpu=self.cpus_per_master_worker,
                    gpu=0,
                    mem=self.mem_per_master_worker,
                    nodelist=self.nodelist,
                    exclude=self.exclude,
                    container_image=self.cluster.cpu_image,
                ),
            ),
            model_worker=TasksGroup(
                count=self.n_nodes * self.n_gpus_per_node,
                scheduling=Scheduling(
                    cpu=self.cpus_per_model_worker,
                    gpu=1,
                    mem=self.mem_per_model_worker,
                    nodelist=self.nodelist,
                    exclude=self.exclude,
                    container_image=self.cluster.gpu_image,
                ),
            ),
        )

    @property
    def _allocation_mode(self):
        return AllocationMode.from_str(self.allocation_mode)

    def _get_rpc_allocations(self) -> List[RPCAllocation]:
        if self.allocation_mode == "manual" and self.nodelist is None:
            logger.warning(
                "Warning: Nodelist is not set in manual allocation mode, "
                "in this case you cannot specify device mesh for each model RPC. "
                "All model RPC will be allocated on GPUs automatically "
                f"allocated according to n_nodes {self.n_nodes} "
                f"and n_gpus_per_node {self.n_gpus_per_node}."
            )

        self._check_legal_allocation_options()

        rpcs = self.rpcs
        if self._allocation_mode.is_decoupled():
            paras = self._allocation_mode.parallel_strat

            gdp, gpp, gtp = paras["gen"]["d"], paras["gen"]["p"], paras["gen"]["m"]
            gen_world_size = gdp * gpp * gtp
            assert (
                gen_world_size < self.n_gpus_per_node
                or gen_world_size % self.n_gpus_per_node == 0
            )
            gen_device_mesh, train_device_mesh = self.global_device_mesh.split(
                gen_world_size
            )

            self.gen_device_mesh = gen_device_mesh
            self.train_device_mesh = train_device_mesh

            rpc_allocs = []
            flag = False
            for rpc in rpcs.values():
                if rpc.is_generate():
                    if gpp != 1:
                        raise NotImplementedError(
                            "vllm/sglang pipeline parallel is not supported yet."
                        )
                    if flag:
                        raise NotImplementedError(
                            "vllm/sglang does not support two generation RPCs for now."
                        )
                    alloc = RPCAllocation(
                        rpc=rpc,
                        device_mesh=gen_device_mesh,
                        parallel=ParallelismConfig(
                            data_parallel_size=gdp,
                            pipeline_parallel_size=gpp,
                            tensor_parallel_size=gtp,
                            use_sequence_parallel=False,
                        ),
                    )
                    flag = True
                else:
                    rpc_name = rpc.name
                    if rpc_name in paras:
                        dp, pp, tp = (
                            paras[rpc_name]["d"],
                            paras[rpc_name]["p"],
                            paras[rpc_name]["m"],
                        )
                    else:
                        if "*" not in paras:
                            raise ValueError(
                                f"RPC {rpc_name} parallel strategy not given, "
                                "expect a `*` to specify the default parallel strategy."
                            )
                        dp, pp, tp = paras["*"]["d"], paras["*"]["p"], paras["*"]["m"]
                    if (
                        dp * pp * tp + gdp * gpp * gtp
                        != self.n_nodes * self.n_gpus_per_node
                    ):
                        raise ValueError(
                            "The multiplication of 3D parallel degrees "
                            "does not equal to the number of gpus. "
                            "Note that the device mesh of vLLM/SGLang should be disjoint from the device mesh of other MFCs, "
                            "so their summation should be equal to the total number of gpus. "
                            f"dp={dp}, pp={pp}, mp={tp}, gen.dp={gdp}, gen.pp={gpp}, gen.mp={gtp}, "
                            f"n_nodes={self.n_nodes}, n_gpus_per_node={self.n_gpus_per_node}"
                        )
                    alloc = RPCAllocation(
                        rpc=rpc,
                        device_mesh=train_device_mesh,
                        parallel=ParallelismConfig(
                            data_parallel_size=dp,
                            pipeline_parallel_size=pp,
                            tensor_parallel_size=tp,
                            use_sequence_parallel=(
                                rpc.interface_type == ModelInterfaceType.TRAIN_STEP
                                and tp > 1
                            ),
                        ),
                    )
                rpc_allocs.append(alloc)
            if not flag:
                raise ValueError(
                    "No generation RPC found. Please use the hybrid train allocation mode."
                )
        elif self._allocation_mode.is_global_hybrid():
            paras = self._allocation_mode.parallel_strat
            rpc_allocs = []
            for rpc_name, rpc in self.rpcs.items():
                if rpc_name in paras:
                    dp, pp, tp = (
                        paras[rpc_name]["d"],
                        paras[rpc_name]["p"],
                        paras[rpc_name]["m"],
                    )
                else:
                    if "*" not in paras:
                        raise ValueError(
                            f"RPC {rpc_name} parallel strategy not given, "
                            "expect a `*` to specify the default parallel strategy."
                        )
                    dp, pp, tp = paras["*"]["d"], paras["*"]["p"], paras["*"]["m"]
                assert dp * pp * tp == self.n_nodes * self.n_gpus_per_node
                alloc = RPCAllocation(
                    rpc=rpc,
                    device_mesh=self.global_device_mesh,
                    parallel=ParallelismConfig(
                        data_parallel_size=dp,
                        pipeline_parallel_size=pp,
                        tensor_parallel_size=tp,
                        use_sequence_parallel=(
                            rpc.interface_type == ModelInterfaceType.TRAIN_STEP
                            and tp > 1
                        ),
                    ),
                )
                rpc_allocs.append(alloc)
        elif self.allocation_mode == "manual":
            if self.nodelist is None:
                raise ValueError(
                    "The 'nodelist' option must be specified when using manual allocation mode."
                )
            rpc_allocs: List[RPCAllocation] = [
                RPCAllocation(
                    rpc=rpc,
                    device_mesh=(
                        make_device_mesh_from_name(
                            self.cluster,
                            self.nodelist,
                            self.allocations[rpc_type].device_mesh,
                            self.global_device_mesh.n_gpus_per_node,
                        )
                        if self.allocations[rpc_type].device_mesh is not None
                        else self.global_device_mesh
                    ),
                    parallel=self.allocations[rpc_type].parallel,
                )
                for rpc_type, rpc in rpcs.items()
            ]
        elif self.allocation_mode == "heuristic":
            rpc_allocs: List[RPCAllocation] = self._heuristic_rpc_allocation()
        else:
            raise NotImplementedError(
                f'Unknown allocation mode "{self.allocation_mode}".'
            )
        return rpc_allocs

    def _get_model_worker_configs(
        self, rpc_allocs: List[RPCAllocation]
    ) -> List[ModelWorker]:
        self._run_model_sanity_check(rpc_allocs)

        model_worker = []
        shard_counter = defaultdict(lambda: 0)

        model_name_to_rpc_allocs: Dict[ModelName, List[RPCAllocation]] = defaultdict(
            list
        )
        for rpc_alloc in rpc_allocs:
            model_name_to_rpc_allocs[rpc_alloc.rpc.model_name].append(rpc_alloc)

        for i, j in itertools.product(range(self.n_nodes), range(self.n_gpus_per_node)):
            mw = ModelWorker(
                base_seed=self.seed,
                shards=[],
                datasets=self.datasets,
                shuffle_dataset=self.shuffle_dataset,
                torch_cache_mysophobia=self.torch_cache_mysophobia,
                cuda_cache_cleanliness=self.cache_clear_freq is not None,
                cuda_cache_clear_freq=self.cache_clear_freq,
                tokenizer_name_or_path=self.tokenizer_name_or_path,
            )

            # decoupled allocation, shortcut case
            if (
                self._allocation_mode.is_decoupled()
                and self.gen_device_mesh.mapping[i, j]
            ):
                gen_rpc_alloc = next(
                    alloc for alloc in rpc_allocs if alloc.rpc.is_generate()
                )
                model_name = gen_rpc_alloc.rpc.model_name
                topo = get_topo(
                    gen_rpc_alloc.parallel,
                    gradient_checkpointing=False,
                    max_prompt_len=(self.max_prompt_len),
                    gradient_accumulation_fusion=False,
                    is_train=False,
                )
                model_cfg = self.models[model_name.role]

                gen_backend_name = ""
                if self._allocation_mode.is_decoupled_vllm():
                    gen_backend_name = "vllm"
                elif self._allocation_mode.is_decoupled_sglang():
                    gen_backend_name = "sglang"
                backend_cfg = getattr(model_cfg, gen_backend_name)

                global GEN_HYBRID_TRAIN_DECOUPLE_ALLOC_WARN
                if (
                    backend_cfg.hybrid_train
                    and not GEN_HYBRID_TRAIN_DECOUPLE_ALLOC_WARN
                ):
                    logger.warning(
                        "hybrid_train=True takes no effect for the decoupled allocation"
                    )
                    GEN_HYBRID_TRAIN_DECOUPLE_ALLOC_WARN = True
                backend_cfg.hybrid_train = False

                if gen_backend_name == "vllm":
                    check_valid_vllm(model_name.role, model_cfg.vllm, rpc_allocs)
                elif gen_backend_name == "sglang":
                    check_valid_sglang(model_name.role, model_cfg.sglang, rpc_allocs)

                shard_idx = shard_counter[model_name]
                dict_args: Dict[str, Any] = asdict(backend_cfg)
                mw.shards.append(
                    StandaloneModelShardAbstraction(
                        id=ModelShardID(
                            model_name=model_name,
                            topo=topo,
                            dp_rank=topo.get_coord(shard_idx).data,
                            pp_rank=topo.get_coord(shard_idx).pipe,
                            tp_rank=topo.get_coord(shard_idx).tensor,
                        ),
                        model=ModelAbstraction(
                            "tokenizer", args=dict(tokenizer_path=model_cfg.path)
                        ),
                        backend=ModelBackendAbstraction(
                            gen_backend_name,
                            args=dict(
                                model_path=model_cfg.path,
                                **dict_args,
                            ),
                        ),
                    )
                )
                shard_counter[model_name] += 1

                model_worker.append(mw)
                continue

            for (
                model_name,
                model_rpc_allocs,
            ) in model_name_to_rpc_allocs.items():
                rpcs = [rpc_alloc.rpc for rpc_alloc in model_rpc_allocs]
                if self._allocation_mode.is_decoupled() and all(
                    rpc.is_generate() for rpc in rpcs
                ):
                    continue
                rpc_alloc = model_rpc_allocs[0]
                model_cfg = self.models[model_name.role]
                model = get_real_model_config(
                    model_path=model_cfg.path,
                    hf_model_family=model_cfg.type._class,
                    is_critic=model_cfg.type.is_critic,
                    init_from_scratch=model_cfg.init_from_scratch,
                    init_critic_from_actor=model_cfg.init_critic_from_actor,
                    dtype="bf16" if model_cfg.bf16 else "fp16",
                )
                hf_config = transformers.AutoConfig.from_pretrained(
                    model_cfg.path,
                    trust_remote_code=True,
                    force_download=True,
                )
                model_config = HF_MODEL_FAMILY_REGISTRY[model_cfg.type._class][
                    "config_from_hf_converter"
                ](hf_config)
                if (
                    model_config.n_kv_heads % rpc_alloc.parallel.tensor_parallel_size
                    != 0
                ) or (
                    model_config.n_q_heads % rpc_alloc.parallel.tensor_parallel_size
                    != 0
                ):
                    raise ValueError(
                        f"The number of KV heads {model_config.n_kv_heads} or "
                        f"Q heads {model_config.n_q_heads} is not"
                        f" divisible by the configured TP size "
                        f"({rpc_alloc.parallel.tensor_parallel_size}). "
                        f"Please decrease TP size."
                    )
                mapping = rpc_alloc.device_mesh.mapping
                gradient_checkpointing = model_cfg.gradient_checkpointing and any(
                    rpc.interface_type == ModelInterfaceType.TRAIN_STEP for rpc in rpcs
                )

                topo = get_topo(
                    rpc_alloc.parallel,
                    gradient_checkpointing=gradient_checkpointing,
                    max_prompt_len=(
                        self.max_prompt_len
                        if any(
                            rpc.interface_type == ModelInterfaceType.GENERATE
                            for rpc in rpcs
                        )
                        else None
                    ),
                    gradient_accumulation_fusion=(model_cfg.backend == "megatron")
                    and (model_cfg.type._class != "bailing"),
                    is_train=any(rpc.is_train() for rpc in rpcs),
                )

                if any(rpc.is_train() for rpc in rpcs):
                    backend = make_train_backend_config(model_cfg, rpc_alloc.parallel)
                elif model_cfg.vllm.hybrid_train and any(
                    rpc.is_generate() for rpc in rpcs
                ):
                    assert len(rpcs) == 1 and rpcs[0].is_generate(), rpcs
                    assert (
                        not model_cfg.sglang.hybrid_train
                    ), "vLLM and SGLang cannot be enabled at the same time"
                    dict_args: Dict[str, Any] = asdict(model_cfg.vllm)
                    check_valid_vllm(model_name.role, model_cfg.vllm, rpc_allocs)
                    backend = ModelBackendAbstraction(
                        "vllm",
                        args=dict(
                            model_path=model_cfg.path,
                            **dict_args,
                        ),
                    )
                elif model_cfg.sglang.hybrid_train and any(
                    rpc.is_generate() for rpc in rpcs
                ):
                    raise NotImplementedError(
                        "SGLang hybrid_train=True is not supported yet."
                    )
                else:
                    backend = make_inf_backend_config(model_cfg, rpc_alloc.parallel)

                if mapping[i, j]:
                    shard_idx = shard_counter[model_name]
                    mw.shards.append(
                        StandaloneModelShardAbstraction(
                            id=ModelShardID(
                                model_name=model_name,
                                topo=topo,
                                dp_rank=topo.get_coord(shard_idx).data,
                                pp_rank=topo.get_coord(shard_idx).pipe,
                                tp_rank=topo.get_coord(shard_idx).tensor,
                            ),
                            model=model,
                            backend=backend,
                            eval_dataset=self.eval_dataset,
                            eval_bs=self.eval_bs,
                        )
                    )
                    shard_counter[model_name] += 1
            model_worker.append(mw)
        return model_worker

    def initial_setup(self) -> ExperimentConfig:

        rpc_allocs = self._get_rpc_allocations()

        resolve_replica_ids(rpc_allocs, self.models)
        resolve_rpc_hooks(
            rpc_allocs, self.models
        )  # inplace modify MFCDefs in rpc allocations

        model_worker = self._get_model_worker_configs(rpc_allocs)

        return ExperimentConfig(
            exp_ctrl=self.exp_ctrl,
            wandb=self.wandb,
            swanlab=self.swanlab,
            tensorboard=self.tensorboard,
            model_rpcs=[rpc_alloc.rpc for rpc_alloc in rpc_allocs],
            model_worker=model_worker,
            auto_eval=self.auto_eval,
            evaluator=self.auto_eval_config,
        )

    def _check_legal_allocation_options(self):
        if self.n_nodes > self.cluster.n_nodes:
            raise ValueError(
                f"Number of used nodes {self.n_nodes} should not be larger than the cluster size {self.cluster.n_nodes}"
            )
        if self.n_gpus_per_node > self.cluster.n_gpus_per_node:
            raise ValueError(
                f"Number of used GPUs per node {self.n_gpus_per_node} should not be larger than the cluster limit {self.cluster.n_gpus_per_node}"
            )
        if self.n_nodes > 1 and self.n_gpus_per_node != self.cluster.n_gpus_per_node:
            raise ValueError(
                f"For distributed experiments, only using all GPUs on each node is allowed."
            )
        if self.n_nodes > 1 and self.mode == "local":
            raise ValueError(
                "Cannot run multi-node experiment in local mode, "
                "please setup slurm for distributed runs."
            )

        if self.n_gpus_per_node != 8 and self.allocation_mode == "heuristic":
            raise ValueError(
                f"Cannot run heuristic allocation with "
                f"n_gpus_per_node {self.n_gpus_per_node}, "
                "please set n_gpus_per_node to 8."
            )

        for rpc_name, rpc in self.rpcs.items():
            if rpc_name != rpc.name:
                raise KeyError(
                    f"RPC name {rpc_name} does not match the name in the MFCDef object {rpc.name}."
                )
            if self.allocation_mode == "manual" and rpc_name not in self.allocations:
                if rpc_name not in self.allocations:
                    raise ValueError(
                        f"RPC {rpc_name} is not in allocations, please implement "
                        f"`allocations()` method in your config class to enable "
                        f"manual allocation."
                    )

            if rpc.model_name.role not in self.models.keys():
                raise ValueError(
                    f"RPC {rpc.name} model name {rpc.model_name.role} is not in models."
                )

    def _run_model_sanity_check(self, rpc_allocs: List[RPCAllocation]):
        for alloc in rpc_allocs:
            check_valid_parallel_batch_size(alloc)
        for role, model in self.models.items():
            check_valid_model_and_path(role, model, self.cluster.fileroot)
            check_valid_optimizer(model)
