# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import dataclasses
import itertools
import os
from collections import defaultdict
from typing import *

import transformers

import realhf.base.logging as logging
from realhf.api.cli_args import AsyncRLOptions, ParallelismConfig
from realhf.api.core.config import (
    AgentAbstraction,
    DatasetAbstraction,
    EnvServiceAbstraction,
    ModelAbstraction,
    ModelBackendAbstraction,
    ModelName,
    ModelShardID,
    StandaloneModelShardAbstraction,
)
from realhf.api.core.dfg import ModelInterfaceType
from realhf.api.core.model_api import (
    HF_MODEL_FAMILY_REGISTRY,
    GenerationHyperparameters,
)
from realhf.api.core.system_api import (
    ExperimentConfig,
    ExperimentScheduling,
    GenerationServer,
    GserverManager,
    ModelWorker,
    RolloutWorker,
    Scheduling,
    TasksGroup,
)
from realhf.api.quickstart.device_mesh import RPCAllocation
from realhf.experiments.common.common import CommonExperimentConfig
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

logger = logging.getLogger("AsyncRLExperimentConfig", "colored")

GEN_HYBRID_TRAIN_DECOUPLE_ALLOC_WARN = False
GEN_WORKER_DEFAULT_CAPACITY = 512


@dataclasses.dataclass
class AsyncRLExperimentConfig(CommonExperimentConfig, AsyncRLOptions):
    @property
    def generation_config(self) -> GenerationHyperparameters:
        raise NotImplementedError()

    @property
    def env(self) -> EnvServiceAbstraction:
        return EnvServiceAbstraction("null")

    @property
    def agent(self) -> AgentAbstraction:
        return AgentAbstraction("null")

    @property
    def gen_backend_args(self) -> Any:
        raise NotImplementedError()

    @property
    def get_backend_type(self) -> str:
        return "sglang"

    def scheduling_setup(self) -> ExperimentScheduling:
        """The resourced occupied by each worker.

        The resource requirements will be sent to SLURM or Ray, while
        being ignored in the local mode.
        """
        gen_world_size = AllocationMode.from_str(self.allocation_mode).get_gen_size()
        train_world_size = self.n_nodes * self.n_gpus_per_node - gen_world_size
        gen_tp_size = AllocationMode.from_str(self.allocation_mode).get_gen_tp_size()
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
                count=train_world_size,
                scheduling=Scheduling(
                    cpu=self.cpus_per_model_worker,
                    gpu=1,
                    mem=self.mem_per_model_worker,
                    nodelist=self.nodelist,
                    exclude=self.exclude,
                    container_image=self.cluster.gpu_image,
                ),
            ),
            generation_server=TasksGroup(
                count=gen_world_size // gen_tp_size,
                scheduling=Scheduling(
                    cpu=self.cpus_per_generation_server,
                    gpu=gen_tp_size,
                    mem=self.mem_per_generation_server,
                    nodelist=self.nodelist,
                    exclude=self.exclude,
                    container_image=self.cluster.gpu_infer_image,
                ),
            ),
            gserver_manager=TasksGroup(
                count=1,
                scheduling=Scheduling(
                    cpu=self.cpus_per_gserver_manager,
                    gpu=0,
                    mem=self.mem_per_gserver_manager,
                    nodelist=self.nodelist,
                    exclude=self.exclude,
                    container_image=self.cluster.cpu_image,
                ),
            ),
            rollout_worker=TasksGroup(
                count=self.n_rollout_workers or train_world_size,
                scheduling=Scheduling(
                    cpu=self.cpus_per_rollout_worker,
                    gpu=0,
                    mem=self.mem_per_rollout_worker,
                    nodelist=self.nodelist,
                    exclude=self.exclude,
                    container_image=self.cluster.cpu_image,
                ),
            ),
        )

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
            if self.gen_device_mesh.mapping[i, j]:
                continue
            mw = ModelWorker(
                base_seed=self.seed,
                shards=[],
                # NOTE: here we use puller stream to wrap the original dataset
                datasets=[
                    DatasetAbstraction(
                        "puller_stream",
                        args=dict(
                            dataset_cfgs=self.datasets,
                            args=self,
                        ),
                    )
                ],
                torch_cache_mysophobia=self.torch_cache_mysophobia,
                cuda_cache_cleanliness=self.cache_clear_freq is not None,
                cuda_cache_clear_freq=self.cache_clear_freq,
                tokenizer_name_or_path=self.tokenizer_name_or_path,
            )
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

    def get_rollout_worker_configs(self, rpc_allocs):
        gen_world_size = AllocationMode.from_str(self.allocation_mode).get_gen_size()
        train_world_size = self.n_nodes * self.n_gpus_per_node - gen_world_size
        gen_rpc_alloc = next(alloc for alloc in rpc_allocs if alloc.rpc.is_generate())
        model_name = gen_rpc_alloc.rpc.model_name

        return [
            RolloutWorker(
                base_seed=self.seed,
                model_name=model_name,
                tokenizer_path=self.tokenizer_name_or_path,
                new_tokens_per_chunk=self.new_tokens_per_chunk,
                env=self.env,
                agent=self.agent,
                datasets=self.datasets,
                rollout_request_timeout=self.flush_request_timeout,
            )
            for _ in range(self.n_rollout_workers or train_world_size)
        ]

    def get_generation_server_configs(self, rpc_allocs):
        am = AllocationMode.from_str(self.allocation_mode)
        gen_world_size = am.get_gen_size()
        gen_tp_size = am.get_gen_tp_size()
        gen_rpc_alloc = next(alloc for alloc in rpc_allocs if alloc.rpc.is_generate())
        model_name = gen_rpc_alloc.rpc.model_name
        model_cfg = self.models[model_name.role]
        return [
            GenerationServer(
                base_seed=self.seed,
                backend_type=self.get_backend_type,
                backend_args=self.gen_backend_args,
                model_path=model_cfg.path,
                tp_size=gen_tp_size,
            )
            for _ in range(gen_world_size // gen_tp_size)
        ]

    def get_gserver_manager_config(self, rpc_allocs):
        am = AllocationMode.from_str(self.allocation_mode)
        gen_world_size = am.get_gen_size()
        gen_tp_size = am.get_gen_tp_size()
        gen_rpc_alloc = next(alloc for alloc in rpc_allocs if alloc.rpc.is_generate())
        model_name = gen_rpc_alloc.rpc.model_name
        train_rpcs = [alloc.rpc for alloc in rpc_allocs if alloc.rpc.is_train()]
        assert all(rpc.n_seqs == train_rpcs[0].n_seqs for rpc in train_rpcs)
        max_concurrent_rollouts = self.max_concurrent_rollouts
        if max_concurrent_rollouts is None:
            max_concurrent_rollouts = train_rpcs[0].n_seqs
        return [
            GserverManager(
                model_name=model_name,
                flush_request_timeout=self.flush_request_timeout,
                n_servers=gen_world_size // gen_tp_size,
                schedule_policy=self.schedule_policy,
                max_head_offpolicyness=self.max_head_offpolicyness,
                train_batch_size=train_rpcs[0].n_seqs,
                max_concurrent_rollouts=max_concurrent_rollouts,
            )
        ]

    def initial_setup(self) -> ExperimentConfig:
        assert self._allocation_mode.is_decoupled(), self._allocation_mode
        rpc_allocs = self._get_rpc_allocations()

        resolve_replica_ids(rpc_allocs, self.models)
        resolve_rpc_hooks(
            rpc_allocs, self.models
        )  # inplace modify MFCDefs in rpc allocations

        return ExperimentConfig(
            exp_ctrl=self.exp_ctrl,
            wandb=self.wandb,
            swanlab=self.swanlab,
            tensorboard=self.tensorboard,
            # NOTE: master and model worker only see RPCs without generation
            model_rpcs=[
                rpc_alloc.rpc
                for rpc_alloc in rpc_allocs
                if not rpc_alloc.rpc.is_generate()
            ],
            model_worker=self._get_model_worker_configs(rpc_allocs),
            generation_server=self.get_generation_server_configs(rpc_allocs),
            gserver_manager=self.get_gserver_manager_config(rpc_allocs),
            rollout_worker=self.get_rollout_worker_configs(rpc_allocs),
            auto_eval=self.auto_eval,
            evaluator=self.auto_eval_config,
        )
