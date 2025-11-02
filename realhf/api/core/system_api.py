# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import dataclasses
import os
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import realhf.api.core.dfg as dfg
from realhf.api.cli_args import (
    AutomaticEvaluator,
    ExperimentSaveEvalControl,
    SwanlabConfig,
    TensorBoardConfig,
    WandBConfig,
)
from realhf.api.core.config import (
    AgentAbstraction,
    DatasetAbstraction,
    EnvServiceAbstraction,
    ModelAbstraction,
    ModelName,
    ModelShardID,
    StandaloneModelShardAbstraction,
)
from realhf.base import constants, topology


class ExpStatus(Enum):
    RUNNING = "RUNNING"
    ABORTED = "ABORTED"
    COMPLETE = "COMPLETE"


@dataclasses.dataclass
class Scheduling:
    # TODO: add partition
    cpu: int
    gpu: int
    mem: int
    nodelist: str = None
    exclude: str = None
    container_image: str = None
    env_vars: Dict[str, str] = dataclasses.field(default_factory=dict)
    # time utils from "https://slurm.schedmd.com/sbatch.html"
    time_limit: Optional[str] = None  # see  "--time" option for format
    begin: Optional[str] = None  # see "--begin" option for format
    deadline: Optional[str] = None  # see "--deadline" option for format


@dataclasses.dataclass
class WorkerInformation:
    """The basic information of an worker.

    To improve config readability, the experiment starter will fill the
    fields, instead of letting the users do so in experiment configs.
    """

    experiment_name: str = ""
    trial_name: str = ""  # Name of the trial of the experiment; e.g. "{USER}-0".
    worker_type: str = ""  # E.g. "policy", "actor", or "trainer".
    worker_index: int = (
        -1
    )  # The index of the worker of the specific type, starting from 0.
    worker_count: int = (
        0  # Total number of workers; hence, 0 <= worker_index < worker_count.
    )
    worker_tag: Optional[str] = (
        None  # For actor and policy worker, can be "training" or "evaluation".
    )
    host_key: Optional[str] = None  # Worker will update and keep this key alive.
    watch_keys: Union[str, List[str]] = (
        None  # Worker will exit if all of the watching keys are gone.
    )

    def system_setup(
        self,
        experiment_name,
        trial_name,
        worker_type,
        worker_index,
        worker_count,
    ):
        """Setup system related worker information, while leaving the rest
        untouched."""
        self.experiment_name = experiment_name
        self.trial_name = trial_name
        self.worker_type = worker_type
        self.worker_index = worker_index
        self.worker_count = worker_count


@dataclasses.dataclass
class ModelWorker:
    base_seed: int
    shards: List[StandaloneModelShardAbstraction]
    # dataset, for source model workers
    tokenizer_name_or_path: Optional[str] = None
    datasets: Optional[List[Union[str, DatasetAbstraction]]] = None
    shuffle_dataset: bool = True
    cuda_cache_cleanliness: bool = True
    cuda_cache_clear_freq: int = 10
    torch_cache_mysophobia: bool = False
    # model_topos and worker_info will be configured automatically
    model_rpcs: List[dfg.MFCDef] = None
    model_topos: Dict[ModelName, topology.ProcessTopology] = None
    msid2mwid: Dict[ModelShardID, int] = None
    data_transfer_pairs: List[Tuple[ModelName, ModelName]] = None
    sync_param_pairs: List[Tuple[ModelName, ModelName]] = None
    # profiling
    profile_mode: bool = False
    worker_info: Optional[WorkerInformation] = None

    def __post_init__(self):
        model_names = [s.id.model_name for s in self.shards]
        if len(set(model_names)) != len(model_names):
            raise ValueError(
                f"ModelWorker cannot have multiple shards of the same model name: {model_names}."
            )


@dataclasses.dataclass
class GenerationServer:
    base_seed: int
    backend_type: str
    backend_args: Any
    model_path: str
    tp_size: int
    worker_info: WorkerInformation = None


@dataclasses.dataclass
class GserverManager:
    model_name: ModelName
    n_servers: int
    schedule_policy: str
    max_head_offpolicyness: int
    train_batch_size: int
    flush_request_timeout: int
    max_concurrent_rollouts: int
    worker_info: WorkerInformation = None


@dataclasses.dataclass
class RolloutWorker:
    base_seed: int
    model_name: ModelName
    tokenizer_path: str
    new_tokens_per_chunk: int
    rollout_request_timeout: int
    env: EnvServiceAbstraction
    agent: AgentAbstraction
    datasets: List[Union[str, DatasetAbstraction]]
    worker_info: WorkerInformation = None


@dataclasses.dataclass
class MasterWorker:
    base_seed: int
    exp_ctrl: ExperimentSaveEvalControl
    # main components
    n_model_workers: int
    shuffle_dataset: bool = True
    model_rpcs: List[dfg.MFCDef] = None
    model_topos: Dict[ModelName, topology.ProcessTopology] = None
    msid2mwid: Dict[ModelShardID | str, int] = None
    data_transfer_pairs: List[Tuple[str, str]] = None
    sync_param_pairs: List[Tuple[str, str]] = None
    worker_info: Optional[WorkerInformation] = None


@dataclasses.dataclass
class TasksGroup:
    count: int
    scheduling: Scheduling


@dataclasses.dataclass
class ExperimentScheduling:
    model_worker: TasksGroup
    master_worker: TasksGroup
    generation_server: TasksGroup | None = None
    gserver_manager: TasksGroup | None = None
    rollout_worker: TasksGroup | None = None
    controller_image: str = None


@dataclasses.dataclass
class ExperimentConfig:
    exp_ctrl: ExperimentSaveEvalControl
    wandb: WandBConfig
    swanlab: SwanlabConfig
    tensorboard: TensorBoardConfig
    # dataflow
    model_rpcs: List[dfg.MFCDef]
    model_worker: List[ModelWorker] = dataclasses.field(default_factory=list)
    generation_server: List[GenerationServer] = dataclasses.field(default_factory=list)
    gserver_manager: List[GserverManager] = dataclasses.field(default_factory=list)
    rollout_worker: List[RolloutWorker] = dataclasses.field(default_factory=list)
    # master_worker will be set automatically
    master_worker: Optional[List[MasterWorker]] = None
    # automatic evaluation
    auto_eval: bool = False
    evaluator: AutomaticEvaluator = dataclasses.field(
        default_factory=AutomaticEvaluator
    )

    def __post_init__(self):
        self.master_worker = [
            MasterWorker(
                base_seed=self.model_worker[0].base_seed,
                exp_ctrl=self.exp_ctrl,
                n_model_workers=len(self.model_worker),
                shuffle_dataset=self.model_worker[0].shuffle_dataset,
            )
        ]

    def lazy_init(self):
        assert self.master_worker is not None and len(self.master_worker) == 1
        model_names = set()
        for w in self.model_worker:
            model_names = model_names.union([s.id.model_name for s in w.shards])
        model_names = sorted(list(model_names))

        assert constants.trial_name() is not None
        assert constants.experiment_name() is not None
        # If verbose set to True here, every worker will print the graph once
        # due to lazy init on workers.
        G = dfg.build_graph(self.model_rpcs, verbose=False)
        for rpc in self.model_rpcs:
            rpc._G = G

        self._validate_model_names(model_names)

        model_topos = self._collect_topos(model_names)
        model_configs = self._collect_model_configs(model_names)

        data_transfer_pairs = self._resolve_data_transfer_pairs(model_names)

        sync_param_pairs = self._resolve_param_realloc_pairs(model_configs, model_topos)

        model_names_to_instantiate = self._resolve_model_names_to_instantiate(
            model_names
        )

        for mw in self.model_worker:
            for s in mw.shards:
                s.should_instantiate = s.id.model_name in model_names_to_instantiate

        msid2mwid = {}
        for i, mw in enumerate(self.model_worker):
            mw.model_topos = model_topos
            for m in mw.shards:
                msid2mwid[m.id] = i
        for m in self.model_worker:
            m.msid2mwid = msid2mwid
            m.data_transfer_pairs = data_transfer_pairs
            m.sync_param_pairs = sync_param_pairs

        for m in self.model_worker:
            m.model_rpcs = self.model_rpcs

        # setup master worker config
        self.master_worker[0].model_rpcs = self.model_rpcs
        self.master_worker[0].model_topos = model_topos
        self.master_worker[0].msid2mwid = msid2mwid
        self.master_worker[0].sync_param_pairs = sync_param_pairs
        self.master_worker[0].data_transfer_pairs = data_transfer_pairs

    def resolve_worker_config(self, worker_type, worker_index):
        return getattr(self, worker_type)[worker_index]

    def set_worker_information(self, experiment_name, trial_name):
        for worker_type, workers in [
            ("model_worker", self.model_worker),
            ("master_worker", self.master_worker),
            ("gserver_manager", self.gserver_manager),
            ("rollout_worker", self.rollout_worker),
            ("generation_server", self.generation_server),
        ]:
            if len(workers) == 0:
                continue
            for i, worker in enumerate(workers):
                system_worker_info = dict(
                    experiment_name=experiment_name,
                    trial_name=trial_name,
                    worker_type=worker_type,
                    worker_index=i,
                    worker_count=len(workers),
                )
                if worker.worker_info is not None:
                    worker.worker_info.system_setup(**system_worker_info)
                else:
                    worker.worker_info = WorkerInformation(**system_worker_info)

    def _collect_topos(
        self, model_names: List[ModelName]
    ) -> Dict[ModelName, topology.ProcessTopology]:
        model_topos = {}
        model_allocations = {}
        for model_name in model_names:
            _this_mws_with_indicies = list(
                filter(
                    lambda i_mw: any(
                        x.id.model_name == model_name for x in i_mw[1].shards
                    ),
                    enumerate(self.model_worker),
                )
            )
            _this_mw_indices, _this_mws = zip(*_this_mws_with_indicies)
            _this_mw_indices = tuple(sorted(_this_mw_indices))
            all_shards: List[StandaloneModelShardAbstraction] = [
                next(filter(lambda x: x.id.model_name == model_name, mw.shards))
                for mw in _this_mws
            ]
            model_topos[model_name] = all_shards[0].id.topo
            model_allocations[model_name] = tuple(sorted(_this_mw_indices))

            ##### Sanity check of parallelism ranks. #####
            ranks = [s.id.parallelism_rank for s in all_shards]
            _topos = [s.id.topo for s in all_shards]
            if set(ranks) != set(list(range(len(_this_mws)))) or any(
                _t.world_size() != _topos[0].world_size() for _t in _topos
            ):
                raise ValueError(
                    f"Parallelism rank check failed: model name {model_name}, "
                    f"model shard ids={[s.id for s in all_shards]}."
                )
            ##### Sanity check of parallelism ranks. #####
        return model_topos

    def _collect_model_configs(
        self, model_names: List[ModelName]
    ) -> Dict[ModelName, ModelAbstraction]:
        model_configs = {}
        for model_name in model_names:
            _this_mws = list(
                filter(
                    lambda mw: any(x.id.model_name == model_name for x in mw.shards),
                    self.model_worker,
                )
            )
            all_shards: List[StandaloneModelShardAbstraction] = [
                next(filter(lambda x: x.id.model_name == model_name, mw.shards))
                for mw in _this_mws
            ]
            model_configs[model_name] = all_shards[0].model
        return model_configs

    def _validate_model_names(self, model_names: List[ModelName]):
        model_names = sorted(model_names)
        _roles = set(mn.role for mn in model_names)
        _replica_ids = {
            _role: sorted([mn.replica_id for mn in model_names if mn.role == _role])
            for _role in _roles
        }
        for v in _replica_ids.values():
            if list(sorted(v)) != list(range(len(v))):
                raise ValueError(
                    f"Model replica ids should be 0, 1, 2, ... for each role: {_replica_ids}."
                )

    def _resolve_data_transfer_pairs(
        self, model_names: List[ModelName]
    ) -> List[Tuple[ModelName, ModelName]]:
        data_transfer_pairs: List[Tuple[ModelName, ModelName]] = []
        G = self.model_rpcs[0]._G
        for edge in G.edges():
            mn1 = G.nodes[edge[0]]["object"].model_name
            mn2 = G.nodes[edge[1]]["object"].model_name
            data_transfer_pairs.append((mn1, mn2))
        src_rpcs = [rpc for rpc in self.model_rpcs if rpc.is_src]
        data_src_rpc = src_rpcs[0]
        for r in src_rpcs:
            if (
                data_src_rpc.model_name,
                r.model_name,
            ) not in data_transfer_pairs:
                data_transfer_pairs.append((data_src_rpc.model_name, r.model_name))
        return data_transfer_pairs

    def _resolve_param_realloc_pairs(
        self, model_configs, model_topos
    ) -> List[Tuple[ModelName, ModelName]]:
        sync_param_pairs: List[Tuple[ModelName, ModelName]] = []
        for rpc in self.model_rpcs:
            for hook in rpc._pre_hooks + rpc._post_hooks:
                if not isinstance(hook, dfg.ParamReallocHook):
                    continue
                other_model_name = (
                    hook.target if hook.target is not None else hook.source
                )
                other_topo = (
                    model_topos[hook.target]
                    if hook.target is not None
                    else model_topos[hook.source]
                )
                self_topo = model_topos[rpc.model_name]
                if (
                    self_topo.get_dim("tensor") % other_topo.get_dim("tensor") != 0
                    and other_topo.get_dim("tensor") % self_topo.get_dim("tensor") != 0
                ):
                    raise ValueError(
                        "To synchronize parameters between two models, "
                        "their model parallel size must be a multiple of each other."
                    )
                if rpc.model_name == other_model_name:
                    raise ValueError(
                        f"Cannot synchronize parameters within the same model "
                        f"(in {rpc}, {rpc.model_name} and {hook.target})."
                    )
                if hook.target is not None:
                    if not (rpc.model_name, hook.target) in sync_param_pairs:
                        sync_param_pairs.append((rpc.model_name, hook.target))
                else:
                    if not (hook.source, rpc.model_name) in sync_param_pairs:
                        sync_param_pairs.append((hook.source, rpc.model_name))
        return sync_param_pairs

    def _resolve_model_names_to_instantiate(
        self, model_names: List[ModelName]
    ) -> List[ModelName]:
        # Mark which shard of the same role should be instantiated.
        roles = set([model_name.role for model_name in model_names])
        role_is_trainable = {role: False for role in roles}
        role_trainable_idx = {}
        role_idx_collection = {role: set() for role in roles}
        for role in roles:
            for rpc in self.model_rpcs:
                if rpc.role != role:
                    continue
                if rpc.interface_type == dfg.ModelInterfaceType.TRAIN_STEP:
                    if role_is_trainable[role]:
                        raise ValueError(
                            f"Multiple train_step for the same role {role} is not allowed."
                        )
                    role_is_trainable[role] = True
                    role_trainable_idx[role] = rpc.model_name.replica_id
                role_idx_collection[role].add(rpc.model_name.replica_id)
        role_cnt = {role: len(v) for role, v in role_idx_collection.items()}

        model_names_to_instantiate = []
        for role in roles:
            if role_is_trainable[role]:
                model_names_to_instantiate.append(
                    ModelName(role, role_trainable_idx[role])
                )
            else:
                model_names_to_instantiate += [
                    ModelName(role, i) for i in range(role_cnt[role])
                ]

        return model_names_to_instantiate


class Experiment:
    """Base class for defining the procedure of an experiment."""

    def scheduling_setup(self) -> ExperimentScheduling:
        """Returns the Scheduling of all workers."""
        raise NotImplementedError()

    def initial_setup(self) -> ExperimentConfig | List[ExperimentConfig]:
        """Returns a list of workers to create when a trial of the experiment
        is initialized."""
        raise NotImplementedError()


ALL_EXPERIMENT_CLASSES = {}


def register_experiment(name, cls):
    assert name not in ALL_EXPERIMENT_CLASSES
    ALL_EXPERIMENT_CLASSES[name] = cls


def make_experiment(name) -> Experiment:
    cls = ALL_EXPERIMENT_CLASSES[name]
    args = cls()
    if args.cluster.config_path:
        from realhf.base.cluster import load_spec_from_file

        load_spec_from_file(args.cluster)
    from realhf.base import name_resolve

    name_resolve.reconfigure(args.cluster.name_resolve)
    return args
