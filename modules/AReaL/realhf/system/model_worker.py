# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import contextlib
import copy
import gc
import itertools
import json
import multiprocessing as mp
import os
import pickle
import queue
import re
import shutil
import socket
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Hashable, List, Optional, Set, Tuple

import numpy as np
import pynvml
import tabulate
import torch
import torch.distributed as dist
import torch.utils.data

import realhf.impl.model.comm.global_comm as global_comm
import realhf.impl.model.comm.param_realloc as param_realloc_comm
from realhf.api.core import data_api, dfg, model_api, system_api
from realhf.api.core.config import ModelName
from realhf.base import (
    constants,
    gpu_utils,
    logging,
    name_resolve,
    names,
    network,
    recover,
    seeding,
    timeutil,
    topology,
)
from realhf.impl.model.nn.real_llm_api import ReaLModel
from realhf.impl.model.utils import cuda_graph
from realhf.system import request_reply_stream, worker_base
from realhf.system.data_manager import DataManager
from realhf.system.redistributor import RedistribStep
from realhf.system.stream_dataset import PullerStreamDataset

# NOTE: Register all implemented datasets and models.
import realhf.impl.dataset  # isort:skip
import realhf.impl.model  # isort:skip

logger = logging.getLogger("Model Worker", "colored")
blogger = logging.getLogger("benchmark")

TIME_RECORD_RPCS = [
    "generate",
    "inference",
    "train_step",
    "initialize",
]
NON_BLOCKING_RPCS = [
    "model_config",
    "fetch",
    "spec",
    "clear_data_cache",
]

# The model worker will poll requests from the master worker for this many seconds.
# Increase the value if the model worker cannot receive concurrent requests in time.
_MODEL_WORKER_POLL_REQUESTS_SECS = 0.1
_MODEL_WORKER_POLL_REQUESTS_INTERVAL_SECS = 0.01


def get_pytorch_profiler(kernel_only: bool, enabled: bool = True):
    if enabled and kernel_only:
        return torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA])
    elif enabled:
        return torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
        )
    else:
        return contextlib.nullcontext()


class NoRequestToHandle(Exception):
    pass


class ModelWorker(worker_base.Worker):
    _setup_counter = -1

    def _configure(self, cfg: system_api.ModelWorker):
        self._setup_counter += 1

        self.config = cfg
        self.model_names = [s.id.model_name for s in cfg.shards]

        self.__experiment_name = self.config.worker_info.experiment_name
        self.__trial_name = self.config.worker_info.trial_name

        self.data_consumers = self.config.model_rpcs[0].data_consumers

        self.__worker_index = cfg.worker_info.worker_index

        seeding.set_random_seed(cfg.base_seed, f"model_worker{self.__worker_index}")

        # Reveal process group identity of this worker to world.
        gpu_utils.reveal_pg_identity(
            self.__experiment_name, self.__trial_name, self.__worker_index
        )
        self.__dist_env_resolved = False

        self.__clear_cache_frequency = timeutil.FrequencyControl(
            frequency_steps=self.config.cuda_cache_clear_freq
        )
        self.torch_cache_mysophobia = cfg.torch_cache_mysophobia

        r = self.config.worker_info

        # recover info
        self.__recover_run, self.__recover_info = recover.load_recover_info(self.args)

        # Whether to enable profiling is controlled by the following environment variables.
        self.__enable_profiler = os.getenv("REAL_DUMP_TRACE", "0") == "1"
        self.__record_performance = os.getenv("REAL_RECORD_PERFORMANCE", "0") == "1"
        self.__enable_memory_dump = os.getenv("REAL_DUMP_MEMORY", "0") == "1"
        self.__performance_recorder = dict()

        # Add an additional subscript pattern for source RPCs.
        self.__has_dataset = False
        self.__dataset_dp_size = self.__dataset_dp_rank = 0
        sub_patterns = [s.id for s in self.config.shards]
        self.src_rpc = src_rpc = [rpc for rpc in self.config.model_rpcs if rpc.is_src][
            0
        ]
        for s in self.config.shards:
            _pp_size = s.id.topo.get_dim("pipe")
            if not (s.id.tp_rank == 0 and s.id.pp_rank == _pp_size - 1):
                continue
            if src_rpc.model_name == s.id.model_name:
                self.__has_dataset = True
                self.__dataset_dp_size = s.id.topo.get_dim("data")
                self.__dataset_dp_rank = s.id.dp_rank
                sub_patterns.append(f"__data{self.__dataset_dp_rank}__")
                break

        if self.__has_dataset:
            name = names.stream_pullers(self.__experiment_name, self.__trial_name)
            name_resolve.add_subentry(name, str(self.__dataset_dp_rank))

        return r

    def _get_recover_ckpt_path(self, role: str):
        if not self.__recover_run:
            return None
        epoch = self.__recover_info.last_step_info.epoch + 1
        epochstep = self.__recover_info.last_step_info.epoch_step + 1
        globalstep = self.__recover_info.last_step_info.global_step + 1
        save_root = constants.get_save_path(self.args)
        if epoch > 0:
            role_path = os.path.join(save_root, role)
            if os.path.exists(role_path):
                model_path = os.path.join(
                    role_path,
                    f"epoch{epoch}epochstep{epochstep}globalstep{globalstep}",
                )
                if not os.path.exists(model_path):
                    raise RuntimeError(
                        f"Guessed checkpoint path {model_path} does not exist. "
                        "Skip loading checkpoints in the recovered run."
                    )
                return model_path
        return None

    @property
    def _tp_rank(self) -> int:
        return constants.tensor_parallel_rank()

    @property
    def _pp_rank(self) -> int:
        return constants.pipe_parallel_rank()

    @property
    def _dp_rank(self) -> int:
        return constants.data_parallel_rank()

    @property
    def _pp_size(self) -> int:
        return constants.pipe_parallel_world_size()

    @property
    def _tp_size(self) -> int:
        return constants.tensor_parallel_world_size()

    @property
    def _dp_size(self) -> int:
        return constants.data_parallel_world_size()

    @property
    def _is_dp_head(self) -> bool:
        return self._tp_rank == 0 and self._pp_rank == self._pp_size - 1

    @property
    def _model(self) -> model_api.Model:
        return self.__models[constants.model_name()]

    @property
    def _interface(self) -> model_api.ModelInterface:
        return self.__interfaces[constants.model_name()]

    @property
    def _eval_dataloader(self) -> torch.utils.data.DataLoader:
        return self.__eval_dataloaders[constants.model_name()]

    @property
    def _module(self) -> torch.nn.Module | ReaLModel:
        return self.__unwrapped_models[constants.model_name()]

    @property
    def _backend(self) -> model_api.ModelBackend:
        return self.__backends[constants.model_name()]

    def __lazy_setup(self):

        # Build stream connecting with master workers.
        self.__stream = request_reply_stream.make_worker_stream(
            self.config.worker_info,
            idx=self.__worker_index,
        )

        self.__pg_info = global_comm.setup_global_comm(
            args=self.args,
            expr_name=self.__experiment_name,
            trial_name=self.__trial_name,
            worker_index=self.__worker_index,
            model_topos=self.config.model_topos,
            msid2mwid=self.config.msid2mwid,
        )

        self.data_manager = DataManager(
            model_topos=self.config.model_topos,
            msid2mwid=self.config.msid2mwid,
            data_transfer_pairs=self.config.data_transfer_pairs,
        )
        self.data_manager.setup_process_groups()

        self.__param_realloc_info = param_realloc_comm.setup_param_realloc(
            model_topos=self.config.model_topos,
            msid2mwid=self.config.msid2mwid,
            param_realloc_pairs=self.config.sync_param_pairs,
        )

        logger.info(
            f"SetUp Information - Model worker {self.__worker_index} runs on "
            f"node {network.gethostname()} (IP {network.gethostip()}) "
            f"device index {self.__pg_info.local_gpu_id}."
        )

        self.__device = (
            torch.device("cuda:0") if constants.use_cuda() else torch.device("cpu")
        )

        for model_name_, topo_ in self.config.model_topos.items():
            rpcs = [
                rpc for rpc in self.config.model_rpcs if rpc.model_name == model_name_
            ]
            assert len(rpcs) >= 1
            is_trainable_model = any(
                [
                    rpc.interface_type == dfg.ModelInterfaceType.TRAIN_STEP
                    for rpc in rpcs
                ]
            )
            param_realloc_comm.set_trainable(model_name_, is_trainable_model)
            constants.set_rank_mapping(model_name_, topo_, self.config.msid2mwid)
            grid = topology.ParallelGrid(
                topology=topo_,
                rank_mapping=constants.rank_mapping_of_model(model_name_),
                process_group=self.__pg_info.model_groups[model_name_],
            )
            constants.set_grid(model_name_, grid)

        # Set up training dataset for source RPCs.
        self.__datasets = []
        if self.__has_dataset:
            datasets = [
                data_api.make_dataset(
                    d,
                    # NOTE: we must use the same seed to ensure the same dataset split
                    self.config.base_seed,
                    self.__dataset_dp_rank,
                    self.__dataset_dp_size,
                    self.config.tokenizer_name_or_path,
                )
                for d in self.config.datasets
            ]
            self.__datasets = datasets

            self.__dataloaders: List[
                torch.utils.data.DataLoader[data_api.SequenceSample]
            ] = []
            for i, d in enumerate(self.__datasets):
                g = torch.Generator()
                g.manual_seed(
                    self.config.base_seed + seeding._seed_from_key(f"__dataloader{i}__")
                )
                dataloader_kwargs = dict(
                    shuffle=self.config.shuffle_dataset,
                    generator=g,
                )
                if not isinstance(d, PullerStreamDataset):
                    dataloader_kwargs["collate_fn"] = data_api.SequenceSample.gather
                    # NOTE: This is *NOT* the actual batch size for training.
                    # It is just a proper size to load data to workers.
                    dataloader_kwargs["batch_size"] = 10240
                else:
                    dataloader_kwargs["batch_size"] = None
                self.__dataloaders.append(
                    torch.utils.data.DataLoader(d, **dataloader_kwargs)
                )

            self.dataset_size = sum(len(d) for d in self.__datasets)

            self.__data_generators = [enumerate(d) for d in self.__dataloaders]

        self.__models: Dict[ModelName, model_api.Model] = dict()
        self.__model_is_handle: Dict[ModelName, bool] = dict()
        self.__interfaces: Dict[ModelName, model_api.ModelInterface] = dict()
        self.__eval_dataloaders: Dict[ModelName, torch.utils.data.DataLoader] = dict()

        self.__backends: Dict[ModelName, model_api.ModelBackend] = dict()
        self.__unwrapped_models: Dict[ModelName, torch.nn.Module | ReaLModel] = dict()

        self.__backend_initialized: Dict[ModelName, bool] = dict()

        recover_ckpt_paths = []
        for s in self.config.shards:
            with constants.model_scope(s.id.model_name):
                self.__backend_initialized[s.id.model_name] = False
                tik = time.perf_counter()
                if self.__recover_run:
                    model_path = self._get_recover_ckpt_path(s.id.model_name.role)
                    if model_path is not None:
                        logger.info(f"Loading checkpoint during recover: {model_path}")
                        recover_ckpt_paths.append(model_path)
                        if s.model.type_ == "real_model":
                            s.model.args["model_path"] = model_path
                            s.model.args["init_critic_from_actor"] = False
                            s.model.args["init_from_scratch"] = False
                        elif constants.parallelism_rank() == 0:
                            logger.warning(
                                f"Unknown how to recover model type {s.model.type_}"
                            )

                        # Recover indices for dynamic dataset
                        for i, d in enumerate(self.__datasets):
                            if (
                                s.id.model_name == self.src_rpc.model_name
                                and self.__has_dataset
                                and hasattr(d, "filter")
                            ):
                                dataset_indices_path = os.path.join(
                                    constants.get_save_path(self.args),
                                    "dataset_indices",
                                    f"{self._dp_rank}_{i}.npy",
                                )
                                if os.path.exists(dataset_indices_path):
                                    indices = np.load(dataset_indices_path).tolist()
                                    logger.info(
                                        f"DP rank {self._dp_rank} updating dataset indices upon recover, "
                                        f"size {len(d.active_indices)} -> {len(indices)}"
                                    )
                                    d.active_indices = indices

                if constants.parallelism_rank() == 0:
                    self.logger.info(
                        f"Making model {s.id.model_name}, configuration {s.model}..."
                    )

                self.__models[s.id.model_name] = model = model_api.make_model(
                    s.model, name=s.id.model_name, device=self.__device
                )
                if self.__recover_run:
                    model.version = copy.deepcopy(self.__recover_info.last_step_info)
                self.__unwrapped_models[s.id.model_name] = model.module
                if s.should_instantiate:
                    if isinstance(model.module, ReaLModel):
                        model.module.instantiate()
                    self.__model_is_handle[s.id.model_name] = False
                else:
                    self.__model_is_handle[s.id.model_name] = True
                self.__backends[s.id.model_name] = model_api.make_backend(s.backend)
                interface_impl = [
                    rpc.interface_impl
                    for rpc in self.config.model_rpcs
                    if rpc.model_name == s.id.model_name
                ]
                assert all(x == interface_impl[0] for x in interface_impl)
                self.__interfaces[s.id.model_name] = model_api.make_interface(
                    interface_impl[0]
                )

                if s.eval_dataset is not None:
                    eval_dataset = data_api.make_dataset(
                        s.eval_dataset,
                        # NOTE: we must use the same seed to ensure the same dataset split
                        self.config.base_seed,
                        s.id.dp_rank,
                        s.id.topo.get_dim("data"),
                        self.__models[s.id.model_name].tokenizer,
                    )
                    eval_dataloader = torch.utils.data.DataLoader(
                        eval_dataset,
                        batch_size=s.eval_bs,
                        collate_fn=data_api.SequenceSample.gather,
                        shuffle=False,
                    )
                else:
                    eval_dataloader = None
                self.__eval_dataloaders[s.id.model_name] = eval_dataloader

        all_recover_ckpt_paths = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(all_recover_ckpt_paths, recover_ckpt_paths)
        recover_ckpt_paths = set(itertools.chain.from_iterable(all_recover_ckpt_paths))
        for model_path in recover_ckpt_paths:
            if dist.get_rank() == 0 and os.path.islink(model_path):
                # Make the base model path persistent if it is a symlink to the recover checkpoint,
                # because we may want to copy huggingface configurations from it, and
                # th next recover save will remove this symlink.
                dst_path = Path(model_path).parent / "_tmp_ckpt"
                shutil.rmtree(dst_path, ignore_errors=True)
                shutil.copytree(model_path, dst_path)
                os.unlink(model_path)
                os.system(f"mv {str(dst_path)} {model_path}")
        dist.barrier()

        self.__request_cache = {}
        self.__ack_cache = {}

        self.__request_queue = queue.Queue(maxsize=10240)
        self.__reply_queue = queue.Queue(maxsize=10240)
        self.__request_sample_size = dict()

        self.__compute_input_queues = {
            model_name: dict(
                train_step=queue.Queue(4),
                inference=queue.Queue(4),
                generate=queue.Queue(4),
            )
            for model_name in self.__models.keys()
        }

    def __handle_one_rpc_hook(self, hook: str, hook_data: Any):
        ret = None

        tik = time.perf_counter()
        if hook == "data_transfer":
            self.__data_transfer_among_workers(hook_data)
        elif hook == "param_realloc":
            self.__param_realloc(hook_data)
        elif hook == "offload":
            # NOTE: Profiling (or cuda synchronization) will cause an overhead ~0.5s.
            m = self.__unwrapped_models[hook_data["model_name"]]
            if not isinstance(m, ReaLModel):
                logger.warning(
                    f"Model {hook_data['model_name']} (type={type(m)}) is not a ReaLModel, "
                    f"so it can't use offload."
                )
                return
            if not m._offloaded:
                m.async_offload()
        elif hook == "save":
            self.__save_model(hook_data)
        elif hook == "evaluate":
            logger.debug(f"hook_data: {hook_data}")
            with constants.model_scope(hook_data["model_name"]):
                ret = self._interface.evaluate(self._model, self._eval_dataloader)
            if ret:
                logger.info(
                    f"Model {hook_data['model_name']} evaluation done. "
                    f"Statistics: {ret}. Time consumption: {time.perf_counter() - tik:.4f}s."
                )
        else:
            raise NotImplementedError(f"Unknown hook {hook}.")

        self._clear_memory()
        blogger.debug(
            f"Model worker {self.__worker_index} handle "
            f"RPC hook {hook} CPU time {time.perf_counter() - tik:.4f}s."
        )
        if constants.use_cuda():
            torch.cuda.synchronize()
        return ret

    def _clear_memory(self, force=False):
        # empty cache to remove large cache blocks, ~0.1s overhead
        if force or self.torch_cache_mysophobia:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

    def handle_all_pre_hooks(self):
        # drain request queues, handle all pending hooks, then recover the queue
        cache = []
        while True:
            try:
                (
                    request,
                    data,
                    handled,
                    res,
                    time_record,
                ) = self.__request_queue.get_nowait()
                request: request_reply_stream.Payload
                if not handled:
                    while len(request.pre_hooks) > 0:
                        assert len(request.pre_hooks) == len(request.pre_hook_data)
                        assert not handled and res is None
                        with constants.model_scope(request.handler.model_name):
                            if constants.parallelism_rank() == 0:
                                logger.debug(
                                    f"Model `{request.handler.model_name}` handling "
                                    f"{len(request.pre_hooks)} pre-hook for request `{request.handle_name}`. "
                                    f"The current hook is `{request.pre_hooks[0]}`. "
                                    f"{self.__request_queue.qsize()} requests left to handle their potential pre-hooks."
                                )
                        tik = time.perf_counter()
                        hook = request.pre_hooks.pop(0)
                        hook_data = request.pre_hook_data.pop(0)
                        self.__handle_one_rpc_hook(hook, hook_data)
                        time_record[
                            f"timeperf/{request.handler.model_name.role}_{request.handle_name}/pre-{hook}"
                        ] += (time.perf_counter() - tik)
                cache.append((request, data, handled, res, time_record))
            except queue.Empty:
                break

        for c in cache:
            self.__request_queue.put_nowait(c)

    def handle_non_blocking_request(self, request: request_reply_stream.Payload):
        assert len(request.pre_hooks) == 0, request
        assert len(request.post_hooks) == 0, request

        if request.handle_name == "model_config":
            if isinstance(
                self.__unwrapped_models[request.handler.model_name],
                ReaLModel,
            ):
                res = self.__unwrapped_models[request.handler.model_name].config
            else:
                res = None
        elif request.handle_name == "fetch":
            dp_rank = int(re.search(r"__data(\d+)__", request.handler).group(1))
            assert self.__has_dataset
            assert isinstance(request.data, int), request.data
            dataset_id = request.data
            # Fetch.
            try:
                self.__dataset_batch_counter, cur_sample = next(
                    self.__data_generators[dataset_id]
                )
            except StopIteration:
                # Upon the first fetch request, filter dataset and create dataloader.
                eval_scores_path = os.path.join(
                    constants.get_save_path(self.args),
                    "dataset_eval_scores.json",
                )
                dataset_indices_path = os.path.join(
                    constants.get_save_path(self.args),
                    "dataset_indices",
                    f"{dp_rank}_{dataset_id}.npy",
                )
                os.makedirs(os.path.dirname(dataset_indices_path), exist_ok=True)
                if hasattr(self.__datasets[dataset_id], "filter") and os.path.exists(
                    eval_scores_path
                ):
                    # Don't filter dataset on the first poll after recover.
                    with open(eval_scores_path, "r", encoding="utf-8") as f:
                        dataset_eval_scores = json.load(f)
                    self.__datasets[dataset_id].filter(dataset_eval_scores)
                    # Save the dataset indices after filtering
                    np.save(
                        dataset_indices_path,
                        self.__datasets[dataset_id].active_indices,
                    )
                g = torch.Generator()
                g = g.set_state(self.__dataloaders[dataset_id].generator.get_state())
                dataloader_kwargs = dict(
                    shuffle=self.config.shuffle_dataset,
                    generator=g,
                )
                if not isinstance(self.__datasets[dataset_id], PullerStreamDataset):
                    dataloader_kwargs["collate_fn"] = data_api.SequenceSample.gather
                    # NOTE: This is *NOT* the actual batch size for training.
                    # It is just a proper size to load data to workers.
                    dataloader_kwargs["batch_size"] = 10240
                else:
                    dataloader_kwargs["batch_size"] = None
                self.__dataloaders[dataset_id] = torch.utils.data.DataLoader(
                    self.__datasets[dataset_id], **dataloader_kwargs
                )
                self.__data_generators[dataset_id] = enumerate(
                    self.__dataloaders[dataset_id]
                )
                self.__dataset_batch_counter, cur_sample = next(
                    self.__data_generators[dataset_id]
                )

            if isinstance(cur_sample, data_api.SequenceSample):
                samples = cur_sample.unpack()
            else:
                assert isinstance(cur_sample, list), type(cur_sample)
                samples = cur_sample

            data_loaded = []
            for x in samples:
                if (
                    self.__recover_run
                    and x.ids[0] in self.__recover_info.hash_vals_to_ignore
                    # Rollout worker has filered once
                    and not isinstance(self.__datasets[dataset_id], PullerStreamDataset)
                ):
                    self.__recover_info.hash_vals_to_ignore.remove(x.ids[0])
                    continue
                if self.data_manager.has_data(x.ids[0]):
                    continue
                data_loaded.append(x.cpu())
                self.data_manager.store(x)
            assert len(set([x.ids[0] for x in data_loaded])) == len(data_loaded)

            meta_sample = None
            birth_times = []
            if len(data_loaded) > 0:
                sample = data_api.SequenceSample.gather(data_loaded)
                meta_sample = sample.meta()
                if "birth_time" in sample.keys:
                    birth_times = (
                        sample.data["birth_time"].flatten().cpu().numpy().tolist()
                    )
                    assert len(birth_times) == meta_sample.bs
                else:
                    birth_times = (
                        time.monotonic_ns()
                        + np.arange(len(data_loaded), dtype=np.int64)
                    ).tolist()

            res = data_api.DataBatchMeta(
                dp_rank=dp_rank,
                meta_sample=meta_sample,
                birth_times=birth_times,
            )
        elif request.handle_name == "spec":
            # Raw dataset without filtering.
            res = {
                "n_datasets": len(self.__datasets),
                "dataset_size": self.dataset_size,
            }
        elif request.handle_name == "clear_data_cache":
            ids = request.data
            self.data_manager.remove(ids)
            gc.collect()
            if (
                self.config.cuda_cache_cleanliness
                and self.__clear_cache_frequency.check()
            ):
                st = time.monotonic()
                self._clear_memory(force=True)
                et = time.monotonic()
                blogger.debug(
                    f"Model worker {self.__worker_index} cleared cache in {et-st:.4f}s. "
                )
            logger.debug(
                "Get clear_data_cache. "
                f"Remaining data in local storage: {self.data_manager.storage_size()}. "
            )
            res = request_reply_stream.NoResponse()
        self.__reply_queue.put_nowait((request, res))
        self.__request_sample_size[request.request_id] = 1

    def handle_blocking_request(
        self,
        request: request_reply_stream.Payload,
        data: Any,
        handled: bool,
        res: Optional[Any],
        time_record: Dict,
    ) -> worker_base.PollResult:
        tik = time.perf_counter()

        assert not handled and res is None, (
            handled,
            res,
            len(request.post_hooks),
        )

        model_name = request.handler.model_name
        with constants.model_scope(model_name):
            if constants.parallelism_rank() == 0:
                blogger.debug(
                    f"Model #{request.handler.model_name}# "
                    f"starts handling request *{request.handle_name}*."
                )
            res = None
            if request.handle_name == "empty":
                # Empty request is used for executing hooks,
                # e.g., data transfer, parameter syncrhonization.
                pass
            elif request.handle_name == "initialize":
                self.__models[request.handler.model_name] = self._backend.initialize(
                    self._model, data
                )
                if self.__recover_run:
                    model_path = self._get_recover_ckpt_path(model_name.role)
                    if model_path is not None:
                        self._backend.load(
                            self.__models[request.handler.model_name], model_path
                        )
                        logger.info(
                            f"Loaded backend states during recover: {model_path}"
                        )
                self.__backend_initialized[request.handler.model_name] = True
                # Offload this model after initialization if any MFC requires offloading.
                for rpc in self.config.model_rpcs:
                    if rpc.model_name != request.handler.model_name:
                        continue
                    if all(
                        not isinstance(hook, dfg.OffloadHook)
                        for hook in rpc._post_hooks
                    ):
                        continue
                    self.__unwrapped_models[request.handler.model_name].async_offload()
                    break
            ############## computation function calls ##############
            elif request.handle_name in ["inference", "generate", "train_step"]:
                res = self.__handle_model_function_calls(request, data)
            else:
                raise NotImplementedError(
                    f"Unknown request type: {request.handle_name}."
                )

            if (
                request.handle_name in TIME_RECORD_RPCS
                and self._is_dp_head
                and self._dp_rank == 0
            ):
                blogger.debug(
                    f"Model #{request.handler.model_name}# handle "
                    f"request *{request.handle_name}*"
                    f" in ${time.perf_counter() - tik:.4f}$s"
                )
        time_record[
            f"timeperf/{request.handler.model_name.role}_{request.handle_name}/main"
        ] += (time.perf_counter() - tik)

        # Handle all post hooks right after the main computation
        if len(request.post_hooks) > 0:
            assert len(request.post_hooks) == len(request.post_hook_data)
            for hook, hook_data in zip(request.post_hooks, request.post_hook_data):
                tik = time.perf_counter()
                ret = self.__handle_one_rpc_hook(hook, hook_data)
                if hook == "evaluate":
                    assert request.handle_name == "train_step", request.handle_name
                    assert isinstance(ret, dict), ret
                    if isinstance(res, dict):
                        res.update(ret)
                    else:
                        res[0].update(ret)
                time_record[
                    f"timeperf/{request.handler.model_name.role}_{request.handle_name}/post-{hook}"
                ] += (time.perf_counter() - tik)

        # update param realloc step after handling post hooks
        if request.handle_name == "train_step":
            tik = time.perf_counter()
            global_step = self.__models[model_name].version.global_step
            realloc_dir = os.path.join(
                constants.get_param_realloc_path(self.args),
                model_name.role,
                str(global_step),
            )
            save_meta = dict(
                model_name=model_name,
                save_backend=False,
                save_dir=realloc_dir,
            )
            self.__save_model(save_meta)
            name = names.model_version(
                self.__experiment_name,
                self.__trial_name,
                model_name.role,
            )
            with constants.model_scope(model_name):
                dist.barrier(group=constants.cpu_parallelism_group())
                if constants.parallelism_rank() == 0:
                    name_resolve.add(name, str(global_step), replace=True)
            time_record[
                f"timeperf/{request.handler.model_name.role}_{request.handle_name}/param-sync-save"
            ] += (time.perf_counter() - tik)

        res = (res, time_record)
        self.__reply_queue.put_nowait((request, res))
        sample_count = data.bs if isinstance(data, data_api.SequenceSample) else 1
        self.__request_sample_size[request.request_id] = sample_count

    def _get_setup_logdir(self, name):
        subdir = os.path.join(
            constants.get_log_path(self.args),
            name,
            f"setup{self._setup_counter}",
        )
        os.makedirs(subdir, exist_ok=True)
        return subdir

    @contextlib.contextmanager
    def __maybe_profile_rpc(self, rpc: dfg.MFCDef):
        # barrier within this model group before and after profiled RPC
        if (
            self.__record_performance
            or self.__enable_profiler
            or self.__enable_memory_dump
        ):
            torch.cuda.synchronize()
            dist.barrier(group=constants.cpu_parallelism_group())
            # pfer can be a null context if enable_profiler is False
            pfer = get_pytorch_profiler(
                kernel_only=False, enabled=self.__enable_profiler
            )
            pfer.__enter__()
            # The pytorch profiler will call cuda synchronize for us.
            tik = time.perf_counter()

        try:
            yield self
        finally:
            # Dump profiler results.
            if (
                self.__record_performance
                or self.__enable_profiler
                or self.__enable_memory_dump
            ):
                pfer.__exit__(None, None, None)
                dist.barrier(group=constants.cpu_parallelism_group())
                torch.cuda.synchronize()
                tok = time.perf_counter()
                rpc_time = tok - tik

            if self.__record_performance:
                if len(self.__performance_recorder) == 0:
                    self.__performance_recorder["info"] = {
                        "pipeline_size": self._pp_size,
                        "model_size": self._tp_size,
                        "data_size": self._dp_size,
                        "rank": constants.parallelism_rank(),
                        "sequence_parallel_enabled": constants.sequence_parallel(),
                        "gradient_checkpointing_enabled": constants.gradient_checkpointing(),
                        "interface_type": str(rpc.interface_type),
                    }
                    self.__performance_recorder["time"] = [rpc_time]
                else:
                    self.__performance_recorder["time"].append(rpc_time)

                with open(
                    os.path.join(
                        self._get_setup_logdir("performance"),
                        f"rpc-mw{self.__worker_index}.txt",
                    ),
                    "a",
                ) as f:
                    f.write(
                        f"rpc: {rpc.name} rank: {dist.get_rank()} time: {rpc_time}\n"
                    )

            if self.__enable_profiler:
                if self._dp_rank == 0 and self._is_dp_head:
                    blogger.info(
                        f"RPC {rpc.name} execution time "
                        f"w/o external data processing: {rpc_time:.2f} secs."
                    )
                    collect_tik = time.perf_counter()
                    blogger.info(
                        f"Collecting system metrics from the profiler. "
                        "This may take for a while..."
                    )

                pfer.export_chrome_trace(
                    os.path.join(
                        self._get_setup_logdir("trace"),
                        f"{rpc.name}_r{dist.get_rank()}.json",
                    )
                )
                if self._dp_rank == 0 and self._is_dp_head:
                    blogger.info(
                        f"System metrics collected. Time consumption:"
                        f" {time.perf_counter() - collect_tik:.2f} secs."
                    )

    def __handle_model_function_calls(
        self, request: request_reply_stream.Payload, data: Any
    ):
        # Check that the model is instantiated and is not empty.
        assert not self.__model_is_handle[
            request.handler.model_name
        ], request.handler.model_name

        input_queue = self.__compute_input_queues[request.handler.model_name][
            request.handle_name
        ]
        rpc: dfg.MFCDef = next(
            rpc for rpc in self.config.model_rpcs if rpc.name == request.data
        )

        data: data_api.SequenceSample = input_queue.get_nowait()

        if self.config.profile_mode:
            data = self._interface.mock(request.handle_name, self._model, data)

        if rpc.input_key_remap:
            data.remap_keys_(rpc.input_key_remap)

        with self.__maybe_profile_rpc(rpc):
            if request.handle_name == "inference":
                res = self._interface.inference(
                    self._model,
                    data,
                    mb_spec=rpc.mb_spec,
                )  # -> SequenceSample
            elif request.handle_name == "train_step":
                res = self._interface.train_step(
                    self._model,
                    data,
                    mb_spec=rpc.mb_spec,
                )  # -> Dict
            elif request.handle_name == "generate":
                res = self._interface.generate(
                    self._model,
                    data,
                    mb_spec=rpc.mb_spec,
                )  # -> SequenceSample
            else:
                raise NotImplementedError(f"Unknown MFC type: {request.handle_name}.")

        eval_scores_path = os.path.join(
            constants.get_save_path(self.args),
            "dataset_eval_scores.json",
        )
        eval_scores = {}
        if isinstance(res, data_api.SequenceSample) and constants.is_dp_head():
            if rpc.output_key_remap:
                res.remap_keys_(rpc.output_key_remap)
            res = res.select(rpc.output_keys)

            # Update scores to update data sample distribution.
            if "scores" in res.metadata:
                # All-gather across the DP rank
                all_scores = [None for _ in range(self._dp_size)]
                local_scores = {i: s for i, s in zip(res.ids, res.metadata["scores"])}
                dist.all_gather_object(
                    all_scores,
                    local_scores,
                    group=constants.data_parallel_group(),
                )
                # Since the device mesh generating "scores" may not overlap
                # with the device mesh of dataset, write all scores into the disk
                # for later usage.

                if os.path.exists(eval_scores_path):
                    with open(eval_scores_path, "r", encoding="utf-8") as f:
                        eval_scores.update(json.load(f))
                for scores in all_scores:
                    eval_scores.update(scores)

                res.metadata.pop("scores")
        dist.barrier(group=constants.cpu_parallelism_group())
        if len(eval_scores) > 0 and self._dp_rank == 0 and self._is_dp_head:
            with open(
                eval_scores_path,
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(eval_scores, f, ensure_ascii=False, indent=4)

        # Store data into storage.
        if self._is_dp_head and isinstance(res, data_api.SequenceSample):
            for x in res.unpack():
                # The input data must exist in the storage, otherwise
                # the model function call will not run.
                self.data_manager.update(x)

        # Only return meta data back to the master worker.
        if isinstance(res, data_api.SequenceSample):
            res = res.meta()

        if constants.use_cuda():
            # Monitoring info. There's an all-gather and an all-reduce
            # over the parallelism group in this function.
            torch.cuda.synchronize()
            if (
                self._model.backend_name != "vllm"
                and self._model.backend_name != "sglang"
            ):
                # Since vLLM/SGLang allocates GPU memory in advance, it is very
                # easy to exceed the 0.95 threshold that triggers a kill.
                # We omit GPU stats logging for vLLM/SGLang.
                self.__log_gpu_stats(request)

        self._clear_memory()
        if constants.use_cuda():
            torch.cuda.synchronize()
        dist.barrier(group=constants.cpu_parallelism_group())
        return res

    def __data_transfer_among_workers(self, hook_data: Dict[str, Any]):
        meta_sample = hook_data["meta_sample"]

        plan = [RedistribStep(**json.loads(x)) for x in hook_data["plan"]]
        self.data_manager.redistribute(meta_sample, plan=plan)

        if hook_data["target"] in self.__models:
            with constants.model_scope(hook_data["target"]):
                local_ids = hook_data["partitioned_ids"][self._dp_rank]
            r = data_api.SequenceSample.gather(
                [
                    self.data_manager.get(_id).to_device(constants.current_device())
                    for _id in local_ids
                ],
                keys=meta_sample.keys,
            )
            self.__compute_input_queues[hook_data["target"]][
                hook_data["handle_name"]
            ].put_nowait(r)

    def __param_realloc(self, hook_data: Dict):
        from_model_name: ModelName = hook_data["from_model_name"]
        to_model_name: ModelName = hook_data["to_model_name"]

        from_topo: topology.ProcessTopology = hook_data["from_topo"]
        to_topo: topology.ProcessTopology = hook_data["to_topo"]

        # NOTE: For the convenience of future developement, we
        # run parameter reallocation with disk save-load by default.
        if os.getenv("REAL_PARAM_REALLOC_IMPL", "DISK") == "DISK":
            if hook_data["eta"] != 1.0:
                raise NotImplementedError("eta != 1.0 is not supported yet.")

            # If the source is not a trainable model, it will not own
            # parameters, so we just release its GPU memory.
            with constants.model_scope(from_model_name):
                from_model_ranks = sorted(constants.parallelism_group_ranks())
            if not param_realloc_comm.is_trainable(from_model_name):
                if dist.get_rank() not in from_model_ranks:
                    return
                if not isinstance(self.__unwrapped_models[from_model_name], ReaLModel):
                    # We can only release the memory of ReaLModel,
                    # because we don't know how to rebuild the parameters otherwise.
                    return
                m = self.__unwrapped_models[from_model_name]
                dummy_tensor = torch.tensor((), dtype=m.dtype, device=m.device)
                for p in m.layers.parameters():
                    p.data = dummy_tensor
                m.contiguous_param = dummy_tensor
                return

            # Get global_step from source model via broadcast,
            # since there are no global_step information on model workers for generation.
            if (
                from_model_name in self.__models
                and dist.get_rank() == from_model_ranks[0]
            ):
                global_step = self.__models[from_model_name].version.global_step
            else:
                global_step = 0
            g = self.__param_realloc_info.param_realloc_model_cpu_group[
                param_realloc_comm.ParamReallocModelPair(from_model_name, to_model_name)
            ]
            global_step = torch.tensor(global_step, device="cpu")
            dist.broadcast(global_step, src=from_model_ranks[0], group=g)
            global_step = int(global_step.item())

            realloc_dir = os.path.join(
                constants.get_param_realloc_path(self.args),
                from_model_name.role,
                str(global_step),
            )
            if from_model_name in self.__unwrapped_models:
                save_meta = dict(
                    model_name=from_model_name,
                    save_backend=False,
                    save_dir=realloc_dir,
                )
                self.__save_model(save_meta)
            dist.barrier(group=g)
            if to_model_name in self.__unwrapped_models:
                load_meta = dict(
                    model_name=to_model_name,
                    load_dir=realloc_dir,
                )
                self.__load_model(load_meta)
                # Remove the reallocated checkpoint.
                with constants.model_scope(to_model_name):
                    dist.barrier(constants.cpu_parallelism_group())
                    if constants.parallelism_rank() == 0:
                        shutil.rmtree(realloc_dir, ignore_errors=True)
                        os.makedirs(realloc_dir, exist_ok=True)
        else:
            logger.warning(
                "[Depreated Warning] Parameter reallocation through "
                "NCCL will be disabled in future versions."
            )
            to_model_config = hook_data["to_model_config"]
            if from_model_name in self.__unwrapped_models:
                m = self.__unwrapped_models[from_model_name]
            else:
                m = self.__unwrapped_models[to_model_name]
            try:
                new_layers, new_param, _ = m.build_reparallelized_layers_async(
                    from_model_name=from_model_name,
                    to_model_name=to_model_name,
                    from_topo=from_topo,
                    to_topo=to_topo,
                    to_model_config=to_model_config,
                    pg_info=self.__param_realloc_info,
                )
            except RuntimeError as e:
                if from_model_name in self.__unwrapped_models:
                    logger.error(f"from model error: {from_model_name}")
                if to_model_name in self.__unwrapped_models:
                    logger.info(f"to model error: {to_model_name}")
                raise e
            if to_model_name in self.__models and param_realloc_comm.is_trainable(
                from_model_name
            ):
                self.__unwrapped_models[to_model_name].patch_reparallelization(
                    (new_layers, new_param), eta=hook_data["eta"]
                )

        if from_model_name in self.__models and not param_realloc_comm.is_trainable(
            from_model_name
        ):
            self.__model_is_handle[from_model_name] = True
        if to_model_name in self.__models and param_realloc_comm.is_trainable(
            from_model_name
        ):
            self.__model_is_handle[to_model_name] = False

    def __save_model(self, hook_data: Dict):
        # NOTE: we should not create the `save_dir` here,
        # because it will be automatically created by our save function.
        # As such, if the checkpoint dir exists, we know that the checkpoint
        # must have been properly saved.
        tik = time.perf_counter()
        # When `recover_only` is True, the model should save an overwrittable checkpoint for recover.
        recover_only = hook_data.get("recover_only", False)
        with constants.model_scope(hook_data["model_name"]):
            if not recover_only:
                save_dir = hook_data["save_dir"]
            else:
                # Remove all previous symlinks.
                save_root = Path(hook_data["save_dir"]).parent
                save_dir = str(save_root / "recover_checkpoint")
                if constants.parallelism_rank() == 0:
                    if os.path.exists(save_root):
                        for fn in os.listdir(save_root):
                            if (save_root / fn).is_dir() and (
                                save_root / fn
                            ).is_symlink():
                                os.unlink(save_root / fn)
                    shutil.rmtree(save_dir, ignore_errors=True)
            dist.barrier(constants.cpu_parallelism_group())
            self._interface.save(self._model, save_dir)
            # The `save` method of the interface may be empty.
            # We only save the backend state if the parameters have been indeed saved.
            if os.path.exists(save_dir) and hook_data.get("save_backend", True):
                self._backend.save(self._model, save_dir)

            t = torch.tensor(
                float(time.perf_counter() - tik),
                dtype=torch.float64,
                device=constants.current_device(),
            )
            dist.all_reduce(
                t, op=dist.ReduceOp.MAX, group=constants.parallelism_group()
            )
            if constants.parallelism_rank() == 0:
                if recover_only and os.path.exists(save_dir):
                    # Create a symlink from "recover_checkpoint" to a directory with step counter,
                    # such that we can directly load it as a persistent checkpoint.
                    os.symlink(save_dir, hook_data["save_dir"])
                logger.info(
                    f"Model {hook_data['model_name']} saved at {hook_data['save_dir']}. "
                    f"Time consumption: {float(t):.4f}s."
                )

    def __load_model(self, hook_data: Dict):
        tik = time.perf_counter()
        with constants.model_scope(hook_data["model_name"]):
            if isinstance(self._model.module, torch.nn.Identity) and isinstance(
                self._backend,
                (
                    model_api.ALL_BACKEND_CLASSES["sglang"],
                    model_api.ALL_BACKEND_CLASSES["vllm"],
                ),
            ):
                # The uninitialized vLLM/SGLang model. Since we create the model
                # inside the vLLM/SGLang backend, the initial param realloc before
                # backend initialization can be ignored.
                return
            if self._model.backend_name in ["vllm", "sglang"]:
                if constants.parallelism_rank() == 0:
                    logger.info(f"Updating {self._model.backend_name} model from disk.")
                module = self._model.module
                module.update_weights_from_disk(hook_data["load_dir"])
            else:
                module: ReaLModel = self.__unwrapped_models[hook_data["model_name"]]
                assert isinstance(module, ReaLModel), type(module)
                module.instantiate()
                module.load_from_hf(hook_data["load_dir"], init_critic_from_actor=False)

            t = torch.tensor(
                float(time.perf_counter() - tik),
                dtype=torch.float64,
                device=constants.current_device(),
            )
            dist.all_reduce(
                t, op=dist.ReduceOp.MAX, group=constants.parallelism_group()
            )
            if constants.parallelism_rank() == 0:
                logger.info(
                    f"Model {hook_data['model_name']} loaded from {hook_data['load_dir']}. "
                    f"Time consumption: {float(t):.4f}s."
                )

    def maybe_post_responses(self):
        ready_to_post = []
        while True:
            try:
                request, res = self.__reply_queue.get_nowait()
                ready_to_post.append((request, res))
            except queue.Empty:
                break

        batch_size = sample_size = 0
        for request, res in ready_to_post:
            # For some requests, do not respond to the master worker.
            if isinstance(res, request_reply_stream.NoResponse):
                continue
            request: request_reply_stream.Payload
            reply = request_reply_stream.Payload(
                handler="master",
                request_id=request.request_id,
                handle_name=request.handle_name,
                data=res,
            )
            self.__stream.post(reply)
            # logger.info(f"handle_name {request.handle_name} Posted req id = {request.request_id}")
            sample_size += self.__request_sample_size.pop(request.request_id)
            batch_size += 1
        return worker_base.PollResult(sample_count=sample_size, batch_count=batch_size)

    def __maybe_receive_one_request(self):
        try:
            r: request_reply_stream.Payload = self.__stream.poll()
            if r.handle_name == "ack":
                self.__ack_cache[r.request_id] = r
            else:
                if r.no_syn:
                    self.__request_queue.put_nowait(
                        (r, r.data, False, None, defaultdict(int))
                    )
                else:
                    self.__stream.post(
                        request_reply_stream.Payload(
                            handler="master",
                            request_id=r.syn_reply_id,
                            handle_name="syn",
                        ),
                    )
                    self.__request_cache[r.ack_reply_id] = r
        except request_reply_stream.NoMessage:
            time.sleep(_MODEL_WORKER_POLL_REQUESTS_INTERVAL_SECS)
            pass

    def maybe_receive_requests(self):
        tik = time.perf_counter()
        while time.perf_counter() - tik < _MODEL_WORKER_POLL_REQUESTS_SECS:
            self.__maybe_receive_one_request()
            cur_ack_ids = list(self.__ack_cache.keys())
            for ack_id in cur_ack_ids:
                if ack_id in self.__request_cache:
                    self.__ack_cache.pop(ack_id)
                    req = self.__request_cache.pop(ack_id)
                    self.__request_queue.put_nowait(
                        (req, req.data, False, None, defaultdict(int))
                    )

    def _poll(self):
        if not self.__dist_env_resolved:
            self.__lazy_setup()
            if constants.use_cuda():
                self._clear_memory(force=True)
                pynvml.nvmlInit()
                self.__nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(
                    self.__pg_info.local_gpu_id
                )
            else:
                self.__nvml_handle = None
            self.__dist_env_resolved = True

        self.maybe_receive_requests()

        r = worker_base.PollResult(0, 0)

        # Prioritize the `reset` and `flush` request.
        # If `flush`, run all the remaining blocking requests.
        # These requested tasks typically involve NCCL communication
        # or GPU computation. We need to ensure that all these tasks
        # are executed in the same order across all model workers.
        flush = False
        for _ in range(self.__request_queue.qsize()):
            request, data, handled, res, time_record = self.__request_queue.get_nowait()
            if request.handle_name == "reset":
                # Pause the worker and wait for the next `configure`
                # command from the controller.
                return self.__experiment_complete_exit()
            elif request.handle_name == "flush":
                flush = True
            elif request.handle_name in NON_BLOCKING_RPCS:
                self.handle_non_blocking_request(request)
            else:
                self.__request_queue.put_nowait(
                    (request, data, handled, res, time_record)
                )

        # Non-blocking requests are usually fast, so we can
        # respond them in a batch without affecting the accuracy
        # of time logging in the master worker.
        r += self.maybe_post_responses()

        if flush:
            # NOTE: We ensure that all model workers have the same set of requests
            # at any time through a TCP-like protocol, i.e., req -> ack -> syn -> resp.
            # Each request is composed of pre-hooks, the main request, and post-hooks.
            # We execute all pre-hooks first because they involve data transfer
            # among workers. Executing them first avoids blocking MFCs that require
            # data from the same set of GPUs but are executed on disjoint GPUs.
            self.handle_all_pre_hooks()

            # Prioritize requests that requires a smaller device mesh.
            rescheduled_requests = []
            other_requests = []
            for _ in range(self.__request_queue.qsize()):
                (
                    request,
                    data,
                    handled,
                    res,
                    time_record,
                ) = self.__request_queue.get_nowait()
                if request.handle_name not in ["inference", "generate", "train_step"]:
                    other_requests.append((request, data, handled, res, time_record))
                else:
                    with constants.model_scope(request.handler.model_name):
                        w = dist.get_world_size(constants.parallelism_group())
                    rescheduled_requests.append(
                        (request, data, handled, res, time_record, w)
                    )
            rescheduled_requests.sort(key=lambda x: x[-1])
            for request, data, handled, res, time_record, _ in rescheduled_requests:
                self.__request_queue.put_nowait(
                    (request, data, handled, res, time_record)
                )
            for request, data, handled, res, time_record in other_requests:
                self.__request_queue.put_nowait(
                    (request, data, handled, res, time_record)
                )

            # Execute one MFC them immediately return the result, such that
            # we can correctly log the time consumption in the master worker.
            while True:
                try:
                    (
                        request,
                        data,
                        handled,
                        res,
                        time_record,
                    ) = self.__request_queue.get_nowait()
                    self.handle_blocking_request(
                        request, data, handled, res, time_record
                    )
                    r += self.maybe_post_responses()
                except queue.Empty:
                    break
        return r

    def __experiment_complete_exit(self):
        # maybe dump profile recorder
        if self.__record_performance:
            with open(
                os.path.join(
                    self._get_setup_logdir("performance"),
                    f"mw{self.__worker_index}.json",
                ),
                "w",
            ) as f:
                json.dump(self.__performance_recorder, f, indent=4)

        self.__stream.close()

        self.__unwrapped_models.clear()

        # Calling backend.destroy removes all hooks and releases the memory.
        for model_name, backend in self.__backends.items():
            backend.destroy(self.__models[model_name])

        self.__models.clear()
        self.__backends.clear()
        self.__interfaces.clear()

        # Reset model worker states.
        self.__dist_env_resolved = False

        if constants.use_cuda():
            before_mem = pynvml.nvmlDeviceGetMemoryInfo(self.__nvml_handle).used

        constants.reset_run()
        topology.destroy_all_comm_groups()
        cuda_graph.destroy_all()

        self._clear_memory(force=True)

        if constants.use_cuda():
            # Record memory.
            after_mem = pynvml.nvmlDeviceGetMemoryInfo(self.__nvml_handle).used
            blogger.debug(
                f"GPU memory used upon experiment complete: "
                f"{before_mem/1024**2:.2f}MB -> {after_mem / 1024**2:.2f}MB"
            )

            self.__nvml_handle = None
            try:
                pynvml.nvmlShutdown()
            except pynvml.nvml.NVMLError_Uninitialized:
                pass
        self.pause()
        return worker_base.PollResult(sample_count=0, batch_count=0)

    # def __recover_save(self):
    #     # store model and dataset states for recover
    #     if self.__dist_env_resolved:

    #         for model_name, model in self.__models.items():
    #             if self.__model_is_handle[model_name]:
    #                 continue
    #             constants._model_name = None  # force quit model_scope
    #             with constants.model_scope(model_name):
    #                 ckpt_save_dir = os.path.join(
    #                     self.__recover_states_root, "ckpt", model_name.role
    #                 )
    #                 # replace old recover ckpt
    #                 logger.info(
    #                     f"saving model {model_name} ckpt for recover at {ckpt_save_dir}. "
    #                     f"epoch {model.version.epoch}, epoch_step {model.version.epoch_step}, "
    #                     f"global step {model.version.global_step}"
    #                 )
    #                 if self.__has_dataset:
    #                     logger.info(
    #                         f"Dataset info: " f"dataset epoch {self.__dataset_epoch}"
    #                     )
    #                 self._interface.save(model, ckpt_save_dir)
    #                 logger.info(f"saving done.")

    # def _exit_hook(self, exit_status: worker_base.WorkerServerStatus):
    #     logger.info(
    #         f"Model worker {self.__worker_index} exit with status {exit_status}."
    #     )
    #     if os.getenv("REAL_SAVE_RECOVER_STATES", "0") != "1":
    #         return
    #     if exit_status == worker_base.WorkerServerStatus.ERROR:
    #         try:
    #             sleep_time = 600
    #             current_sleep_time = 0
    #             while current_sleep_time < sleep_time:
    #                 logger.info(
    #                     f"ERROR exit, waited {current_sleep_time} s for interruption ..."
    #                 )
    #                 time.sleep(10)
    #                 current_sleep_time += 10
    #         except KeyboardInterrupt:
    #             logger.info("Received SIGINT, starting recover save")

    #     self.__recover_save()

    def __log_gpu_stats(self, request: request_reply_stream.Payload):
        # Log GPU utilization and memory statistics.
        utilization = pynvml.nvmlDeviceGetUtilizationRates(self.__nvml_handle)  # bytes
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(self.__nvml_handle)  # bytes
        kill_threshold = float(os.environ.get("REAL_GPU_MEMORY_KILL_THRESHOLD", "1.0"))
        if memory_info.used / memory_info.total > kill_threshold:
            raise RuntimeError(
                f"GPU memory excceeds kill threshold {kill_threshold:.2f}. "
                "This threshold could be adjusted by changing environment "
                'variable "REAL_GPU_MEMORY_KILL_THRESHOLD".'
            )

        torch_mem_stats = torch.cuda.memory_stats(0)

        # All-gather hostname, gpu ID, and stats.
        hostname = socket.gethostname()
        hostname_len = len(hostname)
        assert hostname_len < 64, "hostname should not have more than 64 chars"
        # Encode hostnames into long.
        hostname_np = np.fromstring(
            hostname + "x" * (64 - len(hostname)), dtype=np.int64
        )
        local_mem_stats = torch.tensor(
            [hostname_len, self.__pg_info.local_gpu_id]
            + hostname_np.tolist()
            + [
                torch_mem_stats["allocated_bytes.all.peak"],
                torch_mem_stats["reserved_bytes.all.peak"],
                memory_info.used,
            ],
            dtype=torch.long,
            device="cuda",
        )  # length 2 + 8 + 3 = 13
        mem_stats = local_mem_stats.new_zeros(
            size=(
                dist.get_world_size(constants.parallelism_group()),
                local_mem_stats.shape[0],
            )
        )
        # All-gather memory stats.
        dist.all_gather_into_tensor(
            mem_stats, local_mem_stats, group=constants.parallelism_group()
        )
        mem_stats = mem_stats.cpu().numpy()

        # All-reduce utilization.
        gpu_compute_util = torch.tensor(
            utilization.gpu, dtype=torch.float32, device="cuda"
        )
        dist.all_reduce(gpu_compute_util, group=constants.parallelism_group())
        gpu_compute_util = gpu_compute_util.item() / dist.get_world_size(
            constants.parallelism_group()
        )

        def _decode_hostname(idx):
            hn_np = mem_stats[idx, 2 : 2 + 8]
            l = mem_stats[idx, 0]
            return hn_np.tobytes().decode("utf-8")[:l]

        def _decode_gpu_id(idx):
            return f"{_decode_hostname(idx)}:{mem_stats[idx, 1]}"

        max_used_gpu_id = _decode_gpu_id(np.argmax(mem_stats[:, -1]))
        max_reserved_gpu_id = _decode_gpu_id(np.argmax(mem_stats[:, -2]))
        max_tensor_gpu_id = _decode_gpu_id(np.argmax(mem_stats[:, -3]))

        # NOTE: We only log the peak memory because it's
        # the most important for detecting OOM issues.
        headers = [
            " ",
            "TotalMem",
            "PeakUsedMem",
            "PeakTensorMem",
            "PeakReservedMem",
            "MaxMemUtil",
            "AvgComputeUtil",
        ]
        line1 = [
            "Value",
            f"{memory_info.total / 1024**2:.2f}MB",
            f"{max(mem_stats[:, -1]) / 1024**2:.2f}MB",
            f"{max(mem_stats[:, -3]) / 1024**2:.2f}MB",
            f"{max(mem_stats[:, -2]) / 1024**2:.2f}MB",
            f"{max(mem_stats[:, -1]) / memory_info.total * 100:.2f}%",
            f"{gpu_compute_util:.2f}%",
        ]
        line2 = [
            "GPU ID",
            "-",
            max_used_gpu_id,
            max_tensor_gpu_id,
            max_reserved_gpu_id,
            max_used_gpu_id,
            "-",
        ]

        if self._dp_rank == 0 and self._is_dp_head:
            logger.info(
                f"Aggregated GPU memory stats after MFC `{request.handle_name}`"
                f" within model `{request.handler.model_name}`:\n"
                + tabulate.tabulate(
                    [headers, line1, line2], headers="firstrow", tablefmt="fancy_grid"
                )
            )
