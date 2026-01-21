# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import dataclasses
import gc
import multiprocessing as mp
import os
import pickle
import queue
import random
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple

import pynvml
import pytest
import torch
import torch.distributed as dist
import torch.utils.data

from realhf.api.cli_args import BaseExperimentConfig, NameResolveConfig
from realhf.api.core.data_api import SequenceSample
from realhf.base import constants, gpu_utils, logging, name_resolve, names, topology
from realhf.base.topology import (
    DataPipeTensorParallelTopology,
    ParallelGrid,
    PipeDataTensorParallelTopology,
)

logger = logging.getLogger("testing")

MODEL_NAME = "default"
_DEFAULT_EXPR_NAME = "test"
_DEFAULT_TRIAL_NAME = "test"

TESTING_MODEL_VOCAB_SIZE = 128
TESTING_MODEL_N_POSITIONS = 128
TESTING_MODEL_INTERMEDIATE_SIZE = 32
TESTING_MODEL_HIDDEN_SIZE = 16
TESTING_MODEL_HEAD_DIM = 2
TESTING_MODEL_N_LAYERS = 8
TESTING_MODEL_N_HEADS = 8

TESTING_DATASET_SIZE = 100


class StandaloneTestingProcess(mp.Process):
    """Aims for defining this class:
    + Removing duplicate setup GPU codes in each test.

    Note that `init_global_constants` should be called in `func`.
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        barrier: mp.Barrier,  # type: ignore
        err_queue: mp.Queue,
        func: Callable,
        *args,
        expr_name: str = None,
        dist_backend: Optional[str] = None,
        trial_name: str = None,
        setup_dist_torch: bool = True,
        use_cpu: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        self.err_queue = err_queue
        self.barrier = barrier

        self.expr_name = expr_name if expr_name is not None else _DEFAULT_EXPR_NAME
        self.trial_name = trial_name if trial_name is not None else _DEFAULT_TRIAL_NAME
        self.dist_backend = dist_backend

        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.dist_backend = dist_backend
        self.setup_dist_torch = setup_dist_torch
        self.use_cpu = use_cpu

    def _run(self):
        return self.func(*self.args, **self.kwargs)

    def run(self) -> None:
        assert not torch.cuda.is_initialized()
        constants.set_force_cpu(self.use_cpu)
        if constants.use_cuda():
            torch.cuda.set_device(0)

        from realhf.api.cli_args import NameResolveConfig

        name_resolve.reconfigure(
            NameResolveConfig(
                "nfs", f"/tmp/areal/testing/{self.expr_name}/{self.trial_name}/"
            )
        )

        self.barrier.wait()

        if self.setup_dist_torch:
            # init process group
            gpu_utils.reveal_pg_identity(self.expr_name, self.trial_name, self.rank)
            self.barrier.wait()
            from realhf.impl.model.comm.global_comm import setup_global_comm

            if self.dist_backend is None:
                self.dist_backend = "gloo" if not constants.use_cuda() else "nccl"
            setup_global_comm(
                BaseExperimentConfig(),
                self.expr_name,
                self.trial_name,
                self.rank,
                backend=self.dist_backend,
            )

        # misc setup
        if constants.use_cuda():
            pynvml.nvmlInit()
            pytorch_memory_burnin(self.rank)

        try:
            self._run()
        except Exception as e:
            print(traceback.format_exc(), flush=True)
            self.err_queue.put(e)
            raise e
        if dist.is_initialized():
            dist.destroy_process_group()


class LocalMultiProcessTest:
    """Aims for defining this class:
    1. Defining a barrier and a queue for all sub-processes.
    2. Error handling after launch.
    """

    # NOTE: This is necessary for running pytest, otherwise
    # pytest will exit early before subprocesses terminate.
    mp.set_start_method("spawn", force=True)

    def __init__(
        self,
        world_size: int,
        func: Callable | List[Callable],
        *args,
        expr_name: str = None,
        trial_name: str = None,
        dist_backend: Optional[str] = None,
        timeout_secs: int = 300,
        setup_dist_torch: bool = True,
        **kwargs,
    ):
        self.barrier = mp.Barrier(world_size)
        self.err_queue = mp.Queue(world_size)
        os.environ["REAL_MODE"] = "LOCAL"
        if torch.cuda.is_available():
            os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
            os.environ["GPU_DEVICES_ISOLATED"] = str(1)
        expr_name = expr_name if expr_name is not None else _DEFAULT_EXPR_NAME
        trial_name = trial_name if trial_name is not None else _DEFAULT_TRIAL_NAME
        name_resolve.reconfigure(
            NameResolveConfig("nfs", f"/tmp/areal/testing/{expr_name}/{trial_name}/")
        )
        clear_name_resolve(expr_name, trial_name)
        self.timeout_secs = timeout_secs
        self.processes = []
        if isinstance(func, list):
            assert len(func) == world_size, (len(func), world_size)
        for rank in range(world_size):
            if torch.cuda.is_available():
                os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
            p = StandaloneTestingProcess(
                rank,
                world_size,
                self.barrier,
                self.err_queue,
                func[rank] if isinstance(func, list) else func,
                *args,
                expr_name=expr_name,
                trial_name=trial_name,
                dist_backend=dist_backend,
                setup_dist_torch=setup_dist_torch,
                use_cpu=not constants.use_cuda(),
                **kwargs,
            )
            self.processes.append(p)

    def start(self):
        [p.start() for p in self.processes]

    def wait(self, timeout=None):
        timeout = timeout or self.timeout_secs
        tik = time.time()
        while any([p.is_alive() for p in self.processes]):
            try:
                err = self.err_queue.get_nowait()
                raise err
            except queue.Empty:
                time.sleep(0.1)
            if time.time() - tik > self.timeout_secs:
                raise TimeoutError("Timeout")

    def launch(self, timeout=None):
        self.start()
        try:
            self.wait(timeout)
        except TimeoutError as e:
            self.terminate()
            raise e
        self.terminate()

    def terminate(self):
        [p.terminate() for p in self.processes]
        [p.join() for p in self.processes]


def init_global_constants(
    num_dp=1,
    num_tp=1,
    num_pp=1,
    topo=None,
    model_name=None,
    msid2mwid=None,
    sequence_parallel=False,
    gradient_checkpointing=True,
    gradient_accumulation_fusion=False,
    max_prompt_len=None,
    is_train: bool = True,
    expr_name=None,
    trial_name=None,
):
    expr_name = expr_name if expr_name is not None else _DEFAULT_EXPR_NAME
    trial_name = trial_name if trial_name is not None else _DEFAULT_TRIAL_NAME
    constants.set_experiment_trial_names(expr_name, trial_name)
    model_name = model_name if model_name is not None else MODEL_NAME

    if topo is None:
        if is_train:
            topo = PipeDataTensorParallelTopology(
                num_dp=num_dp,
                num_tp=num_tp,
                num_pp=num_pp,
                sequence_parallel=sequence_parallel,
                gradient_checkpointing=gradient_checkpointing,
                gradient_accumulation_fusion=gradient_accumulation_fusion,
                max_prompt_len=max_prompt_len,
            )
        else:
            topo = DataPipeTensorParallelTopology(
                num_dp=num_dp,
                num_tp=num_tp,
                num_pp=num_pp,
                sequence_parallel=sequence_parallel,
            )
        ws = num_dp * num_tp * num_pp
    else:
        ws = topo.world_size()

    with constants.model_scope(model_name):
        constants.set_rank_mapping(model_name, topo, msid2mwid=msid2mwid)
        wg_ranks = [constants.to_global_pg_rank(i) for i in range(ws)]
        wg = topology.new_or_get_group(ranks=wg_ranks)

        constants.set_parallelism_group(
            model_name=model_name, pgroup=wg, ranks=wg_ranks
        )
        grid = ParallelGrid(
            process_group=wg,
            topology=topo,
            rank_mapping=constants.rank_mapping_of_model(model_name),
        )
        constants.set_grid(model_name=model_name, grid=grid)


def clear_name_resolve(expr_name=None, trial_name=None):
    expr_name = expr_name if expr_name is not None else _DEFAULT_EXPR_NAME
    trial_name = trial_name if trial_name is not None else _DEFAULT_TRIAL_NAME
    name_resolve.clear_subtree(
        names.trial_root(experiment_name=expr_name, trial_name=trial_name)
    )


def pytorch_memory_burnin(rank):
    torch.cuda.set_device(0)
    torch.cuda.init()
    x = torch.randn(1, device="cuda", dtype=torch.float64, requires_grad=True)
    y = x * torch.randn(1000, device="cuda", dtype=torch.float64)
    y.mean().backward()
    del x, y
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()


def clear_gpu_cache():
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()


def get_memory(rank):
    handle = pynvml.nvmlDeviceGetHandleByIndex(rank)
    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    used_memory = memory_info.used / (1024**2)  # Convert bytes to megabytes
    return used_memory


def get_pytorch_profiler(save_fn: str):

    def trace_handler(p: torch.profiler._KinetoProfile):
        # print(
        #     p.key_averages().table(
        #         sort_by="cuda_memory_usage", row_limit=20, max_name_column_width=30, max_src_column_width=30
        #     )
        # )
        p.export_chrome_trace(save_fn)

    return torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        on_trace_ready=trace_handler,
        with_flops=True,
    )


def random_sample(num_sequences: int, seq_len: int, vocab_size: int, seed: int = 1):
    torch.manual_seed(seed)
    return torch.randint(0, vocab_size, (num_sequences, seq_len), dtype=torch.long)


def make_random_packed_batches(
    n_batches,
    batch_size,
    seq_len,
    vocab_size,
    seed: int = 1,
    dp_rank=None,
    dp_size=None,
) -> List[SequenceSample]:
    assert (dp_rank is None and dp_size is None) or (
        dp_rank is not None and dp_size is not None
    )
    if dp_rank is None:
        dp_rank = constants.data_parallel_rank()
        dp_size = constants.data_parallel_world_size()
    assert batch_size % dp_size == 0
    n_seqs = batch_size * n_batches
    seqs = random_sample(batch_size * n_batches, seq_len, vocab_size, seed)
    seqs = seqs[n_seqs * dp_rank // dp_size : n_seqs * (dp_rank + 1) // dp_size]
    x = SequenceSample.from_default(
        seqlens=[seq_len for _ in range(seqs.shape[0])],
        data=dict(
            packed_input_ids=seqs.view(-1),
            prompt_mask=torch.zeros_like(seqs.view(-1), dtype=torch.bool),
        ),
        ids=list(range(seqs.shape[0])),
    )
    return x.split(n_batches)


def make_random_unpacked_batches(
    n_batches,
    batch_size,
    seq_len,
    vocab_size,
    seed: int = 1,
    dp_rank=None,
    dp_size=None,
):
    n_seqs = batch_size * n_batches
    dp_batch_size = batch_size // dp_size
    assert (dp_rank is None and dp_size is None) or (
        dp_rank is not None and dp_size is not None
    )
    if dp_rank is None:
        dp_rank = constants.data_parallel_rank()
        dp_size = constants.data_parallel_world_size()
    assert batch_size % dp_size == 0
    seqs = random_sample(batch_size * n_batches, seq_len, vocab_size, seed)
    seqs = seqs[n_seqs * dp_rank // dp_size : n_seqs * (dp_rank + 1) // dp_size]
    batches = [
        seqs[j * dp_batch_size : (j + 1) * dp_batch_size] for j in range(n_batches)
    ]
    batches = [
        dict(
            input_ids=batch,
            attention_mask=torch.ones_like(batch),
        )
        for batch in batches
    ]
    return batches


def get_free_mem_gb() -> int:
    with open("/proc/meminfo") as file:
        for line in file:
            if "MemFree" in line:
                return int(line.split()[1]) >> 20
