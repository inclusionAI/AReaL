# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

# log format constants
import contextlib
import datetime
import getpass
import os
import pathlib
from pathlib import Path
from typing import *

import numpy as np
import torch

import realhf.base.logging as logging

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from realhf.api.cli_args import BaseExperimentConfig
    from realhf.api.core.config import ModelName
    from realhf.api.core.system_api import ModelShardID
    from realhf.base.topology import ParallelGrid, ProcessTopology


class GlobalMemoryBuffer:
    """Global buffer to avoid dynamic memory allocations.

    Caller should ensure that buffers of the same name are not used
    concurrently.
    """

    def __init__(self):
        self.buffer = {}

    def get_tensor(self, tensor_shape, dtype, name, force_zero: bool = False):
        device = current_device()
        required_len = int(np.prod(tensor_shape))
        if self.buffer.get((name, dtype), None) is None:
            self.buffer[(name, dtype)] = torch.empty(
                required_len,
                dtype=dtype,
                device=device,
                requires_grad=False,
            )
        elif self.buffer[(name, dtype)].numel() < required_len:
            self.buffer[(name, dtype)] = torch.nn.functional.pad(
                self.buffer[(name, dtype)],
                (0, required_len - self.buffer[(name, dtype)].numel()),
                value=0,
            )
        res = self.buffer[(name, dtype)][0:required_len].view(*tensor_shape)
        if force_zero:
            res.zero_()
        return res


# For large models, generation may consume more than 3600s.
# We set a large value to avoid NCCL timeout issues during generaiton.
NCCL_DEFAULT_TIMEOUT = datetime.timedelta(seconds=7200)

# We may want to use CPU for testing even when CUDA is available.
TORCH_FORCE_CPU = False

# constants in experiment instance scope
LOCAL_CACHE_DIR = "/tmp/realhf"
QUICKSTART_EXPR_CACHE_PATH = str(Path(__file__).parent.parent.parent / ".cache")
os.makedirs(QUICKSTART_EXPR_CACHE_PATH, exist_ok=True)
PORT_LOCKFILE_ROOT = os.getenv("AREAL_PORT_LOCKFILE_ROOT", "/tmp/areal/ports/")
os.makedirs(PORT_LOCKFILE_ROOT, exist_ok=True)

PYTORCH_KERNEL_CACHE_PATH = (
    f"{LOCAL_CACHE_DIR}/.cache/{getpass.getuser()}/torch/kernels"
)
TRITON_CACHE_PATH = f"{LOCAL_CACHE_DIR}/.cache/{getpass.getuser()}/triton"
os.makedirs(PYTORCH_KERNEL_CACHE_PATH, exist_ok=True)
os.makedirs(TRITON_CACHE_PATH, exist_ok=True)


def get_cache_path(args: "BaseExperimentConfig") -> str:
    path = f"{args.cluster.fileroot}/.cache/{getpass.getuser()}/{args.experiment_name}/{args.trial_name}"
    os.makedirs(path, exist_ok=True)
    return path


def get_log_root(args: "BaseExperimentConfig") -> str:
    log_root = f"{args.cluster.fileroot}/logs/{getpass.getuser()}"
    os.makedirs(log_root, exist_ok=True)
    return log_root


def get_log_path(args: "BaseExperimentConfig") -> str:
    log_path = f"{args.cluster.fileroot}/logs/{getpass.getuser()}/{args.experiment_name}/{args.trial_name}"
    os.makedirs(log_path, exist_ok=True)
    return log_path


def get_save_root(args: "BaseExperimentConfig") -> str:
    path = f"{args.cluster.fileroot}/checkpoints/{getpass.getuser()}"
    os.makedirs(path, exist_ok=True)
    return path


def get_save_path(args: "BaseExperimentConfig") -> str:
    path = f"{args.cluster.fileroot}/checkpoints/{getpass.getuser()}/{args.experiment_name}/{args.trial_name}"
    os.makedirs(path, exist_ok=True)
    return path


def get_param_realloc_path(args: "BaseExperimentConfig"):
    path = f"{args.cluster.fileroot}/.cache/{getpass.getuser()}/param_realloc"
    os.makedirs(path, exist_ok=True)
    return path


BASE_ENVIRONS = {
    # "PYTHONPATH": "/realhf",
    "REAL_IS_REMOTE": "1",
    # "NCCL_P2P_DISABLE": "1",
    # "NCCL_IB_DISABLE": "1",
    "TOKENIZERS_PARALLELISM": "true",
    "PYTORCH_KERNEL_CACHE_PATH": PYTORCH_KERNEL_CACHE_PATH,
    "TRITON_CACHE_DIR": TRITON_CACHE_PATH,
    # "TORCH_DISTRIBUTED_DEBUG": "DETAIL",
    # "NCCL_SOCKET_IFNAME": "ibp71s0",
    # "GLOO_SOCKET_IFNAME": "ibp71s0",
    # "TORCH_USE_CUDA_DSA": "1",
    # "NCCL_IGNORE_DISABLED_P2P": "1",
    # "CUDA_LAUNCH_BLOCKING": "1",  # NOTE: CUDAGraph Capturing will not work if CUDA_LAUNCH_BLOCKING is set to 1.
    # "NCCL_COMM_BLOCKING": "1",  # NOTE: CUDAGraph Capturing will not work if NCCL_COMM_BLOCKING is set to 1.
    # "NCCL_BLOCKING_WAIT": "1",  # NOTE: CUDAGraph Capturing will not work if NCCL_BLOCKING_WAIT is set to 1.
    # "TORCH_SHOW_CPP_STACKTRACES": "1",
    # "RAY_DEDUP_LOGS": "0",  # disable ray log deduplication
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "OMP_NUM_THREADS": str(min(os.cpu_count(), 32)),
    # torch.distributed.all_reduce does not free the input tensor until
    # the synchronization point. This causes the memory usage to grow
    # as the number of all_reduce calls increases. This env var disables
    # this behavior.
    # Related issue:
    # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573
    "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
    # Whether to enable time mark to plot timelines.
    "REAL_DUMP_TRACE": os.getenv("REAL_DUMP_TRACE", "0"),
    "REAL_DUMP_MEMORY": os.getenv("REAL_DUMP_MEMORY", "0"),
    "REAL_GPU_MEMORY_KILL_THRESHOLD": os.getenv(
        "REAL_GPU_MEMORY_KILL_THRESHOLD", "1.0"
    ),
    "LC_ALL": "C",
    "LANG": "C",
    "NCCL_DEBUG": "WARN",
}
PPU_ENVIRONS = {
    "NCCL_DEBUG": "INFO",
    "NCCL_IB_DISABLE": "1",
    "NCCL_DEBUG_SUBSYS": "INIT",
    "NCCL_SET_THREAD_NAME": "1",
    "NCCL_IB_HCA": "",
    "NCCL_SOCKET_IFNAME": "bond0",
    "PCCL_STATE_MONITOR_DISABLE": "1",
}
NA132_ENVIRONS = {
    "NCCL_SOCKET_IFNAME": "bond0",
    "NCCL_NET_PLUGIN": "",
    "NCCL_IB_GID_INDEX": "3",
    "NCCL_IB_TIMEOUT": "2",
    "NCCL_IB_RETRY_CNT": "7",
    "NCCL_IB_SL": "5",
    "NCCL_IB_TC": "136",
    "NCCL_IB_HCA": "mlx5_bond",
    "NCCL_IB_QPS_PER_CONNECTION": "8",
    "NCCL_SET_THREAD_NAME": "1",
    "NCCL_DEBUG_SUBSYS": "INIT,TUNING,GRAPH",
}


# _model_name will be changed in the model_scope context manager
_model_name: "ModelName" = None

# constants in worker/process scope
_experiment_name = None
_trial_name = None

_grids: Dict["ModelName", "ParallelGrid"] = {}
_pgroups: Dict["ModelName", Any] = (
    {}
)  # torch.distributed.ProcessGroup, not type hint here to avoid importing torch
_cpu_pgroups: Dict["ModelName", Any] = (
    {}
)  # torch.distributed.ProcessGroup, not type hint here to avoid importing torch
_pgroup_ranks: Dict["ModelName", List[int]] = {}
_self_group = None
_rank_mapping: Dict["ModelName", Dict["ModelShardID", int]] = {}
_global_memory_buffer: GlobalMemoryBuffer = GlobalMemoryBuffer()


# TODO: As in Megatron, we can set NCCL group options. Is it necessary?


def reset_run():
    global _model_name, _grids, _pgroups, _pgroup_ranks, _self_group, _rank_mapping, _global_memory_buffer
    _model_name = None
    _grids = {}
    _pgroups = {}
    _pgroup_ranks = {}
    _self_group = None
    _rank_mapping = {}
    _global_memory_buffer = GlobalMemoryBuffer()


@contextlib.contextmanager
def model_scope(model_name: "ModelName"):
    global _model_name
    assert _model_name is None
    _model_name = model_name
    yield
    assert _model_name == model_name
    _model_name = None


@contextlib.contextmanager
def model_scope_disabled():
    global _model_name
    assert _model_name is not None
    t, _model_name = _model_name, None
    yield
    _model_name = t


################# setter functions #################
def set_force_cpu(val: bool):
    global TORCH_FORCE_CPU
    TORCH_FORCE_CPU = val


def set_experiment_trial_names(expr_name: str, trial_name: str):
    global _experiment_name, _trial_name
    if _experiment_name is not None and _experiment_name != expr_name:
        raise RuntimeError("Experiment name has been set.")
    if _trial_name is not None and _trial_name != trial_name:
        raise RuntimeError("Trial name has been set.")
    _experiment_name = expr_name
    _trial_name = trial_name


def set_grid(model_name: "ModelName", grid: "ParallelGrid"):
    global _grids
    if model_name in _grids:
        raise RuntimeError(f"Grid for model {model_name} is already set.")
    _grids[model_name] = grid


def set_parallelism_group(model_name: "ModelName", pgroup, ranks):
    global _pgroups
    if model_name in _pgroups:
        raise RuntimeError(f"Parallelism group for model {model_name} is already set.")
    _pgroups[model_name] = pgroup
    _pgroup_ranks[model_name] = ranks


def set_cpu_parallelism_group(model_name: "ModelName", pgroup):
    global _cpu_pgroups
    if model_name in _cpu_pgroups:
        raise RuntimeError(f"Parallelism group for model {model_name} is already set.")
    _cpu_pgroups[model_name] = pgroup


def set_self_group(pgroup):
    global _self_group
    if _self_group is not None:
        raise RuntimeError("Self group is already set.")
    _self_group = pgroup


def set_rank_mapping(
    model_name: "ModelName",
    topo: "ProcessTopology",
    msid2mwid: Optional[Dict["ModelShardID", int]] = None,
):
    global _rank_mapping
    if model_name in _rank_mapping:
        raise RuntimeError(f"Rank mapping for model {model_name} is already set.")
    if msid2mwid is None:
        _rank_mapping[model_name] = {i: i for i in range(topo.world_size())}
    else:
        msid2mwid = {k: v for k, v in msid2mwid.items() if k.model_name == model_name}
        _rank_mapping[model_name] = {
            topo.get_rank(data=s.dp_rank, tensor=s.tp_rank, pipe=s.pp_rank): mw_id
            for s, mw_id in msid2mwid.items()
        }


################# attribute functions #################
def current_device() -> torch.device:
    global TORCH_FORCE_CPU
    if TORCH_FORCE_CPU or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.cuda.current_device()


def use_cuda() -> bool:
    return not TORCH_FORCE_CPU and torch.cuda.is_available()


def use_te_impl() -> bool:
    try:
        import transformer_engine.pytorch as te

        TE_ENABLED = True
    except ImportError:
        TE_ENABLED = False
    return TE_ENABLED and os.getenv("REAL_LLM_USE_TE") == "1"


def sequence_parallel() -> bool:
    return grid().topology().sequence_parallel


def gradient_accumulation_fusion() -> bool:
    _grad_accum_fusion_available = True
    try:
        import fused_weight_gradient_mlp_cuda
    except ImportError:
        _grad_accum_fusion_available = False
    return _grad_accum_fusion_available and getattr(
        grid().topology(), "gradient_accumulation_fusion", False
    )


def max_prompt_len() -> int:
    return grid().topology().max_prompt_len


def gradient_checkpointing() -> bool:
    return getattr(grid().topology(), "gradient_checkpointing", False)


def has_model_name(name: str) -> bool:
    return name in _grids and _grids[name].global_rank != -1


def self_group():
    global _self_group
    assert _self_group is not None
    return _self_group


def model_name():
    if _model_name == None:
        raise RuntimeError(
            "Global constant `model_name` should be accessed in the `model_scope` context."
        )
    return _model_name


def experiment_name():
    if _experiment_name == None:
        raise RuntimeError("Global constant `experiment_name` is accessed before set.")
    return _experiment_name


def trial_name():
    if _trial_name == None:
        raise RuntimeError("Global constant `trial_name` is accessed before set.")
    return _trial_name


def grid() -> "ParallelGrid":
    if _model_name is None:
        raise RuntimeError("Global constant `model_name` is accessed before set.")
    if _grids.get(_model_name, None) is None:
        raise RuntimeError(f"Grid for model {_model_name} is not set.")
    return _grids[_model_name]


def grid_of_model(model_name: str) -> "ParallelGrid":
    if _grids.get(model_name, None) is None:
        raise RuntimeError(f"Grid for model {model_name} is not set.")
    return _grids[model_name]


def parallelism_group():
    """Returns the 3D parallelism group of a specific model."""
    if _model_name is None:
        raise RuntimeError("Global constant `model_name` is accessed before set.")
    if _pgroups.get(_model_name, None) is None:
        raise RuntimeError(f"Parallelism group for model {_model_name} is not set.")
    return _pgroups[_model_name]


def cpu_parallelism_group():
    """Returns the GLOO 3D parallelism group of a specific model."""
    if _model_name is None:
        raise RuntimeError("Global constant `model_name` is accessed before set.")
    if _cpu_pgroups.get(_model_name, None) is None:
        raise RuntimeError(f"Parallelism group for model {_model_name} is not set.")
    return _cpu_pgroups[_model_name]


def parallelism_group_ranks():
    if _model_name is None:
        raise RuntimeError("Global constant `model_name` is accessed before set.")
    if _pgroup_ranks.get(_model_name, None) is None:
        raise RuntimeError(
            f"Parallelism group ranks for model {_model_name} is not set."
        )
    return _pgroup_ranks[_model_name]


def parallelism_group_size() -> int:
    """The 3D parallelism group size of a specific model, normally dp_size *
    pp_size * tp_size."""
    import torch.distributed as dist

    return dist.get_world_size(group=parallelism_group())


def parallelism_rank() -> int:
    """Return the rank of a specific model in its 3D parallelism group."""
    import torch.distributed as dist

    return dist.get_rank(group=parallelism_group())


def to_global_pg_rank(local_rank: int) -> int:
    global _rank_mapping
    if _rank_mapping is None or model_name() not in _rank_mapping:
        raise RuntimeError("Rank mapping is not set.")
    return _rank_mapping[model_name()][local_rank]


def rank_mapping_of_model(model_name: str) -> Dict["ModelShardID", int]:
    global _rank_mapping
    if _rank_mapping is None or _rank_mapping.get(model_name, None) is None:
        raise RuntimeError(f"Rank mapping for model {model_name} is not set.")
    return _rank_mapping[model_name]


def pipe_parallel_rank() -> int:
    return grid().get_pipe_parallel_rank()


def pipe_parallel_world_size() -> int:
    return grid().get_pipe_parallel_world_size()


def pipe_parallel_group():
    return grid().get_pipe_parallel_group()


def pipe_parallel_cpu_group():
    return grid().pp_proc_group_gloo


def is_last_pipe_stage():
    return pipe_parallel_rank() == pipe_parallel_world_size() - 1


def is_first_pipe_stage():
    return pipe_parallel_rank() == 0


def next_pipe_stage():
    return (pipe_parallel_rank() + 1) % pipe_parallel_world_size()


def prev_pipe_stage():
    return (
        pipe_parallel_world_size() + pipe_parallel_rank() - 1
    ) % pipe_parallel_world_size()


def is_dp_head():
    return is_last_pipe_stage() and tensor_parallel_rank() == 0


def tensor_parallel_rank() -> int:
    """Return the rank inside the tensor parallelism group."""
    return grid().get_tensor_model_parallel_rank()


def tensor_parallel_world_size() -> int:
    """Return the world size of the tensor parallelism group."""
    return grid().get_tensor_model_parallel_world_size()


def tensor_parallel_group():
    """Return the NCCL tensor parallelism process group."""
    return grid().get_tensor_model_parallel_group()


def tensor_parallel_cpu_group():
    """Return the GLOO tensor parallelism process group."""
    return grid().get_tensor_model_parallel_cpu_group()


def tp_and_pp_group():
    """Used as the world group of vLLM."""
    return grid().get_model_parallel_group()


def tp_and_pp_cpu_group():
    return grid().ds_model_proc_group_gloo


def tp_and_pp_rank():
    """Used as the rank in the world group of vLLM."""
    return grid().get_model_parallel_rank()


def tp_and_pp_world_size():
    """Used as the world size of vLLM."""
    return grid().get_model_parallel_world_size()


def data_parallel_rank() -> int:
    return grid().get_data_parallel_rank()


def data_parallel_world_size() -> int:
    return grid().get_data_parallel_world_size()


def data_parallel_group():
    return grid().get_data_parallel_group()


def get_global_memory_buffer():
    global _global_memory_buffer
    assert _global_memory_buffer is not None, "global memory buffer is not set"
    return _global_memory_buffer


def clear_global_memory_buffer():
    global _global_memory_buffer
    _global_memory_buffer = GlobalMemoryBuffer()


def get_repo_path() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent.parent.parent


def get_env_vars(exp_cfg: "BaseExperimentConfig", **kwargs):
    kwargs.update(
        REAL_DUMP_TRACE=os.environ.get("REAL_DUMP_TRACE", "0"),
        REAL_RECORD_PERFORMANCE=os.environ.get("REAL_RECORD_PERFORMANCE", "0"),
        FUNCTIONCALL_SERVICE_DOMAIN=os.getenv("FUNCTIONCALL_SERVICE_DOMAIN", ""),
        REAL_DUMP_MEMORY=os.environ.get("REAL_DUMP_MEMORY", "0"),
        REAL_OSS_TESTCASE_PATH=os.getenv("REAL_OSS_TESTCASE_PATH", ""),
    )
    envvars = {
        **kwargs,
        "REAL_PACKAGE_PATH": str(get_repo_path()),
        **BASE_ENVIRONS,
    }
    if exp_cfg.cluster.cluster_name == "wa180":
        envvars.update(**PPU_ENVIRONS)
    if exp_cfg.cluster.cluster_name == "na132":
        envvars.update(**NA132_ENVIRONS)
    return envvars
