import dataclasses
import enum
import getpass
import os
import time
from typing import Dict, Optional

from areal.api.alloc_mode import AllocationMode, AllocationType
from areal.utils import logging, name_resolve, names

logger = logging.getLogger("Launcher Utils")

LOCAL_CACHE_DIR = "/tmp/areal"
PYTORCH_KERNEL_CACHE_PATH = (
    f"{LOCAL_CACHE_DIR}/.cache/{getpass.getuser()}/torch/kernels/"
)
TRITON_CACHE_PATH = f"{LOCAL_CACHE_DIR}/.cache/{getpass.getuser()}/triton/"
os.makedirs(PYTORCH_KERNEL_CACHE_PATH, exist_ok=True)
os.makedirs(TRITON_CACHE_PATH, exist_ok=True)
BASE_ENVIRONS = {
    "TOKENIZERS_PARALLELISM": "true",
    "PYTORCH_KERNEL_CACHE_PATH": PYTORCH_KERNEL_CACHE_PATH,
    "TRITON_CACHE_DIR": TRITON_CACHE_PATH,
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "PYTHONPATH": os.getenv("PYTHONPATH", ""),
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
    "NCCL_DEBUG": "WARN",
    "NCCL_DEBUG_SUBSYS": "INIT,TUNING,GRAPH",
}
ROLLOUT_SERVER_WAIT_TIMEOUT_SECONDS = 360


def get_env_vars(
    cluster_name: str, additional_env_vars: Optional[str] = None
) -> Dict[str, str]:
    """Returns the environment variables for the cluster."""
    _additional_env_vars = (
        dict(item.split("=") for item in additional_env_vars.split(","))
        if additional_env_vars
        else dict()
    )
    if cluster_name == "na132":
        return {**BASE_ENVIRONS, **NA132_ENVIRONS, **_additional_env_vars}
    else:
        return {**BASE_ENVIRONS, **_additional_env_vars}


class JobState(enum.Enum):
    NOT_FOUND = 0
    PENDING = 1
    RUNNING = 2
    COMPLETED = 3
    FAILED = 4
    CANCELLED = 5

    def active(self):
        return self == self.PENDING or self == self.RUNNING


class JobException(Exception):

    def __init__(self, run_name, worker_type, host, reason: JobState):
        super().__init__(f"Job {run_name}:{worker_type} {reason} at node {host}")
        self.run_name = run_name
        self.worker_type = worker_type
        self.host = host
        self.reason = reason


@dataclasses.dataclass
class JobInfo:
    name: str
    state: JobState
    host: Optional[str] = (
        None  # The host on which the job is/was running. None if the job had not run.
    )
    submit_time: Optional[str] = None
    start_time: Optional[str] = None
    slurm_id: Optional[int] = None  # Slurm only. The Slurm id of the job.


def wait_llm_server_addrs(
    experiment_name: str,
    trial_name: str,
    n_rollout_servers: int,
):
    # Get rollout nodes, find the hosts
    name = names.gen_servers(experiment_name, trial_name)
    start = time.perf_counter()
    while True:
        rollout_addrs = name_resolve.get_subtree(name)
        if len(rollout_addrs) >= n_rollout_servers:
            logger.info(
                f"Found {len(rollout_addrs)} rollout servers: {', '.join(rollout_addrs)}"
            )
            break

        time.sleep(1)
        if time.perf_counter() - start > ROLLOUT_SERVER_WAIT_TIMEOUT_SECONDS:
            raise TimeoutError(
                f"Timeout waiting for rollout servers to be ready. "
                f"Expected {n_rollout_servers} servers, found {len(rollout_addrs)}."
            )
    return rollout_addrs


def validate_config_for_distributed_launcher(config):
    n_nodes = config.cluster.n_nodes
    n_gpus_per_node = config.cluster.n_gpus_per_node
    allocation_mode = config.allocation_mode
    allocation_mode = AllocationMode.from_str(allocation_mode)
    if allocation_mode.type_ == AllocationType.DECOUPLED_TRAIN:
        assert (
            allocation_mode.gen.world_size + allocation_mode.train.world_size
            == n_nodes * n_gpus_per_node
        ), (
            f"#GPUs required for allocation mode {allocation_mode.gen.world_size + allocation_mode.train.world_size} "
            f"is not equal to #GPUs in the config {n_nodes * n_gpus_per_node}."
        )
    if allocation_mode.gen_backend == "sglang":
        # Launcher should launch SGLang servers according to allocation mode.
        assert (
            allocation_mode.gen.pp_size == 1
        ), "Pipeline generation in SGLang is not supported for now."
    elif allocation_mode.gen_backend == "vllm":
        # Launcher should launch vLLM servers according to allocation mode.
        assert (
            allocation_mode.gen.pp_size == 1
        ), "Pipeline generation in vLLM is not supported for now."
        assert (
            allocation_mode.gen.tp_size <= config.cluster.n_gpus_per_node
        ), "Currently only support vLLM TP size less <= #GPUs per node."
