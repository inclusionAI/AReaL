import dataclasses
import enum
import getpass
import os
import pathlib
import shutil
import subprocess
import sys
import time
from pathlib import Path

from areal.api.alloc_mode import AllocationMode, AllocationType
from areal.api.cli_args import (
    SGLangConfig,
)
from areal.utils import logging, name_resolve, names, pkg_version

logger = logging.getLogger("LauncherUtils")

LOCAL_CACHE_DIR = os.getenv("AREAL_CACHE_DIR", f"/tmp/areal-{getpass.getuser()}")
PYTORCH_KERNEL_CACHE_PATH = (
    f"{LOCAL_CACHE_DIR}/.cache/{getpass.getuser()}/torch/kernels/"
)
VLLM_CACHE_ROOT = f"{LOCAL_CACHE_DIR}/.cache/{getpass.getuser()}/vllm/"
TRITON_CACHE_PATH = f"{LOCAL_CACHE_DIR}/.cache/{getpass.getuser()}/triton/"
PYTHONPATH = os.pathsep.join(
    filter(
        None,
        [
            os.getenv("PYTHONPATH", None),
            str(pathlib.Path(__file__).resolve().parent.parent.parent),
        ],
    )
)
os.makedirs(PYTORCH_KERNEL_CACHE_PATH, exist_ok=True)
os.makedirs(VLLM_CACHE_ROOT, exist_ok=True)
os.makedirs(TRITON_CACHE_PATH, exist_ok=True)
BASE_ENVIRONS = {
    "TOKENIZERS_PARALLELISM": "true",
    "PYTORCH_KERNEL_CACHE_PATH": PYTORCH_KERNEL_CACHE_PATH,
    "TRITON_CACHE_DIR": TRITON_CACHE_PATH,
    "VLLM_CACHE_ROOT": VLLM_CACHE_ROOT,
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "PYTHONPATH": PYTHONPATH,
}


def get_env_vars(
    cluster_name: str, additional_env_vars: str | None = None
) -> dict[str, str]:
    """Returns the environment variables for the cluster."""
    _additional_env_vars = (
        dict(item.split("=") for item in additional_env_vars.split(","))
        if additional_env_vars
        else dict()
    )

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
    host: str | None = (
        None  # The host on which the job is/was running. None if the job had not run.
    )
    submit_time: str | None = None
    start_time: str | None = None
    slurm_id: int | None = None  # Slurm only. The Slurm id of the job.


def wait_llm_server_addrs(
    experiment_name: str,
    trial_name: str,
    n_rollout_servers: int = 1,
    timeout: int | None = 1200,
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
        if timeout is not None and time.perf_counter() - start > timeout:
            raise TimeoutError(
                f"Timeout waiting for rollout servers to be ready. "
                f"Expected {n_rollout_servers} servers, found {len(rollout_addrs)}."
            )
    return rollout_addrs


def validate_config_for_launcher(config):
    if not hasattr(config, "actor"):
        raise RuntimeError("`actor` field is required by all yaml configs.")
    allocation_mode = config.allocation_mode
    allocation_mode = AllocationMode.from_str(allocation_mode)
    if allocation_mode.gen_backend in ["sglang", "vllm"]:
        if not hasattr(config, "rollout"):
            raise RuntimeError(
                "`rollout` field is required when `allocation_mode` "
                "includes inference backends like sglang/vllm. "
                "You should use a yaml config for RL rather than SFT, or "
                "remove the inference part from `allocation_mode`."
            )


def validate_config_for_distributed_launcher(config):
    validate_config_for_launcher(config)
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
        if allocation_mode.gen.pp_size > 1:
            raise NotImplementedError(
                "Pipeline generation in SGLang is not supported for now."
            )
    elif allocation_mode.gen_backend == "vllm":
        # Launcher should launch vLLM servers according to allocation mode.
        if allocation_mode.gen.pp_size > 1:
            raise NotImplementedError(
                "Pipeline generation in vLLM is not supported for now."
            )
        if allocation_mode.gen.tp_size > config.cluster.n_gpus_per_node:
            raise ValueError("Currently only support vLLM TP size <= #GPUs per node.")


def _get_sglang_install_info():
    """Get SGLang installation path and patch binary location.

    Returns:
        tuple: (target_path, patch_binary) where target_path is the SGLang
               installation directory and patch_binary is the path to patch command.

    Raises:
        RuntimeError: If SGLang installation path cannot be determined or patch
                     command is not found.
    """
    target_path = None
    sglang_meta = subprocess.check_output(
        [sys.executable, "-m", "pip", "show", "sglang"]
    ).decode("utf-8")
    # Prioritize editable install location, since pip show lists both locations
    # if installed in editable mode.
    for line in sglang_meta.split("\n"):
        line = line.strip()
        if line.startswith("Editable project location: "):
            target_path = str(Path(line.split(": ")[1]) / "sglang")
            break
    else:
        for line in sglang_meta.split("\n"):
            line = line.strip()
            if line.startswith("Location: "):
                target_path = str(Path(line.split(": ")[1]) / "sglang")
                break

    if not target_path or not os.path.exists(target_path):
        raise RuntimeError("Could not determine the installation path of SGLang.")

    patch_binary = shutil.which("patch")
    if not patch_binary:
        raise RuntimeError(
            "Could not locate the `patch` command; SGLang patch application failed."
        )

    return target_path, patch_binary


def _apply_fast_load_patch(config: SGLangConfig):
    """Apply fast load or multithread load patch to SGLang installation.

    Args:
        config: SGLang configuration containing enable_fast_load and
                enable_multithread_load flags.

    Raises:
        RuntimeError: If patch application fails.
    """

    p = Path(os.path.dirname(__file__))
    patch_path = str(
        p.parent.parent
        / "patch"
        / "sglang"
        / "fast_load"
        / f"v{pkg_version.get_version('sglang')}.patch"
    )

    target_path, patch_binary = _get_sglang_install_info()

    result = subprocess.run(
        [patch_binary, "-p1", "-N", "-i", patch_path],
        cwd=target_path,
        capture_output=True,
        text=True,
    )

    output = (result.stdout or "") + (result.stderr or "")
    if result.returncode == 0:
        logger.info(f"Applied SGLang patch {patch_path} to {target_path}")
    elif (
        "Reversed (or previously applied) patch detected" in output
        or "Skipping patch." in output
    ):
        logger.warning(
            f"SGLang patch {patch_path} appears to be already applied for {target_path}."
        )
    else:
        logger.error(
            "Failed to apply SGLang patch %s to %s. Output:\n%s",
            patch_path,
            target_path,
            output.strip(),
        )
        raise RuntimeError(
            f"SGLang patch {patch_path} failed with exit code {result.returncode}."
        )


def _apply_routing_replay_patch():
    """Apply routing replay patch to SGLang installation.

    Raises:
        RuntimeError: If patch file doesn't exist or patch application fails.
    """
    # if sglang version >= 0.5.7, routing_replay is built-in
    if pkg_version.get_version("sglang") >= "0.5.7":
        return

    p = Path(os.path.dirname(__file__))
    routing_replay_patch = (
        p.parent.parent
        / "patch"
        / "sglang"
        / "routing_replay"
        / f"v{pkg_version.get_version('sglang')}.patch"
    )

    if not routing_replay_patch.exists():
        raise RuntimeError(
            f"Routing_replay patch file {routing_replay_patch} does not exist."
        )

    target_path, patch_binary = _get_sglang_install_info()

    result = subprocess.run(
        [patch_binary, "-p2", "-N", "-i", str(routing_replay_patch)],
        cwd=target_path,
        capture_output=True,
        text=True,
    )

    output = (result.stdout or "") + (result.stderr or "")
    if result.returncode == 0:
        logger.info(f"Applied routing_replay patch to {target_path}")
    elif (
        "Reversed (or previously applied) patch detected" in output
        or "Skipping patch." in output
    ):
        logger.warning(
            f"Routing_replay patch {routing_replay_patch} appears to be already applied for {target_path}."
        )
    else:
        logger.error(
            "Failed to apply routing_replay patch %s to %s. Output:\n%s",
            str(routing_replay_patch),
            target_path,
            output.strip(),
        )
        raise RuntimeError(
            f"Routing_replay patch failed with exit code {result.returncode}"
        )


def maybe_apply_sglang_patch(config: SGLangConfig, enable_routing_replay: bool = False):
    """Apply SGLang patches based on configuration.

    Args:
        config: SGLang configuration.
        enable_routing_replay: Whether to apply routing replay patch.
    """
    if config.enable_fast_load or config.enable_multithread_load:
        _apply_fast_load_patch(config)
    if enable_routing_replay:
        _apply_routing_replay_patch()
