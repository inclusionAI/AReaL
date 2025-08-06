import getpass
import importlib.util
import os
import pathlib
import sys
import time
from typing import Dict, List, Optional

import ray
import ray.exceptions
from ray.runtime_env import RuntimeEnv
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from areal.api.cli_args import (
    ClusterSpecConfig,
    LauncherConfig,
    SGLangConfig,
    parse_cli_args,
    to_structured_cfg,
)
from areal.api.io_struct import AllocationMode, AllocationType
from areal.utils.launcher import (
    get_env_vars,
    validate_config_for_distributed_launcher,
    wait_sglang_server_addrs,
)
from areal.utils.ray import get_placement_group_master_ip_and_port
from realhf.base import logging, name_resolve, names
from realhf.scheduler.client import JobException, JobState

logger = logging.getLogger("RayLauncher")

RAY_WAIT_CHECK_TIME_INTERVAL = 5  # seconds
DEFAULT_MAIN_FUNC_NAME = "main"


def run_func(file_path, function_name, *args, **kwargs):
    # Convert the file path to a module name
    module_name = file_path.replace("/", "_").replace(".", "_")

    # Load the module from file path
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    # Get the function and execute it
    try:
        function = getattr(module, function_name)
    except AttributeError as e:
        raise ValueError(
            f"Function '{function_name}' not found in module '{module_name}'. "
            f"Please ensure the name of the main function in your entry point "
            f"is '{function_name}'."
        ) from e
    return function(*args, **kwargs)


class RayLauncher:
    def __init__(self, experiment_name: str, trial_name: str, fileroot: str):
        self.experiment_name = experiment_name
        self.trial_name = trial_name
        self.fileroot = fileroot

        # job_name to ray future
        self.jobs = {}

    @property
    def run_name(self):
        return f"{self.experiment_name}_{self.trial_name}"

    def log_path_of(self, job_name: str) -> str:
        log_path = f"{self.fileroot}/logs/{getpass.getuser()}/{self.experiment_name}/{self.trial_name}"
        os.makedirs(log_path, exist_ok=True)
        return os.path.join(log_path, f"{job_name}.log")

    def submit(
        self,
        job_name: str,
        file_path: str,
        func_name: str,
        args: List[str],  # arguments to pass to the function
        gpus: int,
        cpus: int,
        mem: int,  # MB
        env_vars: Optional[Dict] = None,
        placement_group: Optional[PlacementGroup] = None,
        bundle_index: int = -1,
        kwargs: Optional[
            Dict[str, str]
        ] = None,  # keyword arguments to pass to the function
    ):
        if kwargs is None:
            kwargs = {}
        runtime_env = RuntimeEnv(
            env_vars=env_vars or dict(),
        )
        scheduling_strategy = (
            PlacementGroupSchedulingStrategy(
                placement_group=placement_group,
                placement_group_bundle_index=bundle_index,
                placement_group_capture_child_tasks=True,
            )
            if placement_group is not None
            else "DEFAULT"
        )
        future = ray.remote(
            num_cpus=cpus,
            num_gpus=gpus,
            memory=mem * 1024 * 1024,  # Convert MB to bytes
            runtime_env=runtime_env,
            scheduling_strategy=scheduling_strategy,
        )(run_func).remote(file_path, func_name, *args, **kwargs)
        self.jobs[job_name] = future
        return future

    def submit_array(
        self,
        job_name: str,
        file_path: str,
        func_name: str,
        count: int,
        nodes: int,
        list_args: List[List],
        gpus_per_task: int,
        cpus_per_task: int,
        mem_per_task: int,  # MB
        list_kwargs: List[Dict] | None = None,
        env_vars: Optional[Dict] = None,
        amend_torch_dist_env: bool = False,
    ):
        """Submit an array of jobs to Ray with ray placement groups.

        Note: Here we use `ray.remote` instead of `ray job submit` since `ray job submit`
        does not support placement groups, and can not specify which node to run the job on.
        Therefore we could not know the IP address of jobs for torch distributed initialization.
        """

        if count % nodes != 0:
            raise ValueError(
                f"Count {count} is not divisible by nodes {nodes}. "
                "Please ensure that count is a multiple of nodes."
            )
        assert (
            len(list_args) == count
        ), f"Length of list_args {len(list_args)} does not match count {count}."
        if list_kwargs is not None:
            assert (
                len(list_kwargs) == count
            ), f"Length of list_kwargs {len(list_kwargs)} does not match count {count}."

        tasks_per_node = count // nodes
        gpus_per_node = gpus_per_task * tasks_per_node
        cpus_per_node = cpus_per_task * tasks_per_node
        mem_per_node = mem_per_task * tasks_per_node

        placement_group = ray.util.placement_group(
            bundles=[
                {
                    "CPU": cpus_per_node,
                    "GPU": gpus_per_node,
                    "memory": mem_per_node * 1024 * 1024,  # Convert MB to bytes
                }
            ]
            * nodes,
            strategy="STRICT_SPREAD",
        )
        try:
            ray.get(placement_group.ready(), timeout=30)
        except ray.exceptions.GetTimeoutError as e:
            logger.error(
                "Ray placement group timeout, please check if the resource requirement "
                "for your experiment exceeds the available resources in the cluster. \n"
                f"ray.nodes(): {ray.nodes()} \n"
                f"Placement Group bundles: "
                f"cpus_per_node={cpus_per_node}, gpus_per_node={gpus_per_node}, "
                f"mem_per_node={mem_per_node}MB, nodes={nodes}"
            )
            raise e

        if amend_torch_dist_env:
            host_ip, port = get_placement_group_master_ip_and_port(placement_group)
            logger.info(
                f"Amend torch distributed env vars: "
                f"MASTER_ADDR={host_ip}, PORT={port}"
            )

        futures = []
        for i in range(count):
            args = list_args[i]
            kwargs = list_kwargs[i] if list_kwargs is not None else {}

            # manage environment variables
            env_vars = env_vars or {}
            if "CUDA_VISIBLE_DEVICES" in env_vars:
                logger.warning(
                    "Setting CUDA_VISIBLE_DEVICES before running ray jobs may result in unexpected behavior."
                )

            node_id = i // tasks_per_node
            _env_vars = {
                **env_vars,
            }

            if amend_torch_dist_env:
                assert gpus_per_task == 1
                # NOTE: Here we only provide environment variables for torch distributed
                # initialization, and LOCAL_RANK for torch.device.
                # Other environment variables automatically set by torchrun are not set, and
                # they should be never accessed in trainer code.
                _env_vars.update(
                    {
                        "RANK": str(i),
                        "WORLD_SIZE": str(count),
                        # Ray will automatically isolate CUDA_VISIBLE_DEVICES for each GPU
                        "LOCAL_RANK": "0",
                        "MASTER_ADDR": str(host_ip),
                        "MASTER_PORT": str(port),
                    }
                )
            future = self.submit(
                job_name=f"{job_name}:{i}",
                file_path=file_path,
                func_name=func_name,
                args=args,
                gpus=gpus_per_task,
                cpus=cpus_per_task,
                mem=mem_per_task,
                env_vars=_env_vars,
                placement_group=placement_group,
                bundle_index=node_id,
                kwargs=kwargs,
            )
            futures.append(future)

        return futures

    def stop(self, job_name: str, force: bool = False):
        """Stop a job by name."""
        if job_name in self.jobs:
            future = self.jobs[job_name]
            try:
                ray.cancel(future, force=force)
            except Exception as e:
                logger.error(f"Failed to cancel job {job_name}: {e}")
                return
            self.jobs.pop(job_name, None)
            logger.info(f"Job {job_name} stopped.")
        else:
            logger.warning(f"Job {job_name} not found in running jobs.")

    def stop_all(self, force: bool = False):
        """Stop all jobs."""
        for job_name in list(self.jobs.keys()):
            self.stop(job_name, force=force)
        logger.info("All jobs stopped.")
        self.jobs.clear()

    def wait(
        self, check_status=(JobState.FAILED,), remove_status=(JobState.COMPLETED,)
    ):
        """Check every RAY_WAIT_CHECK_TIME_INTERVAL seconds for the status of all jobs.
        If a ray job returns, its status changes to JobState.COMPLETED.
        If a ray job failed, its status changes to JobState.FAILED.
        If any job is in check_status, stop all jobs at once.
        If any job is in remove status, remove them from job list.
        Return if all jobs are removed from job list, or some job is in check status.
        """
        for status in list(check_status) + list(remove_status):
            assert status in [
                JobState.COMPLETED,
                JobState.FAILED,
            ], "In RayLauncher.wait, we only check completed or failed jobs."
        logger.info(f"Waiting for {len(self.jobs)} jobs.")
        while self.jobs:
            job_status = {}
            for job_name, future in list(self.jobs.items()):
                try:
                    r = ray.get(future, timeout=0.1)
                    logger.info(f"Job {job_name} completed with result: {r}")
                    job_status[job_name] = JobState.COMPLETED
                except ray.exceptions.RayTaskError as e:
                    logger.error(f"Job {job_name} failed with error: {e}.")
                    job_status[job_name] = JobState.FAILED
                except ray.exceptions.GetTimeoutError:
                    continue

            for job_name, status in job_status.items():
                if status in check_status:
                    logger.info(f"Job {job_name} is {status}, stopping all jobs.")
                    self.stop_all(force=True)
                    return
                if status in remove_status:
                    logger.info(f"Job {job_name} is {status}, removed.")
                    self.jobs.pop(job_name)

            time.sleep(RAY_WAIT_CHECK_TIME_INTERVAL)


def ray_main():
    # usage: python -m areal.launcher.ray <entry_point> --config <config_path> [<additional_args>]
    ray.init()
    config, config_file = parse_cli_args(sys.argv[2:])
    config.launcher = to_structured_cfg(config.launcher, LauncherConfig)
    config.cluster = to_structured_cfg(config.cluster, ClusterSpecConfig)
    config.sglang = to_structured_cfg(config.sglang, SGLangConfig)
    validate_config_for_distributed_launcher(config)

    name_resolve.reconfigure(config.cluster.name_resolve)
    name_resolve.clear_subtree(
        names.trial_root(
            experiment_name=config.experiment_name, trial_name=config.trial_name
        )
    )

    n_nodes = config.cluster.n_nodes
    n_gpus_per_node = config.cluster.n_gpus_per_node
    launcher = RayLauncher(
        experiment_name=config.experiment_name,
        trial_name=config.trial_name,
        fileroot=config.cluster.fileroot,
    )
    allocation_mode = config.allocation_mode
    allocation_mode = AllocationMode.from_str(allocation_mode)
    sglang_addrs = []
    n_sglang_nodes = 0
    if allocation_mode.type_ == AllocationType.DECOUPLED_SGLANG:
        # Launcher should launch SGLang servers according to allocation mode.
        sglang_tp_size = allocation_mode.gen_tp_size
        n_sglang_servers = allocation_mode.gen_dp_size
        n_sglang_nodes = allocation_mode.gen_world_size // n_gpus_per_node

        base_seed = config.sglang.random_seed
        sglang_args_list = [
            [sys.argv[2:] + [f"sglang.random_seed={base_seed + i}"]]
            for i in range(n_sglang_servers)
        ]
        sglang_entry_point = str(
            pathlib.Path(__file__).resolve().parent.joinpath("sglang_server.py")
        )
        launcher.submit_array(
            job_name="llm_server",
            file_path=sglang_entry_point,
            func_name=DEFAULT_MAIN_FUNC_NAME,
            count=n_sglang_servers,
            nodes=n_sglang_nodes,
            list_args=sglang_args_list,
            gpus_per_task=sglang_tp_size,
            cpus_per_task=config.launcher.inference_server_cpus_per_gpu
            * sglang_tp_size,
            mem_per_task=config.launcher.inference_server_mem_per_gpu * sglang_tp_size,
            env_vars=get_env_vars(
                config.cluster.cluster_name,
                config.launcher.inference_server_env_vars,
            ),
        )
        # Get SGLang server addresses via name_resolve
        try:
            sglang_addrs = wait_sglang_server_addrs(
                config.experiment_name,
                config.trial_name,
                n_sglang_servers,
            )
        except TimeoutError as e:
            launcher.stop_all(force=True)
            raise e

    trainer_n_nodes = n_nodes - n_sglang_nodes
    trainer_entry_point = sys.argv[1]
    n_trainer_processes = trainer_n_nodes * config.cluster.n_gpus_per_node
    trainer_args_list = [[sys.argv[2:]] for _ in range(n_trainer_processes)]
    if not config.server_only:
        # In ray, we launch trainer in the granularity of processes (1 GPU per process)
        # We amend environment variable similar to torchrun to ensure correct initialization of
        # torch distributed.
        launcher.submit_array(
            job_name="trainer",
            file_path=trainer_entry_point,
            func_name=DEFAULT_MAIN_FUNC_NAME,
            count=trainer_n_nodes * config.cluster.n_gpus_per_node,
            nodes=trainer_n_nodes,
            list_args=trainer_args_list,
            gpus_per_task=1,
            cpus_per_task=config.launcher.trainer_cpus_per_gpu,
            mem_per_task=config.launcher.trainer_mem_per_gpu,
            env_vars=dict(
                **get_env_vars(
                    config.cluster.cluster_name,
                    config.launcher.trainer_env_vars,
                ),
                AREAL_LLM_SERVER_ADDRS=",".join(sglang_addrs),
            ),
            amend_torch_dist_env=True,
        )

    try:
        launcher.wait(check_status=(JobState.COMPLETED, JobState.FAILED))
    except (KeyboardInterrupt, JobException, TimeoutError) as e:
        launcher.stop_all(force=True)
        raise e


if __name__ == "__main__":
    # usage: python -m areal.launcher.ray \
    #   <entry_point> --config <config_path> [<additional_args>] \
    #   launcher.ray.main_func_name=<main_func_name_in_entry_point>
    ray_main()
