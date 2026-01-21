# Copyright 2025 Ant Group Inc.
import copy
import os
import re
import signal
import sys
import threading
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, List

import psutil
import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from realhf.api.cli_args import NameResolveConfig
from realhf.api.core.system_api import Experiment, ExperimentScheduling, TasksGroup
from realhf.base import constants, logging, name_resolve, names
from realhf.system import WORKER_TYPES, load_worker
from realhf.system.worker_base import AsyncWorker, Worker, WorkerServerStatus


# Copied from SGLang
def kill_process_tree(parent_pid, include_parent: bool = True, skip_pid: int = None):
    """Kill the process and all its child processes."""
    # Remove sigchld handler to avoid spammy logs.
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)

    if parent_pid is None:
        parent_pid = os.getpid()
        include_parent = False

    try:
        itself = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        return

    children = itself.children(recursive=True)
    for child in children:
        if child.pid == skip_pid:
            continue
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass

    if include_parent:
        try:
            if parent_pid == os.getpid():
                itself.kill()
                sys.exit(0)

            itself.kill()

            # Sometime processes cannot be killed with SIGKILL (e.g, PID=1 launched by kubernetes),
            # so we send an additional signal to kill them.
            itself.send_signal(signal.SIGQUIT)
        except psutil.NoSuchProcess:
            pass


@ray.remote
class RayWorker:
    """A thin wraper over realhf.system.worker_base.Worker."""

    def __init__(
        self,
        args,
        worker_type: str,
        worker_cls,
        kv_store_name,
    ):
        # Register all datasets and models
        import realhf.impl.dataset  # isort: skip
        import realhf.impl.model  # isort: skip

        os.environ["REAL_MODE"] = "RAY"

        name_recolve_config = NameResolveConfig("ray", ray_actor_name=kv_store_name)
        name_resolve.reconfigure(name_recolve_config)
        self.worker: Worker | AsyncWorker = worker_cls()
        self.worker_type = worker_type
        self.args = args

    def __repr__(self):
        return "".join([c.capitalize() for c in self.worker_type.split("_")])

    def configure(self, cfg: Any, expr_config: Any):

        worker_info = cfg.worker_info
        idx = worker_info.worker_index
        constants.set_experiment_trial_names(
            worker_info.experiment_name, worker_info.trial_name
        )
        self.worker.wandb_config = expr_config.wandb
        self.worker.swanlab_config = expr_config.swanlab
        self.worker.tensorboard_config = expr_config.tensorboard
        self.worker.args = self.args
        self.logger = logging.getLogger(f"{self.worker_type} {idx}", "benchmark")
        self.logger.info(f"Configuring {self.worker_type}...")
        self.worker._configure(cfg)
        self.logger.info(f"Configuring {self.worker_type}... Done.")

    def run_sync(self):
        self.logger.info(f"Running {self.worker_type} lazy initialization...")
        self.worker._poll()
        self.logger.info(f"Running {self.worker_type} lazy initialization... Done.")
        while self.worker.status != WorkerServerStatus.PAUSED:
            self.worker._poll()

    async def run_async(self):
        self.logger.info(f"Running {self.worker_type} lazy initialization...")
        await self.worker._poll_async()
        self.logger.info(f"Running {self.worker_type} lazy initialization... Done.")
        while self.worker.status != WorkerServerStatus.PAUSED:
            await self.worker._poll_async()


def _run_experiment(exp_cfg, expr_name, trial_name):
    # Register all datasets and models
    import realhf.impl.dataset  # isort: skip
    import realhf.impl.model  # isort: skip
    from realhf.api.core.system_api import ALL_EXPERIMENT_CLASSES
    from realhf.system.master_worker import MasterWorker

    constants.set_experiment_trial_names(expr_name, trial_name)

    logger = logging.getLogger(f"RayMasterWorker", "benchmark")

    # Initialize ray in the Ray cluster
    env_vars = constants.get_env_vars(
        exp_cfg,
        WADNB_MODE=exp_cfg.wandb.mode,
        SWANLAB_MODE=exp_cfg.swanlab.mode,
        REAL_MODE="ray",
        REAL_RECOVER_RUN="0",
        REAL_SAVE_RECOVER_STATES="1",
    )
    git_path = Path(__file__).parent.parent / ".git"
    runtime_env = {
        "env_vars": env_vars,
        "working_dir": os.getcwd(),
        "excludes": [str(git_path)],
    }
    logger.info(f"Ray workers runtime env: {runtime_env}")
    ray_log_path = exp_cfg.ray_temp_path
    os.makedirs(ray_log_path, exist_ok=True)
    ray.init(runtime_env=runtime_env, _temp_dir=ray_log_path)
    logger.info(f"Ray log root: {ray_log_path}")
    logger.info("Ray initialized! Ready to run workers.")

    ray_kv_store_name = f"{expr_name}/{trial_name}/ray_kv_store"
    name_recolve_config = NameResolveConfig("ray", ray_actor_name=ray_kv_store_name)
    name_resolve.reconfigure(name_recolve_config)

    name_resolve.clear_subtree(
        names.trial_root(experiment_name=expr_name, trial_name=trial_name)
    )

    # Convert CLI args into worker configurations
    exp_setup = exp_cfg.initial_setup()
    exp_setup.set_worker_information(expr_name, trial_name)
    exp_setup.lazy_init()

    # Initialize all workers
    all_workers = {}

    scheduling: ExperimentScheduling = exp_cfg.scheduling_setup()

    # We assume all nodes have the same resources (CPU, GPU, memory).
    all_available_resources = ray.available_resources()
    all_available_nodes = [
        k
        for k in all_available_resources
        if re.match(r"node:(\b(?:\d{1,3}\.){3}\d{1,3}\b)", k)
    ]
    n_nodes = len(all_available_nodes)
    n_gpus_per_node = int(all_available_resources["GPU"] // n_nodes)
    assert (
        all_available_resources["GPU"] % n_nodes == 0
    ), "AReaL assumes all nodes has the same number of GPUs."

    for worker_type in WORKER_TYPES:
        sch = getattr(scheduling, worker_type)
        if sch is None:
            continue

        available_resources = ray.available_resources()
        cpu = sch.scheduling.cpu * sch.count
        gpu = sch.scheduling.gpu * sch.count
        mem = sch.scheduling.mem * sch.count / 1024  # in GB
        acpu = available_resources.get("CPU", 0)
        agpu = available_resources.get("GPU", 0)
        amem = available_resources.get("memory", 0) / 1024**3
        if acpu < cpu or agpu < gpu or amem < mem:
            logger.critical(
                f"Ray does not have enough resources to launch workers. "
                f"Required: {cpu} CPU, {gpu} GPU, {mem:.2f} GB memory. "
                f"Available: {acpu} CPU, {agpu} GPU, {amem:.2f} GB memory. "
                f"Please launch more Ray nodes otherwise the experiment will get stuck."
            )

        workers = []
        if sch.scheduling.gpu > 0 and n_nodes > 1:
            # When # nodes > 1, for GPU workers, schedule them in granularity of nodes.
            assert (
                n_gpus_per_node % sch.scheduling.gpu == 0
            ), f"Each node should be allocated with identical numbers of {worker_type}."
            n_worker_per_node = int(n_gpus_per_node / sch.scheduling.gpu)
            assert sch.count % n_worker_per_node == 0, (
                f"Total {worker_type} count ({sch.count}) should be divisible by "
                f"the number of workers per node ({n_worker_per_node})."
            )
            n_nodes = int(sch.count / n_worker_per_node)
            placement_group = ray.util.placement_group(
                bundles=[
                    {
                        "CPU": sch.scheduling.cpu * n_worker_per_node,
                        "GPU": sch.scheduling.gpu * n_worker_per_node,
                        "memory": sch.scheduling.mem
                        * 1024**2
                        * n_worker_per_node,  # in bytes
                    }
                ]
                * n_nodes,
            )
            try:
                ray.get(placement_group.ready(), timeout=30)
            except ray.exceptions.GetTimeoutError:
                logger.critical(
                    f"Failed to create placement group for {worker_type}s. "
                    f"Please make sure at least {n_nodes} node "
                    f"has resources for {n_worker_per_node} {worker_type}s."
                )

            for node_id in range(n_nodes):
                # Use a customized packed scheduling method
                # that sequentially allocates nodes.
                for i in range(n_worker_per_node):
                    _idx = node_id * n_worker_per_node + i
                    worker = RayWorker.options(
                        name=f"{worker_type}/{_idx}",
                        num_cpus=sch.scheduling.cpu,
                        num_gpus=sch.scheduling.gpu,
                        memory=sch.scheduling.mem * 1024**2,
                        scheduling_strategy=PlacementGroupSchedulingStrategy(
                            placement_group=placement_group,
                            placement_group_bundle_index=node_id,
                            placement_group_capture_child_tasks=True,
                        ),
                    ).remote(
                        args=exp_cfg,
                        worker_type=worker_type,
                        worker_cls=load_worker(worker_type),
                        kv_store_name=ray_kv_store_name,
                    )
                    workers.append(worker)
        else:
            # Schedule them with SPREAD strategy when
            # 1. CPU workers when n_nodes > 1,
            # to save as much resource as possible on nodes for GPU workers.
            # 2. all workers when n_nodes = 1
            for _idx in range(sch.count):
                worker = RayWorker.options(
                    name=f"{worker_type}/{_idx}",
                    num_cpus=sch.scheduling.cpu,
                    num_gpus=sch.scheduling.gpu,
                    memory=sch.scheduling.mem * 1024**2,
                    scheduling_strategy="SPREAD",
                ).remote(
                    args=exp_cfg,
                    worker_type=worker_type,
                    worker_cls=load_worker(worker_type),
                    kv_store_name=ray_kv_store_name,
                )
                workers.append(worker)
        all_workers[worker_type] = workers

    try:
        # Configure workers
        configure_jobs = []
        for worker_type in all_workers:
            worker_configs = getattr(exp_setup, worker_type)
            workers = all_workers[worker_type]
            assert len(workers) == len(worker_configs), (
                len(workers),
                len(worker_configs),
            )
            jobs = [
                w.configure.remote(c, exp_cfg) for w, c in zip(workers, worker_configs)
            ]
            configure_jobs += jobs

        ray.get(configure_jobs)

        # Run workers
        run_jobs = []
        for worker_type in all_workers:
            workers = all_workers[worker_type]
            if worker_type in ["master_worker", "rollout_worker"]:
                # Only the rollout worker is asynchronous
                jobs = [w.run_async.remote() for w in workers]
            else:
                jobs = [w.run_sync.remote() for w in workers]
            run_jobs += jobs

        ray.get(run_jobs)
    finally:
        ray.shutdown()


class DualOutput:
    def __init__(self, file, terminal):
        self.file = file
        self.terminal = terminal

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)

    def flush(self):
        self.terminal.flush()
        self.file.flush()

    def fileno(self):
        # Return the terminal's fileno to maintain expected behavior
        return self.terminal.fileno()


def run_experiment(exp_cfg, expr_name, trial_name):
    log_path = os.path.join(constants.get_log_path(exp_cfg), "main.log")
    with open(log_path, "a") as f:
        # Create dual output handler
        dual_out = DualOutput(f, sys.stdout)
        dual_err = DualOutput(f, sys.stderr)

        # Redirect stdout and stderr
        with redirect_stdout(dual_out), redirect_stderr(dual_err):
            _run_experiment(exp_cfg, expr_name, trial_name)
