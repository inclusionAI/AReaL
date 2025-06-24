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

        # Use a customized packed scheduling method
        # that sequentially allocates nodes.
        available_nodes = [
            k
            for k in available_resources
            if re.match(r"node:(\b(?:\d{1,3}\.){3}\d{1,3}\b)", k)
        ]
        total_gpus = available_resources["GPU"]
        n_gpus_per_node = int(total_gpus // len(available_nodes))

        count = sch.count
        all_schedules: List[TasksGroup] = []
        for _ in range(sch.count):
            s_ = copy.deepcopy(sch)
            s_.count = 1
            all_schedules.append(s_)

        workers = []

        for node_idx, i in enumerate(range(0, count, n_gpus_per_node)):
            _schedules = all_schedules[i : i + n_gpus_per_node]
            for _idx, sch in enumerate(_schedules):
                # Schedule jobs one-by-one to maintain the order on remote nodes.
                worker = RayWorker.options(
                    name=f"{worker_type}/{_idx + i}",
                    num_cpus=sch.scheduling.cpu,
                    num_gpus=sch.scheduling.gpu,
                    memory=sch.scheduling.mem * 1024**2,
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
