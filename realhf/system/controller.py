# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import copy
import dataclasses
import enum
import functools
import getpass
import json
import os
import re
import socket
import sys
import threading
import time
import traceback
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import colorama
import ray
import ray.util.queue as rq
import torch
from omegaconf import OmegaConf

import realhf.api.core.system_api as system_api
from realhf.base import constants, gpu_utils, logging, name_resolve, names, pkg_version
from realhf.system import WORKER_TYPES, load_worker, worker_base, worker_control
from realhf.system.worker_base import WorkerServerStatus as Wss

flask_available = False
if pkg_version.is_available("flask"):

    from flask import Flask, jsonify

    app = Flask(__name__)

    @app.route("/discovery", methods=["GET"])
    def discovery():
        key = names.metric_server_root(
            constants.experiment_name(), constants.trial_name()
        )
        addresses = name_resolve.get_subtree(key)

        result = []
        if len(addresses) > 0:
            result.append(
                {
                    "targets": addresses,
                    "labels": {
                        "experiment": constants.experiment_name(),
                        "trial": constants.trial_name(),
                    },
                }
            )

        logger.info(f"Discover metric servers: {result}")
        return jsonify(result)

    def start_metric_discovery_server(port: int):
        host_ip = socket.gethostbyname(socket.gethostname())
        logger.info(f"Start metric discovery server: http://{host_ip}:{port}/discovery")
        app.run(debug=False, use_reloader=False, host="0.0.0.0", port=port)

    flask_available = True


CONNECTION_RETRY_AFTER_SECONDS = 360

logger = logging.getLogger("controller", "colored")


@dataclasses.dataclass
class TrialStatus:
    experiment_name: str
    trial_name: str
    running_workers: Dict[str, List[str]] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class TrialHistory:
    experiment_name: str
    trial_name: str
    age_days: int


class ControllerExitStatus(enum.Enum):
    SUCCESS = 0
    TIMEOUT = 1
    INTERRUPTED = 9
    FAIL = 101
    LOST = 102
    UNKNOWN = 404


class Controller:

    def __init__(
        self, experiment_name, trial_name, panel: worker_base.WorkerControlPanel
    ):
        assert "_" not in experiment_name, (
            f"_ not allowed in experiment_name (args: -e) "
            f"{experiment_name}, use '-' instead."
        )
        assert (
            "_" not in trial_name
        ), f"_ not allowed in trial_name (args: -f) {trial_name}, use '-' instead."
        self.experiment_name = experiment_name
        self.trial_name = trial_name

        logger.info("Experiment: %s %s", self.experiment_name, self.trial_name)

        self.__control = panel

    def reconnect(self):
        """Automatically reconnect to workers.

        And list all jobs to scheduler.
        """
        self.__control.auto_connect()

    def __check_consistent_scheduling(
        self,
        scheduling: system_api.ExperimentScheduling,
        setup: system_api.ExperimentConfig,
        verbose=False,
    ):
        # Scheduling and connecting to workers.
        workers_configs = [
            (k, getattr(setup, k), getattr(scheduling, k))
            for k in WORKER_TYPES
            if len(getattr(setup, k)) > 0
        ]

        # Sanity check for scheduling and configuration.
        for _, worker_setups, schedules in workers_configs:
            if not isinstance(schedules, List):
                schedules = [schedules]
            if len(worker_setups) != sum(s.count for s in schedules):
                raise ValueError(
                    f"Configuration and scheduling mismatch. "
                    f"Number of worker configurations: {len(worker_setups)}, "
                    f"Scheduling configs: {schedules}."
                )

        for name, config, schedule in workers_configs:
            count = (
                sum([s.count for s in schedule])
                if isinstance(schedule, list)
                else schedule.count
            )
            if len(config) != count:
                logger.error(
                    "Scheduling and config mismatch, interrupting all workers."
                )
                self.interrupt()
                raise IndexError(
                    f"Configuration has {len(config)} {name}, {count} scheduled."
                )
            if verbose:
                logger.info(f"Configuration has {len(config)} {name}.")

    def start(self, experiment: system_api.Experiment, ignore_worker_error=False):
        if flask_available and experiment.metric_discovery_port > 0:
            server_thread = threading.Thread(
                target=start_metric_discovery_server,
                args=(experiment.metric_discovery_port,),
            )
            server_thread.start()

        if ignore_worker_error:
            check_worker_status = ()
            remove_worker_status = (
                Wss.COMPLETED,
                Wss.ERROR,
                Wss.LOST,
                Wss.UNKNOWN,
                Wss.PAUSED,
            )
        else:
            check_worker_status = (Wss.ERROR, Wss.LOST, Wss.UNKNOWN)
            remove_worker_status = (Wss.COMPLETED, Wss.PAUSED)

        scheduling = experiment.scheduling_setup()
        raw_experiment = copy.deepcopy(experiment)
        setups = experiment.initial_setup()
        if not isinstance(setups, list):
            setups = [setups]

        # Sanity check before launching workers.
        for i, setup in enumerate(setups):
            self.__check_consistent_scheduling(scheduling, setup, verbose=(i == 0))

        worker_counts = [
            (k, len(getattr(setups[0], k)))
            for k in WORKER_TYPES
            if len(getattr(setups[0], k)) > 0
        ]

        name_resolve.add(
            names.trial_registry(self.experiment_name, self.trial_name),
            value=datetime.now().strftime("%Y%m%d"),
            delete_on_exit=False,
            replace=True,
        )
        name_resolve.add(
            names.worker_status(
                experiment_name=self.experiment_name,
                trial_name=self.trial_name,
                worker_name="ctl",
            ),
            value="READY",
            delete_on_exit=True,
        )

        while True:
            try:
                logger.info("Connecting to workers...")
                self.__control.connect(
                    [
                        self.__control.name(name, i)
                        for name, count in worker_counts
                        for i in range(count)
                    ],
                    progress=True,
                    timeout=CONNECTION_RETRY_AFTER_SECONDS,
                    raises_timeout_error=True,
                )
                break

            except TimeoutError:
                logger.info("Connecting to workers timeout. Retrying...")
            except KeyboardInterrupt as e:
                logger.info("Interrupted by user. Stopping all and exiting...")
                raise e

        name_resolve.delete(
            names.worker_status(
                experiment_name=self.experiment_name,
                trial_name=self.trial_name,
                worker_name="ctl",
            )
        )

        # If a log exists, find the last failed setup and run it.
        start_idx = 0
        prev_logfile = os.path.join(constants.get_log_path(experiment), "ctl-0")
        if os.path.exists(prev_logfile):
            with open(prev_logfile, "r") as f:
                for l in f.readlines():
                    match = re.search(r"Entering setup (\d+)/(\d+)", l)
                    if match and int(match.group(2)) == len(setups):
                        last_end_idx = int(match.group(1)) - 1
                        if last_end_idx < len(setups) - 1:
                            start_idx = last_end_idx

        # NOTE: Since worker processes are created and killed by the scheduler,
        # the controller cannot restart a dead worker when error occurs,
        # and it's impossible to continue the experiment when any of the multiple setups fails.
        # We can only relaunch the entire experiment in this case.
        # In particular, while it seems to be possible to continue the experiment if
        # the OOM error occurs, OOM will cause NCCL communication getting stuck (e.g, send/recv),
        # which will finally throw out a C++ exception in the watchdog thread after reaching timeout.
        # We cannot catch this exception, so OOM is irrecoverable.
        for offset, setup in enumerate(setups[start_idx:]):
            i = offset + start_idx

            s = f" Entering setup {i+1}/{len(setups)}... ".center(80, "#")
            logger.info(colorama.Fore.RED + "#" * len(s) + colorama.Style.RESET_ALL)
            logger.info(colorama.Fore.RED + s + colorama.Style.RESET_ALL)
            logger.info(colorama.Fore.RED + "#" * len(s) + colorama.Style.RESET_ALL)

            # Configure workers.
            setup.set_worker_information(
                experiment_name=self.experiment_name, trial_name=self.trial_name
            )
            try:
                for name in WORKER_TYPES:
                    if len(getattr(setup, name)) == 0:
                        continue
                    worker_infos = [x.worker_info for x in getattr(setup, name)]
                    logger.info(f"Configuring Workers: {name}...")

                    self.__control.group_request(
                        "configure",
                        worker_names=[
                            self.__control.name(name, i)
                            for i in range(len(worker_infos))
                        ],
                        worker_kwargs=[
                            dict(worker_info=wi, setup_id=i) for wi in worker_infos
                        ],
                        progress=True,
                    )
            except Exception as e:
                logger.error(f"Configuring Failed: {e}. Exiting Workers.")
                logger.error(traceback.format_exc())
                self.interrupt(wait_timeout=120)
                raise e

            logger.info("Start workers...")
            self.__control.group_request("start")
            logger.info("Started.")
            try:
                self.wait(
                    timeout=None,
                    check_status=check_worker_status,
                    remove_status=remove_worker_status,
                )
            except worker_base.WorkerException as e:
                logger.error(e)
                self.interrupt(wait_timeout=30)
            except KeyboardInterrupt:
                logger.info("Interrupted.")
                self.interrupt(wait_timeout=30)

            s = f" Finishing setup {i+1}/{len(setups)}, pausing workers... ".center(
                80, "#"
            )
            logger.info(colorama.Fore.RED + s + colorama.Style.RESET_ALL)

        logger.info(
            colorama.Fore.YELLOW
            + colorama.Style.BRIGHT
            + "\033[1m"
            + "=" * 80
            + colorama.Style.RESET_ALL
        )
        logger.info(
            colorama.Fore.YELLOW
            + colorama.Style.BRIGHT
            + "\033[1m"
            + (
                f" All {len(setups)} setups are done. "
                "You've done an excellent job! Congrats! "
            ).center(80, "=")
            + colorama.Style.RESET_ALL
        )
        logger.info(
            colorama.Fore.YELLOW
            + colorama.Style.BRIGHT
            + "\033[1m"
            + "=" * 80
            + colorama.Style.RESET_ALL
        )
        logger.info(f"Existing all workers...")
        self.__control.group_request("exit")

    def wait(
        self,
        timeout: Optional[int],
        check_status: Tuple[Wss, ...],
        remove_status: Tuple[Wss, ...],
    ):
        deadline = None if timeout is None else time.time() + timeout
        left = set(self.__control.worker_names)
        num_jobs_left = len(left)
        logger.info(f"Waiting for {num_jobs_left} jobs.")
        current_status = {name: Wss.UNKNOWN for name in self.__control.worker_names}
        while len(left) > 0:
            logger.debug(
                f"JOBS LEFT: {[str(len([l for l in left if job_type in l])) + ' ' + job_type for job_type in set([job_id.split('/')[0] for job_id in left])]}"
            )
            if len(left) < num_jobs_left:
                num_jobs_left = len(left)
                logger.info(f"Waiting for {num_jobs_left} jobs.")
            if deadline is not None and time.time() > deadline:
                raise TimeoutError(
                    f"Timeout waiting for {self.experiment_name, self.trial_name}: {', '.join(sorted(left))}"
                )
            for worker_name, worker_status in self.__control.pulse().items():
                if worker_status in check_status:
                    raise worker_base.WorkerException(
                        worker_name, worker_status, "experiment is running."
                    )
                if worker_status in remove_status:
                    if worker_name in current_status:
                        logger.debug(
                            f"Worker {worker_name} is {worker_status}. Removed from waiting list."
                        )
                        current_status.pop(worker_name)
                    else:
                        pass
                else:
                    if current_status.get(worker_name, None) != worker_status:
                        current_status.update({worker_name: worker_status})
                        logger.debug(
                            f"Update worker status: {worker_name} -> {worker_status}"
                        )

            left = set(current_status.keys())
            time.sleep(10)

    def stop(self):
        """Stop the experiment.

        Note:
            This method assumes that the controller and scheduler is connected to the correct workers. To ensure this,
            call controller.reconnect before your call controller.stop.
        """
        raise NotImplementedError()

    def interrupt(self, wait_timeout=120):
        """Interrupt the experiment."""
        logger.info("Interrupting experiment")
        self.__control.group_request("interrupt", wait_response=False)
        try:
            self.wait(
                timeout=wait_timeout,
                check_status=(),
                remove_status=(
                    Wss.ERROR,
                    Wss.LOST,
                    Wss.COMPLETED,
                    Wss.INTERRUPTED,
                ),
            )
        except TimeoutError:
            raise RuntimeError(f"Fail to interrupt workers, timeout={wait_timeout}.")


def run_ray_worker(
    worker_type,
    idx,
    world_size,
    experiment_name,
    trial_name,
    comm: Tuple[rq.Queue, rq.Queue],
):

    constants.set_experiment_trial_names(experiment_name, trial_name)

    import realhf.api.core.system_api as system_api
    from realhf.api.quickstart.entrypoint import QUICKSTART_CONFIG_CLASSES
    from realhf.base import importing
    from realhf.base.constants import QUICKSTART_EXPR_CACHE_PATH

    if os.path.exists(QUICKSTART_EXPR_CACHE_PATH):
        for exp_cache in os.listdir(QUICKSTART_EXPR_CACHE_PATH):
            target_cache_name = f"{experiment_name}_{trial_name}.json"
            if exp_cache != target_cache_name:
                continue
            cache_file = os.path.join(QUICKSTART_EXPR_CACHE_PATH, target_cache_name)
            with open(cache_file, "r") as f:
                cache = json.load(f)
            usercode_path = cache["usercode_path"]
            exp_cls_args = OmegaConf.create(cache["args"])
            config_name = cache["config_name"]
            # Import user code to register quickstart experiments.
            importing.import_usercode(usercode_path, "_realhf_user_code")
            # Register the internal experiment.
            exp_cls = QUICKSTART_CONFIG_CLASSES[config_name]
            system_api.register_experiment(
                experiment_name, functools.partial(exp_cls, **exp_cls_args)
            )

    # Isolate within the same slurm job, among different jobsteps.
    if torch.cuda.is_initialized():
        raise RuntimeError(
            "CUDA already initialized before isolating CUDA devices. This should not happen."
        )
    gpu_utils.isolate_cuda_device(
        worker_type,
        idx,
        world_size,
        experiment_name,
        trial_name,
    )
    if os.environ.get("CUDA_VISIBLE_DEVICES", None):
        logger.debug("CUDA_VISIBLE_DEVICES: %s", os.environ["CUDA_VISIBLE_DEVICES"])

    # NOTE: Importing these will initialize DeepSpeed/CUDA devices.
    # profiler.import_profiler_registers()
    if worker_type != "master_worker":
        # For master_worker, there could be errors while importing and it is not necessary.
        import realhf.impl.dataset
        import realhf.impl.model
        import realhf.system

    worker_name = f"{worker_type}/{idx}"
    server = worker_control.make_server(
        "ray",
        worker_name=worker_name,
        experiment_name=experiment_name,
        trial_name=trial_name,
        comm=comm,
    )
    worker = load_worker(worker_type)(server=server)
    try:
        worker.run()
    except Exception as e:
        logging.error("Worker %s failed with exception: %s", worker_name, e)
        logging.error(traceback.format_exc())
        raise e


class RayController:
    """A controller that uses Ray to manage workers.

    It uses the basic Controller to configure workers. Besides, it
    launchs all remote workers using Ray, instead of submitting them to
    the scheduler.
    """

    def __init__(self, experiment_name, trial_name):
        # base controller will be lazier initialized when launching workers.
        self.__experiment_name = experiment_name
        self.__trial_name = trial_name
        self.__base_controller = None

        self.__workers_reply_comm = None
        self.__workers_request_comm = None
        self.__workers_ref = None

    def _launch_workers(
        self, worker_counts: List[Tuple[str, int, system_api.TasksGroup]]
    ):
        # Launch remote workers.
        logger.info("Launching remote workers using Ray...")
        self.__workers_ref: Dict[str, ray.ObjectRef] = {}
        self.__workers_request_comm: Dict[str, rq.Queue] = dict()
        self.__workers_reply_comm: Dict[str, rq.Queue] = dict()

        # Count the total required resources and check whether Ray currently has enough of them.
        cpu = gpu = mem = 0.0
        for worker_type, _, schedule in worker_counts:
            if not isinstance(schedule, List):
                schedule = [schedule]
            for s in schedule:
                cpu += s.scheduling.cpu * s.count
                gpu += s.scheduling.gpu * s.count
                mem += s.scheduling.mem * s.count / 1024  # in GB
        available_resources = ray.available_resources()
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

        # Launch ray jobs.
        for worker_type, count, schedule in worker_counts:
            all_schedules: List[system_api.TasksGroup] = []
            if isinstance(schedule, List):
                for s in schedule:
                    for _ in range(s.count):
                        s_ = copy.deepcopy(s)
                        s_.count = 1
                        all_schedules.append(s_)
            else:
                for _ in range(schedule.count):
                    s_ = copy.deepcopy(schedule)
                    s_.count = 1
                    all_schedules.append(s_)
            assert len(all_schedules) == count
            comms = [(rq.Queue(maxsize=8), rq.Queue(maxsize=8)) for _ in all_schedules]
            world_size = len(all_schedules)
            if any(sch.scheduling.gpu > 0 for sch in all_schedules):
                # For GPU jobs, use a customized packed scheduling method
                # that sequentially allocates nodes.
                if not all(
                    sch.scheduling.gpu == all_schedules[0].scheduling.gpu == 1
                    for sch in all_schedules
                ):
                    raise ValueError(
                        "Ray scheduler only supports resource requirements where #GPU=1 or #GPU=0."
                    )
                available_nodes = [
                    k
                    for k in available_resources
                    if re.match(r"node:(\b(?:\d{1,3}\.){3}\d{1,3}\b)", k)
                ]
                total_gpus = available_resources["GPU"]
                if total_gpus % len(available_nodes) != 0:
                    raise ValueError(
                        "Cannot schedule Ray jobs to nodes with heterogeneous numbers of GPUs."
                    )
                n_gpus_per_node = int(total_gpus // len(available_nodes))
                if total_gpus < count:
                    raise RuntimeError(
                        "Available GPUs is smaller than the number of scheduled GPU workers."
                    )

                jobs = []
                for node_idx, i in enumerate(range(0, count, n_gpus_per_node)):
                    _schedules = all_schedules[i : i + n_gpus_per_node]
                    _comms = comms[i : i + n_gpus_per_node]
                    for _idx, (comm, sch) in enumerate(zip(_comms, _schedules)):
                        # Schedule jobs one-by-one to maintain the order on remote nodes.
                        job = ray.remote(
                            num_cpus=sch.scheduling.cpu,
                            num_gpus=sch.scheduling.gpu,
                            memory=sch.scheduling.mem * 1024**2,
                            name=f"{worker_type}/{_idx + i}",
                            resources={available_nodes[node_idx]: 1 / n_gpus_per_node},
                        )(run_ray_worker).remote(
                            worker_type,
                            _idx + i,
                            world_size,
                            self.__experiment_name,
                            self.__trial_name,
                            comm,
                        )
                        try:
                            ray.get(job, timeout=0.1)
                        except ray.exceptions.GetTimeoutError:
                            pass
                        jobs.append(job)
            else:
                # Use the default Ray scheduler, which may have some randomness.
                jobs = [
                    ray.remote(
                        num_cpus=sch.scheduling.cpu,
                        num_gpus=sch.scheduling.gpu,
                        memory=sch.scheduling.mem * 1024**2,
                        name=f"{worker_type}/{idx}",
                    )(run_ray_worker).remote(
                        worker_type,
                        idx,
                        world_size,
                        self.__experiment_name,
                        self.__trial_name,
                        comm,
                    )
                    for idx, (comm, sch) in enumerate(zip(comms, all_schedules))
                ]
            for idx, (job, c) in enumerate(zip(jobs, comms)):
                name = f"{worker_type}/{idx}"
                self.__workers_ref[name] = job
                self.__workers_request_comm[name] = c[0]
                self.__workers_reply_comm[name] = c[1]
            # Perform a poll step on remote jobs to let them raise setup errors,
            # e.g., ImportError, ModuleNotFoundError, etc.
            try:
                ray.get(jobs, timeout=1)
            except ray.exceptions.GetTimeoutError:
                pass
            logger.info(f"Launched {count} {worker_type}.")

        panel = worker_control.make_control(
            "ray",
            self.__experiment_name,
            self.__trial_name,
            request_comms=self.__workers_request_comm,
            reply_comms=self.__workers_reply_comm,
        )
        self.__base_controller = Controller(
            self.__experiment_name, self.__trial_name, panel
        )
        logger.info("All Ray workers are lauched.")

    def start(self, experiment: system_api.Experiment, ignore_worker_error=False):
        scheduling: system_api.ExperimentScheduling = experiment.scheduling_setup()
        setup = experiment.initial_setup()
        if not isinstance(setup, list):
            setup = [setup]
        worker_counts = [
            (k, len(getattr(setup[0], k)), getattr(scheduling, k))
            for k in WORKER_TYPES
            if len(getattr(setup[0], k)) > 0
        ]

        env_vars = constants.get_env_vars(
            experiment,
            REAL_MODE=os.environ.get("REAL_MODE", ""),
            REAL_RECOVER_RUN=os.environ.get("REAL_RECOVER_RUN", ""),
            REAL_SAVE_RECOVER_STATES=os.environ.get("REAL_SAVE_RECOVER_STATES", ""),
        )
        runtime_env = {
            "env_vars": env_vars,
            "working_dir": os.getcwd(),
        }
        logger.info(f"Ray workers runtime env: {runtime_env}")
        ray.init(runtime_env=runtime_env)

        logger.info("Ray initialized! Ready to run workers.")

        try:
            self._launch_workers(worker_counts)
            self.__base_controller.start(experiment, ignore_worker_error)
        except Exception as e:
            ray.shutdown()
            raise e
