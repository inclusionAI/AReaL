# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import os
import re
import signal as signal_module
import subprocess
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import psutil

import realhf.base.logging as logging
from realhf.base import gpu_utils
from realhf.base.constants import get_log_path
from realhf.scheduler.client import (
    JobException,
    JobInfo,
    JobState,
    SchedulerClient,
    SchedulerError,
)

logger = logging.getLogger("Local Scheduler")

JOB_STATE_TO_PROCESS_STATUS = {
    JobState.NOT_FOUND: [],
    JobState.PENDING: [psutil.STATUS_PARKED],
    JobState.RUNNING: [
        psutil.STATUS_RUNNING,
        psutil.STATUS_SLEEPING,
        psutil.STATUS_DISK_SLEEP,
        psutil.STATUS_TRACING_STOP,
        psutil.STATUS_WAKING,
        psutil.STATUS_WAITING,
        psutil.STATUS_LOCKED,
        psutil.STATUS_IDLE,
    ],
    JobState.COMPLETED: [
        psutil.STATUS_DEAD,
        psutil.STATUS_STOPPED,
        psutil.STATUS_ZOMBIE,
    ],
    JobState.FAILED: [],
    JobState.CANCELLED: [],
}

PROCESS_STATUS_TO_JOB_STATE = {}
for job_state, process_statuses in JOB_STATE_TO_PROCESS_STATUS.items():
    for process_status in process_statuses:
        PROCESS_STATUS_TO_JOB_STATE[process_status] = job_state


def terminate_process_and_children(pid: int, signal: Optional[Union[str, int]] = None):
    if signal is None:
        signal = signal_module.SIGKILL
    if isinstance(signal, str):
        signal = getattr(signal_module, signal)
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in children:
            terminate_process_and_children(child.pid)
        parent.send_signal(signal)
    except psutil.NoSuchProcess:
        pass


class LocalSchedulerClient(SchedulerClient):
    """Instead of talking to the scheduler server (the typical behaviour), this
    client starts jobs directly on the local host and keeps a collection of job
    processes."""

    def log_path_of(self, worker_type) -> str:
        return os.path.join(
            get_log_path(self.args),
            f"{worker_type}-0.log",
        )

    def __init__(self, args):
        super().__init__(args)
        self._jobs: Dict[str, subprocess.Popen] = {}
        self._running_worker_types = []

        self._gpu_counter = 0
        self._cuda_devices: List[str] = os.environ.get(
            "CUDA_VISIBLE_DEVICES", ",".join(map(str, range(gpu_utils.gpu_count())))
        ).split(",")

        self._job_counter: Dict[str, int] = defaultdict(int)
        self._job_gpu_cnt: Dict[str, int] = defaultdict(int)
        self._job_env_vars: Dict[str, Dict] = defaultdict(dict)
        self._job_cmd = {}
        self._job_states = {}

        if len(self._cuda_devices) < 1:
            raise RuntimeError(
                f"Local mode can only run when there is at least one GPU. "
                f"CUDA_VISIBLE_DEVICES is currently set to {os.environ['CUDA_VISIBLE_DEVICES']}."
            )

    def __del__(self):
        self.wait(commit=False)

    def submit_array(
        self,
        worker_type: str,
        cmd: str,
        count: int = 1,
        gpu: int = 0,
        env_vars: Optional[Dict] = None,
        **kwargs,
    ):
        if env_vars is None:
            env_vars = {}

        self._job_counter[worker_type] += count
        if worker_type in self._job_gpu_cnt:
            assert self._job_gpu_cnt[worker_type] == (
                gpu
            ), "All workers of the same type must either use GPU or not use GPU."
        else:
            self._job_gpu_cnt[worker_type] = gpu

        if worker_type in self._job_env_vars:
            assert (
                self._job_env_vars[worker_type] == env_vars
            ), "All workers of the same type must have the same env vars."
        else:
            self._job_env_vars[worker_type] = env_vars

        if worker_type in self._job_cmd:
            assert (
                self._job_cmd[worker_type] == cmd
            ), "All workers of the same type must have the same cmd."
        else:
            self._job_cmd[worker_type] = cmd

    def submit(self, worker_type, cmd, **kwargs):
        self.submit_array(worker_type, cmd, count=1, **kwargs)

    def __commit_all(self):
        for worker_type, count, gpu, env_vars in zip(
            self._job_counter.keys(),
            self._job_counter.values(),
            self._job_gpu_cnt.values(),
            self._job_env_vars.values(),
        ):
            os.makedirs(
                os.path.dirname(self.log_path_of(worker_type)),
                exist_ok=True,
                mode=0o775,
            )
            for i in range(count):
                if gpu > 0:
                    # Allocate GPUs in a round-robin manner
                    visible_devices = []
                    for _ in range(gpu):
                        available_device_id = self._gpu_counter % len(
                            self._cuda_devices
                        )
                        self._gpu_counter += 1
                        visible_devices.append(available_device_id)
                    env_vars["CUDA_VISIBLE_DEVICES"] = ",".join(
                        str(self._cuda_devices[j]) for j in visible_devices
                    )
                cmd = (
                    " ".join(str(k) + "=" + str(v) for k, v in env_vars.items())
                    + " stdbuf -oL "
                    + self._job_cmd[worker_type]
                )
                # Run `apps.remote` with a single process.
                # This simulates a multi-prog slurm job with `count` jobsteps, with each jobstep having a single process.
                cmd = cmd.format(
                    jobstep_id=i,
                    n_jobsteps=count,
                    worker_submission_index=0,
                    wprocs_per_jobstep=1,
                    wprocs_in_job=count,
                    wproc_offset=0,
                )
                logger.debug("Starting local process with command: %s", cmd)
                cmd = f"{cmd} | tee -a {self.log_path_of(worker_type)}"
                process = subprocess.Popen(cmd, shell=isinstance(cmd, str))
                self._jobs[f"{worker_type}/{i}"] = process
            self._running_worker_types.append(worker_type)

    def stop(self, worker_type, signal=None):
        assert any(k.startswith(worker_type) for k in self._jobs)
        keys = [k for k, p in self._jobs.items() if k.startswith(worker_type)]
        procs = [p for k, p in self._jobs.items() if k.startswith(worker_type)]
        logger.info(
            f"Stopping local process with signal {signal if signal else 'SIGKILL'}, "
            f"pid: {[p.pid for p in procs]}"
        )
        for p in procs:
            terminate_process_and_children(p.pid, signal=signal)
        for p in procs:
            p.wait()
        for k, p in zip(keys, procs):
            self._jobs.pop(k)
            del p
        self._running_worker_types.remove(worker_type)

    def stop_all(self, signal=None):
        # signal argument is ignored in local stop_all
        for name in self._job_counter:
            self.stop(name, signal=signal)

    def find(self, job_name):
        if job_name in self._jobs:
            return JobInfo(name=job_name, state=JobState.RUNNING, host="localhost")
        else:
            return JobInfo(name=job_name, state=JobState.NOT_FOUND)

    def find_all(self, job_name_regex=".*"):
        rs = []
        for name in self._jobs:
            if re.fullmatch(job_name_regex, name):
                rs.append(self.find(name))
        return rs

    def wait(
        self,
        timeout=None,
        check_status: Tuple[JobState, ...] = (
            JobState.CANCELLED,
            JobState.FAILED,
            JobState.NOT_FOUND,
        ),
        remove_status: Tuple[JobState, ...] = (JobState.COMPLETED,),
        update=False,
        commit=True,
    ):
        if commit:
            self.__commit_all()
        deadline = None if timeout is None else time.time() + timeout
        logger.info(
            "Waiting for %d local running processes, pids: %s",
            len(self._jobs),
            " ".join(str(job.pid) for job in self._jobs.values()),
        )
        left = set(self._jobs.keys())
        num_jobs_left = len(left)

        while len(left) > 0:
            to_remove = []
            if len(left) < num_jobs_left:
                num_jobs_left = len(left)
                logger.info(f"Waiting for {num_jobs_left} jobs.")
            if deadline is not None and time.time() > deadline:
                raise TimeoutError(
                    f"Timeout waiting for {self.run_name}: {', '.join(sorted(left))}"
                )
            # update job states
            for job_name in list(left):
                job = self._jobs[job_name]
                pid = job.pid
                process = psutil.Process(pid)
                self._job_states[job_name] = PROCESS_STATUS_TO_JOB_STATE.get(
                    process.status(), JobState.NOT_FOUND
                )

            for job_name in list(left):
                state = self._job_states[job_name]
                if state in check_status:
                    raise JobException(
                        run_name=self.run_name,
                        worker_type=job_name.split("/")[0],
                        host="local",
                        reason=state,
                    )
                if state in remove_status:
                    logger.info(f"Job {job_name} is {state}.(Removed)")
                    left.remove(job_name)
                    to_remove.append(job_name)

            if update:
                for k in to_remove:
                    self._jobs.pop(k)
                    worker_type = k.split("/")[0]
                    assert worker_type in self._job_counter
                    self._job_counter[worker_type] -= 1
                    if self._job_counter[worker_type] <= 0:
                        assert worker_type in self._job_gpu_cnt
                        assert worker_type in self._job_env_vars
                        assert worker_type in self._job_cmd
                        self._job_counter.pop(worker_type)
                        self._job_gpu_cnt.pop(worker_type)
                        self._job_env_vars.pop(worker_type)
                        self._job_cmd.pop(worker_type)

            time.sleep(2)
