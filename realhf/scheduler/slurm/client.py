# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import fcntl
import os
import re
import select
import subprocess
import threading
import time
from collections import defaultdict
from typing import Dict, List, Literal, Optional, Tuple

import colorama

import realhf.base.logging as logging
from realhf.base.constants import get_log_path
from realhf.scheduler.client import JobException, JobInfo, JobState, SchedulerClient
from realhf.scheduler.evaluator import AutomaticEvaluator
from realhf.scheduler.slurm.utils import (
    SlurmLaunchInfo,
    SlurmResource,
    SlurmResourceNotEnoughException,
    allocate_resources,
)

logger = logging.getLogger("Slurm-scheduler")

SCHEDULING_RETRY_INTERVAL_SECONDS = 30
SCHEDULING_TIMEOUT_MAX_SECONDS = 3600 * 24
SCHEDULER_WAIT_CHECK_TIME_INTERVAL = 5


def monitor_log(
    job_name: str, log_path: str, output_file: str, stop_event: threading.Event
):
    """Monitor a log file and write its contents to the output file with job name prefix."""
    # Wait for log file to be created
    while not os.path.exists(log_path) and not stop_event.is_set():
        time.sleep(0.1)

    if stop_event.is_set():
        return

    # Open the log file and follow it
    with open(log_path, "r") as log_file, open(output_file, "a") as out_file:
        # Store last position
        position = 0
        line_pos = 0

        while not stop_event.is_set():
            log_file.seek(position)
            try:
                new_lines = log_file.readlines()
            except UnicodeDecodeError:
                time.sleep(0.5)
                continue

            if new_lines:
                # Update position
                position = log_file.tell()

                worker_type = job_name.split(":")[1]
                # Write new lines to output file with job name prefix
                for line in new_lines:
                    if line.strip():  # Skip empty lines
                        out_file.write(
                            f"{colorama.Fore.YELLOW + colorama.Style.DIM}({worker_type} Line {line_pos}){colorama.Style.RESET_ALL} {line}"
                        )
                    line_pos += 1
                out_file.flush()

            # Sleep briefly to avoid CPU spinning
            time.sleep(0.1)


class SlurmSchedulerClient(SchedulerClient):
    """Uses Slurm (https://slurm.schedmd.com/overview.html)."""

    def __init__(
        self,
        args,
        schedule_strategy: str,
        evaluator: Optional[AutomaticEvaluator],
        job_group_id: str,
        job_group_index: int,
    ):
        super().__init__(args)

        self.__schedule_strategy = schedule_strategy

        self.__pending_jobs: Dict[str, SlurmLaunchInfo] = dict()
        self.__committed_jobs: Dict[str, SlurmLaunchInfo] = dict()

        self.__submission_counter = defaultdict(int)
        self.__wprocs_counter = defaultdict(int)
        self.__evaluator = evaluator
        self.__job_group_id = job_group_id
        self.__job_group_index = job_group_index

    def submit(self, worker_type, cmd, **kwargs):
        self.submit_array(worker_type, cmd, count=1, **kwargs)

    def submit_array(
        self,
        worker_type: str,
        cmd: str,  # XXX: should be None for workers
        count: int,
        cpu: int = 1,
        gpu: int = 0,
        mem: int = 1024,  # MB
        env_vars: Optional[Dict] = None,
        container_image: Optional[str] = None,
        container_mounts: Optional[str] = None,
        nodelist: Optional[str] = None,
        exclude: Optional[str] = None,
        hostfile: bool = True,
        multiprog: bool = True,
        begin: str = None,
        deadline: str = None,
        time_limit: str = None,
    ):
        container_image = container_image or self.args.cluster.cpu_image
        container_mounts = container_mounts or self.args.cluster.mount
        # record launch information, do not submit to slurm until `wait()` is called
        # NOTE: fractional GPU requirement will be resolved automatically in `__post_init__` of SlurnLaunchInfo
        log_path = os.path.join(
            get_log_path(self.args),
            f"{worker_type}-{self.__submission_counter[worker_type]}.log",
        )
        multiprog_path = os.path.join(
            get_log_path(self.args),
            "slurm",
            "multiprog",
            f"{worker_type}-{self.__submission_counter[worker_type]}.multiprog",
        )
        os.makedirs(os.path.dirname(multiprog_path), exist_ok=True)
        hostfile_path = os.path.join(
            get_log_path(self.args),
            "slurm",
            "hostfile",
            f"{worker_type}-{self.__submission_counter[worker_type]}.hostfile",
        )
        os.makedirs(os.path.dirname(hostfile_path), exist_ok=True)

        launch_info = SlurmLaunchInfo(
            args=self.args,
            worker_type=worker_type,
            wprocs_in_job=count,
            resource_requirement=SlurmResource(mem=mem, cpu=cpu, gpu=gpu),
            cmd=cmd,
            run_name=self.run_name,
            exper_name=self.expr_name,
            trial_name=self.trial_name,
            container_image=container_image,
            container_mounts=container_mounts,
            env_vars=env_vars,
            nodelist=nodelist,
            exclude=exclude,
            hostfile=hostfile,
            multiprog=multiprog,
            worker_submission_idx=self.__submission_counter[worker_type],
            begin=begin,
            deadline=deadline,
            time_limit=time_limit,
            job_group_id=self.__job_group_id,
            job_group_index=self.__job_group_index,
            log_path=log_path,
            multiprog_path=multiprog_path,
            hostfile_path=hostfile_path,
        )

        if (
            launch_info.slurm_name in self.__pending_jobs
            or launch_info.slurm_name in self.__committed_jobs
        ):
            raise ValueError(f"job name {launch_info.slurm_name} already existed.")

        if launch_info.multiprog:
            launch_info = self.__resolve_multiprog_file(launch_info)

        self.__submission_counter[worker_type] += 1
        self.__wprocs_counter[worker_type] += count

        self.__pending_jobs[launch_info.slurm_name] = launch_info
        logger.info(f"Registered Slurm job {launch_info.slurm_name} to scheduler.")

    def __resolve_multiprog_file(self, launch_info: SlurmLaunchInfo):
        worker_type = launch_info.worker_type
        cmd = launch_info.cmd.format(
            jobstep_id="$SLURM_PROCID",
            n_jobsteps=launch_info.n_jobsteps,
            worker_submission_index=self.__submission_counter[worker_type],
            wprocs_per_jobstep=launch_info.wprocs_per_jobstep,
            wprocs_in_job=launch_info.wprocs_in_job,
            wproc_offset=self.__wprocs_counter[worker_type],
        )
        wrap_cmd = "singularity exec "
        if self.args.cluster.cluster_name == "na132":
            wrap_cmd += "--pid "
        if self.args.cluster.gpu_type == "tesla":
            wrap_cmd += "--nv "
        wrap_cmd += "--no-home --writable-tmpfs "
        if len(launch_info.env_vars) > 0:
            wrap_cmd += f"{' '.join([f'--env {k}={v}' for k, v in launch_info.env_vars.items()])} "
        if len(launch_info.container_mounts) > 0:
            wrap_cmd += f"--bind {launch_info.container_mounts} "
        wrap_cmd += f"{launch_info.container_image} "
        wrap_cmd += "bash -c '{}'".format(cmd)
        launch_info.multiprog_content = f"0-{launch_info.n_jobsteps - 1} {wrap_cmd}\n"
        return launch_info

    def __allocate_and_commit_pending_jobs(self):
        """Allocate resources to all pending job specs.

        Generate hostfiles for each job info
        """
        start_time = time.monotonic()
        while True:
            try:
                fp = open(
                    f"{self.args.cluster.fileroot}/logs/slurm_scheduler.lock", "w"
                )
                fcntl.flock(fp, fcntl.LOCK_EX)
                infos = list(self.__pending_jobs.values())
                infos = allocate_resources(infos, strategy=self.__schedule_strategy)
                self.__pending_jobs = {info.slurm_name: info for info in infos}
                # logger.info("Allocated jobs: ")
                # for info in infos:
                #     logger.info(info)
                break
            except SlurmResourceNotEnoughException:
                logger.critical(
                    "Not enough resources to allocate all pending jobs. Retrying ..."
                )
                logger.warning(
                    "Time since start: %d seconds",
                    time.monotonic() - start_time,
                )
                fcntl.flock(fp, fcntl.LOCK_UN)
                time.sleep(SCHEDULING_RETRY_INTERVAL_SECONDS)
                if time.monotonic() - start_time > SCHEDULING_TIMEOUT_MAX_SECONDS:
                    raise TimeoutError(
                        f"Timeout waiting for {self.run_name} to schedule."
                    )
            except Exception as e:
                fcntl.flock(fp, fcntl.LOCK_UN)
                raise e
        try:
            for slurm_name, launch_info in self.__pending_jobs.items():
                launch_info.commit()
                self.__committed_jobs[slurm_name] = launch_info
            self.__pending_jobs = dict()
            states = [None for _ in self.__committed_jobs]
            while JobState.PENDING in states or None in states:
                time.sleep(0.1)
                states = self.__update_all()
            # time.sleep(2)
            fcntl.flock(fp, fcntl.LOCK_UN)
        except Exception as e:
            for launch_info in self.__committed_jobs.values():
                launch_info.cancel()
            fcntl.flock(fp, fcntl.LOCK_UN)
            raise e

    def stop(self, slurm_name: str):
        launch_info = self.__committed_jobs.get(slurm_name, None)
        if launch_info:
            launch_info.cancel()

    def stop_all(self, signal: Literal["SIGINT", "SIGKILL"] = "SIGKILL"):
        for launch_info in self.__committed_jobs.values():
            logger.info(f"Canceling job {launch_info.slurm_name}")
            launch_info.cancel(signal)
        time.sleep(0.2)
        self.wait(
            check_status=(),
            remove_status=(
                JobState.CANCELLED,
                JobState.NOT_FOUND,
                JobState.FAILED,
                JobState.COMPLETED,
            ),
        )

    def find(self, slurm_name: str) -> JobInfo:
        launch_info = self.__committed_jobs.get(slurm_name, None)
        if launch_info is None or launch_info.job_info is None:
            return JobInfo(name=slurm_name, state=JobState.NOT_FOUND)
        else:
            return launch_info.job_info

    def find_all(self, job_name_regex: str = ".*") -> List[JobInfo]:
        self.__update_all()
        infos = []
        for r in self.__committed_jobs.values():
            if r.job_info is None:
                continue
            if re.fullmatch(job_name_regex, r.slurm_name):
                infos.append(r.job_info)
        return infos

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
    ):
        # before wait, commit all remaining pending jobs
        # TODO: grab global file lock to avoid multi-experiment deadlocks
        self.__allocate_and_commit_pending_jobs()
        # Start monitoring threads
        threads = []
        stop_events = []

        merged_log_path = os.path.join(get_log_path(self.args), "main.log")

        for job_name, launch_info in self.__committed_jobs.items():
            stop_event = threading.Event()
            stop_events.append(stop_event)

            # Thread for monitoring the log file
            log_thread = threading.Thread(
                target=monitor_log,
                args=(job_name, launch_info.log_path, merged_log_path, stop_event),
            )
            threads.append(log_thread)
            log_thread.start()

        # begin wait
        deadline = None if timeout is None else time.time() + timeout
        left = set(self.__committed_jobs.keys())
        num_jobs_left = len(left)
        logger.info(
            f"Waiting for {num_jobs_left} jobs. Jobs IDs: "
            f"{','.join(sorted([x.job_info.slurm_id for x in self.__committed_jobs.values()]))}."
        )
        logger.info(
            f"All slurm logs will be merged. To check the real-time output, "
            f"run\n\t`tail -f {merged_log_path}`."
        )
        try:
            while len(left) > 0:
                if len(left) < num_jobs_left:
                    num_jobs_left = len(left)
                    logger.info(f"Waiting for {num_jobs_left} jobs.")
                if self.__evaluator is not None:
                    self.__evaluator.step()
                if deadline is not None and time.time() > deadline:
                    raise TimeoutError(
                        f"Timeout waiting for {self.run_name}: {', '.join(sorted(left))}"
                    )
                try:
                    self.__update_all()
                except subprocess.CalledProcessError:
                    logger.warning(
                        "Calling squeue failed. Check slurm manually if you continue to see this warning."
                    )
                    time.sleep(30)
                    continue
                for job_slurm_name in list(left):
                    launch_info = self.__committed_jobs[job_slurm_name]
                    if launch_info.slurm_id is None:
                        continue
                    if launch_info.job_info.state in check_status:
                        launch_info.show_log()
                        raise JobException(
                            run_name=self.run_name,
                            worker_type=launch_info.worker_type,
                            host=launch_info.job_info.host,
                            reason=launch_info.job_info.state,
                        )
                    if launch_info.job_info.state in remove_status:
                        logger.info(
                            f"Job {launch_info.slurm_name} is {launch_info.job_info.state}.(Removed)"
                        )
                        left.remove(job_slurm_name)
                        if update:
                            self.__committed_jobs.pop(job_slurm_name)
                time.sleep(SCHEDULER_WAIT_CHECK_TIME_INTERVAL)
        finally:
            [s.set() for s in stop_events]
            [t.join() for t in threads]

    def __update_all(self):
        states = []
        for launch_info in self.__committed_jobs.values():
            state = launch_info.update()
            states.append(state)
        return states
