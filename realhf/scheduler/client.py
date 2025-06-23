# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import dataclasses
import enum
import subprocess
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from realhf.api.cli_args import BaseExperimentConfig


class JobState(enum.Enum):
    NOT_FOUND = 0
    PENDING = 1
    RUNNING = 2
    COMPLETED = 3
    FAILED = 4
    CANCELLED = 5

    def active(self):
        return self == self.PENDING or self == self.RUNNING


class SchedulerError(Exception):
    pass


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
    host: str = (
        None  # The host on which the job is/was running. None if the job had not run.
    )
    submit_time: str = None
    start_time: str = None
    slurm_id: str = None  # Slurm only. The Slurm id of the job.


class SchedulerClient:

    def __init__(self, args: "BaseExperimentConfig"):
        self.args = args
        self.expr_name = args.experiment_name
        self.trial_name = args.trial_name
        self.run_name = f"{self.expr_name}_{self.trial_name}"

    def submit(self, worker_type, cmd, **kwargs):
        """Submits a job to the scheduler. Raises exception if the job is
        already running.

        Args:
            worker_type: The worker type to be submitted. The job name is specified when initializing the client.
            cmd (str or List[str]): The command of this job. If this is str, the command is parsed by
                shell; otherwise it is executed directly.
        """
        raise NotImplementedError()

    def submit_array(self, worker_type, cmd, count, **kwargs):
        """Submits an array of jobs to the scheduler.

        Args:
            worker_type: The worker type to be submitted, shared by all jobs.
            cmd: Command template of the jobs that may contain an "{index}" format placeholder.
            count: Number of jobs. The indices of the jobs shall be 0..count-1.
        """
        for index in range(count):
            self.submit(
                worker_type + "_" + str(index),
                cmd.format(index=index, count=count),
                **kwargs,
            )

    def stop(self, job_name):
        """Stops a running job.

        Raises exception if there is no such job, but passes if the job
        has stopped either successfully or not.
        """
        raise NotImplementedError()

    def stop_all(self, signal=None):
        """Stops the whole job."""
        raise NotImplementedError()

    def find(self, job_name) -> Optional[JobInfo]:
        """Gets the status of a job of this job.

        Args:
            job_name: Name of the job.

        Returns:
            A JobInfo if the job is found, or None otherwise.
        """
        raise NotImplementedError()

    def find_all(self, job_name_regex=".*") -> List[JobInfo]:
        """Finds jobs.

        Args:
            job_name_regex: job name regex.

        Returns:
            A list of found JobInfo.
        """
        raise NotImplementedError()

    def wait(self, timeout=None, **kwargs):
        """Waits until all jobs submitted via this client instance finish."""
        raise NotImplementedError()


def remote_worker_cmd(expr_name, trial_name, debug, worker_type):
    # requires information in scheduler package
    return (
        f"python3 {'' if debug else '-O'} -m realhf.apps.remote worker -w {worker_type} "
        f"-e {expr_name} -f {trial_name} -i {{jobstep_id}} -g {{n_jobsteps}} -r {{worker_submission_index}} "
        f"-p {{wprocs_per_jobstep}} -j {{wprocs_in_job}} -o {{wproc_offset}}"
    )


def setup_cmd(expr_name, trial_name, debug):
    bash_cmd = (  # f"pip3 install -e $REAL_PACKAGE_PATH --no-build-isolation && "
        f"python3 {'' if debug else '-O'} -m realhf.apps.remote "
        f"reset_name_resolve -e {expr_name} -f {trial_name}"
    )
    # return f"bash -c \"{bash_cmd}\""
    return bash_cmd


def control_cmd(expr_name, trial_name, debug, ignore_worker_error, controller_type):
    bash_cmd = (  # f"pip3 install -e $REAL_PACKAGE_PATH --no-build-isolation && "
        f"python3 {'' if debug else '-O'} -m realhf.apps.remote controller "
        f"-e {expr_name} -f {trial_name} "
        f"--{'ignore_worker_error' if ignore_worker_error else 'raise_worker_error'} "
        f"--type {controller_type}"
    )
    # return f"bash -c \"{bash_cmd}\""
    return bash_cmd


def make(args: "BaseExperimentConfig", **kwargs) -> SchedulerClient:
    if args.mode == "slurm":
        from realhf.scheduler.slurm.client import SlurmSchedulerClient

        job_group_id = kwargs.get("job_group_id", None)
        job_group_index = kwargs.get("job_group_index", None)
        evaluator = kwargs.get("evaluator", None)
        return SlurmSchedulerClient(
            args,
            args.schedule_strategy,
            evaluator,
            job_group_id,
            job_group_index,
        )
    elif args.mode == "local":
        from realhf.scheduler.local.client import LocalSchedulerClient

        return LocalSchedulerClient(args)
    else:
        raise NotImplementedError(f"Scheduler {mode} not found")
