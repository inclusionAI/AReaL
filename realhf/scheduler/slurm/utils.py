# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

from __future__ import (
    annotations,  # python3.7+ feature to allow self-referencing type hints
)

import collections
import dataclasses
import datetime
import getpass
import json
import math
import os
import shutil
import socket
import subprocess
from typing import Callable, Dict, List, Literal, Optional, Union

import pandas as pd

import realhf.base.logging as logging
import realhf.version as version
from realhf.api.cli_args import BaseExperimentConfig
from realhf.base.constants import get_log_path
from realhf.scheduler.client import JobException, JobInfo, JobState

logger = logging.getLogger("scheduler.slurm.utils")

SQUEUE_FIELDS = [
    "JobID",
    "State",
    "SubmitTime",
    "StartTime",
    "Name",
    "NodeList",
    "UserName",
    "MaxCPUs",
    "cpus-per-task",
    "NumTasks",
    "tres-alloc",
]
STATUS_MAPPING = {
    "RUNNING": JobState.RUNNING,
    "COMPLETING": JobState.RUNNING,
    "PENDING": JobState.PENDING,
    "CANCELLED": JobState.CANCELLED,
    "FAILED": JobState.FAILED,
    "COMPLETED": JobState.COMPLETED,
    "OUT_OF_MEMORY": JobState.FAILED,
    "DEADLINE": JobState.COMPLETED,
    "TIMEOUT": JobState.COMPLETED,
}


class SlurmResourceNotEnoughException(Exception):
    pass


class InvalidGPUTypeException(Exception):
    pass


@dataclasses.dataclass
class SlurmResource:
    # a data class that represents a slurm resource quota
    mem: int = 0
    cpu: int = 0
    gpu: Union[float, int] = 0

    def __str__(self):
        return (
            "SlurmResource: \n"
            + "mem: "
            + str(self.mem)
            + " MB \n"
            + "cpu: "
            + str(self.cpu)
            + " \n"
            + "gpu: "
            + str(self.gpu)
        )

    def __mul__(self, other: int) -> SlurmResource:
        assert isinstance(
            other, int
        ), "ResourceRequirement can only be multiplied by int."
        return SlurmResource(
            mem=self.mem * other,
            cpu=self.cpu * other,
            gpu=self.gpu * other,
        )

    def __rmul__(self, other: int) -> SlurmResource:
        return self.__mul__(other)

    def __add__(self, other: SlurmResource) -> SlurmResource:
        assert isinstance(
            other, SlurmResource
        ), "SlurmResource can only add another SlurmResource instance."
        return SlurmResource(
            mem=self.mem + other.mem,
            cpu=self.cpu + other.cpu,
            gpu=self.gpu + other.gpu,
        )

    def __sub__(self, other: SlurmResource) -> SlurmResource:
        assert isinstance(
            other, SlurmResource
        ), "SlurmResource can only subtract another SlurmResource instance."
        return SlurmResource(
            mem=self.mem - other.mem,
            cpu=self.cpu - other.cpu,
            gpu=self.gpu - other.gpu,
        )

    def __neg__(self) -> SlurmResource:
        return SlurmResource(
            mem=-self.mem,
            cpu=-self.cpu,
            gpu=-self.gpu,
        )

    def __eq__(self, other: SlurmResource) -> bool:
        return self.mem == other.mem and self.cpu == other.cpu and self.gpu == other.gpu

    def __lt__(self, other: SlurmResource) -> bool:
        if self.gpu != other.gpu:
            return self.gpu < other.gpu
        if self.cpu != other.cpu:
            return self.cpu < other.cpu
        if self.mem != other.mem:
            return self.mem < other.mem

    def valid(self) -> bool:
        # check if it is a valid resource requirement
        if self.mem < 0 or self.cpu < 0 or self.gpu < 0:
            return False
        return True


@dataclasses.dataclass
class SlurmLaunchInfo:
    """A SlurmLaunchInfo contains all informantion required to **launch** a
    slurm job.

    Matching one `TasksGroup` in `SchedulingConfig` and one slurm job.

    The naming conventions:
        - `job`: Literally a slurm job with a (maybe non-unique) job name and an unique job ID,
            which may contain multiple job steps and processes. It corresponds an `sbatch` or `srun` call.
            Job names are guaranteed to be unique using the scheduler within this repo.
        - `jobstep`: Literally a slurm job step with a unique job step ID, i.e., ${jobID}.${stepID},
            which corresponds to a running instance `apps.remote` script, but may still contain multiple processes.
            A job step occupies at most one GPU. Processes in the same job step must share the same GPU.
        - `wproc`: A single worker process launched by `apps.remote` script, which may occupy less than 1 GPU.
            A worker just corresponds to a process.
        - `task`: The alias of `jobstep`. It is easier to under stand this concept in the context of `srun` command.
            `--ntasks' is just the number of jobsteps. We use the alternative term `jobstep` to avoid confusion.

    Attributes:
        run_name (str): Identifier of this run, typically ${exp_name}_${trial_name}.
        worker_type (str): Type of workers to be launched, e.g. model_worker, data_worker, etc.
        worker_submission_idx (int): For heterogeneous scheduling, we submit jobs of the same worker_type to slurm
            for multiple times. `worker_submission_idx` is used to distinguish them, so the (global) slurm job name will
            be ${run_name}:${worker_type}:${worker_submission_idx}.
        wprocs_in_job: The number of worker processes in this slurm job (of all job steps).
        n_jobsteps (int): The number of job steps of this slurm job. This is also the group size of the multiprog file.
            Will be resolved automatically according to GPU requirement.
        wprocs_per_jobstep: The number of worker processes in each job step, as well as the number of sub-processes
            spawned by `apps.remote`. Will be resolved automatically according to GPU requirement.

        resource_requirement (SlurmResource): The resource requirement of this job, including all job steps.
        cmd (str): The command to be executed.
        container_image (str): In current PPU setup, container_image should match the format provided by singularity.
            If the image is a file, this string should be the path. If the image is a remote docker image,
            this string should be of format 'docker://<image>'.
        container_mounts (str): .
        env_vars (dict): .
        nodelist (str): .
        exclude (str): .
        partition (str, optional): default to "all".
        time_limit (str, optional): Slurm job time limit.
        begin (str, optional): Scheduled worker start time.
        deadline (str, optional): Scheduled worker end time.
        hostfile (bool): Whether to use hostfile for `--distribution=arbitrary` scheduling.
        hostfile_content (str, optional): The content of the hostfile.
        multiprog (bool): Whether to use multiprog file for `--multi-prog` job submission.
        multiprog_content (str, optional): The content of the multiprog file.
    """

    args: BaseExperimentConfig
    run_name: str
    exper_name: str
    trial_name: str
    worker_type: str
    worker_submission_idx: int
    wprocs_in_job: int
    job_group_id: str
    job_group_index: str

    log_path: str
    multiprog_path: str
    hostfile_path: str

    resource_requirement: SlurmResource
    cmd: str
    container_image: str
    container_mounts: str
    env_vars: dict
    nodelist: str
    exclude: str
    partition: Optional[str] = "all"
    time_limit: Optional[str] = None
    begin: Optional[str] = None
    deadline: Optional[str] = None
    # hostfile
    hostfile: bool = True
    hostfile_content: Optional[str] = None
    # multiprog options, override cmd
    multiprog: bool = True
    multiprog_content: Optional[str] = None

    n_jobsteps: int = None
    wprocs_per_jobstep: int = None

    job_info: Optional[JobInfo] = None

    def __post_init__(self):
        """Resolve fractional GPU resource requirement."""
        gpu_per_worker = self.resource_requirement.gpu
        # assert gpu_per_worker <= 1 and gpu_per_worker >= 0
        if gpu_per_worker < 1 and gpu_per_worker > 0:
            self.resource_requirement.gpu = 1
            self.wprocs_per_jobstep = math.floor(1 / gpu_per_worker)
            self.resource_requirement.cpu *= self.wprocs_per_jobstep
            self.resource_requirement.mem *= self.wprocs_per_jobstep
            self.n_jobsteps = math.ceil(self.wprocs_in_job / self.wprocs_per_jobstep)
            logger.info(f"Resolved fractional GPU requirement for {self.slurm_name}")
            logger.info(
                f"GPU per worker {gpu_per_worker}, workers per jobstep (process size in `apps.remote`) {self.wprocs_per_jobstep}, "
                f"number of jobsteps (instance of running `apps.remote`) {self.n_jobsteps}"
            )
        else:
            self.n_jobsteps = self.wprocs_in_job
            self.wprocs_per_jobstep = 1

    @property
    def slurm_name(self) -> str:
        return f"{self.run_name}:{self.worker_type}:{self.worker_submission_idx}"

    @property
    def slurm_id(self) -> Optional[str]:
        if self.job_info:
            return self.job_info.slurm_id
        else:
            return None

    def show_log(self):
        try:
            terminal_columns = os.get_terminal_size().columns
        except OSError:
            terminal_columns = shutil.get_terminal_size().columns
        logger.info(
            f"Showing log of slurm job: {self.worker_type}-{self.worker_submission_idx}\n\n{'-'*terminal_columns}"
        )
        subprocess.Popen(["tail", "-n50", self.log_path]).wait(timeout=3)
        logger.info(
            f"End of log: {self.worker_type}-{self.worker_submission_idx}\n\n{'-'*terminal_columns}"
        )

    def update(self):
        job_infos = query_jobs(slurm_names=[self.slurm_name])
        job_infos = sorted(
            job_infos,
            key=lambda x: parse_formatted_time(x.submit_time),
            reverse=True,
        )
        self.job_info = job_infos[0] if len(job_infos) > 0 else None
        if self.job_info:
            return self.job_info.state
        else:
            return None

    def cancel(self, signal: Literal["SIGINT", "SIGKILL"] = "SIGKILL"):
        cancel_jobs(slurm_names=[self.slurm_name], signal=signal)
        self.job_info = JobInfo(name=self.slurm_name, state=JobState.CANCELLED)

    def __str__(self):
        s = f"SlurmLaunchInfo [{self.slurm_name}] \n"
        s += f"Resources: [\n{self.resource_requirement}\n]\n"
        s += f"Multiprog Filepath: [{self.multiprog_path}]\n"
        s += f"Multiprog Content: [\n{self.multiprog_content}\n]\n"
        s += f"Hostfile Filepath: [{self.hostfile_path}]\n"
        s += f"Hostfile Content: [\n{self.hostfile_content}\n]\n"
        if self.job_info is None:
            job_info_str = "None"
        else:
            job_info_str = "\n".join(
                [f"{k}: {v}" for k, v in self.job_info.__dict__.items()]
            )
        s += f"Runtime JobInfo: [\n{job_info_str}\n]\n"
        env_var_str = "\n".join([f"{k}: {v}" for k, v in self.env_vars.items()])
        s += f"Env vars: [\n{env_var_str}\n]\n"
        return s

    def commit(self):
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True, mode=0o775)

        ntasks = self.n_jobsteps
        mem = self.resource_requirement.mem
        cpu = self.resource_requirement.cpu
        gpu = self.resource_requirement.gpu

        cmd = self.cmd

        # assert gpu == 1 or gpu == 0, "Slurm job GPU requirement should be resolved to a integer."

        if self.multiprog:
            with open(self.multiprog_path, "w") as f:
                f.write(self.multiprog_content)
        if self.hostfile:
            with open(self.hostfile_path, "w") as f:
                f.write(self.hostfile_content)

        logger.info(
            f'Allocating {ntasks} jobstep(s) "{self.worker_type}" submssion index {self.worker_submission_idx}'
            f" with {cpu} cpu, {gpu} gpu and {mem} MB memory."
        )
        logger.info(f"To check the output, run \n\t`tail -f {self.log_path}`.")

        # Setup sbatch
        # head
        gres_line = ""
        if gpu >= 1:
            assert (gpu * ntasks) % self.args.cluster.n_gpus_per_node == 0
            # In current slurm cluster setup, we can only use "--gres" to
            # allocate PPUs per node. There are no options to allocate customized
            # gres per tasks.
            if self.args.cluster.gpu_type == "ppu":
                gres_line = f"--gres=ppu:{self.args.cluster.n_gpus_per_node}"
            else:
                gres_line = f"--gres=gpu:{self.args.cluster.n_gpus_per_node}"

        srun_env = os.environ.copy()
        job_metadata = {
            "user": srun_env.get("EMAILPREFIX", ""),
            "version": version.__version__,
            "branch": version.__branch__,
            "commit": version.__commit__,
            "dirty": version.__is_dirty__,
            "job_group_id": self.job_group_id,
            "job_group_index": self.job_group_index,
        }
        job_metadata_json = json.dumps(job_metadata)

        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={self.slurm_name}",
            f"#SBATCH --output={self.log_path}",
            "#SBATCH --open-mode=append",
            f"#SBATCH --ntasks={ntasks}",
            f"#SBATCH {gres_line}" if gpu >= 1 else "",
            f"#SBATCH --cpus-per-task={cpu}",
            f"#SBATCH --mem-per-cpu={mem // max(1, cpu)}M",
            "#SBATCH --distribution=arbitrary" if self.hostfile else "",
            # f'#SBATCH --nodelist={spec.nodelist}' if spec.nodelist is not None else "",
            # f'#SBATCH --exclude={spec.exclude}' if spec.exclude is not None else "",
            f"#SBATCH --time={self.time_limit}" if self.time_limit else "",
            f"#SBATCH --begin={self.begin}" if self.begin else "",
            f"#SBATCH --deadline={self.deadline}" if self.deadline else "",
            f"#SBATCH --comment='{job_metadata_json}'",
        ]

        if self.hostfile:
            srun_env["SLURM_HOSTFILE"] = self.hostfile_path
        # Setup step command.
        # add current directory into container mounts to ensure editable mode for realhf package
        srun_flags = [
            f"--ntasks={ntasks}",
            f"--cpus-per-task={cpu}",
            gres_line,
            f"--mem-per-cpu={mem // max(1, cpu)}",
            f"--multi-prog {self.multiprog_path}" if self.multiprog else "",
        ]

        # The `-K` option ensures all job steps within the same job id would be killed if
        # one of them exited with error. This is necessary for recovery.
        if self.multiprog:
            srun_cmd = (
                f'srun --mpi=pmi2 -K -l {" ".join(srun_flags)} {self.multiprog_path}'
            )
        else:
            srun_cmd = f'srun --mpi=pmi2 -K -l {" ".join(srun_flags)} {cmd}'

        lines += [
            'echo "[Runner] StartTime: $(date -u)"',
            'echo "[Runner] Host: $(hostname)"',
            "echo '[Runner] Command: {}'".format(srun_cmd),
            "echo '[Runner] Log: {}'".format(self.log_path),
            'echo "[Runner] CudaVisible: $CUDA_VISIBLE_DEVICES"',
            'echo "[Runner] CudaMpsPerc: $CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"',
            srun_cmd,
            "RETCODE=$?",
            'echo "[Runner] FinishTime: $(date -u)"',
            'echo "[Runner] RetCode: $RETCODE"',
            'echo "[Runner] ------------"',
            "exit $RETCODE",
        ]

        script_strs = "\n".join(list(filter(lambda x: x, lines))) + "\n"
        script = script_strs.encode("ascii")

        def pad_output_str_to_length(s: str, pad_s: str, length: int):
            assert len(pad_s) == 1
            assert len(s) + 2 <= length
            n_pads = (length - len(s) - 2) // 2
            return pad_s * n_pads + " " + s + " " + pad_s * n_pads

        with open(self.log_path, "a") as f:
            f.write(pad_output_str_to_length("SBATCH SCRIPT BEGIN", "=", 80) + "\n")
            f.write(script_strs)
            f.write(pad_output_str_to_length("SBATCH SCRIPT END", "=", 80) + "\n")
            f.write(pad_output_str_to_length("SBATCH JOB INFO BEGIN", "=", 80) + "\n")
            f.write(str(self))
            f.write(pad_output_str_to_length("SBATCH JOB INFO END", "=", 80) + "\n")
            f.write(pad_output_str_to_length("JOB OUTPUT BEGIN", "=", 80) + "\n")
        r = (
            subprocess.check_output(
                ["sbatch", "--parsable"], input=script, env=srun_env
            )
            .decode("ascii")
            .strip()
        )
        self.job_info = JobInfo(name=self.slurm_name, state=JobState.PENDING)


def parse_formatted_time(time_string: str) -> int:
    if time_string == "N/A":
        return -1
    d = datetime.datetime.strptime(time_string, "%Y-%m-%dT%H:%M:%S")
    return int(datetime.datetime.timestamp(d))


def unparse_formatted_time(timestamp: int) -> str:
    if timestamp == -1:
        return "N/A"
    d = datetime.datetime.fromtimestamp(timestamp)
    return d.strftime("%Y-%m-%dT%H:%M:%S")


# slurm command execute and output parsing
def query_jobs(
    slurm_names: Optional[List[str]] = None,
    slurm_ids: Optional[List[str]] = None,
    status: str = "all",
    delimiter: str = "__PSI__",
) -> List[JobInfo]:
    squeue_format = f":.{delimiter},".join(SQUEUE_FIELDS)
    cmd = ["squeue", "-O", squeue_format, f"-t{status}"]
    if slurm_names is not None:
        cmd += ["-n", ",".join(slurm_names)]
    if slurm_ids is not None:
        cmd += ["-j", ",".join([str(s) for s in slurm_ids])]
    output = (
        subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode("ascii").strip()
    )
    rs = []
    for line in output.split("\n")[1:]:
        job_id, state, submit_time, start_time, slurm_name, nodelist, *_ = line.split(
            delimiter
        )
        rs.append(
            JobInfo(
                name=slurm_name,
                state=STATUS_MAPPING[state],
                host=nodelist,
                submit_time=submit_time,
                start_time=start_time,
                slurm_id=job_id.strip(),
            )
        )
    return rs


def cancel_jobs(
    slurm_names: Optional[List[str]] = None,
    slurm_ids: Optional[List[str]] = None,
    signal: Literal["SIGINT", "SIGKILL"] = "SIGKILL",
):
    assert (
        slurm_names is not None or slurm_ids is not None
    ), "Must specify slurm_names or slurm_ids."
    assert not (
        slurm_names and slurm_ids
    ), "Cannot specify both slurm_names and slurm_ids."
    cmd = ["scancel", "-s", signal]
    if slurm_names is not None:
        cmd += ["-n", ",".join(slurm_names)]
    elif slurm_ids is not None:
        cmd += ["-j", ",".join([str(s) for s in slurm_ids])]
    subprocess.check_call(cmd)
    logger.info(
        f"Cancelled Slurm job with signal {signal}: "
        f"slurm identifiers {slurm_names if slurm_ids is None else slurm_ids}"
    )


def _parse_output_status_line(status):
    assert status.startswith("State=")
    status = status.split(" ")[0]
    status = status.split("=")[1]
    return status.split("+")


def _parse_output_tres_line(tres):
    tres = tres.split("=", maxsplit=1)[1]
    tres = tres.split(",")
    res = SlurmResource()
    if len(tres) == 0 or (len(tres) == 1 and tres[0] == ""):
        return SlurmResource()
    for t in tres:
        if t.startswith("mem"):
            if t.endswith("M"):
                res.mem = int(t.split("=")[1].strip("M"))
            elif t.endswith("G"):
                res.mem = int(float(t.split("=")[1].strip("G")) * 1024)
            elif t.endswith("T"):
                res.mem = int(float(t.split("=")[1].strip("T")) * 1024 * 1024)
            else:
                raise ValueError("Unknown memory unit.")
        elif t.startswith("cpu"):
            res.cpu = int(t.split("=")[1])
        elif t.startswith("gres/gpu"):
            prefix, sgpu = t.split("=")
            res.gpu = int(sgpu)
        elif t.startswith("gres/ppu"):
            prefix, sgpu = t.split("=")
            res.gpu = int(sgpu)
        elif t.startswith("billing"):
            # slurm default resource to limit number of
            # tasks in one node
            pass
        else:
            raise NotImplementedError(f"Unknown resource type: {repr(t)}")
    return res


def available_hostnames(
    nodelist: Optional[str] = None,
    exclude: Optional[str] = None,
    partition: Optional[str] = None,
) -> List[str]:
    sinfo_cmd = 'sinfo -o "%N" --noheader'
    if partition:
        sinfo_cmd += f" --partition={partition}"
    all_nodelist: str = (
        subprocess.check_output(sinfo_cmd, shell=True).decode("utf-8").strip()
    )
    all_hostnames: List[str] = (
        subprocess.check_output(
            [
                "scontrol",
                "show",
                "hostnames",
                all_nodelist,
            ]
        )
        .decode("utf-8")
        .strip()
        .split("\n")
    )

    if nodelist is not None:
        valid_hostnames: List[str] = (
            subprocess.check_output(
                [
                    "scontrol",
                    "show",
                    "hostnames",
                    nodelist,
                ]
            )
            .decode("utf-8")
            .strip()
            .split("\n")
        )
    else:
        valid_hostnames = all_hostnames

    if exclude is not None:
        excluded_hostnames: List[str] = (
            subprocess.check_output(
                [
                    "scontrol",
                    "show",
                    "hostnames",
                    exclude,
                ]
            )
            .decode("utf-8")
            .strip()
            .split("\n")
        )
        for hn in excluded_hostnames:
            if hn in valid_hostnames:
                valid_hostnames.remove(hn)

    invalid_hostnames = []
    for hn in valid_hostnames:
        if hn not in all_hostnames:
            logger.warning(
                f"Invalid host name: {hn}. Maybe it is not in this partition/cluster."
            )
            invalid_hostnames.append(hn)

    for hn in invalid_hostnames:
        valid_hostnames.remove(hn)

    return valid_hostnames


def get_all_node_resources() -> Dict[str, SlurmResource]:
    """Execute `scontrol show node` to get all node resources available in the
    slurm cluster.

    Return a list of SlurmResource
    """
    o = subprocess.check_output(["scontrol", "show", "node"]).decode("utf-8")
    nodes = o.split("\n\n")
    all_rres = {}
    for node in nodes:
        if len(node) <= 1:
            continue
        ls = node.split("\n")
        node_name = ls[0].split(" ")[0].split("=")[1]
        ctres = SlurmResource()
        atres = SlurmResource()
        for l in ls:
            l = l.strip("\n").strip()
            if l.startswith("State"):
                status = _parse_output_status_line(l)
                if any(
                    x in status
                    for x in ["DOWN", "DRAIN", "NOT_RESPONDING", "COMPLETING"]
                ):
                    break
            if l.startswith("CfgTRES"):
                ctres = _parse_output_tres_line(l)
            if l.startswith("AllocTRES"):
                atres = _parse_output_tres_line(l)
        rres = ctres - atres
        if rres.valid():
            all_rres[node_name] = rres
        else:
            all_rres[node_name] = SlurmResource()

    return all_rres


def resource_to_string(resources: Dict[str, SlurmResource]) -> str:
    resource_list = [
        {
            **{"NodeName": k},
            **{
                field.name: getattr(r, field.name)
                for field in r.__dataclass_fields__.values()
            },
        }
        for k, r in resources.items()
    ]
    return pd.DataFrame(resource_list).to_string(index=False)


def allocate_resources(
    infos: List[SlurmLaunchInfo],
    strategy: Literal["empty_first", "allocated_first"] = "empty_first",
) -> List[SlurmLaunchInfo]:
    """Allocate all slurm task specs, fill in the hostfile field of the specs.

    All slurm tasks are scheduled in pack. There are two choices of allocating
    strategies. The first is `empty_first`, which means we first allocate
    tasks to nodes with more free resources. The second is `allocated_first`,
    which allocate tasks to nodes with less free resources without exceeding
    resource capacity.
    """
    assert strategy in ["empty_first", "allocated_first"]
    all_resources = get_all_node_resources()
    # sorted by requirements in descending order
    infos = sorted(
        infos, key=lambda x: x.n_jobsteps * x.resource_requirement, reverse=True
    )
    prioritized_hosts = set()
    if len(infos) == 0:
        return infos
    cluster_config = infos[0].args.cluster
    for info_idx, info in enumerate(infos):
        valid_hostnames = available_hostnames(
            nodelist=info.nodelist,
            exclude=info.exclude,
            partition=info.partition,
        )
        valid_hostnames = list(filter(lambda x: x in all_resources, valid_hostnames))
        prioritized_resources = {
            hn: all_resources[hn] for hn in valid_hostnames if hn in prioritized_hosts
        }
        other_resources = {
            hn: all_resources[hn]
            for hn in valid_hostnames
            if hn not in prioritized_hosts
        }
        # sorted by available resources according to chosen strategy
        prioritized_resources = sorted(
            prioritized_resources.items(),
            key=lambda x: x[1],
            reverse=strategy != "allocated_first",
        )
        # if all of the allocated nodes cannot satisfy the requirement,
        # find the new node according to chosen strategy
        other_resources = sorted(
            other_resources.items(),
            key=lambda x: x[1],
            reverse=strategy != "allocated_first",
        )
        valid_resources = prioritized_resources + other_resources
        task_left = info.n_jobsteps
        allocated = dict()
        for hostname, resource in valid_resources:
            tmp = task_left
            while task_left > 0:
                # In current slurm cluster GRES setting,
                # we can only allocate tasks in the granularity of nodes
                # (16 PPUs/8 GPUs by default)
                batched_requirement = info.resource_requirement
                batched_ntasks = 1
                gpu_per_task = info.resource_requirement.gpu
                if gpu_per_task > 0:
                    assert (
                        task_left * gpu_per_task % cluster_config.n_gpus_per_node == 0
                    ), (task_left, gpu_per_task)
                    assert (
                        cluster_config.n_gpus_per_node % gpu_per_task == 0
                    ), gpu_per_task
                    batched_ntasks = int(cluster_config.n_gpus_per_node // gpu_per_task)
                    batched_requirement = batched_ntasks * info.resource_requirement
                try:
                    resource = resource - batched_requirement
                except InvalidGPUTypeException:
                    # InvalidGPUTypeException will be raised when
                    # `resource` and `batched_requirement`
                    # do not have the same GPU type.
                    break
                if not resource.valid():
                    resource += batched_requirement
                    break
                task_left -= batched_ntasks
                prioritized_hosts.add(hostname)
            if tmp - task_left > 0:
                allocated[hostname] = tmp - task_left
            all_resources[hostname] = resource
        if task_left > 0:
            if cluster_config.gpu_type == "ppu" and info.resource_requirement.gpu > 0:
                logger.warning(
                    "For PPU resources, we can only allocate tasks in the "
                    f"granularity of nodes ({cluster_config.n_gpus_per_node} PPUs)"
                )
            logger.warning(
                f'Unable to allocate {info.n_jobsteps} Jobs with name "{info.slurm_name}". '
                f"Resource Requirement of this job is: {dataclasses.asdict(info.resource_requirement)}. "
                f"Valid resources for this job is "
                f"(according to NodeList={info.nodelist}, "
                f"and Exclude={info.exclude}):\n {resource_to_string({k: v for k, v in get_all_node_resources().items() if k in valid_hostnames})}"
            )
            for pinfo in infos[:info_idx]:
                if (
                    len(
                        set(pinfo.hostfile_content.split("\n")).intersection(
                            set(valid_hostnames)
                        )
                    )
                    == 0
                ):
                    continue
                palloc = collections.defaultdict(lambda: 0)
                for _n in pinfo.hostfile_content.split("\n"):
                    palloc[_n] += 1
                logger.warning(
                    f'Found previous job "{pinfo.slurm_name}" (ntasks={pinfo.n_jobsteps}) '
                    f"has been allocated to the same set of nodes. "
                    f"Resource requirement of this job is: {dataclasses.asdict(pinfo.resource_requirement)}, "
                    f"allocation of this job is {dict(palloc)}."
                )
            raise SlurmResourceNotEnoughException()
        hostlist = []
        for hostname, task_num in allocated.items():
            hostlist += [hostname] * task_num
        info.hostfile_content = "\n".join(hostlist)
    return infos


def show_tesla():
    all_rres = get_all_node_resources()
    hostname = socket.gethostname()
    for k in available_hostnames():
        print(k, all_rres[k])


def show_all():
    all_rres = get_all_node_resources()
    for k, v in all_rres.items():
        print(k, v)


if __name__ == "__main__":
    show_all()
