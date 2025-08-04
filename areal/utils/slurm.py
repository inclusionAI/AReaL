import subprocess
from typing import List, Literal, Optional

from realhf.base import logging
from realhf.scheduler.client import JobInfo, JobState

logger = logging.getLogger("Slurm Utils")


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

SBATCH_SCRIPT_TEMPLATE = """#!/bin/bash
{sbatch_options}

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
echo nodes=$nodes

nodes_array=($nodes)
echo node_array=$nodes_array

head_node=${{nodes_array[0]}}
echo head_node=$head_node

# Getting the head node IP address
head_node_ip=$(srun {srun_additional_args} --nodes=1 --ntasks=1 -n1 -c1 --mem=10M --nodelist="$head_node" hostname --ip-address)
echo head_node_ip=$head_node_ip

# Find a free port on the head node
# Wonderful linux command to find a random free port (between 10000 and 60000) by deepseek
trainer_port=$(srun {srun_additional_args} --nodes=1 --ntasks=1 -n1 -c1 --mem=10M --nodelist="$head_node" bash -c "comm -23 <(seq 10000 60000 | sort) <(ss -tan | awk '{{print $4}}' | cut -d':' -f2 | grep '[0-9]\\{{1,5\\}}' | sort -u) | shuf | head -n 1")
echo trainer_port=$trainer_port

# srun commands
{srun_cmds}

wait
"""

SRUN_CMD_TEMPLATE: str = """srun {additional_args} \\
    --nodelist=${{nodes_array[{node_id}]}} --nodes={nodes} --ntasks={ntasks} \\
    --gres=gpu:{n_gpus_per_node} --cpus-per-task={cpus_per_task} --mem-per-cpu={mem_per_cpu}M \\
    {cmd} &
"""

APPTAINER_CMD_TEMPLATE: str = """singularity exec --no-home --writable-tmpfs --nv --pid \\
    --bind {container_mounts} \\
    {container_env_strings} \\
    {container_image} \\
    {cmd}"""


def cancel_jobs(
    slurm_names: Optional[List[str]] = None,
    slurm_ids: Optional[List[int]] = None,
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
        cmd += [",".join(str(s) for s in slurm_ids)]
    subprocess.check_call(cmd)
    logger.info(
        f"Cancelled Slurm job with signal {signal}: "
        f"slurm identifiers {slurm_names if slurm_ids is None else slurm_ids}. CMD: {cmd}"
    )


def query_jobs(
    slurm_names: Optional[List[str]] = None,
    slurm_ids: Optional[List[int]] = None,
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
                slurm_id=int(job_id.strip()),
            )
        )
    return rs


def parse_slurm_nodelist(nodelist: str) -> List[str]:
    return (
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


def get_slurm_host_ip(node: str, srun_addtional_args: str):
    try:
        cmd = f"srun {srun_addtional_args} --immediate=1 --nodes=1 --ntasks=1 -n1 -c1 --mem=10M --nodelist={node} hostname --ip-address"
        return subprocess.check_output(cmd.split(" ")).decode("utf-8").strip()
    except subprocess.CalledProcessError:
        logger.warning(f"Get slurm host IP for node {node} failed.")
