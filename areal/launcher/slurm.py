import getpass
import os
import re
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

import realhf.base.logging as logging
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
from areal.utils.slurm import (
    APPTAINER_CMD_TEMPLATE,
    SBATCH_SCRIPT_TEMPLATE,
    SRUN_CMD_TEMPLATE,
    cancel_jobs,
    query_jobs,
)
from realhf.base import logging, name_resolve, names
from realhf.scheduler.client import JobException, JobInfo, JobState

logger = logging.getLogger("SlurmLauncher")

SLURM_WAIT_CHECK_TIME_INTERVAL = 5


class SlurmLauncher:
    def __init__(
        self, experiment_name: str, trial_name: str, fileroot: str, container_type: str
    ):
        self.experiment_name = experiment_name
        self.trial_name = trial_name
        self.fileroot = fileroot
        self.container_type = container_type

        # slurm_job_id -> JobInfo
        self.jobs: Dict[int, JobInfo] = {}
        self.job_names = []

    @property
    def run_name(self) -> str:
        """Returns the run name of this launcher."""
        return f"{self.experiment_name}_{self.trial_name}"

    def slurm_name(self, job_name: str) -> str:
        """Returns the slurm name of a job."""
        return f"{self.experiment_name}_{self.trial_name}:{job_name}"

    def log_path_of(self, job_name: str) -> str:
        log_path = f"{self.fileroot}/logs/{getpass.getuser()}/{self.experiment_name}/{self.trial_name}"
        os.makedirs(log_path, exist_ok=True)
        return os.path.join(log_path, f"{job_name}.log")

    def sbatch_path_of(self, job_name: str) -> str:
        sbatch_path = f"{self.fileroot}/logs/{getpass.getuser()}/{self.experiment_name}/{self.trial_name}"
        os.makedirs(sbatch_path, exist_ok=True)
        return os.path.join(sbatch_path, f"{job_name}.sh")

    def submit(self, job_name, cmd, **kwargs):
        """Submits and launch a job with SBATCH.

        Args:
            cmd (str or List[str]): The core command to be executed.
        """
        return self.submit_array(job_name, cmd, count=1, **kwargs)

    def find_job_id(self, job_name: str):
        job_name = self.slurm_name(job_name)
        for job_id, job_info in self.jobs.items():
            if job_info.name == job_name:
                return job_id
        return None

    def submit_array(
        self,
        job_name: str,
        cmd: List[str] | str,
        count: int,
        nodes: int,
        n_gpus_per_node: int,
        cpus_per_task: int,
        mem_per_task: int,  # MB
        container_image: str,
        srun_additional_args: str = "",
        container_mounts: Optional[str] = None,
        env_vars: Optional[Dict] = None,
        nodelist: Optional[str] = None,
        exclude: Optional[str] = None,
    ):
        """Submits and launch a job array with SBATCH.
        Note that a job array has one (unique) slurm name, and one (unique) slurm id.

        Args:
            job_name (str): The job name of the job array. The actual slurm name will be
                `<experiment_name>_<trial_name>:<job_name>`.
            cmd (str or List[str]): The core command to be executed.
            count (int): The number of jobs in the array.
        """
        assert job_name not in self.job_names, (
            f"Job {job_name} is already submitted. "
            "Please use a different job name or stop the existing job."
        )
        if isinstance(cmd, str):
            cmd = [cmd]
        assert len(cmd) == count, (
            f"Command length {len(cmd)} does not match the job count {count}. "
            "Please provide a command for each job in the array."
        )
        assert count % nodes == 0, (
            f"Job count {count} must be divisible by the number of nodes {nodes}. "
            "Please adjust the job count or the number of nodes."
        )
        ntasks_per_node = count // nodes
        assert n_gpus_per_node % ntasks_per_node == 0, (
            "GPUs must be evenly distributed across tasks. "
            f"Current #GPUs per node {n_gpus_per_node}, #tasks per node {ntasks_per_node}."
        )

        mem_per_cpu = mem_per_task // cpus_per_task  # MB per CPU
        mem_per_node = (
            mem_per_task * count // nodes + 1024 * 10
        )  # make sure slurm does not run out of resources

        sbatch_options = [
            f"--job-name={self.slurm_name(job_name)}",
            f"--output={self.log_path_of(job_name)}",
            "--open-mode=append",
            "--no-requeue",
            f"--nodes={nodes}-{nodes}",
            f"--ntasks-per-node={ntasks_per_node}",
            f"--gres=gpu:{n_gpus_per_node}",
            f"--cpus-per-task={cpus_per_task}",
            f"--mem={mem_per_node}M",
        ]

        if nodelist:
            sbatch_options.append(f"--nodelist={nodelist}")
        if exclude:
            sbatch_options.append(f"--exclude={exclude}")

        sbatch_options_str = "\n".join([f"#SBATCH {opt}" for opt in sbatch_options])

        if env_vars is None:
            env_vars = dict()
        n_gpus_per_task = n_gpus_per_node // ntasks_per_node
        assert (
            "CUDA_VISIBLE_DEVICES" not in env_vars
        ), "CUDA_VISIBLE_DEVICES should be automatically resolved by Launcher instead of manually assigned."

        srun_cmds = []
        for i in range(count):
            # resolve CUDA_VISIBLE_DEVICES for each task
            gpu_id_start = (i % ntasks_per_node) * n_gpus_per_task
            gpu_id_end = ((i % ntasks_per_node) + 1) * n_gpus_per_task
            node_id = i // ntasks_per_node
            _env_vars = {
                **env_vars,
                "CUDA_VISIBLE_DEVICES": ",".join(
                    str(x) for x in range(gpu_id_start, gpu_id_end)
                ),
            }
            # Prepare the command for each job in the array
            job_cmd = cmd[i]

            if self.container_type == "apptainer":
                env_string = " ".join(
                    "--env {}={}".format(k, v) for k, v in _env_vars.items()
                )
                apptainer_cmd = APPTAINER_CMD_TEMPLATE.format(
                    container_mounts=container_mounts or "",
                    container_env_strings=env_string,
                    container_image=container_image,
                    cmd=job_cmd,
                )
                srun_cmd = SRUN_CMD_TEMPLATE.format(
                    additional_args=srun_additional_args,
                    nodes=1,
                    ntasks=1,
                    node_id=node_id,
                    n_gpus_per_node=n_gpus_per_task,
                    cpus_per_task=cpus_per_task,
                    mem_per_cpu=mem_per_cpu,
                    cmd=apptainer_cmd,
                )
            elif self.container_type == "none":
                env_string = "--export=" + ",".join(
                    "{}={}".format(k, v) for k, v in _env_vars.items()
                )
                srun_additional_args = srun_additional_args + " " + env_string
                srun_cmd = SRUN_CMD_TEMPLATE.format(
                    additional_args=srun_additional_args,
                    nodes=1,
                    ntasks=1,
                    node_id=node_id,
                    n_gpus_per_node=n_gpus_per_task,
                    cpus_per_task=cpus_per_task,
                    mem_per_cpu=mem_per_cpu,
                    cmd=job_cmd,
                )
            else:
                raise ValueError(
                    f"Unsupported container type: {self.container_type}. "
                    "Supported types are 'apptainer' and 'none'."
                )
            srun_cmds.append(srun_cmd)

        srun_cmds = "\n".join(srun_cmds)
        sbatch_script = SBATCH_SCRIPT_TEMPLATE.format(
            sbatch_options=sbatch_options_str,
            srun_additional_args=srun_additional_args,
            srun_cmds=srun_cmds,
        )
        sbatch_file_path = self.sbatch_path_of(f"{job_name}")
        with open(sbatch_file_path, "w") as f:
            f.write(sbatch_script)

        # Submit the job
        try:
            output = (
                subprocess.check_output(["sbatch", sbatch_file_path])
                .decode("utf-8")
                .strip()
            )
            logger.info(
                f"Submitted Slurm job {self.slurm_name(job_name)} to scheduler. To check the output, run \n\t`tail -f {self.log_path_of(job_name)}`."
            )
        except subprocess.CalledProcessError as e:
            logger.warning(
                f"Failed to submit job {self.slurm_name(job_name)}. "
                f"For debugging, please make sure your sbatch command works "
                f"and check generated sbatch file on {sbatch_file_path}."
            )
            logger.error(f"Error message: {e}")
            return

        match = re.search(r"Submitted batch job (\d+)", output)
        slurm_job_id = int(match.group(1)) if match else None
        if slurm_job_id is None:
            logger.warning(
                f"Failed to obtain job id for job {self.slurm_name(job_name)}. "
                f"sbatch output: {output}"
            )
            return

        assert isinstance(slurm_job_id, int)
        self.jobs[slurm_job_id] = JobInfo(
            name=self.slurm_name(job_name),
            state=JobState.PENDING,
            slurm_id=slurm_job_id,
        )
        self._update_all()

    def stop(self, job_name, force=False):
        """Stops a running job.

        Raises exception if there is no such job, but passes if the job
        has stopped either successfully or not.

        Args:
            job_name: The job name of the job array to stop.
                The actual slurm job name will be `<experiment_name>_<trial_name>:<job_name>`.
        """
        signal = "SIGKILL" if force else "SIGTERM"
        job_id = self.find_job_id(job_name)
        if not job_id:
            return
        return cancel_jobs(slurm_ids=[job_id], signal=signal)

    def stop_all(self, force=False):
        """Stops all running jobs."""
        signal = "SIGKILL" if force else "SIGTERM"
        return cancel_jobs(slurm_ids=list(self.jobs.keys()), signal=signal)

    def find(self, job_name) -> JobInfo | None:
        """Gets the status of a job of this job.

        Args:
            job_name: The job name of the job array to find.
                The actual slurm job name will be `<experiment_name>_<trial_name>:<job_name>`.

        Returns:
            A JobInfo if the job is found, or None otherwise.
        """
        self._update_all()
        job_id = self.find_job_id(job_name)
        return self.jobs[job_id] if job_id else None

    def find_all(self, job_name_regex=".*") -> List[JobInfo]:
        """Finds jobs.

        Args:
            job_name_regex: job name regex.

        Returns:
            A list of found JobInfo.
        """
        self._update_all()
        infos = []
        for r in self.jobs.values():
            job_name = r.name.split(":")[-1]  # Extract the job name from slurm name
            if re.fullmatch(job_name_regex, job_name):
                infos.append(r)
        return infos

    def _find_job_with_status(
        self,
        status: List[JobState],
    ) -> List[JobInfo]:
        """Finds jobs with the given status.

        Args:
            status: A list of JobState to filter jobs.

        Returns:
            A list of JobInfo with the given status.
        """
        self._update_all()
        return [r for r in self.jobs.values() if r.state in status]

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
        """Waits until all jobs submitted via this client instance finish."""
        # begin wait
        deadline = None if timeout is None else time.time() + timeout

        num_jobs_left = len(self.jobs)
        left = list(self.jobs.keys())
        logger.info(
            f"Waiting for {num_jobs_left} jobs. Jobs IDs: "
            f"{','.join(sorted([str(x.slurm_id) for x in self.jobs.values()]))}."
        )
        while len(left) > 0:
            if len(left) < num_jobs_left:
                num_jobs_left = len(left)
                logger.info(f"Waiting for {num_jobs_left} jobs.")
            if deadline is not None and time.time() > deadline:
                raise TimeoutError(
                    f"Timeout waiting for {num_jobs_left} jobs. Job ID: "
                    f"{','.join(sorted([str(x.slurm_id) for x in self.jobs.values()]))}."
                )
            self._update_all()
            left = list(self.jobs.keys())
            for slurm_id in list(left):
                slurm_info = self.jobs[slurm_id]
                if slurm_info.slurm_id is None:
                    continue
                if slurm_info.state in check_status:
                    raise JobException(
                        run_name=self.run_name,
                        worker_type=slurm_info.name,
                        host=slurm_info.host,
                        reason=slurm_info.state,
                    )
                if slurm_info.state in remove_status:
                    logger.info(
                        f"Job {slurm_info.name} is {slurm_info.state}. (Removed)"
                    )
                    left.remove(slurm_id)
                    if update:
                        self.jobs.pop(slurm_info.slurm_id)
            time.sleep(SLURM_WAIT_CHECK_TIME_INTERVAL)

    def _update_all(self):
        """Updates the status of all jobs."""
        try:
            slurm_infos = query_jobs(slurm_ids=list(self.jobs.keys()))
            for slurm_info in slurm_infos:
                assert slurm_info.slurm_id is not None
                self.jobs[slurm_info.slurm_id] = slurm_info
        except subprocess.CalledProcessError:
            logger.warning(
                "Calling squeue failed. Check slurm manually if you continue to see this warning."
            )


def slurm_main():
    config, _ = parse_cli_args(sys.argv[2:])
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

    launcher = SlurmLauncher(
        experiment_name=config.experiment_name,
        trial_name=config.trial_name,
        fileroot=config.cluster.fileroot,
        container_type=config.launcher.slurm.container_type,
    )
    allocation_mode = config.allocation_mode
    allocation_mode = AllocationMode.from_str(allocation_mode)
    sglang_cmds = []
    sglang_addrs = []
    n_sglang_nodes = 0
    if allocation_mode.type_ == AllocationType.DECOUPLED_SGLANG:
        # Launcher should launch SGLang servers according to allocation mode.
        sglang_tp_size = allocation_mode.gen_tp_size
        n_sglang_servers = allocation_mode.gen_dp_size
        n_sglang_nodes = allocation_mode.gen_world_size // n_gpus_per_node

        base_seed = config.sglang.random_seed
        sglang_server_cmd_template = f"python3 -m areal.launcher.sglang_server {' '.join(sys.argv[2:])} sglang.random_seed={{seed}}"
        for i in range(n_sglang_servers):
            sglang_cmd = sglang_server_cmd_template.format(
                seed=base_seed + i,
            )
            sglang_cmds.append(sglang_cmd)

        launcher.submit_array(
            job_name="llm_server",
            cmd=sglang_cmds,
            count=n_sglang_servers,
            nodes=n_sglang_nodes,
            n_gpus_per_node=config.cluster.n_gpus_per_node,
            cpus_per_task=config.launcher.inference_server_cpus_per_gpu
            * sglang_tp_size,
            mem_per_task=config.launcher.inference_server_mem_per_gpu * sglang_tp_size,
            srun_additional_args=config.launcher.slurm.srun_additional_args,
            container_image=config.launcher.slurm.inference_server_image,
            container_mounts=config.launcher.slurm.mount,
            env_vars=get_env_vars(
                config.cluster.cluster_name,
                config.launcher.inference_server_env_vars,
            ),
        )
        # Get SGLang server addresses by name resolve
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
    # Here $head_node_ip is the IP address of the first node in the job array.
    # $trainer_port is a free port on the head node.
    # Both of them are obtained in by the SBATCH script.
    trainer_cmd_template = (
        f"torchrun --nnodes={{nnodes}} --nproc-per-node={{nproc_per_node}} --node-rank {{node_rank}} "
        f"--master-addr $head_node_ip --master-port $trainer_port {' '.join(sys.argv[1:])}"
    )

    trainer_cmds = []
    for i in range(trainer_n_nodes):
        # In slurm, we launch trainer in the granularity of nodes with torchrun command.
        trainer_cmds.append(
            trainer_cmd_template.format(
                nnodes=trainer_n_nodes,
                nproc_per_node=config.cluster.n_gpus_per_node,
                node_rank=i,
            )
        )

    if not config.server_only:
        # launch trainers
        launcher.submit_array(
            job_name="trainer",
            cmd=trainer_cmds,
            count=trainer_n_nodes,
            nodes=trainer_n_nodes,
            n_gpus_per_node=config.cluster.n_gpus_per_node,
            cpus_per_task=config.launcher.trainer_cpus_per_gpu
            * config.cluster.n_gpus_per_node,
            mem_per_task=config.launcher.trainer_mem_per_gpu
            * config.cluster.n_gpus_per_node,
            container_image=config.launcher.slurm.trainer_image,
            srun_additional_args=config.launcher.slurm.srun_additional_args,
            container_mounts=config.launcher.slurm.mount,
            env_vars=dict(
                **get_env_vars(
                    config.cluster.cluster_name,
                    config.launcher.trainer_env_vars,
                ),
                AREAL_LLM_SERVER_ADDRS=",".join(sglang_addrs),
            ),
        )

    try:
        launcher.wait(
            check_status=(
                JobState.CANCELLED,
                JobState.FAILED,
                JobState.NOT_FOUND,
                JobState.COMPLETED,
            ),
            remove_status=(),
        )
    except (KeyboardInterrupt, JobException, TimeoutError) as e:
        launcher.stop_all(force=True)
        raise e


if __name__ == "__main__":
    # usage: python -m areal.launcher.slurm <entry_point> \
    #  --config <config_path> [<additional_args>]
    slurm_main()
