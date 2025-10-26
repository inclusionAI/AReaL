import uuid
from collections import defaultdict
from typing import Dict, List

from areal.api.alloc_mode import AllocationMode
from areal.api.cli_args import (
    BaseExperimentConfig,
    ClusterSpecConfig,
    LauncherConfig,
    RecoverConfig,
    SGLangConfig,
    to_structured_cfg,
)
from areal.api.scheduler_api import Job, Scheduler, Worker
from areal.launcher.local import LocalLauncher
from areal.scheduler.rpc.rpc_client import RPCClient
from areal.utils import logging, name_resolve, names
from areal.utils.launcher import get_env_vars
from areal.utils.network import find_free_ports, gethostip
from areal.utils.recover import check_if_recover

logger = logging.getLogger("LocalScheduler")


class LocalScheduler(Scheduler):
    def __init__(self, config: BaseExperimentConfig):
        self.procs = []  # Store subprocess objects
        self.engine_workers: Dict[str, List[str]] = defaultdict(
            list
        )  # role -> [worker_id]
        self.rpc_client = RPCClient()
        self.launcher = LocalLauncher(
            config.experiment_name, config.trial_name, config.cluster.fileroot
        )
        self.config = config

    def create_workers(self, job: Job, *args, **kwargs) -> None:
        config = kwargs.get("config")
        if job.role == "rollout":
            self._create_rollout_workers(job, config)
            return None

        replicas = job.replicas
        master_port = find_free_ports(1, port_range=(10000, 50000))[0]

        for index in range(replicas):
            for task in job.tasks:
                ports = find_free_ports(task.port_count, port_range=(10000, 50000))
                envs = get_env_vars(
                    self.config.cluster.cluster_name,
                )
                extra_envs = task.env_vars if task.env_vars else {}
                extra_envs["PORT_LIST"] = ",".join(map(str, ports))
                envs.update(extra_envs)

                if job.role != "rollout":
                    # For non-rollout workers, set RANK and WORLD_SIZE
                    envs.update(
                        {
                            "RANK": index,
                            "LOCAL_RANK": 0,
                            "WORLD_SIZE": replicas,
                            "MASTER_ADDR": "localhost",
                            "MASTER_PORT": master_port,
                            "NCCL_CUMEM_ENABLE": "0",
                            "NCCL_NVLS_ENABLE": "0",
                        }
                    )
                self.launcher.submit(
                    job_name=f"{job.role}_worker",
                    cmd=f"{task.cmd} --role {job.role} --index {index}",
                    gpu=task.gpu,
                    env_vars=envs,
                )

                if task.type == "worker":
                    worker_id = f"worker_{uuid.uuid4().hex[:8]}"
                    self.rpc_client.register(worker_id, "localhost", ports[0])
                    self.engine_workers.setdefault(job.role, []).append(worker_id)

            logger.info(f"Submitted {job.replicas} tasks for command: {task.cmd}")
        return None

    def _create_rollout_workers(self, job: Job, config):
        config.launcher = to_structured_cfg(config.launcher, LauncherConfig)
        config.recover = to_structured_cfg(config.recover, RecoverConfig)
        config.cluster = to_structured_cfg(config.cluster, ClusterSpecConfig)
        is_recover_run = check_if_recover(config.recover, run_id=0)
        name_resolve.reconfigure(config.cluster.name_resolve)
        name_resolve.clear_subtree(
            names.trial_root(
                experiment_name=config.experiment_name, trial_name=config.trial_name
            )
        )
        alloc_mode = AllocationMode.from_str(config.allocation_mode)

        if job.role == "rollout":
            if alloc_mode.gen_backend == "sglang":
                server_cmd = []
                server_addrs = []
                base_seed = config.sglang.random_seed
                config.sglang = to_structured_cfg(config.sglang, SGLangConfig)
                # each sglang need 2 ports
                ports = find_free_ports(
                    alloc_mode.gen.dp_size * 2, port_range=(10000, 50000)
                )
                host_ip = gethostip()
                host = "localhost" if not config.sglang.enable_metrics else host_ip
                for i in range(alloc_mode.gen.dp_size):
                    config.sglang.random_seed = base_seed + i
                    cmd = SGLangConfig.build_cmd(
                        config.sglang,
                        host=host,
                        tp_size=alloc_mode.gen.tp_size,
                        base_gpu_id=0,
                        port=ports[i * 2],
                        dist_init_addr=f"localhost:{ports[i * 2 + 1]}",
                    )
                    server_cmd.append(cmd)
                    server_addrs.append(f"{host}:{ports[i * 2]}")

                # Launch inference servers.
                self.launcher.submit_array(
                    job_name="rollout_server",
                    cmd=server_cmd,
                    count=alloc_mode.gen.dp_size,
                    gpu=alloc_mode.gen.pp_size * alloc_mode.gen.tp_size,
                    env_vars=get_env_vars(
                        config.cluster.cluster_name,
                        config.launcher.inference_server_env_vars,
                    ),
                )
                logger.info(
                    f"LLM inference server launched at: AREAL_LLM_SERVER_ADDRS={','.join(server_addrs)}"
                )

            task = next((task for task in job.tasks if task.type == "worker"), None)
            if task is not None:
                for i in range(job.replicas):
                    ports = find_free_ports(task.port_count, port_range=(10000, 50000))
                    envs = get_env_vars(
                        config.cluster.cluster_name,
                    )
                    extra_envs = task.env_vars if task.env_vars else {}
                    extra_envs["PORT_LIST"] = ",".join(map(str, ports))
                    extra_envs["AREAL_LLM_SERVER_ADDRS"] = ",".join(server_addrs)
                    extra_envs["AREAL_RECOVER_RUN"] = str(int(is_recover_run))
                    envs.update(extra_envs)
                    self.launcher.submit(
                        job_name="rollout_worker",
                        cmd=f"{task.cmd} --role {job.role} --index {i}",
                        gpu=task.gpu,
                        env_vars=envs,
                    )
                    worker_id = f"worker_{uuid.uuid4().hex[:8]}"
                    self.rpc_client.register(worker_id, "localhost", ports[0])
                    self.engine_workers.setdefault(job.role, []).append(worker_id)
                logger.info(f"Submitted {job.replicas} tasks for command: {task.cmd}")

    def get_workers(self, worker_role, timeout: float = 60.0) -> List[Worker]:
        workers = []
        for worker_id in self.engine_workers.get(worker_role, []):
            if not self.rpc_client.check_health(worker_id, timeout):
                raise TimeoutError(f"Worker {worker_id} check health timeout")
            ip, port = self.rpc_client.get_info(worker_id)
            worker = Worker(id=worker_id, ip=ip, worker_ports=[str(port)])
            workers.append(worker)
        return workers

    def delete_workers(self):
        raise NotImplementedError("LocalScheduler does not support delete_workers")

    # Other methods remain the same
    def create_engine(self, worker_id, engine_obj, *args, **kwargs):
        # launch engine rpc server on the worker
        self.rpc_client.create_engine(worker_id, engine_obj, *args, **kwargs)

    def call_engine(self, worker_id, method, *args, **kwargs):
        ret = self.rpc_client.call_engine(worker_id, method, 3, *args, **kwargs)
        return ret
