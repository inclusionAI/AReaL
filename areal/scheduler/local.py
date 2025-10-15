import getpass
import os
import re
import signal as signal_module
import subprocess
import time
import uuid
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import psutil

from areal.api.alloc_mode import AllocationMode, AllocationType
from areal.api.cli_args import (
    ClusterSpecConfig,
    LauncherConfig,
    RecoverConfig,
    SGLangConfig,
    to_structured_cfg,
)
from areal.api.scheduler_api import Scheduler, Worker
from areal.platforms import current_platform
from areal.scheduler.rpc.rpc_client import RPCClient
from areal.scheduler.rpc.rpc_server import build_rpc_server_start_command
from areal.utils import logging, name_resolve, names
from areal.utils.launcher import JobException, JobInfo, JobState, get_env_vars
from areal.utils.network import find_free_ports, gethostip
from areal.utils.recover import check_if_recover
from areal.launcher.local import LocalLauncher

logger = logging.getLogger("LocalScheduler")

class LocalScheduler(Scheduler):
    def __init__(self, config):
        self.procs = []  # Store subprocess objects
        self.engine_workers: Dict[str, List[str]] = defaultdict(
            list
        )  # role -> [worker_id]
        self.rpc_client = RPCClient()
        self.launcher = LocalLauncher(
            config.experiment_name, config.trial_name, config.cluster.fileroot
        )

    def create_workers(self, worker_role, config, *args, **kwargs) -> None:
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
        logger.info(
            f"experiment_name={config.experiment_name}, "
            f"trial_name={config.trial_name}, fileroot={config.cluster.fileroot}, "
            f"is_recover_run={is_recover_run}"
        )

        server_cmd = []
        server_addrs = []
        if worker_role == "rollout":
            if alloc_mode.gen_backend == "sglang":
                # launch sglang servers
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
                        dist_init_addr=f"localhost:{ports[i*2+1]}",
                    )
                    server_cmd.append(cmd)
                    server_addrs.append(f"{host}:{ports[i * 2]}")

                # Launch inference servers.
                self.launcher.submit_array(
                    job_name="llm_server",
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

                # create rpc server workers
                worker_ports = find_free_ports(
                    alloc_mode.gen.world_size, port_range=(10000, 50000)
                )  # each sglang need 2 ports
                for i in range(alloc_mode.gen.world_size):
                    cmd = build_rpc_server_start_command(worker_ports[i])

                    self.launcher.submit(
                        job_name="rollout_worker",
                        cmd=cmd,
                        gpu=0,
                        env_vars=dict(
                            **get_env_vars(
                                config.cluster.cluster_name,
                                # config.launcher.worker_env_vars,
                            ),
                            AREAL_LLM_SERVER_ADDRS=server_addrs[
                                i % alloc_mode.gen.dp_size
                            ],
                            AREAL_RECOVER_RUN=str(int(is_recover_run)),
                        ),
                    )

                    logger.info(
                        f"RPC server for rollout worker launched at port: {worker_ports[i]}"
                    )

                    worker_id = f"rollout_{i}_{uuid.uuid4().hex[:8]}"
                    self.rpc_client.register(worker_id, "localhost", worker_ports[i])
                    self.engine_workers.setdefault(worker_role, []).append(worker_id)

            else:
                raise NotImplementedError(f"Unsupported allocation mode: {alloc_mode}")
        elif worker_role == "actor":
            if alloc_mode.type_ == AllocationType.DECOUPLED_EVAL:
                gpu = 0
                nprocs = 1
            else:
                gpu = nprocs = alloc_mode.train.world_size

            worker_ports = find_free_ports(alloc_mode.gen.world_size, (10000, 50000))

            self.launcher.submit(
                job_name="trainer",
                cmd=f"torchrun --nnodes 1 --nproc-per-node {nprocs} "
                f"--master-addr localhost --master-port {find_free_ports(1, (10000, 50000))[0]} "
                f"-m areal.scheduler.rpc.rpc_server --rpc_ports {','.join(map(str, worker_ports))}",
                gpu=gpu,
                env_vars=dict(
                    **get_env_vars(
                        config.cluster.cluster_name,
                        config.launcher.trainer_env_vars,
                    ),
                    # AREAL_LLM_SERVER_ADDRS=",".join(server_addrs), # not need?
                    AREAL_RECOVER_RUN=str(int(is_recover_run)),
                ),
            )

            for i in range(alloc_mode.gen.world_size):
                worker_id = f"actor_{i}_{uuid.uuid4().hex[:8]}"
                self.rpc_client.register(worker_id, "localhost", worker_ports[i])
                self.engine_workers.setdefault(worker_role, []).append(worker_id)
        else:
            raise ValueError(f"Unknown worker role: {worker_role}")

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
