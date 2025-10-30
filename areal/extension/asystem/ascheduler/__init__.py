import json
import os
import signal
import time
from dataclasses import asdict
from datetime import datetime
from typing import Any

import requests

from realhf.base import logging

from areal.api.scheduler_api import Job, Scheduler, Worker
from areal.extension.asystem.ascheduler.rpc_client import RPCClient
from areal.utils.errors import FrameworkError

logger = logging.getLogger(__name__)

# Default Asystem API endpoints
DEFAULT_ASYSTEM_SERVER_URL = "http://127.0.0.1:8081"
API_BASE_PATH = "/api/v1"
SCHEDULER_WAIT_CHECK_TIME_INTERVAL = 5  # Seconds

default_envs = {
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "USE_MAX_V2": "1",
    "DISCOVERY_CONFIG_CENTER_TYPE": "FILE",
    "NCCL_CUMEM_ENABLE": "0",
    "NCCL_NVLS_ENABLE": "0",
    "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
    "NCCL_DEBUG": "WARNING",
    "NVTE_FUSED_ATTN": "0",
    "NCCL_MAX_NCHANNELS": "16",
    "NCCL_DEBUG_SUBSYS": "INIT,TUNING,GRAPH",
    "USE_AREAL_LITE": "1",
    "NCCL_SOCKET_IFNAME": "bond0",
    "GLOO_SOCKET_IFNAME": "eth0",
    "NCCL_NET_PLUGIN": "",
    "NCCL_IB_GID_INDEX": "3",
    "NCCL_IB_TIMEOUT": "22",
    "NCCL_IB_RETRY_CNT": "7",
    "NCCL_IB_SL": "5",
    "NCCL_IB_TC": "136",
    "NCCL_IB_HCA": "mlx5_bond",
    "NCCL_SET_THREAD_NAME": "1",
    "NCCL_IB_QPS_PER_CONNECTION": "8",
}


class SchedulerError(Exception):
    """Scheduler related errors"""

    pass


class AsystemScheduler(Scheduler):
    """
    AsystemScheduler implementation that follows the same interface as LocalScheduler.
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.real_package_path = os.environ.get("REAL_PACKAGE_PATH", "")
        assert self.real_package_path != ""
        self.worker_infos = []
        self.submitted_jobs = {}  # job_name -> job_uid
        self.extra_envs = config.get("extra_envs", {})

        # Initialize Asystem client attributes
        self.expr_name = config.get("expr_name", "default_expr")
        self.trial_name = config.get("trial_name", "default_trial")
        self.storage_prefix = config.get("storage_prefix", "")
        self.run_name = f"{self.expr_name}_{self.trial_name}"
        self.endpoint = config.get("endpoint", DEFAULT_ASYSTEM_SERVER_URL).rstrip("/")
        self.api_url = self.endpoint + API_BASE_PATH
        self._session = requests.Session()
        self._submission_counter = 0
        self._worker_registry = {}  # worker_id -> worker_info mapping
        self._addrs = {}
        self.n_gpus_per_node = int(config.get("n_gpus_per_node", "8"))
        # Initialize RPCClient for create_engine and call_engine
        self.rpc_client = RPCClient()
        # 每次实验生成一个uuid
        self.uuid = str(time.time_ns())
        self.timestamp = time.strftime("%Y_%m_%d_%H_%M_%S")
        self.extra_envs["CLUSTER_NAME"] = (
            f"{self.expr_name}_{self.trial_name}_{self.uuid}"
        )
        # 设置 env，用于 antlogs 日志收集
        self.extra_envs["EXPR_NAME"] = f"{self.expr_name}"
        self.extra_envs["TRIAL_NAME"] = f"{self.trial_name}"
        # 设置 env，用于区分 Failover 前后的不同实验
        self.extra_envs["UUID"] = f"{self.timestamp}"
        self.extra_envs["LOG_DIR"] = (
            f"{self.storage_prefix}/experiments/logs/root/{self.expr_name}/{self.trial_name}"
        )
        self.extra_envs["REAL_PACKAGE_PATH"] = self.real_package_path

        # 信号捕获是为了手动跑 trainer.py 增加的功能，用来在 trainer 退出时清理相关 Job
        signal.signal(signal.SIGINT, self.batch_cleanup_jobs)
        signal.signal(signal.SIGTERM, self.batch_cleanup_jobs)

        logger.info(
            f"AsystemScheduler initialized for {self.run_name}. API URL: {self.api_url}"
        )

    def batch_cleanup_jobs(self, signum):
        logger.info(f"signum {signum} received: handle_signals starts")
        for job_name, job_uid in self.submitted_jobs.items():
            try:
                self.stop_job(job_uid)
                logger.info(f"Stopped job: {job_uid}")
            except Exception as e:
                logger.error(f"Error stopping job {job_uid}: {e}")
        logger.info(f"signum {signum} received: handle_signals finished")

        # 重新发送信号并按照默认行为处理
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    def cleanup_jobs(self):
        for job_name, job_uid in self.submitted_jobs.items():
            try:
                self.stop_job(job_uid)
                logger.info(f"Stopped job: {job_uid}")
            except Exception as e:
                logger.error(f"Error stopping job {job_uid}: {e}")
        logger.info("scheduler cleanup_jobs finished")

    def create_workers(self, job: Job, *args, **kwargs):
        """
        启动worker，类似LocalScheduler的create_workers
        """
        logger.info(f"Creating workers with config: {job}")
        assert len(job.tasks) > 0
        for container in job.tasks:
            assert container.gpu <= self.n_gpus_per_node

        assert job.role in ("train", "rollout", "ref")

        # 提交作业到Asystem
        job_info = self.submit_job(job)
        job_name = job_info.get("job_name")
        job_uid = job_info.get("job_uid")

        if job_uid:
            self.submitted_jobs[job.role] = job_uid
            logger.info(f"Job {job_name} submitted with UID: {job_uid}")

            with open("/tmp/job_uids.txt", "a") as f:
                f.write(job_uid + "\n")

            return job_uid
        else:
            raise FrameworkError(
                "FrameworkError", "SchedulerError", f"Failed to submit job: {job_info}"
            )

    def get_workers(self, worker_key, timeout: float = 300.0) -> list[Worker]:
        """
        等待并返回worker列表，包含调度结果，比如ip和engine ports
        """
        logger.info(f"Waiting for workers to be ready (timeout: {timeout}s)")

        if not self.submitted_jobs:
            logger.warning("No jobs submitted yet")
            return []

        # 等待作业启动
        server_infos = self.wait_for_jobs(worker_key, self.submitted_jobs, timeout)

        # 转换为Worker对象列表
        worker_objects = []
        for instance_id, info in server_infos.items():
            worker = Worker(
                id=instance_id,
                ip=info.get("ip", ""),
            )

            # 添加端口信息 - 优先处理ports数组，兼容单个port字段
            if "ports" in info and info["ports"]:
                # 处理端口数组（新格式）
                worker.ports.extend([str(p) for p in info["ports"]])
            elif "port" in info:
                # 处理单个端口（兼容格式）
                worker.ports.append(str(info["port"]))

            worker_objects.append(worker)

        self.worker_infos = [
            (w.id, w.ip, w.ports[0] if w.ports else "") for w in worker_objects
        ]
        logger.info(f"Got {len(worker_objects)} workers ready")

        return worker_objects

    def delete_workers(self, name: str = None):
        """
        Stops a running job.
        """
        if name:
            # 停止指定的作业
            job_uid = self.submitted_jobs.get(name)
            if job_uid is not None:
                self.stop_job(job_uid)
                del self.submitted_jobs[name]
                logger.info(f"Stopped job: {name}")
            else:
                logger.warning(f"Job {name} not found")
        else:
            # 停止所有作业
            for job_name, job_uid in self.submitted_jobs.items():
                try:
                    self.stop_job(job_uid)
                    logger.info(f"Stopped job: {job_name}")
                except Exception as e:
                    logger.error(f"Error stopping job {job_name}: {e}")
            self.submitted_jobs.clear()

    def create_engine(self, worker_id: str, engine_obj: Any, init_config: Any = None):
        """
        远程创建engine实例
        """
        logger.info(f"Creating engine on worker {worker_id}")
        return self.rpc_client.create_engine(worker_id, engine_obj, init_config)

    def call_engine(self, worker_id: str, method: str, *args, **kwargs):
        """
        数据面调用
        """
        logger.info(f"Calling '{method}' on worker {worker_id}")
        return self.rpc_client.call_engine(worker_id, method, 3, *args, **kwargs)

    def call_engine_with_serialized_data(self, worker_id: str, serialized_data: bytes):
        """
        数据面调用（带序列化数据）
        """
        logger.info(f"Calling on worker {worker_id} with serialized data")
        return self.rpc_client.call_engine_with_serialized_data(
            worker_id, serialized_data, 3
        )

    def cleanup(self):
        """
        清理提交的作业和资源
        """
        logger.info("Cleaning up Asystem jobs...")
        try:
            for job_name, job_uid in self.submitted_jobs.items():
                logger.info(f"Cleaning up job {job_name} with UID: {job_uid}")
                self.stop_job(job_uid)
            self.submitted_jobs.clear()
            self.worker_infos.clear()
            logger.info("Cleanup completed successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            raise FrameworkError("FrameworkError", "SchedulerError", e)

    def _request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """Helper method for making HTTP requests to the Asystem server."""
        url = f"{self.api_url}/{endpoint.lstrip('/')}"

        logger.debug(f"Making {method} request to {url}")
        try:
            response = self._session.request(method, url, **kwargs)
            response.raise_for_status()
            return response
        except requests.exceptions.HTTPError as e:
            error_content = "No content"
            try:
                error_content = e.response.json()
            except ValueError:
                error_content = e.response.text
            logger.error(
                f"HTTPError: {e.response.status_code} calling {method} {url}. Response: {error_content}"
            )
            raise FrameworkError("FrameworkError", "SchedulerError", e)
        except requests.exceptions.RequestException as e:
            logger.error(f"RequestException: {e} calling {method} {url}")
            raise FrameworkError("FrameworkError", "SchedulerError", e)

    def _generate_job_name(self) -> str:
        """Generate unique job name"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        job_name = f"{self.expr_name}_{self.trial_name}:{timestamp}"
        self._submission_counter += 1
        return job_name

    def _build_asystem_payload(self, job: Job) -> dict[str, Any]:
        """Build Asystem job payload from SchedulingConfig"""
        job_name = self._generate_job_name()

        # 构建基础标签
        labels = {
            "expr_name": self.expr_name,
            # hack: 先通过trial区分下不同的角色
            "trial_name": self.trial_name + "_" + job.role,
            "run_name": self.run_name,
        }

        # 构建worker groups
        containers = []

        if job.tasks:
            for i, spec in enumerate(job.tasks):
                # 结合默认 envs 和 Scheduling 的 envs
                combined_envs = default_envs.copy()
                combined_envs.update(self.extra_envs)  # 覆盖或新增
                combined_envs.update(spec.env_vars)  # 覆盖或新增
                custom_envs = {
                    "TYPE": spec.type,
                    "ROLE": job.role,
                    "ASTRA_SHARED_PATH": f"{self.storage_prefix}/astate_shared_storage",
                    "WORK_MODE": "GENERATION" if job.role == "rollout" else "TRAINING",
                    "IMAGE": spec.container_image,
                }
                combined_envs.update(custom_envs)

                cmd = (
                    spec.cmd
                    if spec.cmd
                    else (
                        f"bash {self.real_package_path}/areal/scheduler/scripts/launch-worker.sh"
                        if spec.type == "worker"
                        else f"bash {self.real_package_path}/areal/scheduler/scripts/launch-hybrid-server.sh"
                    )
                )
                container = {
                    "name": f"worker_{i}",
                    "command": ["/bin/sh"],
                    "args": ["-c", cmd],
                    "env": combined_envs,
                    "image": spec.container_image,
                    "resourceRequirement": {
                        "cpu": spec.cpu,
                        "memoryMB": spec.mem,
                        "gpuCount": spec.gpu,
                    },
                    "portCount": spec.port_count,
                }
                containers.append(container)

        # 构建worker group
        worker_group = {
            "workerRole": "todo",
            "replicas": job.replicas,
            "containerSpecList": containers,
        }

        # 构建最终payload
        payload = {
            "jobName": job_name,
            "description": f"Job for {self.run_name}",
            "labels": labels,
            "asystemWorkerGroups": [worker_group],
            "scheduleStrategy": (
                job.schedule_strategy if job.schedule_strategy else None
            ),
        }
        if job.schedule_strategy is not None:
            payload["scheduleStrategy"] = asdict(job.schedule_strategy)

        logger.info(f"schedule payload: {payload}")

        return payload

    def _register_worker_info(self, worker_id: str, worker_info: dict[str, Any]):
        """注册worker信息到本地注册表"""
        self._worker_registry[worker_id] = worker_info

        # Register worker with RPCClient for create_engine and call_engine
        ip = worker_info.get("ip")
        ports = worker_info.get("ports", [])
        if ip and ports:
            port = ports[0]  # Use first port for RPC
            self.rpc_client.register(worker_id, ip, port)

        logger.debug(f"Registered worker {worker_id}: {worker_info}")

    def _get_worker_info_by_id(self, worker_id: str) -> dict[str, Any] | None:
        """根据worker_id获取worker信息"""
        return self._worker_registry.get(worker_id)

    def submit_job(self, job: Job) -> dict[str, Any]:
        """
        提交作业到Asystem
        """
        payload = self._build_asystem_payload(job)

        logger.info(f"Submitting job to Asystem: {payload['jobName']}")
        logger.info(f"Job payload: {json.dumps(payload, indent=2)}")

        # 提交作业到Asystem
        response = self._request("POST", "AsystemJobs", json=payload)
        result = response.json()

        # 返回作业信息
        job_uid = result.get("uid")
        if job_uid:
            logger.info(
                f"Job {payload['jobName']} submitted successfully with UID: {job_uid}"
            )
            return {"job_name": payload["jobName"], "job_uid": job_uid}
        else:
            raise FrameworkError(
                "FrameworkError",
                "SchedulerError",
                f"Failed to get job UID from response: {result}",
            )

    def wait_for_jobs(
        self, worker_key, submitted_jobs: dict[str, str], timeout: float = 300.0
    ) -> dict[str, dict[str, Any]]:
        """
        等待作业启动并返回服务器信息
        """
        if not submitted_jobs:
            raise SchedulerError("No jobs have been submitted yet")

        start_time = time.time()
        job_server_infos = {}

        while True:
            # 检查超时
            if timeout and (time.time() - start_time) > timeout:
                raise FrameworkError(
                    "FrameworkError",
                    "SchedulerError",
                    f"Timeout waiting for jobs to start after {timeout} seconds",
                )

            job_uid = submitted_jobs.get(worker_key)
            if not job_uid:
                raise Exception(f"cannot find worker key: {worker_key}")

            response = self._request("GET", f"AsystemJobs/{job_uid}")
            job_info = response.json()
            status = job_info.get("status", "UNKNOWN")
            logger.debug(f"Job {job_uid} status: {status}")
            if status == "RUNNING":
                # 解析资源分布信息
                self._parse_job_infos(job_uid, job_info, job_server_infos)
                logger.info(f"All jobs are running. Server infos: {job_server_infos}")
                return job_server_infos
            elif status in ["FAILED", "CANCELLED"]:
                raise FrameworkError(
                    "FrameworkError",
                    "SchedulerError",
                    f"Job {job_uid} failed with status: {status}",
                )

            time.sleep(SCHEDULER_WAIT_CHECK_TIME_INTERVAL)

    def _parse_job_infos(
        self,
        job_uid: str,
        job_info: dict[str, Any],
        server_infos: dict[str, dict[str, Any]],
    ):
        """解析作业的资源分布信息，包括WorkerGroupStatuses中的端口信息"""

        # 首先处理WorkerGroupStatuses（包含端口信息）
        worker_group_statuses = job_info.get("workerGroupStatuses", [])

        for group_idx, worker_group in enumerate(worker_group_statuses):
            # WorkerGroupStatus是WorkerStatus数组
            for worker_index, worker_status in enumerate(worker_group):
                self._parse_worker_status(
                    job_uid,
                    worker_status,
                    worker_index,
                    server_infos,
                )

    def _parse_worker_status(
        self,
        job_uid: str,
        worker_status: dict[str, Any],
        worker_index: int,
        server_infos: dict[str, dict[str, Any]],
    ):
        """解析单个WorkerStatus"""
        instance_id = f"{job_uid}_worker_{worker_index}"
        ip = worker_status.get("ip", "")

        # 解析容器状态和端口 - 应用新的端口选择逻辑
        container_statuses = worker_status.get("containerStatuses", [])
        ports_list = self._parse_ports_list(container_statuses)
        server_info = {"ip": ip, "ports": ports_list[1]}
        server_infos[instance_id] = server_info

        engine_info = {"ip": ip, "ports": ports_list[0]}
        # 注册worker信息到本地注册表，供create_engine和call_engine使用
        self._register_worker_info(instance_id, engine_info)

        logger.debug(
            f"Parsed worker {instance_id}: engine: {engine_info}, server_info: {server_infos}"
        )

    def _parse_ports_list(self, container_statuses: list) -> list:
        """
        根据新规则选择端口：
        1. 如果有长度为1的端口数组和长度不为1的端口数组，只保留长度为1的端口
        2. 其余情况只保留第一个端口
        """
        ports_list = []

        # 收集所有非空的端口数组
        for container_status in container_statuses:
            container_ports = container_status.get("ports", [])
            ports_list.append(container_ports)

        return ports_list

    def stop_job(self, job_uid: str):
        """停止作业"""
        logger.info(f"Stopping job with UID: {job_uid}")

        try:
            self._request("POST", f"AsystemJobs/{job_uid}/cancel")
            logger.info(f"Job {job_uid} stop request sent successfully")
        except Exception as e:
            logger.error(f"Error stopping job {job_uid}: {e}")
            raise FrameworkError(
                "FrameworkError", "SchedulerError", f"Failed to stop job {job_uid}: {e}"
            )
