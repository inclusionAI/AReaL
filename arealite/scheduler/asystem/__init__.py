from arealite.scheduler.base import Scheduler, Worker
import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, Optional, List
import requests

from arealite.scheduler.base import SchedulingConfig
from arealite.scheduler.rpc.rpc_client import RPCClient

try:
    import cloudpickle
except ImportError:
    import pickle as cloudpickle

logger = logging.getLogger(__name__)

# Default Asystem API endpoints
DEFAULT_ASYSTEM_SERVER_URL = "http://33.215.20.149:8081"
API_BASE_PATH = "/api/v1"
SCHEDULER_WAIT_CHECK_TIME_INTERVAL = 5  # Seconds
logger = logging.getLogger(__name__)

class SchedulerError(Exception):
    """Scheduler related errors"""
    pass

class AsystemScheduler(Scheduler):
    """
    AsystemScheduler implementation that follows the same interface as LocalScheduler.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.worker_infos = []
        self.submitted_jobs = {}  # job_name -> job_uid

        # Initialize Asystem client attributes
        self.expr_name = config.get("expr_name", "default_expr")
        self.trial_name = config.get("trial_name", "default_trial")
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

        logger.info(f"AsystemScheduler initialized for {self.run_name}. API URL: {self.api_url}")

    def _build_rpc_client(self, config):
        pass

    def create_workers(self, worker_key, scheduler_config: SchedulingConfig, *args, **kwargs):
        """
        启动worker，类似LocalScheduler的create_workers
        """
        logger.info(f"Creating workers with config: {scheduler_config}")
        assert len(scheduler_config.specs) > 0
        engine = scheduler_config.specs[0]
        if engine.gpu > self.n_gpus_per_node:
            assert engine.gpu % self.n_gpus_per_node ==0
            engine.gpu = self.n_gpus_per_node
            node_count = engine.gpu // self.n_gpus_per_node
            scheduler_config.replicas = scheduler_config.replicas * node_count

        
        # 提交作业到Asystem
        job_info = self.submit_job(scheduler_config)
        job_name = job_info.get("job_name")
        job_uid = job_info.get("job_uid")
        
        if job_uid:
            self.submitted_jobs[worker_key] = job_uid
            logger.info(f"Job {job_name} submitted with UID: {job_uid}")
        else:
            raise RuntimeError(f"Failed to submit job: {job_info}")

    def get_workers(self, worker_key, timeout: float = 300.0) -> List[Worker]:
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
            
        self.worker_infos = [(w.id, w.ip, w.ports[0] if w.ports else "") for w in worker_objects]
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
        return self.rpc_client.call_engine(worker_id, method, *args, **kwargs)

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
            raise

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
            logger.error(f"HTTPError: {e.response.status_code} calling {method} {url}. Response: {error_content}")
            raise SchedulerError(f"Asystem API request failed: {e}") from e
        except requests.exceptions.RequestException as e:
            logger.error(f"RequestException: {e} calling {method} {url}")
            raise SchedulerError(f"Asystem API request failed: {e}") from e

    def _generate_job_name(self) -> str:
        """Generate unique job name"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        job_name = f"{self.expr_name}_{self.trial_name}:{timestamp}"
        self._submission_counter += 1
        return job_name

    def _build_asystem_payload(self, scheduling_config: SchedulingConfig) -> Dict[str, Any]:
        """Build Asystem job payload from SchedulingConfig"""
        job_name = self._generate_job_name()

        # 构建基础标签
        labels = {
            "expr_name": self.expr_name,
            "trial_name": self.trial_name,
            "run_name": self.run_name,
        }

        # 构建worker groups
        containers = []

        if scheduling_config.specs:
            for i, spec in enumerate(scheduling_config.specs):
                container = {
                    "name": f"worker_{i}",
                    "command": ["/bin/sh"],
                    "args": ["-c", spec.cmd] if spec.cmd else ["-c", "echo 'No command specified'"],
                    "env": spec.env_vars,
                    "image": spec.container_image,
                    "resourceRequirement": {
                        "cpu": spec.cpu,
                        "memoryMB": spec.mem,
                        "gpuCount": spec.gpu,
                    },
                    "portCount": spec.portCount
                }
                containers.append(container)

        # 构建worker group
        worker_group = {
            "workerRole": "todo",
            "replicas": scheduling_config.replicas,
            "containerSpecList": containers,
        }

        # 构建最终payload
        payload = {
            "jobName": job_name,
            "description": f"Job for {self.run_name}",
            "labels": labels,
            "asystemWorkerGroups": [worker_group],
            "scheduleStrategy": scheduling_config.schedule_strategy if scheduling_config.schedule_strategy else None,
        }

        return payload

    def _register_worker_info(self, worker_id: str, worker_info: Dict[str, Any]):
        """注册worker信息到本地注册表"""
        self._worker_registry[worker_id] = worker_info

        # Register worker with RPCClient for create_engine and call_engine
        ip = worker_info.get("ip")
        ports = worker_info.get("ports", [])
        if ip and ports:
            port = ports[0]  # Use first port for RPC
            self.rpc_client.register(worker_id, ip, port)

        logger.debug(f"Registered worker {worker_id}: {worker_info}")

    def _get_worker_info_by_id(self, worker_id: str) -> Optional[Dict[str, Any]]:
        """根据worker_id获取worker信息"""
        return self._worker_registry.get(worker_id)

    def submit_job(self, scheduling_config: SchedulingConfig) -> Dict[str, Any]:
        """
        提交作业到Asystem
        """
        payload = self._build_asystem_payload(scheduling_config)

        logger.info(f"Submitting job to Asystem: {payload['jobName']}")
        logger.info(f"Job payload: {json.dumps(payload, indent=2)}")

        # 提交作业到Asystem
        response = self._request("POST", "AsystemJobs", json=payload)
        result = response.json()

        # 返回作业信息
        job_uid = result.get("uid")
        if job_uid:
            logger.info(f"Job {payload['jobName']} submitted successfully with UID: {job_uid}")
            return {"job_name": payload['jobName'], "job_uid": job_uid}
        else:
            raise SchedulerError(f"Failed to get job UID from response: {result}")

    def wait_for_jobs(self, worker_key, submitted_jobs: Dict[str, str], timeout: float = 300.0) -> Dict[
        str, Dict[str, Any]]:
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
                raise SchedulerError(f"Timeout waiting for jobs to start after {timeout} seconds")

            all_ready = True
            job_uid = submitted_jobs.get(worker_key)
            if not job_uid:
                raise Exception(f"cannot find worker key: {worker_key}")

            response = self._request("GET", f"AsystemJobs/{job_uid}")
            job_info = response.json()
            status = job_info.get("status", "UNKNOWN")
            logger.debug(f"Job {job_uid} status: {status}")
            if status == "RUNNING":
                # 解析资源分布信息
                aicloud_infos = self._parse_legacy_resource_distributions(job_info)
                self._parse_job_infos(job_uid, job_info, aicloud_infos, job_server_infos)
                logger.info(f"All jobs are running. Server infos: {job_server_infos}")
                return job_server_infos
            elif status in ["FAILED", "CANCELLED"]:
                raise SchedulerError(f"Job {job_uid} failed with status: {status}")

            time.sleep(SCHEDULER_WAIT_CHECK_TIME_INTERVAL)

    def _parse_job_infos(self, job_uid: str, job_info: Dict[str, Any], aicloud_infos: List[Dict[str, Any]],
                         server_infos: Dict[str, Dict[str, Any]]):
        """解析作业的资源分布信息，包括WorkerGroupStatuses中的端口信息"""

        # 首先处理WorkerGroupStatuses（包含端口信息）
        worker_group_statuses = job_info.get("workerGroupStatuses", [])

        for group_idx, worker_group in enumerate(worker_group_statuses):
            # WorkerGroupStatus是WorkerStatus数组
            for worker_index, worker_status in enumerate(worker_group):
                self._parse_worker_status(job_uid, worker_status, worker_index, len(worker_group), aicloud_infos,
                                          server_infos)


    def _parse_worker_status(self, job_uid: str, worker_status: Dict[str, Any], worker_index: int, worker_count: int,
                             aicloud_infos: List[Dict[str, Any]], server_infos: Dict[str, Dict[str, Any]]):
        """解析单个WorkerStatus"""
        instance_id = f"{job_uid}_worker_{worker_index}"
        ip = worker_status.get("ip", "")

        # 解析容器状态和端口 - 应用新的端口选择逻辑
        container_statuses = worker_status.get("containerStatuses", [])
        ports_list = self._parse_ports_list(container_statuses)
        # ai云场景
        if len(ports_list) == 1:
            assert worker_count % len(aicloud_infos) == 0
            dp_world_size = worker_count // len(aicloud_infos)
            aicloud_info = aicloud_infos[worker_index // dp_world_size]
            server_info = aicloud_info
        else:
            server_info = {
                "ip": ip,
                "ports": ports_list[1]
            }

        server_infos[instance_id] = server_info

        engine_info = {
            "ip": ip,
            "ports": ports_list[0]
        }
        # 注册worker信息到本地注册表，供create_engine和call_engine使用
        self._register_worker_info(instance_id, engine_info)

        logger.debug(f"Parsed worker {instance_id}: engine: {engine_info}, server_info: {server_infos}")

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

    def _parse_legacy_resource_distributions(self, job_info: Dict[str, Any]):
        """解析legacy resourceDistributions格式（无端口信息）"""
        resource_distributions = job_info.get("resourceDistributions", {})

        # # 处理k8s资源
        # k8s_resources = resource_distributions.get("k8s", [])
        # for i, resource in enumerate(k8s_resources):
        #     instance_id = f"{job_name}_k8s_{i}"
        #     server_infos[instance_id] = {
        #         "ip": resource.get("ip"),
        #         "instance": resource.get("instance"),
        #         "provider": "k8s"
        #     }
        aicloud_infos = []
        # 处理aicloud资源
        aicloud_resources = resource_distributions.get("aicloud", [])
        if aicloud_resources is None:
            return aicloud_infos

        for i, resource in enumerate(aicloud_resources):
            aicloud_infos.append({
                "ip": resource.get("ip"),
                "ports": [8188, ],
            })
        return aicloud_infos

    def stop_job(self, job_uid: str):
        """停止作业"""
        logger.info(f"Stopping job with UID: {job_uid}")

        try:
            response = self._request("DELETE", f"AsystemJobs/{job_uid}")
            logger.info(f"Job {job_uid} stop request sent successfully")
        except Exception as e:
            logger.error(f"Error stopping job {job_uid}: {e}")
            raise SchedulerError(f"Failed to stop job {job_uid}: {e}")

    # def create_engine(self, worker_id: str, engine_obj: Any, init_config: Any = None) -> bool:
    #     """
    #     在指定worker上创建引擎实例
    #     """
    #     logger.info(f"Creating engine on worker {worker_id}")
    #
    #     try:
    #         # 获取worker信息
    #         worker_info = self._get_worker_info_by_id(worker_id)
    #         if not worker_info:
    #             logger.error(f"Worker {worker_id} not found")
    #             return False
    #
    #         ip = worker_info.get("ip")
    #         ports = worker_info.get("ports", [])
    #
    #         if not ip or not ports:
    #             logger.error(f"Invalid worker info for {worker_id}: ip={ip}, ports={ports}")
    #             return False
    #
    #         # 使用第一个端口作为RPC端口
    #         rpc_port = ports[0]
    #         url = f"http://{ip}:{rpc_port}/create_engine"
    #
    #         # 使用cloudpickle序列化引擎对象和初始化配置
    #         payload = (engine_obj, init_config)
    #         serialized_obj = cloudpickle.dumps(payload)
    #
    #         logger.debug(f"Sending create_engine request to {url}")
    #
    #         # 发送HTTP POST请求
    #         response = requests.post(url, data=serialized_obj, timeout=30)
    #
    #         if response.status_code == 200:
    #             logger.info(f"Successfully created engine on worker {worker_id} ({ip}:{rpc_port})")
    #             return True
    #         else:
    #             logger.error(f"Failed to create engine on worker {worker_id}: HTTP {response.status_code}")
    #             logger.error(f"Response: {response.text}")
    #             return False
    #
    #     except Exception as e:
    #         logger.error(f"Error creating engine on worker {worker_id}: {e}")
    #         return False

    # def call_engine(self, worker_id: str, method: str, *args, **kwargs) -> Any:
    #     """
    #     调用指定worker上引擎的方法
    #     """
    #     logger.info(f"Calling method '{method}' on worker {worker_id}")
    #
    #     try:
    #         # 获取worker信息
    #         worker_info = self._get_worker_info_by_id(worker_id)
    #         if not worker_info:
    #             logger.error(f"Worker {worker_id} not found")
    #             raise SchedulerError(f"Worker {worker_id} not found")
    #
    #         ip = worker_info.get("ip")
    #         ports = worker_info.get("ports", [])
    #
    #         if not ip or not ports:
    #             logger.error(f"Invalid worker info for {worker_id}: ip={ip}, ports={ports}")
    #             raise SchedulerError(f"Invalid worker info for {worker_id}")
    #
    #         # 使用第一个端口作为RPC端口
    #         rpc_port = ports[0]
    #         url = f"http://{ip}:{rpc_port}/call"
    #
    #         # 构建请求数据，格式与LocalScheduler的RPCClient保持一致
    #         req = (method, args, kwargs)
    #         serialized_req = cloudpickle.dumps(req)
    #
    #         logger.debug(f"Sending call request to {url}: method={method}")
    #
    #         # 发送HTTP POST请求
    #         response = requests.post(url, data=serialized_req, timeout=30)
    #
    #         if response.status_code == 200:
    #             # 反序列化响应结果
    #             result = cloudpickle.loads(response.content)
    #             logger.info(f"Successfully called '{method}' on worker {worker_id} ({ip}:{rpc_port})")
    #             return result
    #         else:
    #             logger.error(f"Failed to call '{method}' on worker {worker_id}: HTTP {response.status_code}")
    #             logger.error(f"Response: {response.text}")
    #             raise SchedulerError(f"RPC call failed: {response.text}")
    #
    #     except Exception as e:
    #         logger.error(f"Error calling method {method} on worker {worker_id}: {e}")
    #         raise SchedulerError(f"Failed to call method {method}: {e}")