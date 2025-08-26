import abc
import logging
from typing import Any, List, Dict
from dataclasses import dataclass, field


@dataclass
class Worker:
    id: str
    ip: str
    ports: List[str] = field(default_factory=list)


@dataclass
class ContainerSpec:
    cpu: int = 0
    gpu: int = 0
    mem: int = 0
    container_image: str = ""
    cmd: str = ""
    env_vars: Dict[str, str] = field(default_factory=dict)
    portCount: int = 2

@dataclass
class ScheduleStrategy:
    type: str = ""
    uid: str = ""

@dataclass
class SchedulingConfig:
    replicas: int = 0
    specs: List[ContainerSpec] = field(default_factory=list)
    schedule_strategy: ScheduleStrategy=None
    role: str = ""


class Scheduler(abc.ABC):
    def __init__(self, config: dict):
        self.config = config
        self.rpc_client = self._build_rpc_client(config)
        self.worker_map = {}
        logging.info(f"Scheduler initialized with config: {config}")

    @abc.abstractmethod
    def _build_rpc_client(self, config):
        pass

    @abc.abstractmethod
    def create_workers(self, worker_key, scheduler_config, *args, **kwargs) -> str:
        """
        启动worker，返回 [(id, ip, port), ...]
        """
        pass

    @abc.abstractmethod
    def get_workers(self, worker_key, timeout=None) -> List[Worker]:
        """
        等待并返回worker 列表, 包含调度结果, 比如ip和engine ports
        (worker id, ip, ports)
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def delete_workers(self):
        """stop all workers

        Raises exception if there is no such job, but passes if the job
        has stopped either successfully or not.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    async def create_engine(self, worker_id, engine_obj, *args, **kwargs):
        """
        远程创建engine实例
        """
        pass

    @abc.abstractmethod
    def call_engine(self, worker_id, method, *args, **kwargs):
        """
        数据面调用
        """
        pass
