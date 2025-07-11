import abc
import logging


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
    def create_workers(self, scheduler_config: dict) -> list:
        """
        启动worker，返回 [(id, ip, port), ...]
        """
        pass

    @abc.abstractmethod
    def create_engine(self, worker_id, engine_obj):
        """
        远程创建engine实例
        """
        pass

    @abc.abstractmethod
    def call(self, worker_id, method, *args, **kwargs):
        """
        数据面调用
        """
        pass




## ----------------------------------------- ##

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from dataclasses import field, dataclass
from arealite.api.io_struct import (
    FinetuneSpec,
)

@dataclass
class ContainerSpec:
    cpu: int
    gpu: int
    mem: int
    container_image: str = None
    cmd: str = None
    env_vars: Dict[str, str] = field(default_factory=dict)
    port: int = 50000

@dataclass
class SchedulingConfig:
    replicas: int = 0
    specs: List[ContainerSpec] = field(default_factory=list)

@dataclass
class Worker:
    id: str
    ip: str
    ports: List[str] = field(default_factory=list)

class Scheduler(ABC):
    def __init__(self, expr_name: str, trial_name: str):
        self.expr_name = expr_name
        self.trial_name = trial_name
        self.run_name = f"{self.expr_name}_{self.trial_name}"

    def create_workers(self, scheduling_config, **kwargs):
        """
        提交作业， 异步等待作业, 返回作业id
        """
        raise NotImplementedError()

    async def get_workers(self, timeout=None) -> List[Worker]:
        """
        返回engine id, 以及对应的server addr, 将调度结果记录在内存中
        (engine id, server infos<ip, port>})
        """
        raise NotImplementedError()

    async def delete_workers(self, name):
        """Stops a running job.

        Raises exception if there is no such job, but passes if the job
        has stopped either successfully or not.
        """
        raise NotImplementedError()

    async def initialize_engine(self, worker_id: str, engine_obj: Any, init_config: Any | None):
        raise NotImplementedError()

    async def call_engine(self, worker_id: str, method: str, *args, **kwargs):
        """
        call engine's method
        """
        raise NotImplementedError()
