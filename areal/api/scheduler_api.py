import abc
from dataclasses import dataclass, field
from typing import Dict, List


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
    schedule_strategy: ScheduleStrategy = None
    role: str = ""


class Scheduler(abc.ABC):
    def __init__(self, config: dict):
        self.config = config
        self.rpc_client = self._build_rpc_client(config)
        self.worker_map = {}

    @abc.abstractmethod
    def _build_rpc_client(self, config):
        pass

    @abc.abstractmethod
    def create_workers(self, worker_key, scheduler_config, *args, **kwargs) -> str:
        """
        Start workers, return [(id, ip, port), ...]
        """

    @abc.abstractmethod
    def get_workers(self, worker_key, timeout=None) -> List[Worker]:
        """
        Wait and return worker list, including scheduling results such as ip and engine ports
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
        Create engine instance remotely
        """

    @abc.abstractmethod
    def call_engine(self, worker_id, method, *args, **kwargs):
        """
        Data plane call
        """
