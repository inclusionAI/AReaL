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
    port_count: int = 2


@dataclass
class ScheduleStrategy:
    type: str = ""
    uid: str = ""


@dataclass
class SchedulingConfig:
    replicas: int = 0
    specs: List[ContainerSpec] = field(default_factory=list)
    schedule_strategy: ScheduleStrategy | None = None
    role: str = ""


class Scheduler(abc.ABC):
    def create_workers(self, worker_key, scheduler_config, *args, **kwargs) -> str:
        """
        Start workers, return job id
        """

    def get_workers(self, worker_key, timeout=None) -> List[Worker]:
        """
        Wait and return worker list, including scheduling results such as ip and engine ports
        (worker id, ip, ports)
        """
        raise NotImplementedError()

    def delete_workers(self):
        """stop all workers

        Raises exception if there is no such job, but passes if the job
        has stopped either successfully or not.
        """
        raise NotImplementedError()

    async def create_engine(self, worker_id, engine_obj, *args, **kwargs):
        """
        Create engine instance remotely
        """
        raise NotImplementedError()

    def call_engine(self, worker_id, method, *args, **kwargs):
        """
        Data plane call
        """
        raise NotImplementedError()
