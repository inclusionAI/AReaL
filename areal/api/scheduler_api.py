import abc
from dataclasses import dataclass, field
from typing import List, Literal

from areal.api.engine_api import Scheduling


@dataclass
class Worker:
    id: str
    ip: str
    serve_port: str
    extra_ports: List[str] = field(default_factory=list)


@dataclass
class ScheduleStrategy:
    type: Literal["colocation", "separation", ""] = ""
    target: str = ""


@dataclass
class Job:
    replicas: int = 0
    tasks: List[Scheduling] = field(default_factory=list)
    schedule_strategy: ScheduleStrategy | None = None
    role: str = ""


class Scheduler(abc.ABC):
    def create_workers(self, job: Job, *args, **kwargs):
        """
        Start workers
        """
        raise NotImplementedError()

    def get_workers(self, role: str, timeout=None) -> List[Worker]:
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
