import abc
from dataclasses import dataclass, field
from typing import Any

from areal.api.cli_args import SchedulingSpec, SchedulingStrategy


@dataclass
class Worker:
    """
    Represents a worker process in the distributed system.

    Attributes:
        id: Unique identifier for the worker (e.g., "rollout/0", "actor/1").
        ip: IP address where the worker is running.
        ports: List of port numbers (as strings) allocated to this worker for RPC communication.
    """

    id: str
    # worker and engine deploy on the same machine, so ip are the same
    ip: str
    worker_ports: list[str] = field(default_factory=list)
    engine_ports: list[str] = field(default_factory=list)


@dataclass
class Job:
    replicas: int = 0
    tasks: list[SchedulingSpec] = field(default_factory=list)
    scheduling_strategy: SchedulingStrategy | None = None
    role: str = ""


class Scheduler(abc.ABC):
    """
    Abstract base class for schedulers that manage distributed worker processes.

    A scheduler is responsible for:
    - Creating and managing worker processes/containers.
    - Allocating resources (GPUs, ports, memory).
    - Creating and managing engine instances on workers.
    - Facilitating RPC calls to engine methods.
    """

    @abc.abstractmethod
    def create_workers(self, job: Job, *args, **kwargs) -> list[str]:
        """
        Create and start worker processes for a specific role.

        Args:
            scheduler_config: Configuration specifying replicas, resources, and scheduling strategy.
            *args: Additional positional arguments (implementation-specific).
            **kwargs: Additional keyword arguments (implementation-specific).

        Returns:
            List of worker IDs created (e.g., ["rollout/0", "rollout/1"]).

        Raises:
            WorkerCreationError: If worker creation fails.
            ValueError: If scheduler_config is invalid.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_workers(self, role: str, timeout: int | None = None) -> list[Worker]:
        """
        Wait for workers to be ready and return their information.

        This method blocks until all workers for the specified role are ready
        to accept RPC requests, or until the timeout is reached.

        Args:
            role: Role name to query (e.g., "rollout", "actor").
            timeout: Maximum time to wait in seconds. None means use the default timeout.

        Returns:
            List of Worker objects containing worker ID, IP address, and allocated ports.

        Raises:
            WorkerNotFoundError: If no workers exist for the specified role.
            WorkerFailedError: If any worker process has failed.
            WorkerTimeoutError: If timeout is exceeded while waiting for workers.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def delete_workers(self, role: str | None = None):
        """
        Stop and clean up worker processes.

        Args:
            role: Specific role to delete. If None, all workers are deleted.

        Raises:
            WorkerNotFoundError: If the specified role doesn't exist.

        Note:
            This method should gracefully terminate workers and clean up resources.
            It should not raise an exception if workers have already stopped.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    async def create_engine(self, worker_id: str, engine: str, *args, **kwargs) -> Any:
        """
        Create an engine instance on a remote worker.

        The engine parameter is a string import path (e.g., "areal.engine.ppo.actor.FSDPPPOActor")
        that will be dynamically imported and instantiated on the worker.

        Args:
            worker_id: ID of the worker to create the engine on (e.g., "rollout/0").
            engine: Import path to the engine class (e.g., "areal.engine.ppo.actor.FSDPPPOActor").
            *args: Positional arguments passed to engine initialization.
            **kwargs: Keyword arguments passed to engine initialization.

        Returns:
            Result from engine initialization.

        Raises:
            WorkerNotFoundError: If the specified worker doesn't exist.
            WorkerFailedError: If the worker process has failed.
            EngineCreationError: If engine creation or initialization fails.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def call_engine(self, worker_id: str, method: str, *args, **kwargs) -> Any:
        """
        Call a method on an engine instance running on a worker (data plane operation).

        This is the synchronous version. Use `async_call_engine` for async operations.

        Args:
            worker_id: ID of the worker hosting the engine (e.g., "rollout/0").
            method: Name of the method to call on the engine.
            *args: Positional arguments to pass to the method.
            **kwargs: Keyword arguments to pass to the method.

        Returns:
            Result from the engine method call.

        Raises:
            WorkerNotFoundError: If the specified worker doesn't exist.
            WorkerFailedError: If the worker process has failed.
            EngineCallError: If the method call fails.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    async def async_call_engine(
        self, worker_id: str, method: str, *args, **kwargs
    ) -> Any:
        """
        Async version of call_engine for calling engine methods asynchronously.

        This is useful for concurrent operations or when integrating with async frameworks.

        Args:
            worker_id: ID of the worker hosting the engine (e.g., "rollout/0").
            method: Name of the method to call on the engine.
            *args: Positional arguments to pass to the method.
            **kwargs: Keyword arguments to pass to the method.

        Returns:
            Result from the engine method call.

        Raises:
            WorkerNotFoundError: If the specified worker doesn't exist.
            WorkerFailedError: If the worker process has failed.
            EngineCallError: If the method call fails.
        """
        raise NotImplementedError()
