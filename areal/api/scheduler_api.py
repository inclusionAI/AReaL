import abc
from dataclasses import dataclass, field
from typing import Any


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
    ip: str
    ports: list[str] = field(default_factory=list)


@dataclass
class ContainerSpec:
    """
    Resource specification for a worker container/process.

    Attributes:
        cpu: Number of CPU cores to allocate.
        gpu: Number of GPUs to allocate.
        mem: Memory in MB to allocate.
        container_image: Docker container image (for containerized deployments).
        cmd: Command to execute when starting the worker.
        env_vars: Environment variables to set for the worker process.
        port_count: Number of ports to allocate for this worker.
    """

    cpu: int = 0
    gpu: int = 0
    mem: int = 0
    container_image: str = ""
    cmd: str = ""
    env_vars: dict[str, str] = field(default_factory=dict)
    port_count: int = 2


@dataclass
class ScheduleStrategy:
    """
    Scheduling strategy configuration.

    Supported strategies:
        - "new": Allocate new GPUs using round-robin (default).
        - "colocate": Schedule workers on the same GPUs as another role.

    Attributes:
        type: Type of scheduling strategy ("new" or "colocate").
        uid: For "colocate" strategy, the role name to colocate with (e.g., "actor").
            For "new" strategy, this field is optional.
    """

    type: str = ""
    uid: str = ""


@dataclass
class SchedulingConfig:
    """
    Complete configuration for scheduling a group of workers.

    Attributes:
        replicas: Number of worker replicas to create.
        specs: List of container specifications, one per replica (or a single spec for all).
        schedule_strategy: Optional scheduling strategy to use.
        role: Role name for this group of workers (e.g., "rollout", "actor", "critic").
    """

    replicas: int = 0
    specs: list[ContainerSpec] = field(default_factory=list)
    schedule_strategy: ScheduleStrategy | None = None
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
    def create_workers(
        self, role: str, scheduler_config: SchedulingConfig, *args, **kwargs
    ) -> list[str]:
        """
        Create and start worker processes for a specific role.

        Args:
            role: Role name for this group of workers (e.g., "rollout", "actor", "critic").
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
