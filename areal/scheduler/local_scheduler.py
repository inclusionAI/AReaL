"""Local scheduler for managing worker subprocesses on a single GPU node."""

import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import orjson
import psutil

from areal.api.scheduler_api import ContainerSpec, Scheduler, SchedulingConfig, Worker
from areal.scheduler.exceptions import (
    EngineCallError,
    EngineCreationError,
    EngineImportError,
    GPUAllocationError,
    PortAllocationError,
    RPCConnectionError,
    SchedulerError,
    WorkerCreationError,
    WorkerFailedError,
    WorkerNotFoundError,
    WorkerTimeoutError,
)
from areal.scheduler.rpc.serialization import deserialize_value, serialize_value
from areal.utils import logging
from areal.utils.network import find_free_ports, gethostip

logger = logging.getLogger("LocalScheduler")


@dataclass
class WorkerInfo:
    """Internal tracking information for a worker process."""

    worker: Worker  # Public Worker object with id, ip, ports
    process: subprocess.Popen  # The subprocess handle
    role: str  # Worker role (e.g., "rollout", "actor", "critic")
    gpu_devices: list[int]  # Allocated GPU device IDs
    created_at: float  # Timestamp when worker was created
    log_file: str  # Path to stderr log file
    env_vars: dict[str, str] = field(default_factory=dict)  # Environment variables


class LocalScheduler(Scheduler):
    """
    Local scheduler that manages worker subprocesses on a single GPU node.

    This scheduler spawns worker processes running RPC servers and manages their lifecycle.
    It supports different worker types (rollout, actor, critic) through a unified interface.

    Features:
    - Dynamic port allocation
    - Round-robin GPU assignment
    - Process health monitoring
    - Comprehensive error handling
    - Graceful cleanup
    """

    def __init__(
        self,
        gpu_devices: list[int] | None = None,
        log_dir: str = "./logs/workers",
        startup_timeout: float = 30.0,
        health_check_interval: float = 1.0,
    ):
        """
        Initialize the local scheduler.

        Args:
            gpu_devices: List of GPU device IDs to use. If None, uses CUDA_VISIBLE_DEVICES or all GPUs.
            log_dir: Directory for worker log files
            startup_timeout: Maximum time to wait for worker startup (seconds)
            health_check_interval: Interval for health checks (seconds)
        """
        self.gpu_devices = gpu_devices or self._detect_gpus()
        self.log_dir = Path(log_dir)
        self.startup_timeout = startup_timeout
        self.health_check_interval = health_check_interval

        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Track workers by worker_key
        self._workers: dict[str, list[WorkerInfo]] = {}

        # GPU allocation counter for round-robin
        self._gpu_counter = 0

        # Track all allocated ports
        self._allocated_ports = set()

        # HTTP clients for RPC communication
        self._http_client = httpx.Client(timeout=3600.0)  # Sync client - 1 hour timeout
        self._async_http_client = httpx.AsyncClient(timeout=3600.0)  # Async client

        logger.info(
            f"LocalScheduler initialized with GPU devices: {self.gpu_devices}, "
            f"log directory: {self.log_dir}"
        )

    def _detect_gpus(self) -> list[int]:
        """Detect available GPU devices."""
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible:
            try:
                return [int(x) for x in cuda_visible.split(",")]
            except ValueError:
                logger.warning(
                    f"Invalid CUDA_VISIBLE_DEVICES: {cuda_visible}, using default [0]"
                )
                return [0]
        # Default to single GPU
        return [0]

    def _allocate_gpus(self, num_gpus: int) -> list[int]:
        """
        Allocate GPUs using round-robin strategy.

        Args:
            num_gpus: Number of GPUs to allocate

        Returns:
            List of GPU device IDs

        Raises:
            GPUAllocationError: If not enough GPUs available
        """
        if num_gpus > len(self.gpu_devices):
            raise GPUAllocationError(
                f"Requested {num_gpus} GPUs but only {len(self.gpu_devices)} available"
            )

        allocated = []
        for _ in range(num_gpus):
            gpu_id = self.gpu_devices[self._gpu_counter % len(self.gpu_devices)]
            allocated.append(gpu_id)
            self._gpu_counter += 1

        return allocated

    def _get_colocated_gpus(self, target_role: str, worker_idx: int) -> list[int]:
        """
        Get GPU allocation from another role for colocation.

        Args:
            target_role: The role to colocate with
            worker_idx: Index of the worker to get GPUs from

        Returns:
            List of GPU device IDs used by the target worker

        Raises:
            WorkerNotFoundError: If target role doesn't exist
            ValueError: If worker index is out of range
        """
        if target_role not in self._workers:
            raise WorkerNotFoundError(
                f"Cannot colocate with role '{target_role}' - role not found"
            )

        target_workers = self._workers[target_role]
        if worker_idx >= len(target_workers):
            raise ValueError(
                f"Cannot colocate with {target_role}/{worker_idx} - only {len(target_workers)} workers exist"
            )

        return target_workers[worker_idx].gpu_devices

    def _allocate_ports(self, count: int) -> list[int]:
        """
        Allocate free ports.

        Args:
            count: Number of ports to allocate

        Returns:
            List of allocated port numbers

        Raises:
            PortAllocationError: If port allocation fails
        """
        try:
            # Pass a copy of allocated_ports to avoid reference issues
            ports = find_free_ports(count, exclude_ports=set(self._allocated_ports))
            self._allocated_ports.update(ports)
            return ports
        except ValueError as e:
            raise PortAllocationError(str(e)) from e

    def _prepare_worker_specs(
        self, role: str, num_workers: int, specs: list[ContainerSpec] | None
    ) -> list[ContainerSpec]:
        """
        Prepare worker specs for a given number of workers.

        Args:
            role: Worker role name
            num_workers: Number of workers to create
            specs: Optional list of specs

        Returns:
            List of ContainerSpec objects (one per worker)

        Raises:
            WorkerCreationError: If specs configuration is invalid
        """
        if not specs:
            # Default spec: 1 GPU, 2 ports
            return [ContainerSpec(gpu=1, port_count=2)] * num_workers

        # If a single spec is provided, use it for all workers
        if len(specs) == 1:
            return [specs[0]] * num_workers

        # If per-worker specs, validate length matches
        if len(specs) == num_workers:
            return specs

        # Invalid configuration
        raise WorkerCreationError(
            role,
            "Invalid configuration",
            f"specs length ({len(specs)}) must be 1 or equal to replicas ({num_workers})",
        )

    def create_workers(
        self, role: str, scheduler_config: SchedulingConfig, *args, **kwargs
    ) -> list[str]:
        """
        Create worker subprocesses.

        Args:
            role: Role name for this group of workers (e.g., "rollout", "actor", "critic")
            scheduler_config: Scheduling configuration with replicas, specs, and strategy
            *args: Additional arguments passed to worker command
            **kwargs: Additional keyword arguments

        Returns:
            List of worker IDs created (e.g., ["rollout/0", "rollout/1"])

        Raises:
            WorkerCreationError: If worker creation fails
            GPUAllocationError: If GPU allocation fails
            PortAllocationError: If port allocation fails
        """
        if role in self._workers:
            raise WorkerCreationError(
                role,
                "Worker group already exists",
                f"Use delete_workers('{role}') first to remove existing workers",
            )

        # Extract configuration
        num_workers = scheduler_config.replicas
        if num_workers == 0:
            raise WorkerCreationError(
                role, "Invalid configuration", "replicas must be greater than 0"
            )

        # Prepare worker specs
        specs = self._prepare_worker_specs(role, num_workers, scheduler_config.specs)

        # Determine scheduling strategy
        strategy = scheduler_config.schedule_strategy
        if strategy is None:
            strategy_type = "new"
            colocate_role = None
        else:
            strategy_type = strategy.type or "new"
            colocate_role = strategy.uid if strategy_type == "colocate" else None

        logger.info(
            f"Creating {num_workers} workers for role '{role}' "
            f"(strategy: {strategy_type}, colocate_with: {colocate_role})"
        )

        workers = []
        worker_ids = []
        try:
            for idx in range(num_workers):
                worker_id = f"{role}/{idx}"
                spec = specs[idx]

                # Allocate resources based on strategy
                try:
                    # GPU allocation
                    if strategy_type == "colocate":
                        if not colocate_role:
                            raise WorkerCreationError(
                                role,
                                "Invalid strategy",
                                "Colocate strategy requires uid (target role) to be specified",
                            )
                        gpu_devices = self._get_colocated_gpus(colocate_role, idx)
                        logger.debug(
                            f"Worker {worker_id} colocated with {colocate_role}/{idx} on GPUs {gpu_devices}"
                        )
                    else:  # "new" or default
                        gpu_devices = self._allocate_gpus(spec.gpu)
                        logger.debug(
                            f"Worker {worker_id} allocated new GPUs {gpu_devices}"
                        )

                    ports = self._allocate_ports(spec.port_count)
                except (
                    GPUAllocationError,
                    PortAllocationError,
                    WorkerNotFoundError,
                    ValueError,
                ) as e:
                    # Clean up partially created workers
                    self._cleanup_workers(workers)
                    raise WorkerCreationError(
                        role, f"Resource allocation failed for worker {idx}", str(e)
                    ) from e

                # Prepare environment
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_devices))
                env["WORKER_ID"] = worker_id

                # Merge user-provided environment variables from spec
                if spec.env_vars:
                    env.update(spec.env_vars)

                # Prepare log file
                log_file = self.log_dir / f"{worker_id.replace('/', '_')}.log"

                # Build command to start RPC server
                if spec.cmd:
                    # Use custom command from spec
                    cmd = shlex.split(spec.cmd)
                else:
                    # Default: start RPC server
                    cmd = [
                        sys.executable,
                        "-m",
                        "areal.scheduler.rpc.rpc_server",
                        "--port",
                        str(ports[0]),  # Main RPC port
                    ]

                    # Add any additional arguments
                    if args:
                        cmd.extend(args)

                logger.debug(f"Starting worker {worker_id}: {' '.join(cmd)}")

                # Spawn subprocess
                try:
                    with open(log_file, "w") as log_f:
                        process = subprocess.Popen(
                            cmd,
                            env=env,
                            stdout=log_f,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,  # Create new process group
                        )
                except Exception as e:
                    self._cleanup_workers(workers)
                    raise WorkerCreationError(
                        role,
                        f"Failed to spawn subprocess for worker {idx}",
                        str(e),
                    ) from e

                # Check if process started successfully
                time.sleep(0.1)  # Brief delay to catch immediate failures
                if process.poll() is not None:
                    stderr = self._read_log_tail(log_file)
                    self._cleanup_workers(workers)
                    raise WorkerCreationError(
                        role,
                        f"Worker {worker_id} exited immediately with code {process.returncode}",
                        stderr,
                    )

                # Create worker info
                worker = Worker(
                    id=worker_id,
                    ip=gethostip(),
                    ports=[str(p) for p in ports],
                )

                worker_info = WorkerInfo(
                    worker=worker,
                    process=process,
                    role=role,
                    gpu_devices=gpu_devices,
                    created_at=time.time(),
                    log_file=str(log_file),
                    env_vars=env,
                )

                workers.append(worker_info)
                worker_ids.append(worker_id)
                logger.info(
                    f"Worker {worker_id} started (PID: {process.pid}, "
                    f"GPUs: {gpu_devices}, ports: {ports})"
                )

            # Store workers
            self._workers[role] = workers

            logger.info(
                f"Successfully created {len(workers)} workers for role '{role}'"
            )
            return worker_ids

        except Exception as e:
            # Clean up any workers created before the failure
            self._cleanup_workers(workers)
            if isinstance(e, SchedulerError):
                raise
            raise WorkerCreationError(role, "Unexpected error", str(e)) from e

    def get_workers(self, role: str, timeout: float | None = None) -> list[Worker]:
        """
        Get workers and wait for them to be ready.

        Args:
            role: Worker role name
            timeout: Maximum time to wait for workers to be ready (None = use default)

        Returns:
            List of Worker objects

        Raises:
            WorkerNotFoundError: If role doesn't exist
            WorkerFailedError: If any worker process failed
            WorkerTimeoutError: If timeout exceeded waiting for workers
        """
        if role not in self._workers:
            raise WorkerNotFoundError(role)

        workers = self._workers[role]
        timeout = timeout if timeout is not None else self.startup_timeout

        # First check that all processes are still alive
        self._check_worker_health(role)

        # Wait for RPC servers to be ready
        start_time = time.time()
        ready_workers = set()

        while len(ready_workers) < len(workers):
            if time.time() - start_time > timeout:
                raise WorkerTimeoutError(
                    role,
                    timeout,
                )

            for worker_info in workers:
                if worker_info.worker.id in ready_workers:
                    continue

                # Check if process is still alive
                if worker_info.process.poll() is not None:
                    stderr = self._read_log_tail(worker_info.log_file)
                    raise WorkerFailedError(
                        worker_info.worker.id,
                        worker_info.process.returncode,
                        stderr,
                    )

                # Check if RPC server is ready
                if self._is_worker_ready(worker_info):
                    ready_workers.add(worker_info.worker.id)
                    logger.debug(f"Worker {worker_info.worker.id} is ready")

            if len(ready_workers) < len(workers):
                time.sleep(self.health_check_interval)

        logger.info(f"All {len(workers)} workers for role '{role}' are ready")
        return [w.worker for w in workers]

    def _is_worker_ready(self, worker_info: WorkerInfo) -> bool:
        """Check if worker's RPC server is ready via HTTP health check."""
        port = int(worker_info.worker.ports[0])
        url = f"http://{worker_info.worker.ip}:{port}/health"

        try:
            response = self._http_client.get(url, timeout=2.0)
            return response.status_code == 200
        except Exception:
            return False

    def _check_worker_health(self, role: str):
        """
        Check health of all workers in a group.

        Raises:
            WorkerFailedError: If any worker has failed
        """
        if role not in self._workers:
            return

        for worker_info in self._workers[role]:
            returncode = worker_info.process.poll()
            if returncode is not None:
                stderr = self._read_log_tail(worker_info.log_file)
                raise WorkerFailedError(
                    worker_info.worker.id,
                    returncode,
                    stderr,
                )

    def delete_workers(self, role: str | None = None):
        """
        Delete workers and clean up resources.

        Args:
            role: Specific worker role to delete, or None to delete all
        """
        if role is None:
            # Delete all workers
            roles = list(self._workers.keys())
            for r in roles:
                self.delete_workers(r)
            return

        if role not in self._workers:
            logger.warning(f"Worker role '{role}' not found, skipping deletion")
            return

        workers = self._workers[role]
        logger.info(f"Deleting {len(workers)} workers for role '{role}'")

        self._cleanup_workers(workers)

        # Remove from tracking
        del self._workers[role]

        logger.info(f"Successfully deleted workers for role '{role}'")

    def _cleanup_workers(self, workers: list[WorkerInfo]):
        """Clean up worker processes and resources."""
        for worker_info in workers:
            try:
                # Release ports
                for port_str in worker_info.worker.ports:
                    self._allocated_ports.discard(int(port_str))

                # Terminate process tree
                self._terminate_process_tree(worker_info.process.pid)

                logger.debug(f"Cleaned up worker {worker_info.worker.id}")
            except Exception as e:
                logger.error(
                    f"Error cleaning up worker {worker_info.worker.id}: {e}",
                    exc_info=True,
                )

    def _terminate_process_tree(self, pid: int):
        """Terminate a process and all its children."""
        try:
            parent = psutil.Process(pid)
            children = parent.children(recursive=True)

            # Try graceful termination first
            for child in children:
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass

            try:
                parent.terminate()
            except psutil.NoSuchProcess:
                return

            # Wait for graceful termination
            _, alive = psutil.wait_procs([parent] + children, timeout=3)

            # Force kill remaining processes
            for proc in alive:
                try:
                    proc.kill()
                except psutil.NoSuchProcess:
                    pass

        except psutil.NoSuchProcess:
            # Process already gone
            pass
        except Exception as e:
            logger.warning(f"Error terminating process tree {pid}: {e}")

    def _read_log_tail(self, log_file: str, lines: int = 50) -> str:
        """Read the last N lines from a log file."""
        try:
            with open(log_file) as f:
                all_lines = f.readlines()
                return "".join(all_lines[-lines:])
        except Exception as e:
            return f"[Could not read log file: {e}]"

    async def create_engine(
        self,
        worker_id: str,
        engine: str,
        *args,
        **kwargs,
    ) -> Any:
        """
        Create an engine instance on a remote worker.

        The engine parameter is a string import path (e.g., "areal.engine.ppo.actor.FSDPPPOActor")
        that will be dynamically imported and instantiated on the worker.

        Args:
            worker_id: Worker ID in format "role/index"
            engine: Import path to the engine class (e.g., "areal.engine.ppo.actor.FSDPPPOActor")
            *args: Initialization arguments
            **kwargs: Initialization keyword arguments

        Returns:
            Result from engine initialization

        Raises:
            WorkerNotFoundError: If worker doesn't exist
            WorkerFailedError: If worker process has failed
            EngineCreationError: If engine creation fails
        """
        # Verify worker exists and is alive
        worker_info = self._verify_worker_alive(worker_id)

        # Validate engine is a string import path
        if not isinstance(engine, str):
            raise EngineCreationError(
                worker_id,
                f"Engine must be a string import path, got {type(engine)}",
            )

        # Build JSON payload
        payload = {
            "engine": engine,
            "init_args": list(args),
            "init_kwargs": kwargs,
        }

        # Send HTTP request to create engine
        port = int(worker_info.worker.ports[0])
        url = f"http://{worker_info.worker.ip}:{port}/create_engine"

        try:
            logger.info(f"Creating engine '{engine}' on worker '{worker_id}'")

            response = self._http_client.post(
                url,
                content=orjson.dumps(payload),
                headers={"Content-Type": "application/json"},
                timeout=300.0,
            )

            if response.status_code == 200:
                result = response.json()
                logger.info(f"Engine created successfully on worker '{worker_id}'")
                return result.get("result")
            elif response.status_code == 400:
                # Import error or bad request
                error_detail = response.json().get("detail", "Unknown error")
                if "Failed to import" in error_detail:
                    raise EngineImportError(engine, error_detail)
                else:
                    raise EngineCreationError(worker_id, error_detail, 400)
            elif response.status_code == 500:
                # Engine initialization failed
                error_detail = response.json().get("detail", "Unknown error")
                raise EngineCreationError(worker_id, error_detail, 500)
            else:
                raise EngineCreationError(
                    worker_id,
                    f"Unexpected status code: {response.status_code}",
                    response.status_code,
                )

        except httpx.ConnectError as e:
            # Check if worker died
            if worker_info.process.poll() is not None:
                stderr = self._read_log_tail(worker_info.log_file)
                raise WorkerFailedError(
                    worker_id, worker_info.process.returncode, stderr
                ) from e
            raise RPCConnectionError(
                worker_id, worker_info.worker.ip, port, str(e)
            ) from e

        except httpx.TimeoutException as e:
            raise EngineCreationError(worker_id, f"Request timed out: {e}") from e

        except (EngineCreationError, EngineImportError, RPCConnectionError):
            raise

        except Exception as e:
            raise EngineCreationError(worker_id, f"Unexpected error: {str(e)}") from e

    def call_engine(
        self,
        worker_id: str,
        method: str,
        *args,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs,
    ) -> Any:
        """
        Call a method on an engine.

        Args:
            worker_id: Worker ID in format "role/index"
            method: Method name to call
            *args: Method arguments
            max_retries: Maximum number of retry attempts
            retry_delay: Initial delay between retries (exponential backoff)
            **kwargs: Method keyword arguments

        Returns:
            Result from method call

        Raises:
            WorkerNotFoundError: If worker doesn't exist
            WorkerFailedError: If worker process has failed
            EngineCallError: If method call fails
        """
        # Get worker info (initial verification)
        worker_info = self._find_worker_by_id(worker_id)
        if worker_info is None:
            raise WorkerNotFoundError(worker_id)

        # Serialize args and kwargs (convert tensors to SerializedTensor dicts)
        serialized_args = serialize_value(list(args))
        serialized_kwargs = serialize_value(kwargs)

        # Build JSON payload
        payload = {
            "method": method,
            "args": serialized_args,
            "kwargs": serialized_kwargs,
        }

        # Retry logic with exponential backoff
        port = int(worker_info.worker.ports[0])
        url = f"http://{worker_info.worker.ip}:{port}/call"
        last_error = None

        for attempt in range(1, max_retries + 1):
            # Check worker health before each attempt
            if worker_info.process.poll() is not None:
                stderr = self._read_log_tail(worker_info.log_file)
                raise WorkerFailedError(
                    worker_id,
                    worker_info.process.returncode,
                    stderr,
                )

            try:
                logger.debug(
                    f"Calling method '{method}' on worker '{worker_id}' (attempt {attempt})"
                )

                response = self._http_client.post(
                    url,
                    content=orjson.dumps(payload),
                    headers={"Content-Type": "application/json"},
                    timeout=7200.0,  # 2 hours for long-running operations
                )

                result, should_retry, error_msg = self._handle_call_response(
                    response, worker_id, method, attempt
                )
                if not should_retry:
                    if attempt > 1:
                        logger.info(
                            f"Method '{method}' succeeded on worker '{worker_id}' "
                            f"after {attempt} attempts"
                        )
                    return result
                last_error = error_msg

            except Exception as e:
                last_error = self._handle_call_exception(e, worker_info, worker_id)

            # Retry with exponential backoff
            if attempt < max_retries:
                delay = retry_delay * (2 ** (attempt - 1))
                logger.warning(
                    f"Method '{method}' failed on worker '{worker_id}' "
                    f"(attempt {attempt}/{max_retries}): {last_error}. "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)

        # All retries exhausted
        raise EngineCallError(
            worker_id,
            method,
            last_error or "Max retries exceeded",
            attempt=max_retries,
        )

    async def async_call_engine(
        self,
        worker_id: str,
        method: str,
        *args,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs,
    ) -> Any:
        """
        Async version of call_engine for calling engine methods asynchronously.

        Args:
            worker_id: Worker ID in format "role/index"
            method: Method name to call
            *args: Method arguments
            max_retries: Maximum number of retry attempts
            retry_delay: Initial delay between retries (exponential backoff)
            **kwargs: Method keyword arguments

        Returns:
            Result from method call

        Raises:
            WorkerNotFoundError: If worker doesn't exist
            WorkerFailedError: If worker process has failed
            EngineCallError: If method call fails
        """
        # Get worker info (initial verification)
        worker_info = self._find_worker_by_id(worker_id)
        if worker_info is None:
            raise WorkerNotFoundError(worker_id)

        # Route to different endpoint based on method
        port = int(worker_info.worker.ports[0])
        if method == "run_workflow":
            # Special routing for workflow execution
            url = f"http://{worker_info.worker.ip}:{port}/run_workflow"
            # Serialize kwargs for workflow execution
            payload = serialize_value(kwargs)
        else:
            # Standard engine method call
            url = f"http://{worker_info.worker.ip}:{port}/call"
            # Serialize args and kwargs
            serialized_args = serialize_value(list(args))
            serialized_kwargs = serialize_value(kwargs)
            payload = {
                "method": method,
                "args": serialized_args,
                "kwargs": serialized_kwargs,
            }

        last_error = None

        print(url)
        for attempt in range(1, max_retries + 1):
            # Check worker health before each attempt
            if worker_info.process.poll() is not None:
                stderr = self._read_log_tail(worker_info.log_file)
                raise WorkerFailedError(
                    worker_id,
                    worker_info.process.returncode,
                    stderr,
                )

            try:
                logger.debug(
                    f"Async calling method '{method}' on worker '{worker_id}' (attempt {attempt})"
                )

                response = await self._async_http_client.post(
                    url,
                    content=orjson.dumps(payload),
                    headers={"Content-Type": "application/json"},
                    timeout=7200.0,  # 2 hours for long-running operations
                )
                print(response, payload, response.json())

                result, should_retry, error_msg = self._handle_call_response(
                    response, worker_id, method, attempt
                )
                if not should_retry:
                    if attempt > 1:
                        logger.info(
                            f"Method '{method}' succeeded on worker '{worker_id}' "
                            f"after {attempt} attempts"
                        )
                    return result
                last_error = error_msg

            except Exception as e:
                last_error = self._handle_call_exception(e, worker_info, worker_id)

            # Retry with exponential backoff
            if attempt < max_retries:
                delay = retry_delay * (2 ** (attempt - 1))
                logger.warning(
                    f"Method '{method}' failed on worker '{worker_id}' "
                    f"(attempt {attempt}/{max_retries}): {last_error}. "
                    f"Retrying in {delay:.1f}s..."
                )
                import asyncio

                await asyncio.sleep(delay)

        # All retries exhausted
        raise EngineCallError(
            worker_id,
            method,
            last_error or "Max retries exceeded",
            attempt=max_retries,
        )

    def _find_worker_by_id(self, worker_id: str) -> WorkerInfo | None:
        """Find a worker by its ID."""
        for workers in self._workers.values():
            for worker_info in workers:
                if worker_info.worker.id == worker_id:
                    return worker_info
        return None

    def _verify_worker_alive(self, worker_id: str) -> WorkerInfo:
        """
        Verify a worker exists and is alive.

        Args:
            worker_id: Worker ID to verify

        Returns:
            WorkerInfo object

        Raises:
            WorkerNotFoundError: If worker doesn't exist
            WorkerFailedError: If worker process has failed
        """
        worker_info = self._find_worker_by_id(worker_id)
        if worker_info is None:
            raise WorkerNotFoundError(worker_id)

        # Check if process has exited
        if worker_info.process.poll() is not None:
            stderr = self._read_log_tail(worker_info.log_file)
            raise WorkerFailedError(
                worker_id,
                worker_info.process.returncode,
                stderr,
            )

        return worker_info

    def _handle_call_response(
        self, response, worker_id: str, method: str, attempt: int
    ):
        """
        Handle HTTP response from engine call.

        Args:
            response: HTTP response object
            worker_id: Worker ID
            method: Method name being called
            attempt: Current retry attempt number

        Returns:
            Tuple of (result, should_retry, error_message)
            - result: The result from the call if successful, None otherwise
            - should_retry: Whether to retry the request
            - error_message: Error message if failed, None if successful
        """
        if response.status_code == 200:
            result = response.json().get("result")
            # Deserialize result (convert SerializedTensor dicts back to tensors)
            deserialized_result = deserialize_value(result)
            return deserialized_result, False, None
        elif response.status_code == 400:
            # Bad request (e.g., method doesn't exist) - don't retry
            error_detail = response.json().get("detail", "Unknown error")
            raise EngineCallError(worker_id, method, error_detail, attempt)
        elif response.status_code == 500:
            # Engine method failed - don't retry
            error_detail = response.json().get("detail", "Unknown error")
            raise EngineCallError(worker_id, method, error_detail, attempt)
        elif response.status_code == 503:
            # Service unavailable - retry
            return None, True, "Service unavailable"
        else:
            # Other errors - retry
            return None, True, f"HTTP {response.status_code}: {response.text}"

    def _handle_call_exception(
        self, e: Exception, worker_info: WorkerInfo, worker_id: str
    ) -> str:
        """
        Handle exceptions during engine calls and return error message.

        Args:
            e: The exception that occurred
            worker_info: Worker information
            worker_id: Worker ID

        Returns:
            Error message string

        Raises:
            WorkerFailedError: If worker has died
            EngineCallError: If non-retryable error
        """
        if isinstance(e, httpx.ConnectError):
            # Check if worker died
            if worker_info.process.poll() is not None:
                stderr = self._read_log_tail(worker_info.log_file)
                raise WorkerFailedError(
                    worker_id,
                    worker_info.process.returncode,
                    stderr,
                ) from e
            return f"Connection error: {e}"
        elif isinstance(e, httpx.TimeoutException):
            return f"Timeout: {e}"
        elif isinstance(e, EngineCallError):
            raise
        else:
            return f"Unexpected error: {e}"

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.delete_workers()
        except Exception:
            pass
        try:
            self._http_client.close()
        except Exception:
            pass
        try:
            import asyncio

            # Close async client if event loop is available
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self._async_http_client.aclose())
                else:
                    loop.run_until_complete(self._async_http_client.aclose())
            except RuntimeError:
                # No event loop, sync close
                pass
        except Exception:
            pass
