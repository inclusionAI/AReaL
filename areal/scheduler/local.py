import asyncio
import getpass
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiohttp
import orjson
import psutil
import requests

from areal.api.cli_args import BaseExperimentConfig
from areal.api.scheduler_api import Job, Scheduler, SchedulingSpec, Worker
from areal.platforms import current_platform
from areal.scheduler.exceptions import (
    EngineCallError,
    EngineCreationError,
    EngineImportError,
    GPUAllocationError,
    PortAllocationError,
    RPCConnectionError,
    SchedulerError,
    WorkerConfigurationError,
    WorkerCreationError,
    WorkerFailedError,
    WorkerNotFoundError,
    WorkerTimeoutError,
)
from areal.scheduler.rpc.serialization import deserialize_value, serialize_value
from areal.utils import logging
from areal.utils.http import get_default_connector
from areal.utils.launcher import (
    get_env_vars,
)
from areal.utils.network import find_free_ports, gethostip

logger = logging.getLogger("LocalScheduler")


@dataclass
class WorkerInfo:
    worker: Worker
    process: subprocess.Popen
    role: str
    gpu_devices: list[int]
    created_at: float
    log_file: str
    env_vars: dict[str, str] = field(default_factory=dict)


class LocalScheduler(Scheduler):
    """Local scheduler that manages worker subprocesses on a single GPU node.

    This scheduler spawns worker processes running RPC servers and manages their lifecycle.
    It supports different worker types through a unified interface with dynamic port allocation,
    round-robin GPU assignment, process health monitoring, and graceful cleanup.
    """

    def __init__(
        self,
        gpu_devices: list[int] | None = None,
        log_dir: str | None = None,
        startup_timeout: float = 30.0,
        health_check_interval: float = 1.0,
        *,
        fileroot: str | None = None,
        experiment_name: str | None = None,
        trial_name: str | None = None,
        exp_config: BaseExperimentConfig | None = None,
    ):
        self.gpu_devices = gpu_devices or self._detect_gpus()
        if log_dir is not None:
            self.log_dir = Path(log_dir)
        else:
            experiment_name = experiment_name or exp_config.experiment_name
            trial_name = trial_name or exp_config.trial_name
            fileroot = fileroot or exp_config.cluster.fileroot
            assert experiment_name is not None
            assert trial_name is not None
            assert fileroot is not None
            self.log_dir = (
                Path(fileroot)
                / "logs"
                / getpass.getuser()
                / experiment_name
                / trial_name
            )
        self.exp_config = exp_config

        self.startup_timeout = startup_timeout
        self.health_check_interval = health_check_interval

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._workers: dict[str, list[WorkerInfo]] = {}
        self._gpu_counter = 0
        self._allocated_ports = set()
        self._http_client = requests.Session()

        logger.info(
            f"LocalScheduler initialized with GPU devices: {self.gpu_devices}, "
            f"log directory: {self.log_dir}"
        )

    def _detect_gpus(self) -> list[int]:
        cuda_visible = os.environ.get(current_platform.device_control_env_var)
        if current_platform.device_control_env_var and cuda_visible:
            try:
                return [int(x) for x in cuda_visible.split(",")]
            except ValueError:
                logger.warning(
                    f"Invalid {current_platform.device_control_env_var}: {cuda_visible}, using default [0]"
                )
                return [0]
        return [0]

    def _allocate_gpus(self, num_gpus: int) -> list[int]:
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
        try:
            ports = find_free_ports(count, exclude_ports=set(self._allocated_ports))
            self._allocated_ports.update(ports)
            return ports
        except ValueError as e:
            raise PortAllocationError(str(e)) from e

    def _prepare_worker_specs(
        self, role: str, num_workers: int, schedulings: list[SchedulingSpec] | None
    ) -> list[SchedulingSpec]:
        if not schedulings:
            return [
                SchedulingSpec(
                    cpu=1,
                    mem=1024,
                    gpu=1,
                    port_count=2,
                    cmd="python -m areal.scheduler.rpc.rpc_server",
                )
            ] * num_workers

        if len(schedulings) == 1:
            return [schedulings[0]] * num_workers

        if len(schedulings) == num_workers:
            return schedulings

        raise WorkerCreationError(
            role,
            "Invalid configuration",
            f"schedulings length ({len(schedulings)}) must be 1 or equal to replicas ({num_workers})",
        )

    def create_workers(self, job: Job, *args, **kwargs) -> list[str]:
        """Create worker subprocesses.

        Parameters
        ----------
        job : Job
            Job configuration with role, replicas, tasks, and scheduling strategy
        *args
            Additional arguments passed to worker command
        **kwargs
            Additional keyword arguments

        Returns
        -------
        list[str]
            List of worker IDs created (e.g., ["rollout/0", "rollout/1"])

        Raises
        ------
        WorkerCreationError
            If worker creation fails
        GPUAllocationError
            If GPU allocation fails
        PortAllocationError
            If port allocation fails
        """
        role = job.role
        if role in self._workers:
            raise WorkerCreationError(
                role,
                "Worker group already exists",
                f"Use delete_workers('{role}') first to remove existing workers",
            )

        num_workers = job.replicas
        if num_workers == 0:
            raise WorkerCreationError(
                role, "Invalid configuration", "replicas must be greater than 0"
            )

        schedulings = self._prepare_worker_specs(role, num_workers, job.tasks)

        strategy = job.scheduling_strategy
        if strategy is None:
            strategy_type = "separation"
            colocate_role = None
        else:
            strategy_type = strategy.type or "separation"
            colocate_role = strategy.target if strategy_type == "colocation" else None

        logger.info(
            f"Creating {num_workers} workers for role '{role}' "
            f"(strategy: {strategy_type}, colocate_with: {colocate_role})"
        )

        workers = []
        worker_ids = []
        try:
            for idx in range(num_workers):
                worker_id = f"{role}/{idx}"
                scheduling = schedulings[idx]

                try:
                    if strategy_type == "colocation":
                        if not colocate_role:
                            raise WorkerCreationError(
                                role,
                                "Invalid strategy",
                                "Colocation strategy requires target role to be specified",
                            )
                        gpu_devices = self._get_colocated_gpus(colocate_role, idx)
                        logger.debug(
                            f"Worker {worker_id} colocated with {colocate_role}/{idx} on GPUs {gpu_devices}"
                        )
                    else:  # "separation" or default
                        gpu_devices = self._allocate_gpus(scheduling.gpu)
                        logger.debug(
                            f"Worker {worker_id} allocated new GPUs {gpu_devices}"
                        )

                    ports = self._allocate_ports(scheduling.port_count)
                except (
                    GPUAllocationError,
                    PortAllocationError,
                    WorkerNotFoundError,
                    ValueError,
                ) as e:
                    self._cleanup_workers(workers)
                    raise WorkerCreationError(
                        role, f"Resource allocation failed for worker {idx}", str(e)
                    ) from e

                env = get_env_vars(
                    "",
                    ",".join([f"{k}={v}" for k, v in scheduling.env_vars.items()]),
                )
                env[current_platform.device_control_env_var] = ",".join(
                    map(str, gpu_devices)
                )

                if scheduling.env_vars:
                    env.update(scheduling.env_vars)

                log_file = self.log_dir / f"{worker_id.replace('/', '_')}.log"

                if not scheduling.cmd:
                    self._cleanup_workers(workers)
                    raise WorkerCreationError(
                        role,
                        f"SchedulingSpec.cmd is required but not set for worker {worker_id}",
                        "Specify either 'python -m areal.scheduler.rpc.rpc_server' or "
                        "'python -m areal.scheduler.rpc.rpc_server' in your config.",
                    )

                if "--port" in scheduling.cmd:
                    raise WorkerCreationError(
                        role,
                        "Custom command should not include --port argument",
                        "The scheduler automatically allocates and provides the port.",
                    )
                cmd = shlex.split(scheduling.cmd)
                cmd.extend(["--port", str(ports[0])])

                logger.info(f"Starting worker {worker_id}: {' '.join(cmd)}")
                if cmd[0].startswith("python"):
                    cmd[0] = sys.executable

                cmd = (
                    " ".join(str(k) + "=" + str(v) for k, v in env.items())
                    + " stdbuf -oL "
                    + " ".join(cmd)
                )
                cmd = f"{cmd} 2>&1 | tee -a {log_file}"
                try:
                    process = subprocess.Popen(
                        cmd,
                        shell=isinstance(cmd, str),
                        stdout=sys.stdout,
                        stderr=sys.stdout,
                    )
                except Exception as e:
                    self._cleanup_workers(workers)
                    raise WorkerCreationError(
                        role,
                        f"Failed to spawn subprocess for worker {idx}",
                        str(e),
                    ) from e

                time.sleep(0.1)
                if process.poll() is not None:
                    stderr = self._read_log_tail(log_file)
                    self._cleanup_workers(workers)
                    raise WorkerCreationError(
                        role,
                        f"Worker {worker_id} exited immediately with code {process.returncode}",
                        stderr,
                    )

                worker = Worker(
                    id=worker_id,
                    ip=gethostip(),
                    worker_ports=[str(p) for p in ports],
                    engine_ports=[],
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

            self._workers[role] = workers

            logger.info(
                f"Successfully created {len(workers)} workers for role '{role}'"
            )

        except Exception as e:
            self._cleanup_workers(workers)
            if isinstance(e, SchedulerError):
                raise
            raise WorkerCreationError(role, "Unexpected error", str(e)) from e

        for worker_rank, worker_info in enumerate(workers):
            self._configure_worker(worker_info, worker_rank)

        return worker_ids

    def get_workers(self, role: str, timeout: float | None = None) -> list[Worker]:
        """Get workers and wait for them to be ready.

        Parameters
        ----------
        role : str
            Worker role name
        timeout : float, optional
            Maximum time to wait for workers to be ready (None = use default)

        Returns
        -------
        list[Worker]
            List of Worker objects

        Raises
        ------
        WorkerNotFoundError
            If role doesn't exist
        WorkerFailedError
            If any worker process failed
        WorkerTimeoutError
            If timeout exceeded waiting for workers
        """
        if role not in self._workers:
            raise WorkerNotFoundError(role)

        workers = self._workers[role]
        timeout = timeout if timeout is not None else self.startup_timeout

        self._check_worker_health(role)

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

                if worker_info.process.poll() is not None:
                    stderr = self._read_log_tail(worker_info.log_file)
                    raise WorkerFailedError(
                        worker_info.worker.id,
                        worker_info.process.returncode,
                        stderr,
                    )

                if self._is_worker_ready(worker_info):
                    ready_workers.add(worker_info.worker.id)
                    logger.debug(f"Worker {worker_info.worker.id} is ready")

            if len(ready_workers) < len(workers):
                time.sleep(self.health_check_interval)

        logger.info(f"All {len(workers)} workers for role '{role}' are ready")
        return [w.worker for w in workers]

    def _is_worker_ready(self, worker_info: WorkerInfo) -> bool:
        port = int(worker_info.worker.worker_ports[0])
        url = f"http://{worker_info.worker.ip}:{port}/health"

        try:
            response = self._http_client.get(url, timeout=2.0)
            return response.status_code == 200
        except Exception:
            return False

    def _configure_worker(self, worker_info: WorkerInfo, worker_rank: int):
        while not self._is_worker_ready(worker_info):
            time.sleep(0.1)

        worker_id = worker_info.worker.id
        port = int(worker_info.worker.worker_ports[0])
        url = f"http://{worker_info.worker.ip}:{port}/configure"

        try:
            response = self._http_client.post(
                url,
                data=orjson.dumps(
                    serialize_value(
                        dict(
                            config=self.exp_config,
                            role=worker_info.role,
                            rank=worker_rank,
                        )
                    )
                ),
                headers={"Content-Type": "application/json"},
                timeout=300.0,
            )

            if response.status_code == 200:
                logger.info(f"Configuration successfully on worker '{worker_id}'")
                return
            elif response.status_code == 400:
                error_detail = response.json().get("detail", "Unknown error")
                raise WorkerConfigurationError(worker_id, error_detail, str(400))
            elif response.status_code == 500:
                error_detail = response.json().get("detail", "Unknown error")
                raise WorkerConfigurationError(worker_id, error_detail, str(500))
            else:
                raise WorkerConfigurationError(
                    worker_id,
                    f"Unexpected status code: {response.status_code}",
                    str(response.status_code),
                )

        except requests.exceptions.ConnectionError as e:
            if worker_info.process.poll() is not None:
                stderr = self._read_log_tail(worker_info.log_file)
                raise WorkerFailedError(
                    worker_id, worker_info.process.returncode, stderr
                ) from e
            raise RPCConnectionError(
                worker_id, worker_info.worker.ip, port, str(e)
            ) from e

        except requests.exceptions.Timeout as e:
            raise WorkerConfigurationError(worker_id, f"Request timed out: {e}") from e

        except WorkerConfigurationError:
            raise

        except Exception as e:
            raise WorkerConfigurationError(
                worker_id, f"Unexpected error: {str(e)}"
            ) from e

    def _check_worker_health(self, role: str):
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
        """Delete workers and clean up resources.

        Parameters
        ----------
        role : str, optional
            Specific worker role to delete, or None to delete all
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

        del self._workers[role]

        logger.info(f"Successfully deleted workers for role '{role}'")

    def _cleanup_workers(self, workers: list[WorkerInfo]):
        for worker_info in workers:
            try:
                for port_str in worker_info.worker.worker_ports:
                    self._allocated_ports.discard(int(port_str))

                self._terminate_process_tree(worker_info.process.pid)

                logger.debug(f"Cleaned up worker {worker_info.worker.id}")
            except Exception as e:
                logger.error(
                    f"Error cleaning up worker {worker_info.worker.id}: {e}",
                    exc_info=True,
                )

    def _terminate_process_tree(self, pid: int):
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
        except psutil.Error as e:
            logger.warning(f"Error terminating process tree {pid}: {e}", exc_info=True)
        except Exception:
            import traceback

            logger.warning(
                f"Error terminating process tree {pid}: {traceback.format_exc()}"
            )

    def _read_log_tail(self, log_file: str, lines: int = 50) -> str:
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
        """Create an engine instance on a remote worker.

        The engine parameter is a string import path (e.g., "areal.engine.ppo.actor.FSDPPPOActor")
        that will be dynamically imported and instantiated on the worker.

        Parameters
        ----------
        worker_id : str
            Worker ID in format "role/index"
        engine : str
            Import path to the engine class (e.g., "areal.engine.ppo.actor.FSDPPPOActor")
        *args
            Initialization arguments
        **kwargs
            Initialization keyword arguments

        Returns
        -------
        Any
            Result from engine initialization

        Raises
        ------
        WorkerNotFoundError
            If worker doesn't exist
        WorkerFailedError
            If worker process has failed
        EngineCreationError
            If engine creation fails
        """
        # Verify worker exists and is alive
        worker_info = self._verify_worker_alive(worker_id)

        # Validate engine is a string import path
        if not isinstance(engine, str):
            raise EngineCreationError(
                worker_id,
                f"Engine must be a string import path, got {type(engine)}",
            )

        # Build JSON payload with serialized args and kwargs
        payload = {
            "engine": engine,
            "init_args": serialize_value(list(args)),
            "init_kwargs": serialize_value(kwargs),
        }

        # Send HTTP request to create engine
        port = int(worker_info.worker.worker_ports[0])
        url = f"http://{worker_info.worker.ip}:{port}/create_engine"

        try:
            logger.info(f"Creating engine '{engine}' on worker '{worker_id}'")

            timeout = aiohttp.ClientTimeout(total=300.0)
            async with aiohttp.ClientSession(
                timeout=timeout,
                read_bufsize=1024 * 1024 * 10,
                connector=get_default_connector(),
            ) as session:
                async with session.post(
                    url,
                    data=orjson.dumps(payload),
                    headers={"Content-Type": "application/json"},
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(
                            f"Engine created successfully on worker '{worker_id}'"
                        )
                        return result.get("result")
                    elif response.status == 400:
                        # Import error or bad request
                        error_detail = (await response.json()).get(
                            "detail", "Unknown error"
                        )
                        if "Failed to import" in error_detail:
                            raise EngineImportError(engine, error_detail)
                        else:
                            raise EngineCreationError(worker_id, error_detail, 400)
                    elif response.status == 500:
                        # Engine initialization failed
                        error_detail = (await response.json()).get(
                            "detail", "Unknown error"
                        )
                        raise EngineCreationError(worker_id, error_detail, 500)
                    else:
                        raise EngineCreationError(
                            worker_id,
                            f"Unexpected status code: {response.status}",
                            response.status,
                        )

        except (aiohttp.ClientConnectionError, aiohttp.ClientConnectorError) as e:
            if worker_info.process.poll() is not None:
                stderr = self._read_log_tail(worker_info.log_file)
                raise WorkerFailedError(
                    worker_id, worker_info.process.returncode, stderr
                ) from e
            raise RPCConnectionError(
                worker_id, worker_info.worker.ip, port, str(e)
            ) from e

        except asyncio.TimeoutError as e:
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
        """Call a method on an engine.

        Parameters
        ----------
        worker_id : str
            Worker ID in format "role/index"
        method : str
            Method name to call
        *args
            Method arguments
        max_retries : int, optional
            Maximum number of retry attempts, by default 3
        retry_delay : float, optional
            Initial delay between retries (exponential backoff), by default 1.0
        **kwargs
            Method keyword arguments

        Returns
        -------
        Any
            Result from method call

        Raises
        ------
        WorkerNotFoundError
            If worker doesn't exist
        WorkerFailedError
            If worker process has failed
        EngineCallError
            If method call fails
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
        port = int(worker_info.worker.worker_ports[0])
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
        """Async version of call_engine for calling engine methods asynchronously.

        Parameters
        ----------
        worker_id : str
            Worker ID in format "role/index"
        method : str
            Method name to call
        *args
            Method arguments
        max_retries : int, optional
            Maximum number of retry attempts, by default 3
        retry_delay : float, optional
            Initial delay between retries (exponential backoff), by default 1.0
        **kwargs
            Method keyword arguments

        Returns
        -------
        Any
            Result from method call

        Raises
        ------
        WorkerNotFoundError
            If worker doesn't exist
        WorkerFailedError
            If worker process has failed
        EngineCallError
            If method call fails
        """
        # Get worker info (initial verification)
        worker_info = self._find_worker_by_id(worker_id)
        if worker_info is None:
            raise WorkerNotFoundError(worker_id)

        # Route to different endpoint based on method
        port = int(worker_info.worker.worker_ports[0])
        if method == "export_stats":
            url = f"http://{worker_info.worker.ip}:{port}/export_stats"
            payload = None
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
                logger.info(
                    f"Async calling method '{method}' on worker '{worker_id}' (attempt {attempt})"
                )

                timeout = aiohttp.ClientTimeout(total=7200.0)
                async with aiohttp.ClientSession(
                    timeout=timeout,
                    read_bufsize=1024 * 1024 * 10,
                    connector=get_default_connector(),
                ) as session:
                    async with session.post(
                        url,
                        data=orjson.dumps(payload),
                        headers={"Content-Type": "application/json"},
                    ) as response:
                        # Handle response inline since aiohttp json() is async
                        if response.status == 200:
                            result_data = (await response.json()).get("result")
                            deserialized_result = deserialize_value(result_data)
                            if attempt > 1:
                                logger.info(
                                    f"Method '{method}' succeeded on worker '{worker_id}' "
                                    f"after {attempt} attempts"
                                )
                            return deserialized_result
                        elif response.status == 400:
                            # Bad request (e.g., method doesn't exist) - don't retry
                            error_detail = (await response.json()).get(
                                "detail", "Unknown error"
                            )
                            raise EngineCallError(
                                worker_id, method, error_detail, attempt
                            )
                        elif response.status == 500:
                            # Engine method failed - don't retry
                            error_detail = (await response.json()).get(
                                "detail", "Unknown error"
                            )
                            raise EngineCallError(
                                worker_id, method, error_detail, attempt
                            )
                        elif response.status == 503:
                            # Service unavailable - retry
                            last_error = "Service unavailable"
                        else:
                            # Other errors - retry
                            response_text = await response.text()
                            last_error = f"HTTP {response.status}: {response_text}"

            except (aiohttp.ClientConnectionError, aiohttp.ClientConnectorError) as e:
                # Check if worker died
                if worker_info.process.poll() is not None:
                    stderr = self._read_log_tail(worker_info.log_file)
                    raise WorkerFailedError(
                        worker_id,
                        worker_info.process.returncode,
                        stderr,
                    ) from e
                last_error = f"Connection error: {e}"
            except asyncio.TimeoutError as e:
                last_error = f"Timeout: {e}"
            except EngineCallError:
                raise
            except Exception as e:
                last_error = f"Unexpected error: {e}"

            # Retry with exponential backoff
            if attempt < max_retries:
                delay = retry_delay * (2 ** (attempt - 1))
                logger.warning(
                    f"Method '{method}' failed on worker '{worker_id}' "
                    f"(attempt {attempt}/{max_retries}): {last_error}. "
                    f"Retrying in {delay:.1f}s..."
                )

                await asyncio.sleep(delay)

        # All retries exhausted
        raise EngineCallError(
            worker_id,
            method,
            last_error or "Max retries exceeded",
            attempt=max_retries,
        )

    def _find_worker_by_id(self, worker_id: str) -> WorkerInfo | None:
        for workers in self._workers.values():
            for worker_info in workers:
                if worker_info.worker.id == worker_id:
                    return worker_info
        return None

    def _verify_worker_alive(self, worker_id: str) -> WorkerInfo:
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
        if isinstance(e, requests.exceptions.ConnectionError):
            # Check if worker died
            if worker_info.process.poll() is not None:
                stderr = self._read_log_tail(worker_info.log_file)
                raise WorkerFailedError(
                    worker_id,
                    worker_info.process.returncode,
                    stderr,
                ) from e
            return f"Connection error: {e}"
        elif isinstance(e, requests.exceptions.Timeout):
            return f"Timeout: {e}"
        elif isinstance(e, EngineCallError):
            raise
        else:
            return f"Unexpected error: {e}"

    def __del__(self):
        try:
            self.delete_workers()
        except Exception:
            pass
        try:
            self._http_client.close()
        except Exception:
            pass
