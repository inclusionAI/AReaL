import abc
import os
import shlex
import subprocess
import sys
import time
import traceback
from concurrent.futures import Future
from typing import Any

import ray
import requests

from areal.api import InferenceEngine, TrainEngine
from areal.api.cli_args import BaseExperimentConfig
from areal.infra.rpc.rtensor import RTensor
from areal.infra.rpc.serialization import deserialize_value, serialize_value
from areal.infra.utils.proc import kill_process_tree
from areal.utils import logging, name_resolve, seeding
from areal.utils.data import (
    broadcast_tensor_container,
    tensor_container_to,
)
from areal.utils.dynamic_import import import_from_string
from areal.utils.network import find_free_ports


class RayServer(abc.ABC):
    """
    Ray actor base class that all Ray actors under RayScheduler should inherit from
    """

    def __init__(self, config: BaseExperimentConfig, **kwargs):
        self._engines: dict[str, TrainEngine | InferenceEngine] = {}
        self._default_engine_name: str | None = None  # For backward compatibility
        self._allocated_port = set()
        self.config: BaseExperimentConfig = config

    def _get_device(self):
        # lazy resolve the device inside worker process
        from areal.infra.platforms import current_platform

        return current_platform.current_device()

    def _should_broadcast_payload(
        self,
        engine: TrainEngine | InferenceEngine,
        rpc_meta: dict[str, Any] | None,
    ) -> bool:
        default_broadcast = isinstance(engine, TrainEngine) and engine.initialized
        if rpc_meta is None:
            return default_broadcast
        if not isinstance(rpc_meta, dict):
            raise ValueError(
                f"Invalid rpc_meta: expected dict or None, got {type(rpc_meta)}"
            )
        broadcast = rpc_meta.get("broadcast", default_broadcast)
        if not isinstance(broadcast, bool):
            raise ValueError(
                f"Invalid rpc_meta.broadcast: expected bool, got {type(broadcast)}"
            )
        return broadcast

    def ping(self) -> str:
        return "ok"

    def alloc_ports(self, count: int):
        ports = find_free_ports(count, exclude_ports=self._allocated_port)
        self._allocated_port.update(ports)
        return ports

    def configure(self, config: BaseExperimentConfig, role: str, rank: int) -> None:
        name_resolve.reconfigure(config.cluster.name_resolve)
        # Set seed for any TrainEngine instances
        for engine in self._engines.values():
            if isinstance(engine, TrainEngine):
                seeding.set_random_seed(config.seed, key=f"{role}{rank}")
                break
        self.logger.info(f"RayRPCServer configured for role role={role}, rank={rank}")

    def set_env(self, env: dict[str, str]) -> None:
        for k, v in env.items():
            os.environ[str(k)] = str(v)

    def post_init(self, **kwargs) -> Any:
        # the HTTPLauncher needs this, but keeping this here for interface compatibility
        # launched after the actor has been deployed
        pass

    @abc.abstractmethod
    def create_engine(
        self,
        engine: str,
        *init_args,
        engine_name: str | None = None,
        **init_kwargs,
    ) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def call(self, method: str, *args, engine_name: str | None = None, **kwargs) -> Any:
        raise NotImplementedError()

    @abc.abstractmethod
    def destroy(self) -> None:
        raise NotImplementedError()

    def __ray_shutdown__(self):
        self.destroy()


@ray.remote
class RayRPCServer(RayServer):
    """
    Ray engine container. Represents either:
    - one training world rank, or
    - one rollout instance

    Supports multiple named engines per worker for colocation scenarios.

    Placement group scheduling is controlled by the scheduler.
    The actor is only responsible for the engine lifecycle and method calls
    within this process.
    """

    def __init__(self, config: BaseExperimentConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.logger = logging.getLogger("RayRPCServer")

    def create_engine(
        self,
        engine: str,
        *init_args,
        engine_name: str | None = None,
        **init_kwargs,
    ) -> None:
        try:
            engine_class = import_from_string(engine)
            self.logger.debug(f"Initializing engine {engine_class}")
            if not issubclass(engine_class, (TrainEngine, InferenceEngine)):
                raise TypeError(
                    f"Engine class must be a TrainEngine or InferenceEngine, but got {engine_class}"
                )
            engine = engine_class(*init_args, **init_kwargs)

            # Use engine_name if provided, otherwise generate a default name
            if engine_name is None:
                engine_name = f"engine_{len(self._engines)}"
            self._engines[engine_name] = engine

            # Track first engine as default for backward compatibility
            if self._default_engine_name is None:
                self._default_engine_name = engine_name

            self.logger.info(
                f"RayRPCServer Engine '{engine}' instantiated as '{engine_name}'!"
            )
        except Exception as e:
            self.logger.error(
                f"RayRPCServer failed to create engine '{engine}' : {e}\n"
                f"{traceback.format_exc()}"
            )
            raise

    def call(
        self,
        method: str,
        *args,
        engine_name: str | None = None,
        rpc_meta: dict[str, Any] | None = None,
        **kwargs,
    ) -> Any:
        self.logger.debug(
            f"Calling {method} on engine '{engine_name}' with arguments {args=} {kwargs=}"
        )

        # Resolve engine: use provided name, fallback to default, or error
        if engine_name is None:
            engine_name = self._default_engine_name
        if engine_name is None or engine_name not in self._engines:
            raise RuntimeError(
                f"Engine '{engine_name}' not found. "
                f"Available engines: {list(self._engines.keys())}. "
                "Call create_engine() first."
            )
        engine = self._engines[engine_name]

        raw_args = list(args)
        raw_kwargs = kwargs.copy()
        # Fetch remote tensors
        args = RTensor.localize(raw_args)
        kwargs = RTensor.localize(raw_kwargs)

        try:
            should_broadcast = self._should_broadcast_payload(
                engine=engine, rpc_meta=rpc_meta
            )
            if should_broadcast:
                device = self._get_device()
                args = tensor_container_to(args, device)
                args = broadcast_tensor_container(
                    args,
                    src_rank=engine.current_data_parallel_head(),
                    group=engine.context_and_model_parallel_group,
                )
                kwargs = tensor_container_to(kwargs, device)
                kwargs = broadcast_tensor_container(
                    kwargs,
                    src_rank=engine.current_data_parallel_head(),
                    group=engine.context_and_model_parallel_group,
                )
        except Exception as e:
            self.logger.error(
                f"RayRPCServer broadcast failed for '{method}': {e}\n"
                f"{traceback.format_exc()}"
            )
            raise

        try:
            fn = getattr(engine, method)
            # Re-establish current device in RPC execution context before
            # invoking engine methods that may issue object collectives.
            if (
                isinstance(engine, TrainEngine)
                and engine.initialized
                and self._get_device().type != "cpu"
            ):
                from areal.infra.platforms import current_platform

                current_platform.set_device(current_platform.current_device())
            result = fn(*args, **kwargs)
            if isinstance(result, Future):
                result = result.result()
            # Convert all tensors to RTensors and store the tensor locally
            result = RTensor.remotize(result, node_addr="")
            # put back to cpu to mimic RPCServer encode/decode
            result = tensor_container_to(result, "cpu")
            self.logger.debug(f"Successfully completed RayRPCServer call {result}")
            return result
        except Exception as e:
            self.logger.error(
                f"RayRPCServer Engine method '{method}' failed: {e}\n"
                f"{traceback.format_exc()}"
            )
            raise

    def destroy(self) -> None:
        # Destroy all engines
        for engine_name, engine in list(self._engines.items()):
            try:
                engine.destroy()
                self.logger.info(f"RayRPCServer Engine '{engine_name}' destroyed")
            except Exception as e:
                self.logger.error(
                    f"RayRPCServer error destroying engine '{engine_name}': {e}"
                )
        self._engines.clear()
        self._default_engine_name = None
        ray.actor.exit_actor()


@ray.remote
class RayHTTPLauncher(RayServer):
    """
    Ray implementation of a launcher to launch proxy servers and any HTTP servers
    """

    REQUIRED_ARGS = ("command", "worker_index", "role")

    def __init__(self, config: BaseExperimentConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.logger = logging.getLogger("RayHTTPLauncher")

        missing = [k for k in self.REQUIRED_ARGS if k not in kwargs]
        if missing:
            raise TypeError(f"Missing required kwargs: {missing}")

        self.command = kwargs["command"]
        self.worker_index = kwargs["worker_index"]
        self.role = kwargs["role"]
        self.worker_ip = ray.util.get_node_ip_address()
        self.worker_port = None
        self.worker_process: subprocess.Popen | None = None

    def post_init(self, **kwargs):
        self.worker_port = kwargs.get("port", self.alloc_ports(1)[0])
        self.worker_process = self.launch_server(port=self.worker_port)

    def create_engine(
        self,
        engine: str,
        *init_args,
        engine_name: str | None = None,
        **init_kwargs,
    ) -> None:
        self.logger.debug(f"Initializing engine {engine}")
        payload = {
            "engine": engine,
            "engine_name": engine_name,
            "init_args": serialize_value(list(init_args)),
            "init_kwargs": serialize_value(init_kwargs),
        }
        return self._post_request("create_engine", payload)

    def call(self, method: str, *args, engine_name: str | None = None, **kwargs) -> Any:
        self.logger.debug(
            f"Calling {method} on engine '{engine_name}' with arguments {args=} {kwargs=}"
        )

        payload = {
            "method": method,
            "engine_name": engine_name,
            "args": serialize_value(list(args)),
            "kwargs": serialize_value(kwargs),
        }
        return self._post_request("call", payload)

    def destroy(self) -> None:
        if self.worker_process and self.worker_process.poll() is None:
            kill_process_tree(self.worker_process.pid, timeout=3, graceful=True)

        # Destroy all engines
        for engine_name, engine in list(self._engines.items()):
            try:
                engine.destroy()
                self.logger.info(f"RayRPCServer Engine '{engine_name}' destroyed")
            except Exception as e:
                self.logger.error(
                    f"RayRPCServer error destroying engine '{engine_name}': {e}"
                )
        self._engines.clear()
        self._default_engine_name = None
        ray.actor.exit_actor()

    def launch_server(self, port):
        # keeping this as a separate function to support Awex server launches later
        if not self.command:
            raise RuntimeError(
                f"Command was not given to {self.__class__.__name__}.launch_server. Cannot launch without command."
            )

        cmd = [sys.executable, "-m"]
        cmd.extend(shlex.split(self.command))
        cmd.extend(["--port", str(port)])

        cmd.extend(["--experiment-name", self.config.experiment_name])
        cmd.extend(["--trial-name", self.config.trial_name])
        cmd.extend(["--role", self.role])
        cmd.extend(["--worker-index", str(self.worker_index)])

        cluster_config = self.config.cluster
        name_resolve = self.config.cluster.name_resolve

        cmd.extend(["--name-resolve-type", name_resolve.type])
        cmd.extend(["--nfs-record-root", name_resolve.nfs_record_root])
        cmd.extend(["--etcd3-addr", name_resolve.etcd3_addr])
        cmd.extend(["--fileroot", str(cluster_config.fileroot)])

        _env = os.environ.copy()
        self.worker_process = subprocess.Popen(
            cmd, env=_env, stdout=sys.stdout, stderr=subprocess.STDOUT
        )

        try:
            self._check_health()
        except Exception as e:
            self.logger.error(e)
            kill_process_tree(self.worker_process.pid, timeout=3, graceful=True)
            raise RuntimeError(f"Could not launch server with command {cmd}")

        return self.worker_process

    def _post_request(
        self,
        endpoint,
        payload,
        http_timeout: float = 7200.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        url = f"{self.url}/{endpoint}"
        # adapted from local scheduler
        for attempt in range(1, max_retries + 1):
            if self.worker_process and self.worker_process.poll() is not None:
                raise RuntimeError("Worker has terminated")

            response = requests.post(url, json=payload, timeout=http_timeout)

            if response.status_code == 200:
                result = response.json().get("result")
                deserialized_result = deserialize_value(result)
                return deserialized_result
            elif response.status_code in [400, 500]:
                error_detail = response.json().get("detail", "unknown error")
                return error_detail

            # otherwise retry
            if attempt < max_retries:
                delay = retry_delay * (2 ** (attempt - 1))
                self.logger.warning(
                    f"Calling url {url} failed on actor '{ray.runtime_context.get_runtime_context().current_actor}' "
                    f"(attempt {attempt}/{max_retries}): {response.json()}. "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)

    @property
    def url(self):
        return f"http://{self.worker_ip}:{self.worker_port}"

    def _check_health(self, timeout: float = 60.0):
        url = f"{self.url}/health"
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.worker_process and self.worker_process.poll() is not None:
                raise RuntimeError("Server process exited before becoming healthy")

            try:
                r = requests.get(url, timeout=2.0)
                if r.status_code == 200:
                    return
            except requests.RequestException:
                # expected during startup
                pass
            time.sleep(1)

        raise RuntimeError(f"Health check timed out for {url}")
