"""GatewayRolloutController — parallel implementation to RolloutController.

Routes inference and pause/continue traffic through the gateway HTTP stack
(Gateway → Router → Data Proxy → inference backend).
All servers are launched as worker processes via the scheduler.  Inference
server processes are forked through RPCGuard (a lightweight process manager).
"""

from __future__ import annotations

import sys
import time
import traceback
from collections.abc import AsyncGenerator
from threading import Lock
from typing import TYPE_CHECKING, Any

from openai.types.chat import ChatCompletion, ChatCompletionChunk

if TYPE_CHECKING:
    from areal.api.scheduler_api import Scheduler, Worker

from areal.api.io_struct import LocalInfServerInfo
from areal.experimental.rollout_service.controller.config import GatewayControllerConfig
from areal.utils import logging

logger = logging.getLogger("RolloutController")


class GatewayRolloutController:
    """Rollout controller that routes everything through the gateway HTTP stack.

    This is a **parallel** implementation to ``RolloutController`` (NOT a
    subclass).  It is duck-type compatible: the trainer can use either one
    without code changes.

    All servers (inference backend, Router, Data Proxy, Gateway) are launched
    as worker sub-processes via the scheduler.  The controller talks to them
    directly over HTTP — no engine creation or RPC calls on workers.

    The inference backend is determined from ``alloc_mode.gen_backend``
    (currently ``"sglang"`` is supported; ``"vllm"`` is planned).
    """

    # Worker role suffixes for each service type
    _INF_SUFFIX = "-inf"
    _ROUTER_SUFFIX = "-router"
    _DATA_PROXY_SUFFIX = "-data-proxy"
    _GATEWAY_SUFFIX = "-gateway"

    def __init__(
        self,
        config: GatewayControllerConfig,
        scheduler: Scheduler,
    ) -> None:
        self.config = config
        self.scheduler = scheduler

        # Worker management
        self.workers: list[Worker] = []
        self.server_infos: list[LocalInfServerInfo] = []
        self._worker_role: str = ""

        # Addresses resolved after initialization
        self._inf_addrs: list[str] = []
        self._router_addr: str = ""
        self._data_proxy_addrs: list[str] = []
        self._gateway_addr: str = ""

        # Worker ID mapping (data proxy addr → router-assigned worker_id)
        self._worker_ids: dict[str, str] = {}  # dp_addr -> worker_id

        # Version management
        self._version_lock = Lock()
        self._version = 0

        # WorkflowExecutor (created in initialize)
        self._workflow_executor = None

        # Staleness manager (created in initialize)
        self._staleness_manager = None

        # Track which service roles were created for cleanup
        self._service_roles: list[str] = []

        # Track services forked directly via RPCGuard /fork (raw_cmd mode).
        # Each entry: (guard_addr, role, worker_index) for /kill_forked_worker.
        self._forked_services: list[tuple[str, str, int]] = []

        # Proxy compatibility (no-ops — gateway IS the proxy)
        self._proxy_started = False
        self.proxy_workers: list = []
        self.proxy_addrs: list[str] = []

    # -- Initialize --------------------------------------------------------

    def initialize(
        self,
        role: str,
        alloc_mode: Any = None,
        server_args: dict[str, Any] | None = None,
        server_infos: list[LocalInfServerInfo] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        from areal.infra.utils.concurrent import run_async_task

        self._worker_role = role
        run_async_task(
            self._async_initialize,
            alloc_mode,
            server_args,
            server_infos,
            *args,
            **kwargs,
        )

        # Register data proxies in the router
        self._register_data_proxies_in_router()

        # Create WorkflowExecutor directly (no intermediate engine)
        from areal.infra.workflow_executor import WorkflowExecutor

        self._workflow_executor = WorkflowExecutor(
            config=self.config,
            inference_engine=self,
        )
        self._workflow_executor.initialize()

        # Create staleness manager
        from areal.infra.staleness_manager import StalenessManager

        max_concurrent = (
            self.config.max_concurrent_rollouts or self.config.consumer_batch_size
        )
        self._staleness_manager = StalenessManager(
            version_provider=self,
            max_concurrent_rollouts=max_concurrent,
            consumer_batch_size=self.config.consumer_batch_size,
            max_staleness=self.config.max_head_offpolicyness,
        )

        logger.info("GatewayRolloutController initialized (role=%s)", role)

    async def _async_initialize(
        self,
        alloc_mode: Any,
        server_args: dict[str, Any] | None,
        server_infos: list[LocalInfServerInfo] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Launch all servers as worker processes via the scheduler.

        Creates four groups of workers:
        1. Inference servers (dp_size replicas, GPU)
        2. Router (1 replica, CPU)
        3. Data Proxies (dp_size replicas, CPU)
        4. Gateway (1 replica, CPU)
        """
        from dataclasses import asdict

        from areal.api.cli_args import SchedulingSpec, SchedulingStrategy
        from areal.api.scheduler_api import Job

        dp_size = alloc_mode.gen.dp_size if alloc_mode is not None else 1
        cfg = self.config

        # Determine inference backend from allocation mode
        inf_backend = alloc_mode.gen_backend if alloc_mode is not None else "sglang"

        # -- 1. Launch inference servers --------------------------------------
        if server_infos is not None:
            # Pre-existing servers — skip inference worker creation
            self.server_infos = server_infos
            self._inf_addrs = [
                f"http://{info.host}:{info.port}" for info in server_infos
            ]
            logger.info(
                "Using %d pre-existing server_infos, skipping inference worker creation",
                len(server_infos),
            )
        else:
            # -- RPCGuard workers for inference server launch ---
            # Workers run areal.experimental.rollout_service.guard (the RPCGuard
            # Flask app).  We then POST to each guard to allocate a port,
            # build the backend-specific launch command, and POST /fork
            # with raw_cmd to start the inference server as a subprocess.
            import requests

            inf_spec = SchedulingSpec(**asdict(cfg.scheduling_spec[0]))
            if alloc_mode is not None:
                inf_spec.cpu *= alloc_mode.gen_instance_size
                inf_spec.mem *= alloc_mode.gen_instance_size
                if inf_spec.gpu > 0:
                    inf_spec.gpu = alloc_mode.gen_instance_size

            # Env vars inherited by forked data-proxy children
            inf_spec.env_vars["AREAL_DP_TOKENIZER_PATH"] = cfg.tokenizer_path
            inf_spec.env_vars["AREAL_DP_ADMIN_API_KEY"] = cfg.admin_api_key
            inf_spec.env_vars["AREAL_DP_LOG_LEVEL"] = cfg.log_level
            inf_spec.env_vars["AREAL_DP_REQUEST_TIMEOUT"] = str(cfg.request_timeout)

            # Override cmd to launch RPCGuard instead of RPC server
            inf_spec.cmd = "python -m areal.experimental.rollout_service.guard"

            inf_role = f"{self._worker_role}{self._INF_SUFFIX}"
            inf_job = Job(
                replicas=dp_size,
                tasks=[inf_spec for _ in range(dp_size)],
                scheduling_strategy=SchedulingStrategy(),
                role=inf_role,
            )

            # 1a. Create RPCGuard workers
            self.scheduler.create_workers(job=inf_job)
            self._service_roles.append(inf_role)
            inf_workers = self.scheduler.get_workers(role=inf_role)
            self.workers = inf_workers
            logger.info("RPCGuard workers ready: %s", [w.id for w in inf_workers])

            tp_size = alloc_mode.gen.tp_size if alloc_mode is not None else 1

            # 1b. Build backend-specific config and launch command builder
            if inf_backend in ("sglang", None):
                from areal.api.cli_args import SGLangConfig

                sglang_config = SGLangConfig(
                    model_path=cfg.model_path or cfg.tokenizer_path,
                )
                if server_args:
                    for k, v in server_args.items():
                        if hasattr(sglang_config, k):
                            object.__setattr__(sglang_config, k, v)

                def _build_launch_cmd(host: str, port: int) -> list[str]:
                    return SGLangConfig.build_cmd(
                        sglang_config=sglang_config,
                        tp_size=tp_size,
                        base_gpu_id=0,
                        host=host,
                        port=port,
                    )

            elif inf_backend == "vllm":
                # Placeholder — raise early so we never reach here, but
                # the structure is ready for a future vLLM implementation.
                raise NotImplementedError(
                    "vLLM backend is not yet supported by the gateway "
                    "rollout controller."
                )
            else:
                raise ValueError(f"Unsupported inference backend: {inf_backend!r}")

            # 1c. For each RPCGuard worker: alloc port, build cmd, fork server
            for rank, worker in enumerate(inf_workers):
                guard_addr = f"http://{worker.ip}:{worker.worker_ports[0]}"

                # Allocate a free port on the guard node for inference server
                resp = requests.post(
                    f"{guard_addr}/alloc_ports",
                    json={"count": 1},
                    timeout=30,
                )
                resp.raise_for_status()
                port_data = resp.json()
                inf_host = port_data["host"]
                inf_port = port_data["ports"][0]

                cmd = _build_launch_cmd(inf_host, inf_port)

                # Fork inference server via RPCGuard raw_cmd mode
                resp = requests.post(
                    f"{guard_addr}/fork",
                    json={
                        "role": "inf-server",
                        "worker_index": rank,
                        "raw_cmd": cmd,
                    },
                    timeout=30,
                )
                resp.raise_for_status()

                addr = f"http://{inf_host}:{inf_port}"
                self._inf_addrs.append(addr)
                self.server_infos.append(
                    LocalInfServerInfo(
                        host=inf_host,
                        port=inf_port,
                        process=None,  # type: ignore[arg-type]  # RPCGuard manages process
                    )
                )

            # Wait for inference servers to be healthy
            for i, addr in enumerate(self._inf_addrs):
                self._wait_for_service(
                    f"{addr}/health", f"InfServer-{i}", timeout=cfg.setup_timeout
                )
        logger.info("Inference servers: %s", self._inf_addrs)

        inf_role = f"{self._worker_role}{self._INF_SUFFIX}"
        has_inf_workers = inf_role in self._service_roles

        # -- 2. Launch Router -----------------------------------------------
        router_cmd = [
            sys.executable,
            "-m",
            "areal.experimental.rollout_service.router",
            "--admin-api-key",
            cfg.admin_api_key,
            "--routing-strategy",
            cfg.routing_strategy,
            "--poll-interval",
            str(cfg.poll_interval),
            "--log-level",
            cfg.log_level,
        ]

        if has_inf_workers:
            guard_addr = (
                f"http://{self.workers[0].ip}:{self.workers[0].worker_ports[0]}"
            )
            router_host, router_port = self._fork_on_guard(
                guard_addr=guard_addr,
                role="router",
                worker_index=0,
                raw_cmd=router_cmd,
            )
            self._router_addr = f"http://{router_host}:{router_port}"
        else:
            router_spec = SchedulingSpec(
                gpu=0,
                cpu=1,
                mem=4,
                cmd=" ".join(router_cmd),
            )
            router_role = f"{self._worker_role}{self._ROUTER_SUFFIX}"
            router_job = Job(
                replicas=1,
                tasks=[router_spec],
                scheduling_strategy=SchedulingStrategy(),
                role=router_role,
            )
            self.scheduler.create_workers(job=router_job)
            self._service_roles.append(router_role)

            router_workers = self.scheduler.get_workers(role=router_role)
            self._router_addr = (
                f"http://{router_workers[0].ip}:{router_workers[0].worker_ports[0]}"
            )
            self._wait_for_service(f"{self._router_addr}/health", "Router")
        logger.info("Router: %s", self._router_addr)

        # -- 3. Launch Data Proxies -----------------------------------------
        #   When inference workers were created by the controller, data proxies
        #   are **forked** from them (similar to RolloutController.start_proxy),
        #   ensuring colocation on the same node/GPUs.
        #   When pre-existing server_infos were provided (e.g. tests), data
        #   proxies are created as standalone workers instead.
        dp_role = f"{self._worker_role}{self._DATA_PROXY_SUFFIX}"

        if has_inf_workers:
            # Fork data proxies from inference workers (colocated deployment)
            dp_command = "areal.experimental.rollout_service.data_proxy"
            worker_ids = self.scheduler.fork_workers(
                role=dp_role,
                target_role=inf_role,
                command=dp_command,
            )
            self._service_roles.append(dp_role)
            logger.info("Data proxy workers forked: %s", worker_ids)

            dp_workers = self.scheduler.get_workers(role=dp_role)
            self._data_proxy_addrs = [
                f"http://{w.ip}:{w.worker_ports[0]}" for w in dp_workers
            ]

            # Configure each data proxy with its corresponding inference backend.
            for dp_addr, inf_addr in zip(self._data_proxy_addrs, self._inf_addrs):
                self._configure_data_proxy_backend(dp_addr, inf_addr)
        else:
            # Standalone data proxies (pre-existing server_infos)
            for i, inf_addr in enumerate(self._inf_addrs):
                dp_spec = SchedulingSpec(
                    gpu=0,
                    cpu=1,
                    mem=4,
                    cmd=(
                        f"python -m areal.experimental.rollout_service.data_proxy"
                        f" --backend-addr {inf_addr}"
                        f" --tokenizer-path {cfg.tokenizer_path}"
                        f" --admin-api-key {cfg.admin_api_key}"
                        f" --log-level {cfg.log_level}"
                        f" --request-timeout {cfg.request_timeout}"
                    ),
                )
                dp_i_role = f"{dp_role}-{i}"
                dp_job = Job(
                    replicas=1,
                    tasks=[dp_spec],
                    scheduling_strategy=SchedulingStrategy(),
                    role=dp_i_role,
                )
                self.scheduler.create_workers(job=dp_job)
                self._service_roles.append(dp_i_role)

                dp_workers = self.scheduler.get_workers(role=dp_i_role)
                dp_addr = f"http://{dp_workers[0].ip}:{dp_workers[0].worker_ports[0]}"
                self._data_proxy_addrs.append(dp_addr)

        # Wait for all data proxies to be healthy
        for i, dp_addr in enumerate(self._data_proxy_addrs):
            self._wait_for_service(f"{dp_addr}/health", f"DataProxy-{i}")

        # -- 4. Launch Gateway ----------------------------------------------
        gw_cmd = [
            sys.executable,
            "-m",
            "areal.experimental.rollout_service.gateway",
            "--admin-api-key",
            cfg.admin_api_key,
            "--router-addr",
            self._router_addr,
            "--forward-timeout",
            str(cfg.request_timeout),
            "--log-level",
            cfg.log_level,
        ]

        if has_inf_workers:
            guard_addr = (
                f"http://{self.workers[0].ip}:{self.workers[0].worker_ports[0]}"
            )
            gw_host, gw_port = self._fork_on_guard(
                guard_addr=guard_addr,
                role="gateway",
                worker_index=0,
                raw_cmd=gw_cmd,
            )
            self._gateway_addr = f"http://{gw_host}:{gw_port}"
        else:
            gw_spec = SchedulingSpec(
                gpu=0,
                cpu=1,
                mem=4,
                cmd=" ".join(gw_cmd),
            )
            gw_role = f"{self._worker_role}{self._GATEWAY_SUFFIX}"
            gw_job = Job(
                replicas=1,
                tasks=[gw_spec],
                scheduling_strategy=SchedulingStrategy(),
                role=gw_role,
            )
            self.scheduler.create_workers(job=gw_job)
            self._service_roles.append(gw_role)

            gw_workers = self.scheduler.get_workers(role=gw_role)
            self._gateway_addr = (
                f"http://{gw_workers[0].ip}:{gw_workers[0].worker_ports[0]}"
            )
            self._wait_for_service(f"{self._gateway_addr}/health", "Gateway")
        logger.info("Gateway: %s", self._gateway_addr)

    # -- Service health checks & registration ------------------------------

    def _wait_for_service(
        self, url: str, name: str, timeout: float | None = None
    ) -> None:
        """Wait for a service to become healthy."""
        import requests

        timeout = timeout or self.config.setup_timeout
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                resp = requests.get(url, timeout=2)
                if resp.status_code == 200:
                    logger.info("%s is ready at %s", name, url)
                    return
            except requests.RequestException:
                pass
            time.sleep(0.1)
        raise TimeoutError(f"{name} did not become healthy at {url} within {timeout}s")

    def _configure_data_proxy_backend(self, dp_addr: str, backend_addr: str) -> None:
        """Configure a data proxy's inference backend address after fork.

        Called after ``fork_workers`` to tell each data proxy which inference
        server to connect to.  The data proxy exposes a ``/configure_backend``
        endpoint that re-initialises its backend connection.
        """
        import requests

        resp = requests.post(
            f"{dp_addr}/configure_backend",
            json={"backend_addr": backend_addr},
            headers={"Authorization": f"Bearer {self.config.admin_api_key}"},
            timeout=30,
        )
        resp.raise_for_status()
        logger.info("Configured data proxy %s -> backend %s", dp_addr, backend_addr)

    def _register_data_proxies_in_router(self) -> None:
        """Register all data proxy workers in the router and store their worker IDs."""
        import requests

        for dp_addr in self._data_proxy_addrs:
            resp = requests.post(
                f"{self._router_addr}/register_worker",
                json={"worker_addr": dp_addr},
                headers={"Authorization": f"Bearer {self.config.admin_api_key}"},
                timeout=5,
            )
            resp.raise_for_status()
            worker_id = resp.json().get("worker_id")
            if worker_id:
                self._worker_ids[dp_addr] = worker_id
            logger.info(
                "Registered data proxy %s in router (worker_id=%s)", dp_addr, worker_id
            )

    # -- Destroy -----------------------------------------------------------

    def destroy(self) -> None:
        """Tear down all services and release resources."""
        # Destroy workflow executor
        if self._workflow_executor is not None:
            self._workflow_executor.destroy()
            self._workflow_executor = None

        # Kill services forked directly via RPCGuard /fork (router, gateway)
        for guard_addr, role, worker_index in reversed(self._forked_services):
            try:
                self._kill_forked_service(guard_addr, role, worker_index)
            except Exception:
                logger.error(
                    "Error killing forked service %s/%d: %s",
                    role,
                    worker_index,
                    traceback.format_exc(),
                )
        self._forked_services.clear()

        # RPCGuard's shutdown `finally` block automatically kills all
        # forked children (inference servers, data proxies), so no explicit teardown needed.
        # Delete all service workers via scheduler
        for role in reversed(self._service_roles):
            try:
                self.scheduler.delete_workers(role=role)
                logger.info("Workers deleted for role: %s", role)
            except Exception:
                logger.error(
                    "Error deleting workers for role %s: %s",
                    role,
                    traceback.format_exc(),
                )

        self._service_roles.clear()
        self.workers.clear()
        self.server_infos.clear()
        self._inf_addrs.clear()
        self._data_proxy_addrs.clear()
        self._worker_ids.clear()
        self._router_addr = ""
        self._gateway_addr = ""
        self._staleness_manager = None

    # -- Version management ------------------------------------------------

    def set_version(self, version: int) -> None:
        with self._version_lock:
            self._version = version
        # TODO: Weight-update forwarding (set_version broadcast to workers via
        # gateway HTTP) has been removed. Re-implement when the gateway
        # natively supports weight synchronisation.

    def get_version(self) -> int:
        with self._version_lock:
            return self._version

    # -- Capacity ----------------------------------------------------------

    def get_capacity(self) -> int:
        return self.staleness_manager.get_capacity()

    # -- Submit / Wait / Batch ---------------------------------------------

    def submit(
        self,
        data: dict[str, Any],
        workflow: Any,
        workflow_kwargs: dict[str, Any] | None = None,
        should_accept_fn: Any = None,
        task_id: int | None = None,
        is_eval: bool = False,
        group_size: int = 1,
        proxy_addr: str | None = None,
    ) -> int:
        if proxy_addr is None:
            proxy_addr = self._gateway_addr
        resolved_workflow = self._resolve_workflow(
            workflow,
            workflow_kwargs,
            group_size,
            proxy_addr=proxy_addr,
            controller=self,
        )
        resolved_accept_fn = self._resolve_should_accept_fn(should_accept_fn)
        return self.workflow_executor.submit(
            data,
            workflow=resolved_workflow,
            should_accept_fn=resolved_accept_fn,
            task_id=task_id,
            is_eval=is_eval,
        )

    def wait(
        self,
        count: int,
        timeout: float | None = None,
        raise_timeout: bool = True,
    ) -> list[dict[str, Any] | None]:
        return self.workflow_executor.wait(
            count, timeout=timeout, raise_timeout=raise_timeout
        )

    def rollout_batch(
        self,
        data: list[dict[str, Any]],
        workflow: Any,
        workflow_kwargs: dict[str, Any] | None = None,
        should_accept_fn: Any = None,
        group_size: int = 1,
    ) -> list[dict[str, Any]]:
        if not self._gateway_addr:
            raise RuntimeError(
                "GatewayRolloutController.initialize() must be called first"
            )
        proxy_addr = self._gateway_addr
        resolved_workflow = self._resolve_workflow(
            workflow,
            workflow_kwargs,
            group_size,
            proxy_addr=proxy_addr,
            controller=self,
        )
        resolved_accept_fn = self._resolve_should_accept_fn(should_accept_fn)
        for item in data:
            self.workflow_executor.submit(
                data=item,
                workflow=resolved_workflow,
                should_accept_fn=resolved_accept_fn,
            )
        results = self.workflow_executor.wait(count=len(data))
        # Return list of trajectories (matching RolloutController API)
        return [r for r in results if r is not None]

    def prepare_batch(
        self,
        dataloader: Any,
        workflow: Any,
        workflow_kwargs: dict[str, Any] | None = None,
        should_accept_fn: Any = None,
        group_size: int = 1,
        dynamic_bs: bool = False,
    ) -> list[dict[str, Any]]:
        if not self._gateway_addr:
            raise RuntimeError(
                "GatewayRolloutController.initialize() must be called first"
            )
        proxy_addr = self._gateway_addr
        resolved_workflow = self._resolve_workflow(
            workflow,
            workflow_kwargs,
            group_size,
            proxy_addr=proxy_addr,
            controller=self,
        )
        resolved_accept_fn = self._resolve_should_accept_fn(should_accept_fn)
        results = self.workflow_executor.prepare_batch(
            dataloader=dataloader,
            workflow=resolved_workflow,
            should_accept_fn=resolved_accept_fn,
            dynamic_bs=dynamic_bs,
        )
        # Return list of trajectories (matching RolloutController API)
        return [r for r in results if r is not None]

    async def chat_completion(
        self,
        messages: list[dict],
        session_api_key: str | None = None,
        **kwargs,
    ) -> ChatCompletion | AsyncGenerator[ChatCompletionChunk, None]:
        """Send a chat completion request through the gateway HTTP stack.

        Parameters
        ----------
        messages : list[dict]
            OpenAI-style chat messages.
        session_api_key : str | None
            If provided, authenticate as this session; otherwise use the
            admin API key from the controller config.
        **kwargs
            Optional overrides: ``temperature``, ``top_p``,
            ``max_completion_tokens``, ``stream``.

        Returns
        -------
        ChatCompletion | AsyncGenerator[ChatCompletionChunk, None]
            When ``stream=False`` (default): parsed OpenAI ChatCompletion object.
            When ``stream=True``: async generator yielding ChatCompletionChunk.
        """
        import aiohttp

        stream = kwargs.get("stream", False)
        body: dict[str, Any] = {
            "messages": messages,
            "temperature": kwargs.get("temperature", 1.0),
            "top_p": kwargs.get("top_p", 1.0),
            "max_completion_tokens": kwargs.get("max_completion_tokens", 512),
            "stream": stream,
        }
        # Forward extra body params (e.g. chat_template_kwargs)
        extra_body = kwargs.get("extra_body")
        if extra_body and isinstance(extra_body, dict):
            body.update(extra_body)

        api_key = (
            session_api_key
            if session_api_key is not None
            else self.config.admin_api_key
        )
        url = f"{self._gateway_addr}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        if stream:
            return self._stream_chat_completion(url, body, headers)

        # Non-streaming path
        timeout = aiohttp.ClientTimeout(total=self.config.request_timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=body, headers=headers) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(
                        f"Gateway /chat/completions returned {resp.status}: {text}"
                    )
                resp_json = await resp.json()

        return ChatCompletion.model_validate(resp_json)

    async def _stream_chat_completion(
        self,
        url: str,
        body: dict[str, Any],
        headers: dict[str, str],
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        """Parse SSE stream from the gateway into ChatCompletionChunk objects."""
        import aiohttp

        timeout = aiohttp.ClientTimeout(total=self.config.request_timeout)
        session = aiohttp.ClientSession(timeout=timeout)
        try:
            resp = await session.post(url, json=body, headers=headers)
            if resp.status != 200:
                text = await resp.text()
                await resp.release()
                await session.close()
                raise RuntimeError(
                    f"Gateway /chat/completions returned {resp.status}: {text}"
                )

            async for line in resp.content:
                decoded = line.decode("utf-8").strip()
                if not decoded or not decoded.startswith("data: "):
                    continue
                payload = decoded[len("data: ") :]
                if payload == "[DONE]":
                    break
                import json as _json

                chunk_data = _json.loads(payload)
                yield ChatCompletionChunk.model_validate(chunk_data)

            await resp.release()
        finally:
            await session.close()

    # -- Pause / Resume ----------------------------------------------------

    def pause(self) -> None:
        """Pause dispatcher + pause all workers."""
        from areal.infra.utils.concurrent import run_async_task

        if self._workflow_executor is not None:
            self._workflow_executor.pause()
        run_async_task(self.pause_generation)

    def resume(self) -> None:
        """Resume all workers + resume dispatcher."""
        from areal.infra.utils.concurrent import run_async_task

        run_async_task(self.continue_generation)
        if self._workflow_executor is not None:
            self._workflow_executor.resume()

    async def pause_generation(self, worker_id: str | None = None) -> None:
        """Pause generation on a specific worker, or all workers if worker_id is None."""
        if not self._gateway_addr:
            return
        if worker_id is not None:
            await self._async_gateway_http_post(f"/pause_generation/{worker_id}", {})
        else:
            for wid in self._worker_ids.values():
                await self._async_gateway_http_post(f"/pause_generation/{wid}", {})

    async def continue_generation(self, worker_id: str | None = None) -> None:
        """Continue generation on a specific worker, or all workers if worker_id is None."""
        if not self._gateway_addr:
            return
        if worker_id is not None:
            await self._async_gateway_http_post(f"/continue_generation/{worker_id}", {})
        else:
            for wid in self._worker_ids.values():
                await self._async_gateway_http_post(f"/continue_generation/{wid}", {})

    # -- Weight updates (not yet implemented in gateway) --------------------

    async def init_weights_update_group(self, meta: Any) -> None:
        """Initialize NCCL weight-update group on all workers.

        Not yet implemented — the gateway HTTP stack does not support
        weight synchronisation yet.
        """
        raise NotImplementedError(
            "init_weights_update_group is not yet supported by the gateway rollout controller"
        )

    async def update_weights_from_distributed(
        self, meta: Any, param_specs: Any
    ) -> None:
        """Trigger a distributed (NCCL/XCCL) weight update on all workers.

        Not yet implemented — the gateway HTTP stack does not support
        weight synchronisation yet.
        """
        raise NotImplementedError(
            "update_weights_from_distributed is not yet supported by the gateway rollout controller"
        )

    async def update_weights_from_disk(self, meta: Any) -> None:
        """Trigger a disk-based weight update on all workers.

        Not yet implemented — the gateway HTTP stack does not support
        weight synchronisation yet.
        """
        raise NotImplementedError(
            "update_weights_from_disk is not yet supported by the gateway rollout controller"
        )

    # -- Stats -------------------------------------------------------------

    def export_stats(self) -> dict[str, float]:
        """Return local WorkflowExecutor stats."""
        return {}

    def config_perf_tracer(self, config: Any = None, role: str = "") -> None:
        """No-op — gateway does not have per-worker perf tracing."""

    def save_perf_tracer(self, step: int | None = None, force: bool = False) -> None:
        """No-op."""

    # -- Proxy compatibility (gateway IS the proxy) ------------------------

    def start_proxy(self) -> None:
        """No-op — gateway already acts as the proxy."""

    def start_proxy_gateway(self) -> None:
        """No-op — gateway already acts as the proxy gateway."""

    @property
    def proxy_gateway_addr(self) -> str:
        return self._gateway_addr

    # -- Properties --------------------------------------------------------

    @property
    def worker_ids(self) -> dict[str, str]:
        """Return mapping from data proxy address to router-assigned worker_id."""
        return dict(self._worker_ids)

    @property
    def staleness_manager(self):
        return self._staleness_manager

    @property
    def workflow_executor(self):
        if self._workflow_executor is None:
            raise RuntimeError(
                "GatewayRolloutController.initialize() must be called first"
            )
        return self._workflow_executor

    @property
    def dispatcher(self):
        return self.workflow_executor.dispatcher

    @property
    def runner(self):
        return self.dispatcher.runner

    # -- Workflow resolution helpers ----------------------------------------

    def _wrap_openai_agent(self, agent: Any, proxy_addr: str):
        """Wrap an agent workflow in OpenAIProxyWorkflow (HTTP mode only).

        Parameters
        ----------
        agent : Any | None
            The agent workflow to wrap (any class with async run() method).
            ``None`` is valid when ``mode='online'``.
        proxy_addr : str
            HTTP address of the proxy server (required)
        """
        from areal.experimental.openai import OpenAIProxyWorkflow

        openai_cfg = self.config.openai
        mode = getattr(openai_cfg, "mode", "inline")
        admin_api_key = self.config.admin_api_key
        turn_discount = getattr(openai_cfg, "turn_discount", 1.0)
        export_style = getattr(openai_cfg, "export_style", "individual")
        subproc_max_workers = getattr(openai_cfg, "subproc_max_workers", 4)

        return OpenAIProxyWorkflow(
            mode=mode,
            agent=agent,
            proxy_addr=proxy_addr,
            admin_api_key=admin_api_key,
            discount=turn_discount,
            export_style=export_style,
            subproc_max_workers=subproc_max_workers,
            proxy_gateway_addr=self._gateway_addr,
        )

    @staticmethod
    def _resolve_workflow(
        workflow,
        workflow_kwargs=None,
        group_size=1,
        proxy_addr=None,
        controller=None,
    ):
        """Resolve a WorkflowLike to a RolloutWorkflow instance.

        Handles both RolloutWorkflow types (cases 1-3) and agent-like
        workflows that need wrapping in OpenAIProxyWorkflow (cases 4-5).

        Parameters
        ----------
        workflow : WorkflowLike
            A RolloutWorkflow instance, class, import path string,
            agent class, or agent instance.
        workflow_kwargs : dict, optional
            Keyword arguments passed to the workflow/agent constructor.
        group_size : int
            Number of times to run the workflow per input.
        proxy_addr : str, optional
            HTTP address of the proxy server, required for agent workflows.
        controller : GatewayRolloutController, optional
            The controller instance, required for agent workflows (_wrap_openai_agent).
        """
        from areal.api.workflow_api import RolloutWorkflow
        from areal.utils.dynamic_import import import_from_string

        if workflow is None:
            raise ValueError("workflow must be specified")

        resolved: RolloutWorkflow

        # 1. Already a RolloutWorkflow instance
        if isinstance(workflow, RolloutWorkflow):
            resolved = workflow

        # 2. RolloutWorkflow class
        elif isinstance(workflow, type) and issubclass(workflow, RolloutWorkflow):
            if workflow_kwargs is None:
                raise ValueError("workflow_kwargs required when workflow is a class")
            resolved = workflow(**workflow_kwargs)

        # 3. String import path
        elif isinstance(workflow, str):
            imported = import_from_string(workflow)
            if isinstance(imported, type) and issubclass(imported, RolloutWorkflow):
                if workflow_kwargs is None:
                    raise ValueError(
                        "workflow_kwargs required when workflow is a class"
                    )
                resolved = imported(**workflow_kwargs)
            elif isinstance(imported, RolloutWorkflow):
                resolved = imported
            else:
                # Treat as agent-like workflow (needs proxy wrapping)
                if proxy_addr is None or controller is None:
                    raise ValueError(
                        f"proxy_addr and controller are required for agent workflows "
                        f"(non-RolloutWorkflow). Got workflow={workflow!r}"
                    )
                if isinstance(imported, type):
                    agent = imported(**(workflow_kwargs or {}))
                else:
                    agent = imported
                resolved = controller._wrap_openai_agent(agent, proxy_addr=proxy_addr)

        # 4. Callable class (agent-like workflow)
        elif isinstance(workflow, type):
            if proxy_addr is None or controller is None:
                raise ValueError(
                    "proxy_addr and controller are required for agent workflows "
                    "(non-RolloutWorkflow). "
                    "Ensure proxy workers are initialized via RolloutController.start_proxy()."
                )
            agent = workflow(**(workflow_kwargs or {}))
            resolved = controller._wrap_openai_agent(agent, proxy_addr=proxy_addr)

        # 5. Instance of agent-like workflow
        else:
            if proxy_addr is None or controller is None:
                raise ValueError(
                    "proxy_addr and controller are required for agent workflows "
                    "(non-RolloutWorkflow). "
                    "Ensure proxy workers are initialized via RolloutController.start_proxy()."
                )
            resolved = controller._wrap_openai_agent(workflow, proxy_addr=proxy_addr)

        if group_size > 1:
            from areal.infra.remote_inf_engine import GroupedRolloutWorkflow

            resolved = GroupedRolloutWorkflow(
                resolved, group_size, logging.getLogger("RolloutController")
            )

        return resolved

    @staticmethod
    def _resolve_should_accept_fn(should_accept_fn):
        """Resolve should_accept_fn to a callable or None."""
        if should_accept_fn is None or callable(should_accept_fn):
            return should_accept_fn
        if isinstance(should_accept_fn, str):
            from areal.utils.dynamic_import import import_from_string

            func = import_from_string(should_accept_fn)
            if not callable(func):
                raise TypeError(f"Imported {should_accept_fn!r} is not callable")
            return func
        raise TypeError(f"Invalid should_accept_fn type: {type(should_accept_fn)}")

    # -- Internal HTTP helpers ---------------------------------------------

    def _fork_on_guard(
        self,
        guard_addr: str,
        role: str,
        worker_index: int,
        raw_cmd: list[str],
        health_path: str = "/health",
    ) -> tuple[str, int]:
        """Fork a process on a RPCGuard worker via ``/fork`` with ``raw_cmd``.

        Returns ``(host, port)`` of the forked service and records the entry
        in ``_forked_services`` for cleanup.
        """
        import requests

        resp = requests.post(
            f"{guard_addr}/alloc_ports",
            json={"count": 1},
            timeout=30,
        )
        resp.raise_for_status()
        port_data = resp.json()
        host = port_data["host"]
        port = port_data["ports"][0]

        cmd = list(raw_cmd) + ["--host", host, "--port", str(port)]

        resp = requests.post(
            f"{guard_addr}/fork",
            json={
                "role": role,
                "worker_index": worker_index,
                "raw_cmd": cmd,
            },
            timeout=30,
        )
        resp.raise_for_status()

        self._forked_services.append((guard_addr, role, worker_index))

        addr = f"http://{host}:{port}"
        self._wait_for_service(f"{addr}{health_path}", role)

        return host, port

    def _kill_forked_service(
        self, guard_addr: str, role: str, worker_index: int
    ) -> None:
        import requests

        try:
            resp = requests.post(
                f"{guard_addr}/kill_forked_worker",
                json={"role": role, "worker_index": worker_index},
                timeout=10,
            )
            if resp.status_code == 200:
                logger.info("Killed forked service %s/%d", role, worker_index)
            else:
                logger.warning(
                    "Failed to kill forked service %s/%d: %s",
                    role,
                    worker_index,
                    resp.text,
                )
        except requests.RequestException as exc:
            logger.error(
                "Error killing forked service %s/%d: %s", role, worker_index, exc
            )

    def _gateway_http_post(self, endpoint: str, payload: dict[str, Any]) -> None:
        """Make a synchronous HTTP POST to the gateway with admin auth.

        Use ``_async_gateway_http_post`` from async contexts to avoid blocking
        the event loop.
        """
        import requests

        url = f"{self._gateway_addr}{endpoint}"
        try:
            resp = requests.post(
                url,
                json=payload,
                headers={"Authorization": f"Bearer {self.config.admin_api_key}"},
                timeout=self.config.request_timeout,
            )
            if resp.status_code >= 400:
                logger.warning(
                    "Gateway %s returned %s: %s", endpoint, resp.status_code, resp.text
                )
        except requests.RequestException as exc:
            logger.error("Failed to POST %s: %s", endpoint, exc)

    async def _async_gateway_http_post(
        self, endpoint: str, payload: dict[str, Any]
    ) -> None:
        """Make a non-blocking HTTP POST to the gateway with admin auth."""
        import httpx

        url = f"{self._gateway_addr}{endpoint}"
        try:
            async with httpx.AsyncClient(timeout=self.config.request_timeout) as client:
                resp = await client.post(
                    url,
                    json=payload,
                    headers={"Authorization": f"Bearer {self.config.admin_api_key}"},
                )
                if resp.status_code >= 400:
                    logger.warning(
                        "Gateway %s returned %s: %s",
                        endpoint,
                        resp.status_code,
                        resp.text,
                    )
        except httpx.HTTPError as exc:
            logger.error("Failed to POST %s: %s", endpoint, exc)
