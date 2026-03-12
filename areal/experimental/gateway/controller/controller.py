"""GatewayRolloutController — parallel implementation to RolloutController.

Routes ALL traffic (inference, weight updates, pause/continue) through the
gateway HTTP stack (Gateway → Router → Data Proxy → SGLang).
Scheduler is used ONLY to launch SGLang servers and gateway micro-services.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import traceback
from threading import Lock
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from areal.api.io_struct import (
        LocalInfServerInfo,
        ModelRequest,
        ModelResponse,
    )
    from areal.api.scheduler_api import Scheduler, Worker

from areal.experimental.gateway.controller.config import GatewayControllerConfig
from areal.experimental.gateway.controller.inf_engine import GatewayInfEngine

logger = logging.getLogger("GatewayRolloutController")


class GatewayRolloutController:
    """Rollout controller that routes everything through the gateway HTTP stack.

    This is a **parallel** implementation to ``RolloutController`` (NOT a
    subclass).  It is duck-type compatible: the trainer can use either one
    without code changes.
    """

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

        # Version management
        self._version_lock = Lock()
        self._version = 0

        # Gateway micro-services (background threads)
        self._router_server = None
        self._router_thread: threading.Thread | None = None
        self._data_proxy_servers: list = []
        self._data_proxy_threads: list[threading.Thread] = []
        self._gateway_server = None
        self._gateway_thread: threading.Thread | None = None
        self._gateway_services_started = False

        # Inference engine (routes through gateway HTTP)
        self._gateway_inf_engine: GatewayInfEngine | None = None

        # Staleness manager (created in initialize)
        self._staleness_manager = None

        # Proxy compatibility (no-ops — gateway IS the proxy)
        self._proxy_started = False
        self.proxy_workers: list = []
        self.proxy_addrs: list[str] = []

    # -- Naming helpers (match RolloutController conventions) ---------------

    def _engine_name(self, rank: int) -> str:
        return f"{self._worker_role}/{rank}"

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

        # Start gateway micro-services
        self._start_gateway_services()

        # Create inference engine pointed at the gateway
        gateway_addr = f"http://127.0.0.1:{self.config.gateway_port}"
        self._gateway_inf_engine = GatewayInfEngine(gateway_addr, self.config)
        self._gateway_inf_engine.initialize()

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
        """Create workers and launch SGLang servers via scheduler."""
        from dataclasses import asdict

        from areal.api.cli_args import SchedulingSpec, SchedulingStrategy
        from areal.api.scheduler_api import Job

        # Build scheduling spec
        sch_spec = SchedulingSpec(**asdict(self.config.scheduling_spec[0]))
        if alloc_mode is not None:
            sch_spec.cpu *= alloc_mode.gen_instance_size
            sch_spec.mem *= alloc_mode.gen_instance_size
            if sch_spec.gpu > 0:
                sch_spec.gpu = alloc_mode.gen_instance_size

        dp_size = alloc_mode.gen.dp_size if alloc_mode is not None else 1

        scheduling_strategy = self.config.scheduling_strategy
        if scheduling_strategy is None:
            scheduling_strategy = SchedulingStrategy()

        job = Job(
            replicas=dp_size,
            tasks=[sch_spec for _ in range(dp_size)],
            scheduling_strategy=scheduling_strategy,
            role=self._worker_role,
        )

        if server_infos is not None:
            # Pre-existing servers — skip worker creation / server launch
            self.server_infos = server_infos
            logger.info(
                "Using %d pre-existing server_infos, skipping worker creation",
                len(server_infos),
            )
        else:
            logger.info("Creating workers via scheduler...")
            self.scheduler.create_workers(job=job)
            self.workers = self.scheduler.get_workers(role=job.role)
            logger.info("Workers ready: %s", [w.id for w in self.workers])

            # Launch SGLang servers on workers
            tasks = []
            for rank, worker in enumerate(self.workers):
                tasks.append(
                    self.scheduler.async_call_engine(
                        worker_id=worker.id,
                        method="launch_server",
                        engine_name=self._engine_name(rank),
                        server_args=server_args or {},
                    )
                )
            self.server_infos = await asyncio.gather(*tasks)

        logger.info("SGLang servers: %s", self.server_infos)

    # -- Gateway services lifecycle ----------------------------------------

    def _start_gateway_services(self) -> None:
        """Start Router, Data Proxies, and Gateway as background threads."""

        from areal.experimental.gateway.data_proxy.app import (
            create_app as create_data_proxy_app,
        )
        from areal.experimental.gateway.data_proxy.config import DataProxyConfig
        from areal.experimental.gateway.gateway.app import (
            create_app as create_gateway_app,
        )
        from areal.experimental.gateway.gateway.config import GatewayConfig
        from areal.experimental.gateway.router.app import (
            create_app as create_router_app,
        )
        from areal.experimental.gateway.router.config import RouterConfig

        cfg = self.config

        # 1. Start Router
        router_cfg = RouterConfig(
            host=cfg.router_host,
            port=cfg.router_port,
            admin_api_key=cfg.admin_api_key,
            poll_interval=cfg.poll_interval,
            routing_strategy=cfg.routing_strategy,
            log_level=cfg.log_level,
        )
        router_app = create_router_app(router_cfg)
        self._router_thread = self._start_uvicorn_thread(
            router_app, cfg.router_host, cfg.router_port, "Router"
        )

        # Wait for router to be ready
        self._wait_for_service(f"http://127.0.0.1:{cfg.router_port}/health", "Router")

        # 2. Start one Data Proxy per worker
        for i, info in enumerate(self.server_infos):
            dp_port = cfg.data_proxy_base_port + i
            backend_addr = f"http://{info.host}:{info.port}"
            dp_cfg = DataProxyConfig(
                host=cfg.data_proxy_host,
                port=dp_port,
                backend_addr=backend_addr,
                tokenizer_path=cfg.tokenizer_path,
                log_level=cfg.log_level,
                request_timeout=cfg.request_timeout,
                max_resubmit_retries=cfg.max_resubmit_retries,
                resubmit_wait=cfg.resubmit_wait,
                admin_api_key=cfg.admin_api_key,
            )
            dp_app = create_data_proxy_app(dp_cfg)
            thread = self._start_uvicorn_thread(
                dp_app, cfg.data_proxy_host, dp_port, f"DataProxy-{i}"
            )
            self._data_proxy_threads.append(thread)

            # Wait for data proxy to be ready
            self._wait_for_service(
                f"http://127.0.0.1:{dp_port}/health", f"DataProxy-{i}"
            )

            # Register data proxy in router
            self._register_worker_in_router(f"http://127.0.0.1:{dp_port}")

        # 3. Start Gateway
        gw_cfg = GatewayConfig(
            host=cfg.gateway_host,
            port=cfg.gateway_port,
            admin_api_key=cfg.admin_api_key,
            router_addr=f"http://127.0.0.1:{cfg.router_port}",
            forward_timeout=cfg.request_timeout,
            log_level=cfg.log_level,
        )
        gw_app = create_gateway_app(gw_cfg)
        self._gateway_thread = self._start_uvicorn_thread(
            gw_app, cfg.gateway_host, cfg.gateway_port, "Gateway"
        )

        # Wait for gateway to be ready
        self._wait_for_service(f"http://127.0.0.1:{cfg.gateway_port}/health", "Gateway")

        self._gateway_services_started = True
        logger.info("All gateway services started")

    def _start_uvicorn_thread(
        self, app: Any, host: str, port: int, name: str
    ) -> threading.Thread:
        """Start a uvicorn server in a daemon thread."""
        import uvicorn

        server_holder: list = []

        def serve():
            config = uvicorn.Config(app, host=host, port=port, log_level="warning")
            server = uvicorn.Server(config)
            server_holder.append(server)
            server.run()

        thread = threading.Thread(target=serve, name=name, daemon=True)
        thread.start()

        # Wait for server to start
        deadline = time.monotonic() + self.config.setup_timeout
        while time.monotonic() < deadline:
            if (
                server_holder
                and hasattr(server_holder[0], "started")
                and server_holder[0].started
            ):
                break
            time.sleep(0.05)

        if name == "Router":
            self._router_server = server_holder[0] if server_holder else None
        elif name == "Gateway":
            self._gateway_server = server_holder[0] if server_holder else None
        elif name.startswith("DataProxy"):
            if server_holder:
                self._data_proxy_servers.append(server_holder[0])

        return thread

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

    def _register_worker_in_router(self, worker_addr: str) -> None:
        """Register a data proxy worker in the router."""
        import requests

        router_url = f"http://127.0.0.1:{self.config.router_port}/register_worker"
        resp = requests.post(
            router_url,
            json={"worker_addr": worker_addr},
            headers={"Authorization": f"Bearer {self.config.admin_api_key}"},
            timeout=5,
        )
        resp.raise_for_status()
        logger.info("Registered worker %s in router", worker_addr)

    def _stop_gateway_services(self) -> None:
        """Stop all gateway micro-services."""
        for server in (
            [self._gateway_server] + self._data_proxy_servers + [self._router_server]
        ):
            if server is not None:
                server.should_exit = True

        for thread in (
            [self._gateway_thread] + self._data_proxy_threads + [self._router_thread]
        ):
            if thread is not None:
                thread.join(timeout=5.0)

        self._gateway_server = None
        self._gateway_thread = None
        self._data_proxy_servers.clear()
        self._data_proxy_threads.clear()
        self._router_server = None
        self._router_thread = None
        self._gateway_services_started = False

    @property
    def callback_addr(self) -> str:
        """Return gateway address as 'host:port' for training controller callbacks."""
        from areal.utils.network import gethostip

        host = self.config.gateway_host
        if host == "0.0.0.0":
            host = gethostip()
        return f"{host}:{self.config.gateway_port}"

    # -- Destroy -----------------------------------------------------------

    def destroy(self) -> None:
        # Destroy inference engine
        if self._gateway_inf_engine is not None:
            self._gateway_inf_engine.destroy()
            self._gateway_inf_engine = None

        # Stop gateway services
        self._stop_gateway_services()

        # Delete workers via scheduler
        if self._worker_role:
            try:
                self.scheduler.delete_workers(role=self._worker_role)
                self.workers.clear()
                self.server_infos.clear()
                logger.info("Workers deleted")
            except Exception:
                logger.error("Error deleting workers: %s", traceback.format_exc())

        self._staleness_manager = None

    # -- Version management ------------------------------------------------

    def set_version(self, version: int) -> None:
        with self._version_lock:
            self._version = version
            if self._gateway_inf_engine is not None:
                self._gateway_inf_engine.set_version(version)
        # Broadcast to all workers via gateway
        if self._gateway_services_started:
            self._gateway_http_post("/set_version", {"version": version})

    def get_version(self) -> int:
        with self._version_lock:
            return self._version

    # -- Capacity ----------------------------------------------------------

    def get_capacity(self) -> int:
        return self.staleness_manager.get_capacity()

    # -- Submit / Wait / Batch (delegate to GatewayInfEngine) --------------

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
        return self._engine.submit(
            data=data,
            workflow=workflow,
            workflow_kwargs=workflow_kwargs,
            should_accept_fn=should_accept_fn,
            group_size=group_size,
            task_id=task_id,
            is_eval=is_eval,
        )

    def wait(
        self,
        count: int,
        timeout: float | None = None,
        raise_timeout: bool = True,
    ) -> list[dict[str, Any] | None]:
        return self._engine.wait(count, timeout=timeout, raise_timeout=raise_timeout)

    def rollout_batch(
        self,
        data: list[dict[str, Any]],
        workflow: Any,
        workflow_kwargs: dict[str, Any] | None = None,
        should_accept_fn: Any = None,
        group_size: int = 1,
    ) -> dict[str, Any]:
        return self._engine.rollout_batch(
            data=data,
            workflow=workflow,
            workflow_kwargs=workflow_kwargs,
            should_accept_fn=should_accept_fn,
            group_size=group_size,
        )

    def prepare_batch(
        self,
        dataloader: Any,
        workflow: Any,
        workflow_kwargs: dict[str, Any] | None = None,
        should_accept_fn: Any = None,
        group_size: int = 1,
        dynamic_bs: bool = False,
    ) -> dict[str, Any]:
        return self._engine.prepare_batch(
            dataloader=dataloader,
            workflow=workflow,
            workflow_kwargs=workflow_kwargs,
            should_accept_fn=should_accept_fn,
            group_size=group_size,
            dynamic_bs=dynamic_bs,
        )

    async def agenerate(self, req: ModelRequest) -> ModelResponse:
        return await self._engine.agenerate(req)

    async def chat_completion(self, messages, session_api_key=None, **kwargs):
        return await self._engine.chat_completion(
            messages, session_api_key=session_api_key, **kwargs
        )

    # -- Pause / Resume ----------------------------------------------------

    def pause(self) -> None:
        """Pause dispatcher + broadcast pause to all workers."""
        if self._gateway_inf_engine is not None:
            self._gateway_inf_engine.pause()
        if self._gateway_services_started:
            self._gateway_http_post("/pause_generation", {})

    def resume(self) -> None:
        """Broadcast resume to all workers + resume dispatcher."""
        if self._gateway_services_started:
            self._gateway_http_post("/continue_generation", {})
        if self._gateway_inf_engine is not None:
            self._gateway_inf_engine.resume()

    async def pause_generation(self) -> None:
        if self._gateway_services_started:
            self._gateway_http_post("/pause_generation", {})

    async def continue_generation(self) -> None:
        if self._gateway_services_started:
            self._gateway_http_post("/continue_generation", {})

    # -- Weight updates (route through gateway HTTP) -----------------------

    async def init_weights_update_group(self, meta: Any) -> None:
        self._gateway_http_post(
            "/init_weights_update_group",
            {"meta": meta} if not isinstance(meta, dict) else meta,
        )

    async def update_weights_from_distributed(
        self, meta: Any, param_specs: Any
    ) -> None:
        payload = {
            "meta": meta if isinstance(meta, dict) else str(meta),
            "param_specs": param_specs
            if isinstance(param_specs, list)
            else str(param_specs),
        }
        self._gateway_http_post("/update_weights_from_distributed", payload)

    async def update_weights_from_disk(self, meta: Any) -> None:
        self._gateway_http_post(
            "/update_weights_from_disk",
            {"meta": meta} if not isinstance(meta, dict) else meta,
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
        return f"http://127.0.0.1:{self.config.gateway_port}"

    # -- Properties --------------------------------------------------------

    @property
    def staleness_manager(self):
        return self._staleness_manager

    @property
    def dispatcher(self):
        return self._engine.workflow_executor.dispatcher

    @property
    def runner(self):
        return self.dispatcher.runner

    @property
    def _engine(self) -> GatewayInfEngine:
        if self._gateway_inf_engine is None:
            raise RuntimeError(
                "GatewayRolloutController.initialize() must be called first"
            )
        return self._gateway_inf_engine

    # -- Internal HTTP helpers ---------------------------------------------

    def _gateway_http_post(self, endpoint: str, payload: dict[str, Any]) -> None:
        """Make an HTTP POST to the gateway with admin auth."""
        import requests

        url = f"http://127.0.0.1:{self.config.gateway_port}{endpoint}"
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
