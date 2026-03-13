"""GatewayRolloutController — parallel implementation to RolloutController.

Routes ALL traffic (inference, weight updates, pause/continue) through the
gateway HTTP stack (Gateway → Router → Data Proxy → SGLang).
All servers are launched as worker processes via the scheduler.
"""

from __future__ import annotations

import asyncio
import logging
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

    All servers (SGLang, Router, Data Proxy, Gateway) are launched as
    worker sub-processes via the scheduler.  The controller talks to them
    directly over HTTP — no engine creation or RPC calls on workers.
    """

    # Worker role suffixes for each service type
    _SGLANG_SUFFIX = "-sglang"
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
        self._sglang_addrs: list[str] = []
        self._router_addr: str = ""
        self._data_proxy_addrs: list[str] = []
        self._gateway_addr: str = ""

        # Version management
        self._version_lock = Lock()
        self._version = 0

        # Inference engine (routes through gateway HTTP)
        self._gateway_inf_engine: GatewayInfEngine | None = None

        # Staleness manager (created in initialize)
        self._staleness_manager = None

        # Track which service roles were created for cleanup
        self._service_roles: list[str] = []

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

        # Register data proxies in the router
        self._register_data_proxies_in_router()

        # Create inference engine pointed at the gateway
        self._gateway_inf_engine = GatewayInfEngine(self._gateway_addr, self.config)
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
        """Launch all servers as worker processes via the scheduler.

        Creates four groups of workers:
        1. SGLang inference servers (dp_size replicas, GPU)
        2. Router (1 replica, CPU)
        3. Data Proxies (dp_size replicas, CPU)
        4. Gateway (1 replica, CPU)
        """
        from dataclasses import asdict

        from areal.api.cli_args import SchedulingSpec, SchedulingStrategy, SGLangConfig
        from areal.api.scheduler_api import Job

        dp_size = alloc_mode.gen.dp_size if alloc_mode is not None else 1
        cfg = self.config

        # -- 1. Launch SGLang servers ----------------------------------------
        if server_infos is not None:
            # Pre-existing servers — skip SGLang worker creation
            self.server_infos = server_infos
            self._sglang_addrs = [
                f"http://{info.host}:{info.port}" for info in server_infos
            ]
            logger.info(
                "Using %d pre-existing server_infos, skipping SGLang worker creation",
                len(server_infos),
            )
        else:
            sglang_spec = SchedulingSpec(**asdict(cfg.scheduling_spec[0]))
            if alloc_mode is not None:
                sglang_spec.cpu *= alloc_mode.gen_instance_size
                sglang_spec.mem *= alloc_mode.gen_instance_size
                if sglang_spec.gpu > 0:
                    sglang_spec.gpu = alloc_mode.gen_instance_size

            # Build SGLang launch command from server_args
            sglang_cmd = " ".join(SGLangConfig.build_cmd_from_args(server_args or {}))
            sglang_spec.cmd = sglang_cmd

            sglang_role = f"{self._worker_role}{self._SGLANG_SUFFIX}"
            sglang_job = Job(
                replicas=dp_size,
                tasks=[sglang_spec for _ in range(dp_size)],
                scheduling_strategy=SchedulingStrategy(),
                role=sglang_role,
            )
            self.scheduler.create_workers(job=sglang_job)
            self._service_roles.append(sglang_role)

            sglang_workers = self.scheduler.get_workers(role=sglang_role)
            self.workers = sglang_workers
            self._sglang_addrs = [
                f"http://{w.ip}:{w.worker_ports[0]}" for w in sglang_workers
            ]

            # Wait for SGLang servers to be healthy
            for i, addr in enumerate(self._sglang_addrs):
                self._wait_for_service(
                    f"{addr}/health", f"SGLang-{i}", timeout=cfg.setup_timeout
                )

            # Build server_infos from worker addresses
            from areal.api.io_struct import LocalInfServerInfo

            self.server_infos = [
                LocalInfServerInfo(
                    host=w.ip,
                    port=int(w.worker_ports[0]),
                    process=None,  # type: ignore[arg-type]  # managed by scheduler
                )
                for w in sglang_workers
            ]

        logger.info("SGLang servers: %s", self._sglang_addrs)

        # -- 2. Launch Router -----------------------------------------------
        router_spec = SchedulingSpec(
            gpu=0,
            cpu=1,
            mem=4,
            cmd=(
                f"python -m areal.experimental.gateway.router"
                f" --admin-api-key {cfg.admin_api_key}"
                f" --routing-strategy {cfg.routing_strategy}"
                f" --poll-interval {cfg.poll_interval}"
                f" --log-level {cfg.log_level}"
            ),
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
        for i, sglang_addr in enumerate(self._sglang_addrs):
            dp_spec = SchedulingSpec(
                gpu=0,
                cpu=1,
                mem=4,
                cmd=(
                    f"python -m areal.experimental.gateway.data_proxy"
                    f" --backend-addr {sglang_addr}"
                    f" --tokenizer-path {cfg.tokenizer_path}"
                    f" --log-level {cfg.log_level}"
                    f" --request-timeout {cfg.request_timeout}"
                ),
            )
            dp_role = f"{self._worker_role}{self._DATA_PROXY_SUFFIX}-{i}"
            dp_job = Job(
                replicas=1,
                tasks=[dp_spec],
                scheduling_strategy=SchedulingStrategy(),
                role=dp_role,
            )
            self.scheduler.create_workers(job=dp_job)
            self._service_roles.append(dp_role)

            dp_workers = self.scheduler.get_workers(role=dp_role)
            dp_addr = f"http://{dp_workers[0].ip}:{dp_workers[0].worker_ports[0]}"
            self._data_proxy_addrs.append(dp_addr)
            self._wait_for_service(f"{dp_addr}/health", f"DataProxy-{i}")

        logger.info("Data Proxies: %s", self._data_proxy_addrs)

        # -- 4. Launch Gateway ----------------------------------------------
        gw_spec = SchedulingSpec(
            gpu=0,
            cpu=1,
            mem=4,
            cmd=(
                f"python -m areal.experimental.gateway.gateway"
                f" --admin-api-key {cfg.admin_api_key}"
                f" --router-addr {self._router_addr}"
                f" --forward-timeout {cfg.request_timeout}"
                f" --log-level {cfg.log_level}"
            ),
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

    def _register_data_proxies_in_router(self) -> None:
        """Register all data proxy workers in the router."""
        import requests

        for dp_addr in self._data_proxy_addrs:
            resp = requests.post(
                f"{self._router_addr}/register_worker",
                json={"worker_addr": dp_addr},
                headers={"Authorization": f"Bearer {self.config.admin_api_key}"},
                timeout=5,
            )
            resp.raise_for_status()
            logger.info("Registered data proxy %s in router", dp_addr)

    @property
    def callback_addr(self) -> str:
        """Return gateway address as 'host:port' for training controller callbacks."""
        # Strip the http:// prefix to match the host:port format
        addr = self._gateway_addr
        if addr.startswith("http://"):
            addr = addr[len("http://") :]
        return addr

    # -- Destroy -----------------------------------------------------------

    def destroy(self) -> None:
        # Destroy inference engine
        if self._gateway_inf_engine is not None:
            self._gateway_inf_engine.destroy()
            self._gateway_inf_engine = None

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
        self._sglang_addrs.clear()
        self._data_proxy_addrs.clear()
        self._router_addr = ""
        self._gateway_addr = ""
        self._staleness_manager = None

    # -- Version management ------------------------------------------------

    def set_version(self, version: int) -> None:
        with self._version_lock:
            self._version = version
            if self._gateway_inf_engine is not None:
                self._gateway_inf_engine.set_version(version)
        # Broadcast to all workers via gateway
        if self._gateway_addr:
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
        if self._gateway_addr:
            self._gateway_http_post("/pause_generation", {})

    def resume(self) -> None:
        """Broadcast resume to all workers + resume dispatcher."""
        if self._gateway_addr:
            self._gateway_http_post("/continue_generation", {})
        if self._gateway_inf_engine is not None:
            self._gateway_inf_engine.resume()

    async def pause_generation(self) -> None:
        if self._gateway_addr:
            self._gateway_http_post("/pause_generation", {})

    async def continue_generation(self) -> None:
        if self._gateway_addr:
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
        return self._gateway_addr

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
