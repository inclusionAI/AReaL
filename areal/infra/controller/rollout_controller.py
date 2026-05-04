from __future__ import annotations

import asyncio
import concurrent.futures
import shutil
import threading
import time
import traceback
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from threading import Lock
from typing import Any

from flask import Flask, jsonify, request
from torchdata.stateful_dataloader import StatefulDataLoader
from werkzeug.serving import make_server

from areal.api import (
    InferenceEngine,
    Job,
    LocalInfServerInfo,
    ModelRequest,
    ModelResponse,
    ParamSpec,
    RolloutWorkflow,
    Scheduler,
    WeightUpdateMeta,
    Worker,
    WorkflowLike,
)
from areal.api.alloc_mode import ModelAllocation
from areal.api.cli_args import (
    InferenceEngineConfig,
    PerfTracerConfig,
    SchedulingSpec,
)
from areal.infra.rpc.serialization import deserialize_value
from areal.infra.utils.concurrent import run_async_task
from areal.utils import logging, perf_tracer
from areal.utils.data import cycle_dataloader
from areal.utils.dynamic_import import import_from_string
from areal.utils.network import find_free_ports, format_hostport, gethostip
from areal.utils.perf_tracer import trace_perf

from ..staleness_manager import StalenessManager
from ..workflow_executor import BatchTaskDispatcher, TaskIdGenerator

logger = logging.getLogger("RolloutController")


# NOTE: remote task input has a slightly different
# type annotation, which disallows workflow object or types
@dataclass
class _RemoteRolloutTaskInput:
    task_id: int
    data: dict[str, Any]
    workflow: str | None
    workflow_kwargs: dict[str, Any]
    should_accept_fn: str | None
    is_eval: bool = False
    group_size: int = 1
    proxy_addr: str | None = None


@dataclass
class _RemoteRolloutResult:
    task_id: int
    trajectory: dict[str, Any]


class RolloutController:
    def __init__(
        self,
        inf_engine: type[InferenceEngine],
        config: InferenceEngineConfig,
        scheduler: Scheduler,
    ):
        self.inf_engine = inf_engine
        self.config = config
        self.scheduler = scheduler

        # Parse allocation from config.backend
        self.rollout_alloc = ModelAllocation.from_str(config.backend)

        # Worker management
        self.workers: list[Worker] = []  # List of Worker objects from scheduler
        self.server_infos: list[LocalInfServerInfo] = []
        self._worker_role: str

        # Round-robin scheduling
        self._current_worker_idx = 0

        # State
        self._version_lock = Lock()
        self._version = 0

        self._task_id_generator = TaskIdGenerator()

        # Use provided staleness manager or create a default one
        # The manager will be properly initialized in initialize()
        self._staleness_manager: StalenessManager | None = None

        # Dispatcher will be initialized in initialize() after staleness_manager is ready
        self._dispatcher: (
            BatchTaskDispatcher[_RemoteRolloutTaskInput, _RemoteRolloutResult] | None
        ) = None

        # HTTP callback server
        self._callback_app: Flask | None = None
        self._callback_server = None
        self._callback_server_thread: threading.Thread | None = None
        self._callback_port: int | None = None
        self._callback_host: str | None = None
        self._callback_loop: asyncio.AbstractEventLoop | None = None
        self._callback_loop_ready = threading.Event()

        # Pending fire-and-forget NCCL update futures. Tracked so the
        # tensor update handler can drain them before dispatching its own
        # RPC, preventing engine-thread queue starvation.
        self._pending_xccl_futures: list[concurrent.futures.Future] = []
        self._xccl_futures_lock = threading.Lock()

        # Task completion futures
        self._pending_futures: dict[int, asyncio.Future] = {}
        self._futures_lock = threading.Lock()

        # Proxy worker management (for AgentWorkflow support)
        self.proxy_workers: list[Worker] = []
        self.proxy_addrs: list[str] = []
        self._proxy_started = False

        # Proxy gateway server (for online/external access)
        self._proxy_gateway_app = None
        self._proxy_gateway_server = None
        self._proxy_gateway_thread: threading.Thread | None = None
        self._proxy_gateway_port: int | None = None
        self._proxy_gateway_host: str | None = None

    @property
    def _proxy_role(self) -> str:
        """Generate a unique proxy role name based on the worker role.

        This avoids collisions when multiple controllers (e.g., rollout and
        eval-rollout) each fork proxy workers into the same scheduler.
        """
        if not hasattr(self, "_worker_role"):
            raise RuntimeError(
                "Cannot access _proxy_role before initialize() is called"
            )
        return f"proxy-{self._worker_role}"

    def _proxy_engine_name(self, rank: int) -> str:
        """Generate engine name for a proxy worker rank."""
        return f"{self._proxy_role}/{rank}"

    def _engine_name(self, rank: int) -> str:
        """Generate engine name for a worker rank.

        Engine names follow the "role/index" format (e.g., "rollout/0", "rollout/1").
        """
        return f"{self._worker_role}/{rank}"

    def initialize(
        self,
        role: str,
        server_args: dict[str, Any] | None = None,
        server_infos: list[LocalInfServerInfo] | None = None,
        *args,
        **kwargs,
    ):
        # Get scheduling config from kwargs or use defaults
        # Schedule inference engines in the granularity of instance sizes,
        # usually TP x PP.
        self._worker_role = role

        instance_size = (
            self.rollout_alloc.parallel.tp_size * self.rollout_alloc.parallel.pp_size
        )
        dp_size = self.rollout_alloc.parallel.dp_size

        # The first element of `self.config.scheduling_spec` is the resource spec
        # of workers, aka the RPC server process. Since a worker exactly matches
        # to a single engine instance in the local environment, we can dirrectly
        # use the spec of engines  as the spec of workers here. Engine scheduling
        # specs are ignored.
        sch_spec = SchedulingSpec(**asdict(self.config.scheduling_spec[0]))
        sch_spec.cpu *= instance_size
        sch_spec.mem *= instance_size
        if sch_spec.gpu > 0:
            sch_spec.gpu = instance_size

        if sch_spec.ray_placement_strategy == "shared":
            # do not support shared placement for rollout
            logger.warning(
                "Placement strategy 'shared' is not supported for rollouts. Forcing to 'separate' strategy"
            )
            sch_spec.ray_placement_strategy = "separate"

        job = Job(
            replicas=dp_size,
            tasks=[sch_spec for _ in range(dp_size)],
            scheduling_strategy=self.config.scheduling_strategy,
            role=self._worker_role,
        )

        # Call async scheduler methods synchronously
        run_async_task(
            self._async_initialize, job, server_args, server_infos, *args, **kwargs
        )

        # Initialize staleness manager for global capacity control
        max_concurrent_rollouts = (
            self.config.max_concurrent_rollouts or self.config.consumer_batch_size
        )
        consumer_batch_size = self.config.consumer_batch_size
        self._staleness_manager = StalenessManager(
            version_provider=self,
            max_concurrent_rollouts=max_concurrent_rollouts,
            consumer_batch_size=consumer_batch_size,
            max_staleness=self.config.max_head_offpolicyness,
        )

        # Create and initialize the dispatcher
        qsize = self.config.queue_size or max_concurrent_rollouts * 16
        self._dispatcher = BatchTaskDispatcher[
            _RemoteRolloutTaskInput, _RemoteRolloutResult
        ](
            max_queue_size=qsize,
            task_factory=self._create_submit_callback,
            staleness_manager=self._staleness_manager,
            enable_tracing=self.config.enable_rollout_tracing,
        )
        # Initialize the dispatcher's async task runner
        self._dispatcher.initialize(logger=logger)

        # Start callback server for weight sync coordination
        self._start_callback_server()

    async def _async_initialize(
        self,
        job: Job,
        server_args: dict[str, Any],
        server_infos: list[LocalInfServerInfo] | None = None,
        *args,
        **kwargs,
    ):
        # Create workers via scheduler
        logger.info("Creating workers via scheduler...")
        worker_ids = self.scheduler.create_workers(job=job)
        logger.info(f"Workers created: {worker_ids}")

        # Wait for workers to be ready
        logger.info("Waiting for workers to be ready...")
        self.workers = self.scheduler.get_workers(role=job.role)
        logger.info(f"Workers ready: {[w.id for w in self.workers]}")

        # Get engine class path for dynamic import on workers
        engine_class = self.inf_engine

        # Create and initialize engines on workers
        logger.info("Creating engines...")
        tasks = [
            self.scheduler.create_engine(
                worker_id=worker.id,
                engine=f"{engine_class.__module__}.{engine_class.__name__}",
                engine_name=self._engine_name(rank),
                config=self.config,
            )
            for rank, worker in enumerate(self.workers)
        ]
        await asyncio.gather(*tasks)
        logger.info("Engine created on all workers!")

        logger.info("Calling engine initialization...")
        if server_infos is not None:
            # Connecting to existing local servers for evaluation
            self.server_infos = server_infos
            assert len(self.server_infos) == len(self.workers), (
                len(self.server_infos),
                len(self.workers),
            )
            tasks = [
                self.scheduler.async_call_engine(
                    worker_id=worker.id,
                    method="initialize",
                    engine_name=self._engine_name(rank),
                    # args in `engine_api`
                    engine_id=str(rank),
                    addr=f"{info.host}:{info.port}",
                    *args,
                    **kwargs,
                )
                for rank, (worker, info) in enumerate(
                    zip(self.workers, self.server_infos)
                )
            ]
            await asyncio.gather(*tasks)
        else:
            self.server_infos = await self._collective_rpc_async(
                "launch_server", server_args=server_args
            )
            tasks = [
                self.scheduler.async_call_engine(
                    worker_id=worker.id,
                    method="initialize",
                    engine_name=self._engine_name(rank),
                    # args in `engine_api`
                    engine_id=str(rank),
                    *args,
                    **kwargs,
                )
                for rank, worker in enumerate(self.workers)
            ]
            await asyncio.gather(*tasks)

        logger.info("All engines are initialized...")

    def destroy(self):
        # Stop background threads and shutdown the async task runner
        if self._dispatcher is not None:
            self._dispatcher.destroy()

        self._stop_callback_server()

        self._collective_rpc("destroy", http_timeout=60.0)

        # Delete workers via scheduler
        if hasattr(self, "_worker_role"):
            try:
                self.scheduler.delete_workers(role=self._worker_role)
                self.workers.clear()
                logger.info("Workers deleted")
            except Exception:
                logger.error(f"Error deleting workers: {traceback.format_exc()}")

        # Delete proxy workers if initialized
        if self._proxy_started:
            try:
                self.scheduler.delete_workers(role=self._proxy_role)
                self.proxy_workers.clear()
                self.proxy_addrs.clear()
                self._proxy_started = False
                logger.info("Proxy workers deleted")
            except Exception:
                logger.error(f"Error deleting proxy workers: {traceback.format_exc()}")

        # Shutdown proxy gateway if initialized
        self._stop_proxy_gateway()
        with self._futures_lock:
            self._pending_futures.clear()

    def start_proxy(self) -> None:
        """Initialize proxy workers for AgentWorkflow support.

        Creates proxy workers colocated with rollout workers. Each proxy worker
        runs a ProxyRolloutServer that connects to the same inference server
        as its corresponding rollout worker.
        """
        if self._proxy_started:
            logger.warning("Proxy workers already initialized")
            return

        if not self.server_infos:
            raise RuntimeError(
                "Cannot initialize proxy workers: rollout not initialized. "
                "Call initialize() first."
            )

        run_async_task(self._async_start_proxy)
        self._proxy_started = True

    async def _async_start_proxy(self) -> None:
        """Async implementation of proxy worker initialization."""
        command = "areal.experimental.openai.proxy.proxy_rollout_server"
        worker_ids = self.scheduler.fork_workers(
            role=self._proxy_role,
            target_role=self._worker_role,
            command=command,
        )
        logger.info(f"Proxy workers forked: {worker_ids}")

        self.proxy_workers = self.scheduler.get_workers(role=self._proxy_role)
        logger.info(f"Proxy workers: {[w.id for w in self.proxy_workers]}")

        engine_class = f"{self.inf_engine.__module__}.{self.inf_engine.__name__}"

        create_tasks = []
        for rank, worker in enumerate(self.proxy_workers):
            create_tasks.append(
                self.scheduler.create_engine(
                    worker_id=worker.id,
                    engine=engine_class,
                    engine_name=self._proxy_engine_name(rank),
                    config=self.config,
                )
            )
        await asyncio.gather(*create_tasks)
        logger.info("Proxy engines created")

        init_tasks = []
        for rank, (worker, server_info) in enumerate(
            zip(self.proxy_workers, self.server_infos, strict=True)
        ):
            init_tasks.append(
                self.scheduler.async_call_engine(
                    worker_id=worker.id,
                    method="initialize",
                    engine_name=self._proxy_engine_name(rank),
                    addr=f"{server_info.host}:{server_info.port}",
                )
            )
            self.proxy_addrs.append(
                f"http://{format_hostport(worker.ip, int(worker.worker_ports[0]))}"
            )
        await asyncio.gather(*init_tasks)

        logger.info(f"Proxy servers initialized. Addresses: {self.proxy_addrs}")

    def get_proxy_addr(self, rank: int) -> str:
        """Get the proxy server address for a given rollout worker rank.

        Parameters
        ----------
        rank : int
            The rank of the rollout worker

        Returns
        -------
        str
            The HTTP address of the corresponding proxy server
        """
        if not self._proxy_started:
            raise RuntimeError(
                "Proxy workers not initialized. Call start_proxy() first."
            )
        if rank >= len(self.proxy_addrs):
            raise IndexError(
                f"Invalid rank {rank}, only {len(self.proxy_addrs)} proxy workers"
            )
        return self.proxy_addrs[rank]

    def start_proxy_gateway(self) -> None:
        """Start the proxy gateway for external access.

        Creates a FastAPI server that routes requests to backend proxy
        workers. Requires ``start_proxy()`` to have been called first.
        """
        if not self._proxy_started:
            raise RuntimeError(
                "Proxy workers not initialized. Call start_proxy() first."
            )
        if self._proxy_gateway_host is not None:
            logger.warning("Proxy gateway already running")
            return

        from areal.api.cli_args import OpenAIProxyConfig
        from areal.experimental.openai.proxy.proxy_gateway import (
            create_proxy_gateway_app,
        )

        openai_cfg = self.config.openai or OpenAIProxyConfig()

        app = create_proxy_gateway_app(
            proxy_addrs=self.proxy_addrs,
            admin_api_key=openai_cfg.admin_api_key,
        )

        self._proxy_gateway_port = find_free_ports(1)[0]
        self._proxy_gateway_host = gethostip()
        self._proxy_gateway_app = app

        def serve():
            import uvicorn

            try:
                config = uvicorn.Config(
                    app,
                    host="0.0.0.0",
                    port=self._proxy_gateway_port,
                    log_level="warning",
                )
                server = uvicorn.Server(config)
                self._proxy_gateway_server = server
                server.run()
            except Exception:
                logger.error("Proxy gateway thread crashed", exc_info=True)

        self._proxy_gateway_thread = threading.Thread(target=serve, daemon=True)
        self._proxy_gateway_thread.start()

        # Wait for uvicorn to bind the port before propagating the address
        # to worker engines via collective RPC.
        import time

        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if (
                self._proxy_gateway_server is not None
                and self._proxy_gateway_server.started
            ):
                break
            time.sleep(0.05)
        else:
            raise RuntimeError(
                "Proxy gateway failed to start within 10s. "
                f"Cannot propagate address "
                f"{self._proxy_gateway_host}:{self._proxy_gateway_port}"
            )

        logger.info(
            "Proxy gateway started on "
            f"{self._proxy_gateway_host}:{self._proxy_gateway_port}"
        )

        # Propagate proxy_gateway_addr to all rollout worker engines
        # so that _resolve_workflow can pick it up for online mode.
        self._collective_rpc(
            "set_proxy_gateway_addr",
            addr=self.proxy_gateway_addr,
        )

    @property
    def proxy_gateway_addr(self) -> str:
        """Single URL for external users."""
        if self._proxy_gateway_host is None:
            raise RuntimeError("Proxy gateway not started")
        return f"http://{format_hostport(self._proxy_gateway_host, self._proxy_gateway_port)}"

    def _stop_proxy_gateway(self) -> None:
        """Stop the proxy gateway server if running."""
        if self._proxy_gateway_host is None:
            return
        logger.info("Stopping proxy gateway...")
        if self._proxy_gateway_server is not None:
            self._proxy_gateway_server.should_exit = True
        if self._proxy_gateway_thread is not None:
            self._proxy_gateway_thread.join(timeout=30.0)
            if self._proxy_gateway_thread.is_alive():
                logger.warning(
                    "Proxy gateway thread did not exit within 30s; "
                    "daemon thread will be killed on process exit"
                )
        self._proxy_gateway_app = None
        self._proxy_gateway_server = None
        self._proxy_gateway_thread = None
        self._proxy_gateway_port = None
        self._proxy_gateway_host = None

    def _start_callback_server(self):
        """Start Flask HTTP server to receive callbacks from RolloutCallback."""
        if self._callback_server is not None:
            logger.warning("Callback server already running")
            return

        app = Flask(__name__)
        app.logger.disabled = True

        @app.route("/callback/init_weights_group", methods=["POST"])
        def init_weights_group():
            payload = request.get_json() or {}
            meta = deserialize_value(payload.get("meta"))
            asyncio.run_coroutine_threadsafe(
                self.init_weights_update_group(meta), self._callback_loop
            ).result()
            return jsonify({"status": "ok"})

        @app.route("/callback/update_weights_xccl", methods=["POST"])
        def update_weights():
            _t0 = time.time()
            payload = request.get_json() or {}
            meta = deserialize_value(payload.get("meta"))
            param_specs = deserialize_value(payload.get("param_specs"))
            _n_specs = len(param_specs) if param_specs else 0
            logger.info(
                f"[DiagMTP] /callback/update_weights_xccl ENTERED "
                f"(n_param_specs={_n_specs}, version={getattr(meta, 'version', '?')})"
            )
            # Fire-and-forget: schedule the NCCL weight update as a background
            # task and return HTTP 200 immediately. This prevents infrastructure
            # proxy timeouts (504) since the full NCCL transfer chain can take
            # >60s. NCCL broadcast is collective — when the training side's
            # broadcast handle completes, the receive side has the data.
            # Inspired by verl's pattern of decoupling HTTP from NCCL ops.
            fut = asyncio.run_coroutine_threadsafe(
                self.update_weights_from_distributed(meta, param_specs),
                self._callback_loop,
            )
            with self._xccl_futures_lock:
                self._pending_xccl_futures = [
                    f for f in self._pending_xccl_futures if not f.done()
                ]
                self._pending_xccl_futures.append(fut)
                _n_pending = len(self._pending_xccl_futures)
            logger.info(
                f"[DiagMTP] /callback/update_weights_xccl returning HTTP 200 "
                f"(fire-and-forget, handler took {time.time() - _t0:.3f}s, "
                f"pending_xccl_futures={_n_pending})"
            )
            return jsonify({"status": "ok"})

        @app.route("/callback/update_weights_disk", methods=["POST"])
        def update_weights_disk():
            payload = request.get_json() or {}
            meta = deserialize_value(payload.get("meta"))
            asyncio.run_coroutine_threadsafe(
                self.update_weights_from_disk(meta), self._callback_loop
            ).result()
            return jsonify({"status": "ok"})

        @app.route("/callback/pause_generation", methods=["POST"])
        def pause_generation():
            _t0 = time.time()
            logger.info("[DiagMTP] /callback/pause_generation ENTERED")
            asyncio.run_coroutine_threadsafe(
                self.pause_generation(), self._callback_loop
            ).result()
            logger.info(
                f"[DiagMTP] /callback/pause_generation completed in {time.time() - _t0:.3f}s"
            )
            return jsonify({"status": "ok"})

        @app.route("/callback/continue_generation", methods=["POST"])
        def continue_generation():
            _t0 = time.time()
            logger.info("[DiagMTP] /callback/continue_generation ENTERED")
            asyncio.run_coroutine_threadsafe(
                self.continue_generation(), self._callback_loop
            ).result()
            logger.info(
                f"[DiagMTP] /callback/continue_generation completed in {time.time() - _t0:.3f}s"
            )
            return jsonify({"status": "ok"})

        @app.route("/callback/update_weights_tensor", methods=["POST"])
        def update_weights_tensor():
            _t0 = time.time()
            logger.info(
                "[DiagMTP] /callback/update_weights_tensor handler ENTERED "
                f"(flask_thread={threading.current_thread().name})"
            )
            payload = request.get_json() or {}
            _t1 = time.time()
            logger.info(
                f"[DiagMTP] payload parsed in {_t1 - _t0:.3f}s, "
                f"payload_keys={list(payload.keys())}, "
                f"payload_size_bytes={len(str(payload))}"
            )
            serialized_payload = deserialize_value(payload.get("serialized_payload"))
            _t2 = time.time()
            _sp_keys = (
                list(serialized_payload.keys())
                if isinstance(serialized_payload, dict)
                else "N/A"
            )
            _n_snt = (
                len(serialized_payload.get("serialized_named_tensors", []))
                if isinstance(serialized_payload, dict)
                else 0
            )
            _snt_b64_len = (
                sum(len(s) for s in serialized_payload.get("serialized_named_tensors", []))
                if isinstance(serialized_payload, dict)
                else 0
            )
            logger.info(
                f"[DiagMTP] deserialize_value completed in {_t2 - _t1:.3f}s, "
                f"serialized_payload type={type(serialized_payload).__name__}, "
                f"keys={_sp_keys}, n_serialized_tensors={_n_snt}, "
                f"total_b64_bytes={_snt_b64_len} ({_snt_b64_len / 1024 / 1024:.2f} MB)"
            )
            # BLOCKING: MTP tensor update must complete before returning.
            # Following verl/slime's fully-blocking weight update pattern.
            # Unlike NCCL updates (fire-and-forget for concurrent collective
            # participation), tensor updates are rank-0-only unilateral
            # operations that can safely block.
            # Check callback_loop health before scheduling
            _loop = self._callback_loop
            _loop_running = _loop is not None and _loop.is_running()
            _loop_closed = _loop is not None and _loop.is_closed()
            logger.info(
                f"[DiagMTP] _callback_loop status: running={_loop_running}, "
                f"closed={_loop_closed}, loop={_loop}"
            )
            logger.info(
                "[DiagMTP] Scheduling update_weights_from_tensor coroutine "
                "on _callback_loop (BLOCKING with .result())..."
            )
            _t3 = time.time()
            try:
                fut = asyncio.run_coroutine_threadsafe(
                    self.update_weights_from_tensor(serialized_payload),
                    self._callback_loop,
                )
                logger.info(
                    f"[DiagMTP] Coroutine scheduled in {time.time() - _t3:.3f}s, "
                    f"fut={fut}, calling .result() to block..."
                )
                fut.result()
                _t4 = time.time()
                logger.info(
                    f"[DiagMTP] .result() completed in {_t4 - _t3:.3f}s "
                    f"(total handler time: {_t4 - _t0:.3f}s). "
                    "Returning HTTP 200."
                )
            except Exception as e:
                logger.error(
                    f"[DiagMTP] .result() raised exception after "
                    f"{time.time() - _t3:.3f}s: {type(e).__name__}: {e}",
                    exc_info=True,
                )
                raise
            return jsonify({"status": "ok"})

        @app.route("/callback/read_weights_by_name", methods=["POST"])
        def read_weights_by_name_route():
            payload = request.get_json() or {}
            names = payload.get("names", []) or []
            truncate_size = int(payload.get("truncate_size", 8) or 8)
            try:
                fut = asyncio.run_coroutine_threadsafe(
                    self.read_weights_by_name(
                        names=names,
                        truncate_size=truncate_size,
                    ),
                    self._callback_loop,
                )
                entries = fut.result()
                return jsonify({"entries": entries})
            except Exception as _e:
                logger.warning(
                    "[DiagMTP] /callback/read_weights_by_name FAILED: %s",
                    _e,
                )
                return jsonify({"entries": [], "error": str(_e)})

        @app.route("/callback/rollout_complete", methods=["POST"])
        def rollout_complete():
            payload = request.get_json() or {}
            task_id = payload.get("task_id")
            try:
                self._resolve_task_future(task_id)
                return jsonify({"status": "ok"})
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        # [v32] /callback/get_mtp_weight_norm (DEPRECATED STUB)
        # ------------------------------------------------------------
        # v32 attempted to read MTP weights back from sglang via
        # /get_weights_by_name, but MiMoForCausalLM does not override
        # get_weights_by_name and the scheduler routes the call to
        # tp_worker (target), not draft_worker (where MTP layers
        # actually live). Architecturally unfixable from our side.
        # Kept as a 200-stub so older training images calling the old
        # route get a deterministic "deprecated" signal rather than
        # a 404-wrapped-as-500.
        @app.route("/callback/get_mtp_weight_norm", methods=["POST"])
        def get_mtp_weight_norm():
            return jsonify(
                {"error": "deprecated_v32_route_use_get_mtp_probe"}
            ), 200

        # ------------------------------------------------------------
        # [v33] /callback/get_mtp_probe
        # ------------------------------------------------------------
        # Deterministic inference probe. Posts /generate to
        # server_infos[0] with temperature=0, top_p=1, top_k=1,
        # max_new_tokens=1, return_logprob=1 on a fixed prompt, and
        # returns the first input_token_logprob as a float.
        #
        # Payload: {"version": <int>}
        # Response: {"version": <int>, "logprob": <float>,
        #            "server": <host:port>, "prompt": <str>}
        @app.route("/callback/get_mtp_probe", methods=["POST"])
        def get_mtp_probe():
            payload = request.get_json() or {}
            _version = payload.get("version")
            _srv = None
            _prompt_v33 = "fixed_token_seq_v35"
            _probe_ids_v35 = [1, 100, 200, 300, 400, 500, 600, 700]
            try:
                if not self.server_infos:
                    return jsonify(
                        {"error": "no server_infos",
                         "version": _version,
                         "server": None}
                    ), 200
                _s0 = self.server_infos[0]
                _srv = f"{_s0.host}:{_s0.port}"
                try:
                    import requests as _rq_v33c
                except Exception as _e_imp:
                    return jsonify(
                        {"error": f"import fail: {_e_imp!r}",
                         "version": _version,
                         "server": _srv}
                    ), 200
                _url = f"http://{_srv}/generate"
                _req = {
                    "input_ids": _probe_ids_v35,
                    "sampling_params": {
                        "temperature": 0.0,
                        "top_p": 1.0,
                        "top_k": 1,
                        "max_new_tokens": 1,
                    },
                    "return_logprob": True,
                    "logprob_start_len": 0,
                }
                try:
                    _r = _rq_v33c.post(
                        _url, json=_req, timeout=120.0,
                        proxies={"http": None, "https": None},
                    )
                except Exception as _e_http:
                    return jsonify(
                        {"error": f"http fail: {_e_http!r}",
                         "version": _version,
                         "server": _srv, "url": _url}
                    ), 200
                if _r.status_code != 200:
                    return jsonify(
                        {"error": f"sglang status={_r.status_code}",
                         "version": _version,
                         "server": _srv, "url": _url,
                         "body": _r.text[:400]}
                    ), 200
                try:
                    _j = _r.json()
                except Exception as _e_js:
                    return jsonify(
                        {"error": f"json fail: {_e_js!r}",
                         "version": _version,
                         "server": _srv}
                    ), 200
                _item = _j if isinstance(_j, dict) else (
                    _j[0] if isinstance(_j, list) and _j else {}
                )
                _meta = _item.get("meta_info", {}) if isinstance(_item, dict) else {}
                _itl = _meta.get("input_token_logprobs", None)
                _lp = None
                if isinstance(_itl, list) and _itl:
                    for _e in _itl:
                        if isinstance(_e, (list, tuple)) and _e:
                            _cand = _e[0]
                            if isinstance(_cand, (int, float)):
                                _lp = float(_cand)
                                break
                        elif isinstance(_e, (int, float)):
                            _lp = float(_e)
                            break
                if _lp is None:
                    return jsonify(
                        {"error": "no_input_token_logprob",
                         "version": _version,
                         "server": _srv,
                         "meta_keys": list(_meta.keys()) if isinstance(_meta, dict) else None}
                    ), 200
                return jsonify(
                    {"version": _version,
                     "logprob": _lp,
                     "server": _srv,
                     "prompt": _prompt_v33}
                ), 200
            except Exception as _e:
                logger.warning(
                    f"[v33] get_mtp_probe unexpected: {_e!r}"
                )
                return jsonify(
                    {"error": repr(_e), "version": _version,
                     "server": _srv}
                ), 200

        # ------------------------------------------------------------
        # [v38] /callback/get_draft_probe
        # ------------------------------------------------------------
        # Output-sequence probe: unlike v33 which reads
        # input_token_logprobs[0] (target-model only), this probe
        # drives /generate with max_new_tokens=32, temperature=0,
        # top_k=1, return_logprob=1 and returns:
        #   - output_ids   (first 8 generated token ids)
        #   - output_lps   (per-position logprob of generated tokens)
        #   - last_lp      (last position logprob)
        #   - meta_keys    (raw meta_info keys, for field discovery)
        #   - spec_fields  (any meta_info key containing 'spec' or
        #                   'accept' or 'verify' or 'draft')
        # When draft+MTP heads change behavior, output_ids or
        # output_lps MUST change.  If target is frozen but heads
        # drift, the joint sequence changes => H3 confirmed.
        @app.route("/callback/get_draft_probe", methods=["POST"])
        def get_draft_probe_v38():
            payload = request.get_json() or {}
            _version = payload.get("version")
            _srv = None
            _probe_ids_v38 = [1, 100, 200, 300, 400, 500, 600, 700]
            try:
                if not self.server_infos:
                    return jsonify(
                        {"error": "no server_infos",
                         "version": _version,
                         "server": None}
                    ), 200
                _s0 = self.server_infos[0]
                _srv = f"{_s0.host}:{_s0.port}"
                try:
                    import requests as _rq_v38
                except Exception as _e_imp38:
                    return jsonify(
                        {"error": f"import fail: {_e_imp38!r}",
                         "version": _version,
                         "server": _srv}
                    ), 200
                _url = f"http://{_srv}/generate"
                _req = {
                    "input_ids": _probe_ids_v38,
                    "sampling_params": {
                        "temperature": 0.0,
                        "top_p": 1.0,
                        "top_k": 1,
                        "max_new_tokens": 32,
                    },
                    "return_logprob": True,
                    "logprob_start_len": 0,
                }
                try:
                    _r = _rq_v38.post(
                        _url, json=_req, timeout=120.0,
                        proxies={"http": None, "https": None},
                    )
                except Exception as _e_http38:
                    return jsonify(
                        {"error": f"http fail: {_e_http38!r}",
                         "version": _version,
                         "server": _srv, "url": _url}
                    ), 200
                if _r.status_code != 200:
                    return jsonify(
                        {"error": f"sglang status={_r.status_code}",
                         "version": _version,
                         "server": _srv,
                         "body": _r.text[:400]}
                    ), 200
                try:
                    _j = _r.json()
                except Exception as _e_js38:
                    return jsonify(
                        {"error": f"json fail: {_e_js38!r}",
                         "version": _version,
                         "server": _srv}
                    ), 200
                _item = _j if isinstance(_j, dict) else (
                    _j[0] if isinstance(_j, list) and _j else {}
                )
                _meta = _item.get("meta_info", {}) if isinstance(_item, dict) else {}
                _out_text = _item.get("text", None) if isinstance(_item, dict) else None
                _otl = _meta.get("output_token_logprobs", None) if isinstance(_meta, dict) else None
                _out_ids = []
                _out_lps = []
                if isinstance(_otl, list):
                    for _e in _otl[:32]:
                        _lp_i = None
                        _id_i = None
                        if isinstance(_e, (list, tuple)) and len(_e) >= 2:
                            _cand_lp = _e[0]
                            _cand_id = _e[1]
                            if isinstance(_cand_lp, (int, float)):
                                _lp_i = float(_cand_lp)
                            if isinstance(_cand_id, int):
                                _id_i = int(_cand_id)
                        if _id_i is not None:
                            _out_ids.append(_id_i)
                        if _lp_i is not None:
                            _out_lps.append(_lp_i)
                _last_lp = _out_lps[-1] if _out_lps else None
                _sum_lp = sum(_out_lps) if _out_lps else None
                _meta_keys = list(_meta.keys()) if isinstance(_meta, dict) else []
                _spec_fields = {}
                if isinstance(_meta, dict):
                    for _k, _v in _meta.items():
                        _kl = str(_k).lower()
                        if ("spec" in _kl or "accept" in _kl
                                or "verify" in _kl or "draft" in _kl
                                or "jump" in _kl):
                            try:
                                _spec_fields[str(_k)] = _v
                            except Exception:
                                _spec_fields[str(_k)] = repr(_v)
                return jsonify(
                    {"version": _version,
                     "server": _srv,
                     "out_ids_first8": _out_ids[:8],
                     "out_ids_len": len(_out_ids),
                     "out_lps_first4": _out_lps[:4],
                     "last_lp": _last_lp,
                     "sum_lp": _sum_lp,
                     "out_text_head": (_out_text[:60] if isinstance(_out_text, str) else None),
                     "meta_keys": _meta_keys,
                     "spec_fields": _spec_fields}
                ), 200
            except Exception as _e38:
                logger.warning(
                    f"[v38] get_draft_probe unexpected: {_e38!r}"
                )
                return jsonify(
                    {"error": repr(_e38), "version": _version,
                     "server": _srv}
                ), 200

        @app.errorhandler(Exception)
        def handle_error(e):
            logger.error(
                f"Callback handler error: {e} "
                f"(url={request.url}, method={request.method}, "
                f"path={request.path}, endpoint={request.endpoint})"
            )
            return jsonify({"error": str(e)}), 500

        self._callback_port = find_free_ports(1)[0]
        self._callback_host = gethostip()
        self._callback_app = app
        self._callback_server = make_server(
            self._callback_host, self._callback_port, app, threaded=False
        )

        # Suppress Werkzeug access logs (e.g., "POST /callback/rollout_complete 200 -")
        # Override log_request directly on the request handler class
        self._callback_server.RequestHandlerClass.log_request = (
            lambda self, *args, **kwargs: None
        )

        # Also configure Werkzeug logger level for any other log messages
        import logging as stdlib_logging

        werkzeug_logger = stdlib_logging.getLogger("werkzeug")
        werkzeug_logger.setLevel(stdlib_logging.WARNING)

        def run_async_loop():
            """Run a dedicated asyncio event loop in a background thread.

            This loop processes coroutines scheduled via
            asyncio.run_coroutine_threadsafe(). Unlike the original design
            which used run_until_complete() from the werkzeug handler thread,
            a dedicated running loop supports both blocking (.result()) and
            fire-and-forget patterns — critical for avoiding proxy/infra
            timeouts on long-running NCCL weight transfers.
            """
            self._callback_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._callback_loop)
            self._callback_loop_ready.set()
            self._callback_loop.run_forever()

        self._callback_loop_thread = threading.Thread(
            target=run_async_loop, daemon=True
        )
        self._callback_loop_thread.start()
        self._callback_loop_ready.wait()

        def serve_forever():
            logger.info(
                f"Callback server started on {format_hostport(self._callback_host, self._callback_port)}"
            )
            self._callback_server.serve_forever()

        self._callback_server_thread = threading.Thread(
            target=serve_forever, daemon=True
        )
        self._callback_server_thread.start()

    def _stop_callback_server(self):
        """Stop the callback server if running."""
        if self._callback_server is not None:
            logger.info("Stopping callback server...")
            self._callback_server.shutdown()
            if self._callback_loop is not None:
                self._callback_loop.call_soon_threadsafe(self._callback_loop.stop)
                self._callback_loop.close()
            self._callback_server = None
            self._callback_app = None
            self._callback_server_thread = None
            self._callback_port = None
            self._callback_host = None
            self._callback_loop = None
            self._callback_loop_ready.clear()

    @property
    def callback_addr(self) -> str:
        """Return callback server address as 'host:port'."""
        if self._callback_host is None or self._callback_port is None:
            raise RuntimeError("Callback server not started")
        return format_hostport(self._callback_host, self._callback_port)

    def _resolve_task_future(self, task_id: int):
        """Resolve a pending future with the task result."""
        with self._futures_lock:
            future = self._pending_futures.pop(task_id, None)
        if future:
            future.get_loop().call_soon_threadsafe(future.set_result, None)

    def _collective_rpc(self, method: str, *args, **kwargs) -> list[Any]:
        return run_async_task(self._collective_rpc_async, method, *args, **kwargs)

    async def _collective_rpc_async(self, method: str, *args, **kwargs) -> list[Any]:
        return await self._generic_collective_rpc_async(
            method, self.workers, self._engine_name, *args, **kwargs
        )

    def _proxy_collective_rpc(self, method: str, *args, **kwargs) -> list[Any]:
        return run_async_task(self._proxy_collective_rpc_async, method, *args, **kwargs)

    async def _proxy_collective_rpc_async(
        self, method: str, *args, **kwargs
    ) -> list[Any]:
        return await self._generic_collective_rpc_async(
            method, self.proxy_workers, self._proxy_engine_name, *args, **kwargs
        )

    async def _generic_collective_rpc_async(
        self,
        method: str,
        workers: list[Worker],
        engine_name_fn: Callable[[int], str],
        *args,
        **kwargs,
    ) -> list[Any]:
        import time as _time

        _t0 = _time.time()
        _worker_ids = [w.id for w in workers]
        logger.info(
            f"[DiagMTP] _generic_collective_rpc_async ENTERED: "
            f"method={method}, n_workers={len(workers)}, workers={_worker_ids}"
        )
        tasks = [
            self.scheduler.async_call_engine(
                worker_id=worker.id,
                method=method,
                engine_name=engine_name_fn(rank),
                *args,
                **kwargs,
            )
            for rank, worker in enumerate(workers)
        ]
        logger.info(
            f"[DiagMTP] _generic_collective_rpc_async: "
            f"{len(tasks)} tasks created for method={method}, "
            f"calling asyncio.gather..."
        )
        try:
            results = await asyncio.gather(*tasks)
            _elapsed = _time.time() - _t0
            logger.info(
                f"[DiagMTP] _generic_collective_rpc_async COMPLETED: "
                f"method={method} in {_elapsed:.3f}s"
            )
            return results
        except Exception as e:
            _elapsed = _time.time() - _t0
            logger.error(
                f"[DiagMTP] _generic_collective_rpc_async FAILED: "
                f"method={method} after {_elapsed:.3f}s: "
                f"{type(e).__name__}: {e}",
                exc_info=True,
            )
            raise

    def _choose_worker(self) -> tuple[Worker, int]:
        """Choose a worker for the next request using round-robin scheduling.

        Returns
        -------
        tuple[Worker, int]
            The chosen worker object and its rank
        """
        if not self.workers:
            raise RuntimeError("No workers available to choose from.")
        worker = self.workers[self._current_worker_idx]
        rank = self._current_worker_idx
        self._current_worker_idx = (self._current_worker_idx + 1) % len(self.workers)
        return worker, rank

    def _resolve_workflow_str(self, workflow: WorkflowLike | None) -> str | None:
        """Resolve workflow to a string import path.

        Handles RolloutWorkflow, agent workflow instances/classes, string paths,
        and ``None`` (online mode).
        """
        # None workflow = online mode (config-driven)
        if workflow is None:
            return None

        # String paths - return as-is
        if isinstance(workflow, str):
            return workflow

        # RolloutWorkflow classes
        elif isinstance(workflow, type) and issubclass(workflow, RolloutWorkflow):
            return f"{workflow.__module__}.{workflow.__name__}"

        # RolloutWorkflow instances
        elif isinstance(workflow, RolloutWorkflow):
            return f"{workflow.__module__}.{workflow.__class__.__name__}"

        # Agent-like workflow classes
        elif isinstance(workflow, type):
            return f"{workflow.__module__}.{workflow.__name__}"

        # Agent-like workflow instances
        else:
            return f"{workflow.__module__}.{workflow.__class__.__name__}"

    def _resolve_should_accept_fn(
        self, should_accept_fn: Callable[[dict[str, Any]], bool] | str | None
    ):
        if callable(should_accept_fn):
            raise RuntimeError(
                "If given, `should_accept_fn` must be an importable string path, e.g., 'my_module.filter_func'."
            )
        if should_accept_fn is not None:
            try:
                import_from_string(should_accept_fn)
            except Exception:
                raise RuntimeError(
                    f"Failed to import `should_accept_fn` from string path: {should_accept_fn}"
                )
        return should_accept_fn

    def _rollout_stats(self) -> str:
        stats = self._staleness_manager.get_stats()
        return (
            f"enqueued: {stats.enqueued}, "
            f"running: {stats.running}, "
            f"accepted: {stats.accepted}, "
            f"rejected: {stats.rejected}."
        )

    def _create_submit_callback(self, pending_task: _RemoteRolloutTaskInput):
        async def _submit_then_wait() -> _RemoteRolloutResult | None:
            # Choose worker via round-robin
            worker, rank = self._choose_worker()
            engine_name = self._engine_name(rank)

            # NOTE: No need to call `on_rollout_submitted` here.
            # This function will be passed to `BatchTaskDispather` where
            # `on_rollout_submitted` will be called upon dispatching
            task_id = pending_task.task_id

            manager = self.staleness_manager

            try:
                # Set future for this task
                future = asyncio.get_event_loop().create_future()
                with self._futures_lock:
                    self._pending_futures[task_id] = future

                proxy_addr = pending_task.proxy_addr
                if self._proxy_started and proxy_addr is None:
                    proxy_addr = self.get_proxy_addr(rank)
                engine_task_id = await self.scheduler.async_call_engine(
                    worker.id,
                    "submit",
                    engine_name=engine_name,
                    data=pending_task.data,
                    workflow=pending_task.workflow,
                    workflow_kwargs=pending_task.workflow_kwargs,
                    should_accept_fn=pending_task.should_accept_fn,
                    http_timeout=self.config.request_timeout,
                    is_eval=pending_task.is_eval,
                    group_size=pending_task.group_size,
                    task_id=task_id,
                    callback_addr=f"http://{self.callback_addr}/callback/rollout_complete",
                    proxy_addr=proxy_addr,
                )

                assert task_id == engine_task_id, (task_id, engine_task_id)

                # Wait for callback to resolve the future
                await asyncio.wait_for(future, timeout=self.config.request_timeout)

                # Fetch the result
                result = await self.scheduler.async_call_engine(
                    worker.id,
                    "wait_for_task",
                    engine_name=engine_name,
                    task_id=engine_task_id,
                    timeout=0.1,  # A short time to prevent blocking other requests
                    raise_timeout=False,
                    http_timeout=self.config.request_timeout,
                )

                traj = result
                if traj is not None:
                    manager.on_rollout_accepted()
                    if self.config.enable_rollout_tracing:
                        logger.info(
                            f"Finish and accept rollout. {self._rollout_stats()}"
                        )
                    return _RemoteRolloutResult(task_id=task_id, trajectory=traj)

                manager.on_rollout_rejected()
                if self.config.enable_rollout_tracing:
                    logger.info(f"Finish but reject rollout. {self._rollout_stats()}")
                return None

            except TimeoutError:
                if task_id is not None:
                    with self._futures_lock:
                        self._pending_futures.pop(task_id, None)
                manager.on_rollout_rejected()
                logger.error(f"Rollout timed out after {self.config.request_timeout}s")
                return None
            except Exception as exc:
                if task_id is not None:
                    with self._futures_lock:
                        self._pending_futures.pop(task_id, None)
                manager.on_rollout_rejected()
                logger.error("Workflow execution failed: %s", exc, exc_info=True)
                return None

        return _submit_then_wait

    def get_capacity(self):
        return self.staleness_manager.get_capacity()

    def submit(
        self,
        data: dict[str, Any],
        workflow: WorkflowLike,
        workflow_kwargs: dict[str, Any] | None = None,
        should_accept_fn: str | None = None,
        task_id: int | None = None,
        is_eval: bool = False,
        group_size: int = 1,
        proxy_addr: str | None = None,
    ) -> int:
        workflow_str = self._resolve_workflow_str(workflow)
        should_accept_fn = self._resolve_should_accept_fn(should_accept_fn)
        if workflow_kwargs is None:
            workflow_kwargs = {}

        # NOTE: RolloutController does not support `should_accept_fn`
        # If the workflow's result should be aborted,
        # `arun_episode` should return None instead.
        if task_id is None:
            task_id = self._task_id_generator.next()
        task_input = _RemoteRolloutTaskInput(
            data=data,
            workflow=workflow_str,
            workflow_kwargs=workflow_kwargs,
            should_accept_fn=should_accept_fn,
            task_id=task_id,
            is_eval=is_eval,
            group_size=group_size,
            proxy_addr=proxy_addr,
        )

        # Delegate to dispatcher
        self.dispatcher.submit_task_input(task_input)
        return task_id

    def wait(
        self, count: int, timeout: float | None = None, raise_timeout: bool = True
    ) -> list[dict[str, Any] | None]:
        # Delegate to dispatcher and extract trajectories
        results = self.dispatcher.wait_results(count, timeout, raise_timeout)
        # Log and trace
        if self.config.enable_rollout_tracing:
            logger.info("Rollout results are ready!")

        return [r.trajectory if r is not None else None for r in results]

    @trace_perf("rollout_controller.rollout_batch", category="scheduler")
    def rollout_batch(
        self,
        data: list[dict[str, Any]],
        workflow: WorkflowLike,
        workflow_kwargs: dict[str, Any] | None = None,
        should_accept_fn: str | None = None,
        group_size: int = 1,
    ) -> list[dict[str, Any]]:
        perf_tracer.instant(
            "rollout_controller.rollout_batch",
            category="scheduler",
            args={"data": len(data)},
        )
        for item in data:
            self.submit(
                data=item,
                workflow=workflow,
                workflow_kwargs=workflow_kwargs,
                should_accept_fn=should_accept_fn,
                group_size=group_size,
            )
        results = self.wait(count=len(data))
        # Return list of trajectories
        return [r for r in results if r is not None]

    @trace_perf("rollout_controller.prepare_batch", category="scheduler")
    def prepare_batch(
        self,
        dataloader: StatefulDataLoader,
        workflow: WorkflowLike,
        workflow_kwargs: dict[str, Any] | None = None,
        should_accept_fn: str | None = None,
        group_size: int = 1,
        dynamic_bs: bool = False,
    ) -> list[dict[str, Any]]:
        """Prepare a batch with controlled staleness.

        Continuously submits from dataloader and waits for results, ensuring at least
        two batches are pending to maximize overlap.

        See :meth:`~areal.api.engine_api.InferenceEngine.prepare_batch` for parameters.
        """

        workflow_str = self._resolve_workflow_str(workflow)
        if workflow_kwargs is None:
            workflow_kwargs = {}

        def task_input_generator():
            for data in cycle_dataloader(dataloader):
                for item in data:
                    yield _RemoteRolloutTaskInput(
                        data=item,
                        workflow=workflow_str,
                        workflow_kwargs=workflow_kwargs,
                        should_accept_fn=should_accept_fn,
                        task_id=self._task_id_generator.next(),
                        group_size=group_size,
                    )

        if not hasattr(self, "data_generator"):
            self.data_generator = task_input_generator()

        # Delegate to dispatcher
        assert dataloader.batch_size is not None
        results = self.dispatcher.active_submit_and_wait(
            self.data_generator, batch_size=dataloader.batch_size, dynamic_bs=dynamic_bs
        )

        # Return list of trajectories
        trajectories = [r.trajectory if r is not None else None for r in results]
        return [t for t in trajectories if t is not None]

    async def agenerate(self, req: ModelRequest) -> ModelResponse:
        """Asynchronously generate a response for the given request.

        This method provides direct access to the inference engine's generation capabilities
        for single requests, bypassing the workflow system.

        Parameters
        ----------
        req : ModelRequest
            The model request containing input data and generation parameters

        Returns
        -------
        ModelResponse
            The generated response from the model
        """
        # Choose worker and delegate
        worker, rank = self._choose_worker()

        # Call agenerate on engine via scheduler
        return await self.scheduler.async_call_engine(
            worker_id=worker.id,
            method="agenerate",
            engine_name=self._engine_name(rank),
            req=req,
        )

    async def init_weights_update_group(self, meta: WeightUpdateMeta) -> None:
        tasks = [
            self.scheduler.async_call_engine(
                worker_id=worker.id,
                method="init_weights_update_group",
                engine_name=self._engine_name(rank),
                meta=meta,
                xccl_group_ranks=[rank],
            )
            for rank, worker in enumerate(self.workers)
        ]
        await asyncio.gather(*tasks)

    async def update_weights_from_distributed(
        self, meta: WeightUpdateMeta, param_specs: list[ParamSpec]
    ):
        import time as _time

        _t0 = _time.time()
        _n_specs = len(param_specs) if param_specs else 0
        _spec_names = [s.name for s in param_specs[:5]] if param_specs else []
        logger.info(
            f"[DiagMTP] async update_weights_from_distributed ENTERED "
            f"(n_specs={_n_specs}, version={getattr(meta, 'version', '?')}, "
            f"spec_names={_spec_names}...)"
        )
        try:
            await self._collective_rpc_async(
                "update_weights_from_distributed", meta=meta, param_specs=param_specs
            )
            logger.info(
                f"[DiagMTP] async update_weights_from_distributed COMPLETED "
                f"in {_time.time() - _t0:.3f}s"
            )
        except Exception as e:
            logger.error(
                f"[DiagMTP] async update_weights_from_distributed FAILED "
                f"after {_time.time() - _t0:.3f}s: {type(e).__name__}: {e}",
                exc_info=True,
            )
            raise

    async def update_weights_from_disk(self, meta: WeightUpdateMeta):
        meta.clear_checkpoint_after_load = False
        await self._collective_rpc_async("update_weights_from_disk", meta=meta)
        shutil.rmtree(meta.path, ignore_errors=True)

    async def pause_generation(self):
        await self._collective_rpc_async("pause_generation")

    async def continue_generation(self):
        await self._collective_rpc_async("continue_generation")

    async def update_weights_from_tensor(self, serialized_payload: dict) -> None:
        """Update EAGLE draft model MTP weights via tensor update path.

        Receives pre-serialized tensor data from the training side and
        delegates to inference engine workers which send the serialized
        payload directly to the SGLang server's /update_weights_from_tensor
        endpoint.

        Before dispatching the tensor update RPC, drains all pending
        fire-and-forget NCCL update futures. This ensures the worker's
        engine thread queue is clear, preventing the tensor update from
        being queued behind slow NCCL tasks (which would cause an
        indefinite hang). Follows verl/slime's pattern of fully completing
        all weight updates before proceeding.

        Parameters
        ----------
        serialized_payload : dict
            Pre-serialized payload for /update_weights_from_tensor.
        """
        # Drain all pending NCCL update futures before dispatching the
        # tensor update. The NCCL updates and tensor update both go through
        # the worker's single engine thread queue (via async_call_engine →
        # /call → _submit_to_engine_thread). If NCCL tasks are still queued,
        # the tensor update gets stuck behind them indefinitely.
        with self._xccl_futures_lock:
            pending = list(self._pending_xccl_futures)
            self._pending_xccl_futures.clear()

        if pending:
            _drain_t0 = time.time()
            logger.info(
                f"[DiagMTP] Draining {len(pending)} pending NCCL futures "
                f"before tensor update..."
            )
            done_count = 0
            for i, fut in enumerate(pending):
                _fut_t0 = time.time()
                try:
                    await asyncio.wrap_future(fut)
                    done_count += 1
                    logger.info(
                        f"[DiagMTP] Drained future {i + 1}/{len(pending)} "
                        f"in {time.time() - _fut_t0:.3f}s (done={done_count})"
                    )
                except Exception as e:
                    logger.warning(
                        f"[DiagMTP] Pending NCCL future {i + 1}/{len(pending)} "
                        f"raised after {time.time() - _fut_t0:.3f}s: {e}"
                    )
                    done_count += 1
            logger.info(
                f"[DiagMTP] Drained {done_count}/{len(pending)} NCCL futures "
                f"in {time.time() - _drain_t0:.3f}s"
            )

        import time as _time

        _t0 = _time.time()
        _n_workers = len(self.workers)
        _worker_ids = [w.id for w in self.workers]
        logger.info(
            f"[DiagMTP] async update_weights_from_tensor ENTERED on "
            f"_callback_loop (n_workers={_n_workers}, workers={_worker_ids})"
        )
        try:
            await self._collective_rpc_async(
                "update_weights_from_tensor_serialized",
                serialized_payload=serialized_payload,
            )
            logger.info(
                f"[DiagMTP] async update_weights_from_tensor COMPLETED "
                f"in {_time.time() - _t0:.3f}s"
            )
        except Exception as e:
            logger.error(
                f"[DiagMTP] async update_weights_from_tensor FAILED "
                f"after {_time.time() - _t0:.3f}s: {type(e).__name__}: {e}",
                exc_info=True,
            )
            raise

    async def read_weights_by_name(
        self,
        names,
        truncate_size: int = 8,
    ) -> list:
        """[v28] Delegate SGLang HTTP read-by-name.

        Uses a lightweight worker RPC to fetch RemoteInfEngine
        addresses, then calls SGLang's /get_weights_by_parameter_name
        directly over HTTP from the controller process.
        """
        import requests as _v28_requests
        entries: list = []
        try:
            addr_list = await self._collective_rpc_async(
                "get_addresses", http_timeout=60.0,
            )
        except Exception as _e_addr:
            logger.warning(
                "[DiagMTP] read_weights_by_name: addr RPC failed: %s",
                _e_addr,
            )
            addr_list = []
        flat_addrs: list = []
        for a in addr_list or []:
            if isinstance(a, (list, tuple)):
                flat_addrs.extend(a)
            elif a:
                flat_addrs.append(a)
        if not flat_addrs:
            return entries
        addr0 = flat_addrs[0]
        base = (
            addr0 if str(addr0).startswith("http")
            else f"http://{addr0}"
        )
        for nm in names:
            try:
                resp = _v28_requests.post(
                    f"{base}/get_weights_by_parameter_name",
                    json={
                        "name": nm,
                        "truncate_size": truncate_size,
                    },
                    timeout=15,
                    proxies={"http": None, "https": None},
                )
                body = resp.text[:400]
                first8 = None
                dtype = None
                try:
                    _j = resp.json()
                    if isinstance(_j, list):
                        first8 = _j[:8]
                    elif isinstance(_j, dict):
                        first8 = (
                            _j.get("values")
                            or _j.get("first8")
                        )
                        dtype = _j.get("dtype")
                except Exception:
                    pass
                entries.append({
                    "name": nm,
                    "status": resp.status_code,
                    "first8": first8,
                    "dtype": dtype,
                    "body": body,
                })
            except Exception as _e_http:
                entries.append({
                    "name": nm,
                    "status": -1,
                    "first8": None,
                    "dtype": None,
                    "body": f"err: {_e_http}",
                })
        return entries

    def set_version(self, version: int) -> None:
        with self._version_lock:
            self._version = version
            self._collective_rpc("set_version", version=version, http_timeout=60.0)
            if self._proxy_started:
                self._proxy_collective_rpc(
                    "set_version", version=version, http_timeout=60.0
                )

    def get_version(self) -> int:
        with self._version_lock:
            return self._version

    def pause(self):
        self.dispatcher.pause()
        self._collective_rpc("pause", http_timeout=60.0)

    def resume(self):
        self._collective_rpc("resume", http_timeout=60.0)
        self.dispatcher.resume()

    def export_stats(self) -> dict[str, float]:
        all_raw_stats = self._collective_rpc(method="export_stats", http_timeout=60.0)
        stats = defaultdict(float)
        counts = defaultdict(int)

        for raw_stats in all_raw_stats:
            for k, v in raw_stats.items():
                if k.endswith("__count"):
                    counts[k] += v
                else:
                    stats[k] += v * raw_stats.get(k + "__count", 0)

        # Average non-count stats
        final_stats = {}
        for k, v in stats.items():
            count_key = k + "__count"
            if count_key in counts and counts[count_key] > 0:
                final_stats[k] = v / counts[count_key]
        return final_stats

    def config_perf_tracer(self, config: PerfTracerConfig, role: str) -> None:
        async def _call():
            tasks = [
                self.scheduler.async_call_engine(
                    worker_id=worker.id,
                    method="config_perf_tracer",
                    engine_name=self._engine_name(rank),
                    rank=rank,
                    role=role,
                    config=config,
                )
                for rank, worker in enumerate(self.workers)
            ]
            return await asyncio.gather(*tasks)

        run_async_task(_call)

    def save_perf_tracer(self, step: int | None = None, force: bool = False) -> None:
        self._collective_rpc("save_perf_tracer", step=step, force=force)

    @property
    def staleness_manager(self):
        return self._staleness_manager

    @property
    def dispatcher(
        self,
    ) -> BatchTaskDispatcher[_RemoteRolloutTaskInput, _RemoteRolloutResult]:
        """Get the task dispatcher, ensuring initialization has been called."""
        if self._dispatcher is None:
            raise RuntimeError(
                "RolloutController.initialize() must be called before scheduling rollouts."
            )
        return self._dispatcher

    @property
    def runner(self):
        """For backward compatibility. The runner is now owned by the dispatcher."""
        return self.dispatcher.runner
