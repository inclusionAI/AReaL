"""ZMQ DEALER Worker for the Agent Service.

Receives tasks from the Router via a ZMQ DEALER socket, processes them
by calling :meth:`AgentService.run_episode`, and pushes results back via
a ZMQ PUSH socket.

A minimal HTTP server (FastAPI/uvicorn) is kept for Scheduler readiness
probes (``GET /health``) and runtime configuration (``POST /configure``).
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import threading
import time
import uuid
from typing import Any

import zmq

# FastAPI Request must be importable from module globals because
# `from __future__ import annotations` stringifies type hints.
# FastAPI resolves endpoint parameter annotations via get_type_hints(),
# which only searches module-level globals.
try:
    from fastapi import Request as _FastAPIRequest  # noqa: F401
except ImportError:
    _FastAPIRequest = None  # FastAPI not installed

from areal.utils import logging, seeding

from .config import AgentServiceConfig
from .schemas import HealthResponse
from .service import AgentService

logger = logging.getLogger("AgentWorker")

# Environment variable names for worker configuration
ENV_AGENT_IMPORT_PATH_INTERNAL = "AREAL_AGENT_IMPORT_PATH_INTERNAL"
ENV_AGENT_REUSE_INTERNAL = "AREAL_AGENT_REUSE_INTERNAL"
ENV_AGENT_INIT_KWARGS_INTERNAL = "AREAL_AGENT_INIT_KWARGS_INTERNAL"

# ZMQ address env vars (set by AgentController)
ENV_ZMQ_TASK_ADDR = "AREAL_ZMQ_TASK_ADDR"
ENV_ZMQ_RESULT_ADDR = "AREAL_ZMQ_RESULT_ADDR"
ENV_AGENT_WORKER_ID = "AREAL_AGENT_WORKER_ID"


def _get_agent_config_from_env() -> tuple[str | None, bool, dict[str, Any]]:
    """Get agent configuration from internal environment variables.

    Returns
    -------
    tuple[str | None, bool, dict[str, Any]]
        (agent_import_path, agent_reuse, agent_init_kwargs)
    """
    agent_import_path = os.environ.get(ENV_AGENT_IMPORT_PATH_INTERNAL) or None
    agent_reuse = os.environ.get(ENV_AGENT_REUSE_INTERNAL, "false").lower() in (
        "true",
        "1",
    )
    agent_init_kwargs: dict[str, Any] = {}
    kwargs_str = os.environ.get(ENV_AGENT_INIT_KWARGS_INTERNAL, "")
    if kwargs_str:
        try:
            agent_init_kwargs = json.loads(kwargs_str)
        except json.JSONDecodeError as e:
            logger.warning(
                "Invalid JSON for agent_init_kwargs, using empty dict: %s", e
            )
    return agent_import_path, agent_reuse, agent_init_kwargs


# ---------------------------------------------------------------------------
# WorkerZMQ
# ---------------------------------------------------------------------------


class WorkerZMQ:
    """ZMQ DEALER consumer that processes tasks from the Router.

    Parameters
    ----------
    task_addr : str
        Router's req_backend address (Worker's DEALER connects here).
    result_addr : str
        Router's res_frontend address (Worker's PUSH connects here).
    service : AgentService
        The agent service used to process tasks.
    worker_id : str | None
        Unique worker identity. Auto-generated if not provided.
    """

    def __init__(
        self,
        task_addr: str,
        result_addr: str,
        service: AgentService,
        worker_id: str | None = None,
        task_timeout: float = 300.0,
    ) -> None:
        self._task_addr = task_addr
        self._result_addr = result_addr
        self._service = service
        self._worker_id = worker_id or f"worker-{uuid.uuid4().hex[:8]}"
        self._task_timeout = task_timeout
        self._running = threading.Event()
        self._ctx: zmq.Context | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def start(self) -> None:
        """Start the ZMQ event loop (blocking)."""
        self._running.set()
        self._ctx = zmq.Context()
        self._loop = asyncio.new_event_loop()
        try:
            self._run()
        finally:
            self._loop.close()
            self._loop = None

    def stop(self) -> None:
        """Signal the ZMQ loop to stop."""
        self._running.clear()

    def _run(self) -> None:
        assert self._ctx is not None
        assert self._loop is not None

        # DEALER socket — unique identity for Router to address us
        dealer = self._ctx.socket(zmq.DEALER)
        dealer.setsockopt(zmq.IDENTITY, self._worker_id.encode())
        dealer.setsockopt(zmq.LINGER, 0)
        dealer.setsockopt(zmq.RCVHWM, 1)  # fair load balancing: pull one at a time
        dealer.connect(self._task_addr)

        # PUSH socket for results
        push = self._ctx.socket(zmq.PUSH)
        push.setsockopt(zmq.LINGER, 2000)  # keep in-flight results on shutdown
        push.connect(self._result_addr)

        # Send READY message to register with Router
        # DEALER sends [empty_frame, payload] — empty frame is the delimiter
        ready_msg = {"type": "READY", "worker_id": self._worker_id}
        dealer.send_multipart([b"", json.dumps(ready_msg).encode()])
        logger.info("Worker %s sent READY to Router", self._worker_id)

        poller = zmq.Poller()
        poller.register(dealer, zmq.POLLIN)

        try:
            while self._running.is_set():
                socks = dict(poller.poll(timeout=1000))
                if dealer in socks:
                    frames = dealer.recv_multipart()
                    # DEALER receives: [empty_frame, payload]
                    if len(frames) >= 2:
                        payload = frames[1]
                        self._process_task(payload, push)
        finally:
            dealer.close()
            push.close()
            if self._ctx:
                self._ctx.term()
                self._ctx = None
            logger.info("Worker %s stopped", self._worker_id)

    def _process_task(self, payload: bytes, push_sock: zmq.Socket) -> None:
        """Process one task and send result via PUSH."""
        assert self._loop is not None
        t_total = time.monotonic()
        try:
            task = json.loads(payload)
        except json.JSONDecodeError:
            logger.warning("Worker: malformed task payload, skipping")
            return

        task_id = task.get("task_id", "<unknown>")
        t_agent = time.monotonic()
        try:
            result = self._loop.run_until_complete(
                asyncio.wait_for(
                    self._service.run_episode(
                        data=task.get("data", {}),
                        session_url=task.get("session_url", ""),
                        agent_kwargs=task.get("agent_kwargs"),
                        agent_import_path=task.get("agent_import_path"),
                    ),
                    timeout=self._task_timeout,
                )
            )
            agent_elapsed = time.monotonic() - t_agent
            total_elapsed = time.monotonic() - t_total
            logger.info(
                "task=%s agent_run=%.4fs total=%.4fs status=success",
                task_id,
                agent_elapsed,
                total_elapsed,
            )
            push_sock.send_json(
                {"task_id": task_id, "status": "success", "result": result}
            )
        except TimeoutError:
            total_elapsed = time.monotonic() - t_total
            logger.warning(
                "task=%s total=%.4fs status=timeout (exceeded %.1fs)",
                task_id,
                total_elapsed,
                self._task_timeout,
            )
            push_sock.send_json(
                {
                    "task_id": task_id,
                    "status": "error",
                    "error": f"Task timed out after {self._task_timeout}s",
                }
            )
            return
        except Exception as e:
            total_elapsed = time.monotonic() - t_total
            logger.warning(
                "task=%s total=%.4fs status=error err=%s",
                task_id,
                total_elapsed,
                e,
            )
            push_sock.send_json(
                {"task_id": task_id, "status": "error", "error": str(e)}
            )


# ---------------------------------------------------------------------------
# Minimal HTTP server (Scheduler readiness)
# ---------------------------------------------------------------------------


def _run_http_server(service: AgentService, http_port: int) -> None:
    """Run a minimal FastAPI server for health and configuration endpoints."""
    import uvicorn
    from fastapi import FastAPI

    from areal.infra.rpc.serialization import deserialize_value

    app = FastAPI(title="Agent Worker")

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        health_info = await service.health_check()
        return HealthResponse(**health_info)

    @app.post("/configure")
    async def configure(raw_request: _FastAPIRequest):
        data = await raw_request.json()
        config = deserialize_value(data.get("config"))
        rank = data.get("rank", 0)
        seeding.set_random_seed(config.seed, key=f"agent{rank}")
        return {"status": "success"}

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=http_port,
        log_level="warning",
        access_log=False,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for the Agent Worker process (internal, launched by Scheduler)."""
    import argparse

    from areal.utils.network import find_free_ports, gethostip

    parser = argparse.ArgumentParser(description="Agent Service Worker")
    parser.add_argument("--port", type=int, default=0)
    args, _ = parser.parse_known_args()

    port = args.port if args.port != 0 else find_free_ports(1)[0]
    host = gethostip()

    # Read agent config from env vars
    agent_import_path, agent_reuse, agent_init_kwargs = _get_agent_config_from_env()

    # Read ZMQ addresses from env vars (set by AgentController)
    task_addr = os.environ.get(ENV_ZMQ_TASK_ADDR, "tcp://localhost:5556")
    result_addr = os.environ.get(ENV_ZMQ_RESULT_ADDR, "tcp://localhost:5557")
    worker_id = os.environ.get(ENV_AGENT_WORKER_ID, f"worker-{host}-{port}")

    # Initialize AgentService
    config = AgentServiceConfig()
    service = AgentService(
        agent_import_path=agent_import_path,
        config=config,
        agent_reuse=agent_reuse,
        agent_init_kwargs=agent_init_kwargs,
    )

    loop = asyncio.new_event_loop()
    loop.run_until_complete(service.start())

    # Start HTTP health server in background thread
    http_thread = threading.Thread(
        target=_run_http_server,
        args=(service, port),
        daemon=True,
    )
    http_thread.start()

    # Create and start ZMQ worker (blocking)
    worker = WorkerZMQ(
        task_addr=task_addr,
        result_addr=result_addr,
        service=service,
        worker_id=worker_id,
        task_timeout=config.task_timeout,
    )

    def _shutdown(signum, frame):
        logger.info("Worker received signal %d, shutting down", signum)
        # NOTE: threading.Event.clear() is thread-safe; the ZMQ poller will exit
        # on its next 1s timeout. There is a theoretical race where a signal
        # arrives while the ZMQ context is being torn down, but in practice
        # the poller timeout bounds the window and no additional synchronization
        # is needed.
        worker.stop()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    logger.info(
        "Starting Worker %s (http=%d, task=%s, result=%s)",
        worker_id,
        port,
        task_addr,
        result_addr,
    )
    worker.start()  # blocks until stopped

    loop.run_until_complete(service.stop())
    loop.close()


if __name__ == "__main__":
    main()
