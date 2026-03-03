"""Gateway process for the Agent Service — stateless HTTP↔ZMQ bridge.

The Gateway accepts HTTP requests and bridges them to Agent Workers
via ZMQ through a Router process. It never holds HTTP connections
waiting for a worker — ``/submit`` returns immediately with a
``task_id`` and callers poll ``/result/{task_id}`` for the outcome.

ZMQ Architecture
----------------
- PUSH socket: Gateway → Router's req_frontend (PULL)
- PULL socket: Gateway ← Router's res_backend (PUSH)

Endpoints
---------
- POST /submit           — Submit an episode, returns task_id immediately.
- GET  /result/{task_id} — Poll for a task result.
- GET  /health           — Health check with pending/completed counts.
- GET  /metrics          — Request metrics (submitted, completed, errors, avg latency).
- POST /configure        — Accept Scheduler configuration (no-op).
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import zmq
from fastapi import FastAPI, HTTPException

from areal.utils import logging

from .schemas import (
    RunEpisodeRequest,
    SubmitEpisodeResponse,
    TaskResultResponse,
)

logger = logging.getLogger("Gateway")


# ------------------------------------------------------------------
# Task entry (internal bookkeeping)
# ------------------------------------------------------------------


@dataclass
class _TaskEntry:
    """Internal tracking state for a submitted task."""

    status: str  # "pending", "completed", "error", "timeout"
    result: float | dict[str, float] | None = None
    error: str | None = None
    submitted_at: float = field(default_factory=time.time)
    _metrics_recorded: bool = False
    completed_at: float | None = None


# ------------------------------------------------------------------
# Background ZMQ result-receiver thread
# ------------------------------------------------------------------


def _result_receiver_thread(
    ctx: zmq.Context,
    addr: str,
    running: threading.Event,
    results: dict[str, _TaskEntry],
    results_lock: threading.Lock,
) -> None:
    """Continuously receive results from the Router and update *results*.

    Runs in a daemon thread.  Uses ``RCVTIMEO`` to periodically check
    the *running* flag so the thread shuts down cleanly.

    Parameters
    ----------
    ctx : zmq.Context
        Shared ZMQ context (thread-safe for socket creation).
    addr : str
        Router's ``res_backend`` address to connect to.
    running : threading.Event
        Cleared when the Gateway is shutting down.
    results : dict[str, _TaskEntry]
        Shared task-result mapping (protected by *results_lock*).
    results_lock : threading.Lock
        Lock guarding *results*.
    """
    sock = ctx.socket(zmq.PULL)
    sock.setsockopt(zmq.LINGER, 2000)
    sock.setsockopt(zmq.RCVTIMEO, 1000)  # 1s poll for shutdown check
    sock.connect(addr)
    logger.info("Result receiver connected to %s", addr)

    while running.is_set():
        try:
            data: dict[str, Any] = sock.recv_json()
        except zmq.Again:
            continue  # timeout, re-check running flag

        task_id = data.get("task_id")
        if not task_id:
            logger.warning("Result receiver: got result without task_id, dropping")
            continue

        with results_lock:
            entry = results.get(task_id)
            if entry is None:
                logger.warning(
                    "Result receiver: unknown task_id %s, dropping",
                    task_id,
                )
                continue
            if data.get("status") == "error":
                entry.status = "error"
                entry.error = data.get("error")
                entry.completed_at = time.time()
            else:
                entry.status = "completed"
                entry.result = data.get("result")
                entry.completed_at = time.time()

        logger.debug("Result receiver: updated task %s → %s", task_id, entry.status)

    sock.close()
    logger.info("Result receiver thread stopped")


# ------------------------------------------------------------------
# Periodic cleanup (asyncio task)
# ------------------------------------------------------------------


async def _cleanup_loop(
    results: dict[str, _TaskEntry],
    results_lock: threading.Lock,
    task_timeout: float,
    result_ttl: float,
) -> None:
    """Periodically timeout pending tasks and evict stale results.

    Parameters
    ----------
    results : dict[str, _TaskEntry]
        Shared task-result mapping.
    results_lock : threading.Lock
        Lock guarding *results*.
    task_timeout : float
        Seconds after which a pending task is marked ``"timeout"``.
    result_ttl : float
        Seconds after which a terminal result is evicted.
    """
    while True:
        await asyncio.sleep(30)
        now = time.time()
        with results_lock:
            to_delete: list[str] = []
            for task_id, entry in results.items():
                age = now - entry.submitted_at
                if entry.status == "pending" and age > task_timeout:
                    entry.status = "timeout"
                    entry.error = "Task timed out"
                    logger.warning(
                        "Cleanup: task %s timed out after %.1fs", task_id, age
                    )
                elif entry.status in ("completed", "error", "timeout"):
                    evict_ref = (
                        entry.completed_at
                        if entry.completed_at is not None
                        else entry.submitted_at
                    )
                    if now - evict_ref > result_ttl:
                        to_delete.append(task_id)
            for task_id in to_delete:
                del results[task_id]
            if to_delete:
                logger.debug("Cleanup: evicted %d stale results", len(to_delete))


# ------------------------------------------------------------------
# FastAPI application factory
# ------------------------------------------------------------------


def _resolve_zmq_addr(cli_value: str | None, env_var: str, default: str) -> str:
    """Return the ZMQ address from CLI arg, env var, or default (in that order)."""
    if cli_value:
        return cli_value
    return os.environ.get(env_var, default)


def create_app(
    *,
    req_frontend_addr: str = "tcp://127.0.0.1:5555",
    res_backend_addr: str = "tcp://127.0.0.1:5558",
    task_timeout: float = 300.0,
    result_ttl: float = 300.0,
) -> FastAPI:
    """Build and return a FastAPI application with ZMQ bridge lifecycle.

    Parameters
    ----------
    req_frontend_addr : str
        Router's PULL address — the Gateway PUSHes requests here.
    res_backend_addr : str
        Router's PUSH address — the Gateway PULLs results from here.
    task_timeout : float
        Seconds before a pending task is marked timed-out.
    result_ttl : float
        Seconds before a terminal result is evicted from memory.

    Returns
    -------
    FastAPI
        The configured application, ready for ``uvicorn.run``.
    """

    # Shared mutable state (populated in lifespan)
    results: dict[str, _TaskEntry] = {}
    results_lock = threading.Lock()
    push_lock = threading.Lock()

    # Placeholders assigned during lifespan
    push_socket_holder: list[zmq.Socket] = []

    # -- Metrics counters (simple in-process) --
    _counters: dict[str, Any] = {
        "submitted": 0,
        "completed": 0,
        "errors": 0,
        "latency_sum": 0.0,
    }

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # --- Setup ZMQ ---
        ctx = zmq.Context()

        try:
            push_sock = ctx.socket(zmq.PUSH)
            push_sock.setsockopt(zmq.LINGER, 0)
            push_sock.setsockopt(zmq.SNDHWM, 10000)
            push_sock.connect(req_frontend_addr)
            push_socket_holder.append(push_sock)
        except Exception:
            push_sock.close()
            raise

        running = threading.Event()
        running.set()

        recv_thread = threading.Thread(
            target=_result_receiver_thread,
            args=(ctx, res_backend_addr, running, results, results_lock),
            daemon=True,
            name="gateway-result-recv",
        )
        recv_thread.start()

        cleanup_task = asyncio.create_task(
            _cleanup_loop(results, results_lock, task_timeout, result_ttl)
        )

        logger.info(
            "Gateway lifespan started (push→%s, pull←%s, timeout=%.0fs, ttl=%.0fs)",
            req_frontend_addr,
            res_backend_addr,
            task_timeout,
            result_ttl,
        )
        yield

        # --- Teardown ---
        running.clear()
        recv_thread.join(timeout=5)
        if recv_thread.is_alive():
            logger.warning(
                "Result receiver thread did not exit cleanly; possible resource leak"
            )
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass
        push_sock.close()
        push_socket_holder.clear()
        ctx.term()
        logger.info("Gateway lifespan stopped")

    app = FastAPI(
        title="Agent Service Gateway",
        description="Stateless HTTP↔ZMQ bridge for Agent Service",
        version="2.0.0",
        lifespan=lifespan,
    )

    # -- Endpoints -------------------------------------------------------

    @app.post("/submit", response_model=SubmitEpisodeResponse)
    async def submit_episode(request: RunEpisodeRequest) -> SubmitEpisodeResponse:
        """Submit an episode for execution. Returns immediately with a task_id."""
        if not push_socket_holder:
            raise HTTPException(status_code=503, detail="Gateway not initialized")

        task_id = str(uuid.uuid4())
        payload = {**request.model_dump(), "task_id": task_id}

        with push_lock:
            push_socket_holder[0].send_json(payload)

        with results_lock:
            results[task_id] = _TaskEntry(status="pending")

        _counters["submitted"] += 1
        logger.info("Submitted task %s", task_id)
        return SubmitEpisodeResponse(task_id=task_id)

    @app.get("/result/{task_id}", response_model=TaskResultResponse)
    async def get_result(task_id: str) -> TaskResultResponse:
        """Poll for the result of a previously submitted task."""
        with results_lock:
            entry = results.get(task_id)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

        # Update metrics on first observation of terminal state
        if entry.status == "completed" and not entry._metrics_recorded:
            entry._metrics_recorded = True
            _counters["completed"] += 1
            _counters["latency_sum"] += time.time() - entry.submitted_at
        elif entry.status in ("error", "timeout") and not entry._metrics_recorded:
            entry._metrics_recorded = True
            _counters["errors"] += 1

        return TaskResultResponse(
            task_id=task_id,
            status=entry.status,
            result=entry.result,
            error=entry.error,
        )

    @app.get("/health")
    async def health() -> dict[str, Any]:
        """Health check with pending/completed counts."""
        with results_lock:
            pending = sum(1 for e in results.values() if e.status == "pending")
            completed = sum(1 for e in results.values() if e.status == "completed")
        return {
            "status": "ok",
            "pending": pending,
            "completed": completed,
        }

    @app.get("/metrics")
    async def metrics() -> dict[str, Any]:
        """Request metrics: total submitted, completed, errors, avg latency."""
        completed = _counters["completed"]
        avg_latency = (
            round(_counters["latency_sum"] / completed, 4) if completed > 0 else 0.0
        )
        return {
            "total_submitted": _counters["submitted"],
            "total_completed": completed,
            "total_errors": _counters["errors"],
            "avg_latency_s": avg_latency,
        }

    @app.post("/configure")
    async def configure() -> dict[str, str]:
        """Accept Scheduler configuration (no-op for Gateway)."""
        return {"status": "ok"}

    return app


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------


def main() -> None:
    """Entry point for the Gateway process (launched by Scheduler).

    Reads ZMQ addresses from CLI args / environment variables and starts
    the uvicorn server.  Uses ``parse_known_args()`` to tolerate extra
    arguments injected by the Scheduler (e.g. ``--experiment-name``).
    """
    import argparse

    import uvicorn

    from areal.utils.network import find_free_ports, gethostip

    parser = argparse.ArgumentParser(description="Agent Service Gateway")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--zmq-req-frontend-addr", type=str, default=None)
    parser.add_argument("--zmq-res-backend-addr", type=str, default=None)
    parser.add_argument("--task-timeout", type=float, default=300.0)
    parser.add_argument("--result-ttl", type=float, default=300.0)
    args, _ = parser.parse_known_args()

    host = gethostip() if args.host == "0.0.0.0" else args.host
    port = args.port if args.port != 0 else find_free_ports(1)[0]

    req_frontend = _resolve_zmq_addr(
        args.zmq_req_frontend_addr,
        "AREAL_ZMQ_REQ_FRONTEND_ADDR",
        "tcp://127.0.0.1:5555",
    )
    res_backend = _resolve_zmq_addr(
        args.zmq_res_backend_addr,
        "AREAL_ZMQ_RES_BACKEND_ADDR",
        "tcp://127.0.0.1:5558",
    )

    app = create_app(
        req_frontend_addr=req_frontend,
        res_backend_addr=res_backend,
        task_timeout=args.task_timeout,
        result_ttl=args.result_ttl,
    )

    logger.info(
        "Starting Gateway on %s:%d (push→%s, pull←%s)",
        host,
        port,
        req_frontend,
        res_backend,
    )
    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            log_level="warning",
            access_log=False,
        )
    except KeyboardInterrupt:
        logger.info("Shutting down Gateway")


if __name__ == "__main__":
    main()
