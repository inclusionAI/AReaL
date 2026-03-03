"""ZMQ Router process for the Agent Service.

The Router sits between the Gateway and Agent Workers, implementing a
ROUTER/DEALER pattern for request distribution and result forwarding.

Architecture
------------
- PULL socket (req_frontend) ← Gateway PUSHes requests
- ROUTER socket (req_backend) → Workers connect via DEALER
- PULL socket (res_frontend) ← Workers PUSH results
- PUSH socket (res_backend) → Gateway PULLs results

Workers send a ``{"type": "READY"}`` message on startup via their DEALER
socket. The Router tracks worker identities and uses a pluggable
:class:`RoutingStrategy` to decide which worker receives each task.

Endpoints
---------
- GET  /health    — Health check.
- POST /configure — Accept Scheduler configuration (no-op).
"""

from __future__ import annotations

import json
import threading
from abc import ABC, abstractmethod

import zmq
from fastapi import FastAPI

from areal.utils import logging

logger = logging.getLogger("Router")

# ---------------------------------------------------------------------------
# Routing strategies
# ---------------------------------------------------------------------------


class RoutingStrategy(ABC):
    """Abstract base class for worker selection strategies."""

    @abstractmethod
    def select_worker(self, task: dict, workers: list[str]) -> str:
        """Return the worker identity string to route the task to.

        Parameters
        ----------
        task : dict
            The incoming task payload.
        workers : list[str]
            List of registered worker identity strings.

        Returns
        -------
        str
            The chosen worker identity.
        """


class RoundRobinStrategy(RoutingStrategy):
    """Cycle through available workers in order."""

    def __init__(self) -> None:
        self._idx = 0

    def select_worker(self, task: dict, workers: list[str]) -> str:
        if not workers:
            raise RuntimeError("No workers registered")
        worker = workers[self._idx % len(workers)]
        self._idx += 1
        return worker


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


class Router:
    """ZMQ Router that distributes tasks to workers and forwards results.

    Parameters
    ----------
    req_frontend_addr : str
        ZMQ PULL bind address for incoming requests from Gateway.
    req_backend_addr : str
        ZMQ ROUTER bind address for worker DEALER connections.
    res_frontend_addr : str
        ZMQ PULL bind address for incoming results from workers.
    res_backend_addr : str
        ZMQ PUSH bind address for forwarding results to Gateway.
    strategy : RoutingStrategy
        Worker selection strategy (default: :class:`RoundRobinStrategy`).
    http_port : int
        Port for the minimal HTTP health/configure server.
    """

    def __init__(
        self,
        req_frontend_addr: str = "tcp://*:5555",
        req_backend_addr: str = "tcp://*:5556",
        res_frontend_addr: str = "tcp://*:5557",
        res_backend_addr: str = "tcp://*:5558",
        strategy: RoutingStrategy | None = None,
        http_port: int = 8301,
    ) -> None:
        self._req_frontend_addr = req_frontend_addr
        self._req_backend_addr = req_backend_addr
        self._res_frontend_addr = res_frontend_addr
        self._res_backend_addr = res_backend_addr
        self._strategy = strategy or RoundRobinStrategy()
        self._http_port = http_port

        self._running = False
        self._known_workers: list[str] = []
        self._lock = threading.Lock()
        self._pending_queue: list[dict] = []
        self._pending_queue_maxsize: int = 1000

        self._ctx: zmq.Context | None = None
        self._routing_thread: threading.Thread | None = None
        self._result_thread: threading.Thread | None = None
        self._http_thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the ZMQ sockets, routing/result threads, and HTTP server."""
        if self._running:
            logger.warning("Router already running")
            return

        self._running = True
        self._ctx = zmq.Context()

        self._routing_thread = threading.Thread(
            target=self._routing_loop, daemon=True, name="router-routing"
        )
        self._result_thread = threading.Thread(
            target=self._result_loop, daemon=True, name="router-result"
        )
        self._http_thread = threading.Thread(
            target=self._run_http, daemon=True, name="router-http"
        )

        self._routing_thread.start()
        self._result_thread.start()
        self._http_thread.start()

        logger.info(
            "Router started (req_fe=%s, req_be=%s, res_fe=%s, res_be=%s, http=%d)",
            self._req_frontend_addr,
            self._req_backend_addr,
            self._res_frontend_addr,
            self._res_backend_addr,
            self._http_port,
        )

    def stop(self) -> None:
        """Signal threads to stop and clean up ZMQ resources."""
        if not self._running:
            return
        self._running = False

        if self._routing_thread is not None:
            self._routing_thread.join(timeout=5)
        if self._result_thread is not None:
            self._result_thread.join(timeout=5)
        # HTTP thread is daemon; it will exit when the process exits.

        if self._ctx is not None:
            self._ctx.term()
            self._ctx = None

        logger.info("Router stopped")

    # ------------------------------------------------------------------
    # Routing loop (background thread)
    # ------------------------------------------------------------------

    def _routing_loop(self) -> None:
        """Poll req PULL + ROUTER sockets and route tasks to workers."""
        assert self._ctx is not None

        # PULL socket: receives requests from Gateway
        req_pull = self._ctx.socket(zmq.PULL)
        req_pull.setsockopt(zmq.LINGER, 0)
        req_pull.setsockopt(zmq.RCVHWM, 10000)
        req_pull.bind(self._req_frontend_addr)

        # ROUTER socket: workers connect via DEALER
        router_sock = self._ctx.socket(zmq.ROUTER)
        router_sock.setsockopt(zmq.LINGER, 0)
        router_sock.setsockopt(zmq.SNDHWM, 10000)
        router_sock.bind(self._req_backend_addr)

        poller = zmq.Poller()
        poller.register(req_pull, zmq.POLLIN)
        poller.register(router_sock, zmq.POLLIN)

        logger.info("Routing loop started")
        try:
            while self._running:
                socks = dict(poller.poll(timeout=1000))

                # --- handle worker messages (READY, etc.) ---
                if router_sock in socks:
                    frames = router_sock.recv_multipart()
                    # frames: [identity, empty, payload]
                    if len(frames) >= 3:
                        identity = frames[0]
                        payload = frames[2]
                        self._handle_worker_message(identity, payload)

                        # Drain pending queue now that a worker is registered
                        with self._lock:
                            workers_now = list(self._known_workers)
                        while self._pending_queue and workers_now:
                            queued_msg = self._pending_queue.pop(0)
                            try:
                                worker_id = self._strategy.select_worker(
                                    queued_msg, workers_now
                                )
                                router_sock.send_multipart(
                                    [
                                        worker_id.encode(),
                                        b"",
                                        json.dumps(queued_msg).encode(),
                                    ]
                                )
                                logger.info(
                                    "Router: dispatched queued task %s \u2192 worker %s",
                                    queued_msg.get("task_id", "<unknown>"),
                                    worker_id,
                                )
                            except (zmq.ZMQError, RuntimeError) as e:
                                logger.error(
                                    "Router: failed to dispatch queued task: %s", e
                                )

                # --- handle incoming requests from Gateway ---
                if req_pull in socks:
                    raw = req_pull.recv()
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        logger.warning(
                            "Router: dropping malformed request (not valid JSON)"
                        )
                        continue

                    with self._lock:
                        workers = list(self._known_workers)

                    if not workers:
                        if len(self._pending_queue) < self._pending_queue_maxsize:
                            self._pending_queue.append(msg)
                            logger.warning(
                                "Router: no workers registered, queued task %s (%d pending)",
                                msg.get("task_id", "<unknown>"),
                                len(self._pending_queue),
                            )
                        else:
                            logger.error(
                                "Router: pending queue full (%d), dropping task %s",
                                self._pending_queue_maxsize,
                                msg.get("task_id", "<unknown>"),
                            )
                        continue

                    try:
                        worker_id = self._strategy.select_worker(msg, workers)
                    except RuntimeError:
                        logger.warning(
                            "Router: strategy raised RuntimeError, dropping task %s",
                            msg.get("task_id", "<unknown>"),
                        )
                        continue

                    try:
                        router_sock.send_multipart(
                            [worker_id.encode(), b"", json.dumps(msg).encode()]
                        )
                    except zmq.ZMQError as e:
                        logger.error(
                            "Router: failed to send task %s to worker %s: %s",
                            msg.get("task_id", "<unknown>"),
                            worker_id,
                            e,
                        )
                        # H3: Remove unresponsive worker
                        with self._lock:
                            if worker_id in self._known_workers:
                                self._known_workers.remove(worker_id)
                                logger.warning(
                                    "Router: removed unresponsive worker %s from routing table",
                                    worker_id,
                                )
                        continue
                    logger.debug(
                        "Router: routed task %s → worker %s",
                        msg.get("task_id", "<unknown>"),
                        worker_id,
                    )
        finally:
            req_pull.close()
            router_sock.close()
            logger.info("Routing loop stopped")

    def _handle_worker_message(self, identity: bytes, payload: bytes) -> None:
        """Process a message received from a worker on the ROUTER socket."""
        worker_id = identity.decode(errors="replace")
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            logger.warning(
                "Router: non-JSON message from worker %s, ignoring", worker_id
            )
            return

        msg_type = data.get("type", "")
        if msg_type == "READY":
            with self._lock:
                if worker_id not in self._known_workers:
                    self._known_workers.append(worker_id)
                    logger.info(
                        "Router: worker %s registered (%d total)",
                        worker_id,
                        len(self._known_workers),
                    )
        else:
            logger.debug(
                "Router: unknown message type '%s' from worker %s", msg_type, worker_id
            )

    # ------------------------------------------------------------------
    # Result forwarding loop (background thread)
    # ------------------------------------------------------------------

    def _result_loop(self) -> None:
        """Forward results from workers (PULL) to Gateway (PUSH)."""
        assert self._ctx is not None

        res_pull = self._ctx.socket(zmq.PULL)
        res_pull.setsockopt(zmq.LINGER, 2000)
        res_pull.bind(self._res_frontend_addr)

        res_push = self._ctx.socket(zmq.PUSH)
        res_push.setsockopt(zmq.LINGER, 2000)
        res_push.bind(self._res_backend_addr)

        poller = zmq.Poller()
        poller.register(res_pull, zmq.POLLIN)

        logger.info("Result loop started")
        try:
            while self._running:
                socks = dict(poller.poll(timeout=1000))
                if res_pull in socks:
                    raw = res_pull.recv()
                    res_push.send(raw)
        finally:
            res_pull.close()
            res_push.close()
            logger.info("Result loop stopped")

    # ------------------------------------------------------------------
    # HTTP health server (background thread)
    # ------------------------------------------------------------------

    def _run_http(self) -> None:
        """Run a minimal HTTP server for health checks."""
        import uvicorn

        app = _create_http_app(self)
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=self._http_port,
            log_level="warning",
            access_log=False,
        )


def _create_http_app(router: Router) -> FastAPI:
    """Build a minimal FastAPI application for health and configuration.

    Parameters
    ----------
    router : Router
        The Router instance (used for status introspection).

    Returns
    -------
    FastAPI
        The configured application.
    """
    app = FastAPI(
        title="Agent Service Router",
        description="ZMQ Router for Agent Service task distribution",
        version="1.0.0",
    )

    @app.get("/health")
    def health():
        """Health check."""
        with router._lock:
            worker_count = len(router._known_workers)
        return {
            "status": "ok",
            "workers_registered": worker_count,
        }

    @app.post("/configure")
    def configure():
        """Accept Scheduler configuration (no-op for Router)."""
        return {"status": "ok"}

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for the Router process (launched by Scheduler)."""
    import argparse
    import signal

    parser = argparse.ArgumentParser(description="Agent Service Router")
    parser.add_argument("--port", type=int, default=8301)
    parser.add_argument("--req-frontend-addr", type=str, default="tcp://*:5555")
    parser.add_argument("--req-backend-addr", type=str, default="tcp://*:5556")
    parser.add_argument("--res-frontend-addr", type=str, default="tcp://*:5557")
    parser.add_argument("--res-backend-addr", type=str, default="tcp://*:5558")
    args, _ = parser.parse_known_args()

    router = Router(
        req_frontend_addr=args.req_frontend_addr,
        req_backend_addr=args.req_backend_addr,
        res_frontend_addr=args.res_frontend_addr,
        res_backend_addr=args.res_backend_addr,
        strategy=RoundRobinStrategy(),
        http_port=args.port,
    )

    def _shutdown(signum, frame):
        logger.info("Router received signal %d, shutting down", signum)
        router.stop()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    router.start()
    try:
        signal.pause()
    except AttributeError:
        # signal.pause() not available on Windows; block with Event instead
        threading.Event().wait()


if __name__ == "__main__":
    main()
