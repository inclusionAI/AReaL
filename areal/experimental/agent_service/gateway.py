"""Gateway process for the Agent Service.

The Gateway acts as a transparent reverse proxy, forwarding
``/run_episode`` requests to Agent Worker processes using round-robin
load balancing.  There is no internal queue or dispatcher — each
request is forwarded directly to a healthy worker via HTTP.

On worker failure, the Gateway retries the request on the next
healthy worker (up to ``max_retries`` attempts).

Endpoints
---------
- POST /run_episode       — Forward an episode request to a worker.
- GET  /health            — Health check with worker pool statistics.
- POST /register_worker   — Register a new Agent Worker.
- POST /unregister_worker — Remove an Agent Worker from the pool.
- GET  /workers           — List all registered workers.
- GET  /metrics           — Request metrics (count, latency, per-worker stats).
"""

from __future__ import annotations

import asyncio
import math
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import Any

import aiohttp
from fastapi import FastAPI, HTTPException

from areal.utils import logging

from .config import GatewayConfig
from .schemas import (
    RegisterWorkerRequest,
    RunEpisodeRequest,
    RunEpisodeResponse,
)
from .worker_pool import WorkerPoolManager

logger = logging.getLogger("Gateway")


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------


class _Metrics:
    """Simple in-process request metrics."""

    def __init__(self) -> None:
        self.total_requests: int = 0
        self.total_success: int = 0
        self.total_errors: int = 0
        self.total_retries: int = 0
        self._latencies: deque[float] = deque(maxlen=10_000)
        self._per_worker: dict[str, dict[str, int]] = {}

    def record(
        self,
        worker_id: str,
        latency: float,
        *,
        success: bool,
    ) -> None:
        self.total_requests += 1
        self._latencies.append(latency)
        if success:
            self.total_success += 1
        else:
            self.total_errors += 1

        if worker_id not in self._per_worker:
            self._per_worker[worker_id] = {"requests": 0, "errors": 0}
        self._per_worker[worker_id]["requests"] += 1
        if not success:
            self._per_worker[worker_id]["errors"] += 1

    def record_retry(self) -> None:
        self.total_retries += 1

    def snapshot(self) -> dict[str, Any]:
        latencies = list(self._latencies)
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        if latencies:
            sorted_lat = sorted(latencies)
            idx = max(0, math.ceil(len(sorted_lat) * 0.99) - 1)
            p99_latency = sorted_lat[idx]
        else:
            p99_latency = 0.0
        return {
            "total_requests": self.total_requests,
            "total_success": self.total_success,
            "total_errors": self.total_errors,
            "total_retries": self.total_retries,
            "avg_latency_s": round(avg_latency, 4),
            "p99_latency_s": round(p99_latency, 4),
            "per_worker": dict(self._per_worker),
        }


# ------------------------------------------------------------------
# Gateway
# ------------------------------------------------------------------


class Gateway:
    """Transparent reverse proxy that forwards requests to Agent Workers.

    The Gateway owns a :class:`WorkerPoolManager` for round-robin worker
    selection and an :class:`aiohttp.ClientSession` for forwarding requests.
    There is no internal queue — each request is forwarded directly.

    Parameters
    ----------
    config : GatewayConfig
        Gateway configuration (host, port, timeouts, retries, etc.).
    """

    def __init__(
        self,
        config: GatewayConfig,
    ) -> None:
        self._config = config

        self._pool: WorkerPoolManager | None = None
        self._session: aiohttp.ClientSession | None = None
        self._metrics = _Metrics()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Initialize WorkerPoolManager and HTTP session."""
        self._pool = WorkerPoolManager(
            worker_timeout=self._config.worker_timeout,
            health_check_interval=self._config.health_check_interval,
        )
        await self._pool.start()
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self._config.worker_timeout),
        )
        logger.info(
            "Gateway started on %s:%d (max_retries=%d)",
            self._config.host,
            self._config.port,
            self._config.max_retries,
        )

    async def stop(self) -> None:
        """Stop the WorkerPool health check and close the HTTP session."""
        if self._pool is not None:
            await self._pool.stop()
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None
        logger.info("Gateway stopped")

    # ------------------------------------------------------------------
    # Proxy logic
    # ------------------------------------------------------------------

    async def _forward_request(
        self,
        request_data: dict[str, Any],
    ) -> RunEpisodeResponse:
        """Forward a request to a worker with round-robin + retry.

        Tries up to ``max_retries`` workers.  On each failure, the
        worker is marked unhealthy and the next healthy worker is tried.

        Args:
            request_data: Serialized request body to forward.

        Returns:
            RunEpisodeResponse from the worker.

        Raises:
            HTTPException: If all retries are exhausted.
        """
        assert self._pool is not None
        assert self._session is not None

        last_error: str | None = None

        for attempt in range(self._config.max_retries):
            worker = await self._pool.next_worker()
            if worker is None:
                last_error = "No healthy workers available"
                logger.warning(
                    "forward_request: attempt %d/%d — no healthy workers",
                    attempt + 1,
                    self._config.max_retries,
                )
                break

            t0 = time.monotonic()
            try:
                url = f"{worker.address}/run_episode"
                async with self._session.post(url, json=request_data) as resp:
                    elapsed = time.monotonic() - t0

                    if resp.status != 200:
                        body = await resp.text()
                        self._metrics.record(worker.worker_id, elapsed, success=False)
                        await self._pool.mark_unhealthy(worker.worker_id)
                        last_error = f"Worker {worker.worker_id} returned HTTP {resp.status}: {body}"
                        logger.warning(
                            "forward_request: attempt %d/%d — worker %s returned HTTP %d after %.2fs",
                            attempt + 1,
                            self._config.max_retries,
                            worker.worker_id,
                            resp.status,
                            elapsed,
                        )
                        if attempt < self._config.max_retries - 1:
                            self._metrics.record_retry()
                            continue
                        break

                    data: dict[str, Any] = await resp.json()
                    parsed = RunEpisodeResponse(**data)
                    self._metrics.record(worker.worker_id, elapsed, success=True)
                    logger.info(
                        "forward_request: worker %s completed in %.2fs",
                        worker.worker_id,
                        elapsed,
                    )
                    return parsed
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                elapsed = time.monotonic() - t0
                self._metrics.record(worker.worker_id, elapsed, success=False)
                await self._pool.mark_unhealthy(worker.worker_id)
                last_error = f"Worker {worker.worker_id} failed: {exc}"
                logger.warning(
                    "forward_request: attempt %d/%d — worker %s failed after %.2fs: %s",
                    attempt + 1,
                    self._config.max_retries,
                    worker.worker_id,
                    elapsed,
                    exc,
                )
                if attempt < self._config.max_retries - 1:
                    self._metrics.record_retry()

        # All retries exhausted
        raise HTTPException(
            status_code=503,
            detail=f"All workers failed after {self._config.max_retries} "
            f"attempts. Last error: {last_error}",
        )

    # ------------------------------------------------------------------
    # FastAPI application factory
    # ------------------------------------------------------------------

    def create_app(self) -> FastAPI:
        """Build and return a FastAPI application with lifespan management.

        Returns
        -------
        FastAPI
            The configured application, ready for ``uvicorn.run``.
        """
        gateway = self  # capture for lifespan closure

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            await gateway.start()
            yield
            await gateway.stop()

        app = FastAPI(
            title="Agent Service Gateway",
            description=(
                "Reverse proxy that forwards run_episode requests to Agent Workers"
            ),
            version="2.0.0",
            lifespan=lifespan,
        )

        # -- endpoints ------------------------------------------------

        @app.post("/run_episode", response_model=RunEpisodeResponse)
        async def run_episode(
            request: RunEpisodeRequest,
        ) -> RunEpisodeResponse:
            """Forward an episode request to a worker."""
            if gateway._pool is None:
                raise HTTPException(status_code=503, detail="Gateway not initialized")
            logger.info("run_episode: received request")
            return await gateway._forward_request(request.model_dump())

        @app.get("/health")
        async def health():
            """Health check with worker pool statistics."""
            if gateway._pool is None:
                raise HTTPException(status_code=503, detail="Gateway not initialized")
            stats = await gateway._pool.get_stats()
            return {"status": "ok", "workers": stats}

        @app.post("/register_worker")
        async def register_worker(req: RegisterWorkerRequest):
            """Register a new Agent Worker with the pool."""
            if gateway._pool is None:
                raise HTTPException(status_code=503, detail="Gateway not initialized")
            await gateway._pool.register_worker(req.worker_id, req.address)
            return {"status": "ok"}

        @app.post("/unregister_worker")
        async def unregister_worker(req: RegisterWorkerRequest):
            """Remove an Agent Worker from the pool."""
            if gateway._pool is None:
                raise HTTPException(status_code=503, detail="Gateway not initialized")
            await gateway._pool.unregister_worker(req.worker_id)
            return {"status": "ok"}

        @app.get("/workers")
        async def list_workers():
            """List all registered workers."""
            if gateway._pool is None:
                raise HTTPException(status_code=503, detail="Gateway not initialized")
            workers = await gateway._pool.get_all_workers()
            return [w.model_dump() for w in workers]

        @app.post("/configure")
        async def configure():
            """Accept Scheduler configuration (no-op for Gateway)."""
            return {"status": "success"}

        @app.get("/metrics")
        async def metrics():
            """Return request metrics."""
            return gateway._metrics.snapshot()

        return app


def main() -> None:
    """Entry point for the Gateway process (launched by Scheduler)."""
    import argparse

    import uvicorn

    from areal.utils.network import find_free_ports, gethostip

    parser = argparse.ArgumentParser(description="Agent Service Gateway")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=0)
    args, _ = parser.parse_known_args()

    host = gethostip() if args.host == "0.0.0.0" else args.host
    port = args.port if args.port != 0 else find_free_ports(1)[0]

    config = GatewayConfig(
        host=host,
        port=port,
    )
    gateway = Gateway(config=config)

    logger.info("Starting Gateway on %s:%d", host, port)
    try:
        uvicorn.run(
            gateway.create_app(),
            host="0.0.0.0",
            port=port,
            log_level="warning",
            access_log=False,
        )
    except KeyboardInterrupt:
        logger.info("Shutting down Gateway")


if __name__ == "__main__":
    main()
