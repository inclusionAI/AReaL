"""Gateway process for the Agent Service.

The Gateway receives ``/run_episode`` requests from callers (e.g.
OpenAIProxyWorkflow), queues them internally via the :class:`Router`, and
dispatches them to registered Agent Worker processes over HTTP.

Endpoints
---------
- POST /run_episode  — Enqueue an episode request for dispatch to a worker.
- GET  /health       — Health check with worker pool statistics.
- POST /register_worker   — Register a new Agent Worker.
- POST /unregister_worker — Remove an Agent Worker from the pool.
- GET  /workers      — List all registered workers.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from areal.utils import logging

from .config import GatewayConfig
from .router import Router
from .schemas import (
    DispatchRequest,
    RegisterWorkerRequest,
    RunEpisodeRequest,
    RunEpisodeResponse,
)
from .worker_pool import WorkerPoolManager

logger = logging.getLogger("Gateway")


# ------------------------------------------------------------------
# Gateway
# ------------------------------------------------------------------


class Gateway:
    """Central dispatcher that bridges HTTP callers to Agent Workers.

    The Gateway owns a :class:`WorkerPoolManager` and a :class:`Router`.
    Incoming ``/run_episode`` requests are placed onto the Router's bounded
    queue; the Router's background dispatcher forwards them to idle workers.

    Parameters
    ----------
    config : GatewayConfig
        Gateway configuration (host, port, queue_size, etc.).
    """

    def __init__(
        self,
        config: GatewayConfig,
    ) -> None:
        self._config = config

        self._pool: WorkerPoolManager | None = None
        self._router: Router | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Initialize WorkerPoolManager and Router, then start dispatching."""
        self._pool = WorkerPoolManager(
            worker_timeout=self._config.worker_timeout,
        )
        self._router = Router(
            pool=self._pool,
            queue_size=self._config.queue_size,
            worker_timeout=self._config.worker_timeout,
        )
        await self._router.start()
        logger.info(
            "Gateway started on %s:%d (queue_size=%d)",
            self._config.host,
            self._config.port,
            self._config.queue_size,
        )

    async def stop(self) -> None:
        """Stop the Router (cancels dispatcher and closes HTTP session)."""
        if self._router is not None:
            await self._router.stop()
        logger.info("Gateway stopped")

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
            description="Gateway that dispatches run_episode requests to Agent Workers",
            version="1.0.0",
            lifespan=lifespan,
        )

        # -- endpoints ------------------------------------------------

        @app.post("/run_episode", response_model=RunEpisodeResponse)
        async def run_episode(request: RunEpisodeRequest) -> RunEpisodeResponse:
            """Enqueue an episode request for dispatch to a worker."""
            if gateway._router is None:
                raise HTTPException(status_code=503, detail="Gateway not initialized")
            try:
                dispatch_req = DispatchRequest(
                    data=request.data,
                    session_url=request.session_url,
                    agent_kwargs=request.agent_kwargs,
                    agent_import_path=request.agent_import_path,
                )
                response = await gateway._router.enqueue(dispatch_req)
                return RunEpisodeResponse(
                    status=response.status,
                    result=response.result,
                    error=response.error,
                )
            except asyncio.QueueFull:
                raise HTTPException(
                    status_code=503,
                    detail="Request queue is full, try again later",
                )

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

        return app


def main() -> None:
    """Entry point for the Gateway process (internal, launched by Scheduler)."""
    import argparse

    import uvicorn

    from areal.utils.network import find_free_ports, gethostip

    parser = argparse.ArgumentParser(description="Agent Service Gateway")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--queue-size", type=int, default=1000)
    args = parser.parse_args()

    host = gethostip() if args.host == "0.0.0.0" else args.host
    port = args.port if args.port != 0 else find_free_ports(1)[0]

    config = GatewayConfig(
        host=host,
        port=port,
        queue_size=args.queue_size,
    )
    gateway = Gateway(config=config)

    logger.info(
        "Starting Gateway on %s:%d (queue_size=%d)",
        host,
        port,
        config.queue_size,
    )
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
