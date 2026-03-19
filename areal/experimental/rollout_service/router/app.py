"""Router service — stateful routing, session pinning, worker registry.

The Router is a separate FastAPI service from the Gateway.
It owns worker health state, session→worker mappings, and routing strategy.
It never proxies traffic — it only answers routing queries.
"""

from __future__ import annotations

import asyncio
import hmac

import httpx
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from areal.experimental.rollout_service.router.config import RouterConfig
from areal.experimental.rollout_service.router.state import (
    CapacityManager,
    SessionRegistry,
    WorkerRegistry,
)
from areal.experimental.rollout_service.router.strategies import get_strategy
from areal.utils import logging

logger = logging.getLogger("RolloutRouter")


# =============================================================================
# Auth helpers (same pattern as data proxy)
# =============================================================================


def _extract_bearer_token(request: Request) -> str:
    """Extract API token from Authorization header."""
    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip()
    raise HTTPException(
        status_code=401,
        detail="Missing or malformed Authorization header.",
    )


def _require_admin_key(request: Request, admin_key: str) -> str:
    """Validate that the request carries the admin API key."""
    token = _extract_bearer_token(request)
    if not hmac.compare_digest(token, admin_key):
        raise HTTPException(status_code=403, detail="Invalid admin API key.")
    return token


# =============================================================================
# Request models
# =============================================================================


class RegisterWorkerRequest(BaseModel):
    worker_addr: str


class DeleteWorkerRequest(BaseModel):
    worker_addr: str | None = None
    worker_id: str | None = None


class RouteRequest(BaseModel):
    api_key: str | None = None
    path: str | None = None
    session_id: str | None = None


class RegisterSessionRequest(BaseModel):
    session_api_key: str
    session_id: str
    worker_addr: str


# =============================================================================
# App factory
# =============================================================================


def create_app(config: RouterConfig) -> FastAPI:
    """Factory that creates the router FastAPI app."""

    worker_registry = WorkerRegistry()
    session_registry = SessionRegistry()
    capacity_manager = CapacityManager()
    strategy = get_strategy(config.routing_strategy)

    async def _poll_workers() -> None:
        """Background task: periodically poll worker /health endpoints."""
        while True:
            workers = await worker_registry.get_all_workers()
            for w in workers:
                try:
                    async with httpx.AsyncClient(
                        timeout=config.worker_health_timeout
                    ) as client:
                        resp = await client.get(f"{w.worker_addr}/health")
                        await worker_registry.update_health(
                            w.worker_addr, resp.status_code == 200
                        )
                except Exception:
                    await worker_registry.update_health(w.worker_addr, False)
            await asyncio.sleep(config.poll_interval)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info(
            "Router starting — strategy=%s, poll_interval=%.1fs",
            config.routing_strategy,
            config.poll_interval,
        )
        poll_task = asyncio.create_task(_poll_workers())
        app.state.worker_registry = worker_registry
        app.state.session_registry = session_registry
        app.state.capacity_manager = capacity_manager
        app.state.strategy = strategy
        yield
        poll_task.cancel()
        try:
            await poll_task
        except asyncio.CancelledError:
            pass
        logger.info("Router shutting down")

    app = FastAPI(title="AReaL Router", lifespan=lifespan)

    # Expose registries on app.state for tests that bypass lifespan
    app.state.worker_registry = worker_registry
    app.state.session_registry = session_registry
    app.state.capacity_manager = capacity_manager
    app.state.strategy = strategy

    # =========================================================================
    # Health
    # =========================================================================

    @app.get("/health")
    async def health():
        all_workers = await worker_registry.get_all_workers()
        session_count = await session_registry.count()
        capacity = await capacity_manager.get_capacity()
        return {
            "status": "ok",
            "workers": len(all_workers),
            "sessions": session_count,
            "capacity": capacity,
            "strategy": config.routing_strategy,
        }

    # =========================================================================
    # Worker management (admin key required)
    # =========================================================================

    @app.post("/register_worker")
    async def register_worker(body: RegisterWorkerRequest, request: Request):
        _require_admin_key(request, config.admin_api_key)
        worker_id = await worker_registry.register(body.worker_addr)
        logger.info("Worker registered: %s (id=%s)", body.worker_addr, worker_id)
        return {"status": "ok", "worker_id": worker_id}

    @app.delete("/delete_worker")
    async def delete_worker(body: DeleteWorkerRequest, request: Request):
        _require_admin_key(request, config.admin_api_key)
        if body.worker_id is not None:
            worker_addr = await worker_registry.deregister_by_id(body.worker_id)
            if worker_addr is None:
                raise HTTPException(
                    status_code=404, detail=f"Worker ID {body.worker_id} not found"
                )
            revoked = await session_registry.revoke_by_worker(worker_addr)
            logger.info(
                "Worker deleted by id: %s addr=%s (revoked %d sessions)",
                body.worker_id,
                worker_addr,
                revoked,
            )
            return {"status": "ok", "sessions_revoked": revoked}
        elif body.worker_addr is not None:
            await worker_registry.deregister(body.worker_addr)
            revoked = await session_registry.revoke_by_worker(body.worker_addr)
            logger.info(
                "Worker deleted: %s (revoked %d sessions)",
                body.worker_addr,
                revoked,
            )
            return {"status": "ok", "sessions_revoked": revoked}
        else:
            raise HTTPException(
                status_code=422,
                detail="Either 'worker_id' or 'worker_addr' must be provided",
            )

    # =========================================================================
    # Routing (admin key required)
    # =========================================================================

    @app.post("/route")
    async def route(body: RouteRequest, request: Request):
        _require_admin_key(request, config.admin_api_key)
        # 0. session_id lookup takes precedence
        if body.session_id is not None:
            worker = await session_registry.lookup_by_id(body.session_id)
            if worker is None:
                raise HTTPException(status_code=404, detail="Session not found")
            return {"worker_addr": worker}

        if body.api_key is None:
            raise HTTPException(
                status_code=422,
                detail="Either 'api_key' or 'session_id' must be provided",
            )

        # 1. Admin key → pick healthy worker via strategy
        if hmac.compare_digest(body.api_key, config.admin_api_key):
            healthy = await worker_registry.get_healthy_workers()
            if not healthy:
                raise HTTPException(status_code=503, detail="No healthy workers")
            worker = strategy.pick(healthy)
            if worker is None:
                raise HTTPException(status_code=503, detail="No healthy workers")
            return {"worker_addr": worker.worker_addr}

        # 2. Session key → pinned worker
        pinned = await session_registry.lookup_by_key(body.api_key)
        if pinned is not None:
            # Check if pinned worker is healthy
            all_workers = await worker_registry.get_all_workers()
            worker_map = {w.worker_addr: w for w in all_workers}
            w = worker_map.get(pinned)
            if w is None or not w.is_healthy:
                raise HTTPException(status_code=503, detail="Pinned worker unhealthy")
            return {"worker_addr": pinned}

        # 3. Unknown key
        raise HTTPException(status_code=404, detail="Unknown API key")

    # =========================================================================
    # Session registration (admin key required)
    # =========================================================================

    @app.post("/register_session")
    async def register_session(body: RegisterSessionRequest, request: Request):
        _require_admin_key(request, config.admin_api_key)
        await session_registry.register_session(
            body.session_api_key, body.session_id, body.worker_addr
        )
        return {"status": "ok"}

    # =========================================================================
    # Worker listing (admin key required)
    # =========================================================================

    @app.get("/workers")
    async def list_workers(request: Request):
        _require_admin_key(request, config.admin_api_key)
        all_workers = await worker_registry.get_all_workers()
        return {
            "workers": [
                {
                    "worker_id": w.worker_id,
                    "addr": w.worker_addr,
                    "healthy": w.is_healthy,
                    "active_requests": w.active_requests,
                }
                for w in all_workers
            ]
        }

    # =========================================================================
    # Worker resolution by ID (admin key required)
    # =========================================================================

    @app.get("/resolve_worker/{worker_id}")
    async def resolve_worker(worker_id: str, request: Request):
        """Resolve a worker_id to its address.

        Returns the worker address for a given worker ID.
        Used by the gateway to target specific workers for
        pause/continue generation.
        """
        _require_admin_key(request, config.admin_api_key)
        worker = await worker_registry.get_by_id(worker_id)
        if worker is None:
            raise HTTPException(
                status_code=404, detail=f"Worker ID {worker_id} not found"
            )
        return {"worker_id": worker.worker_id, "worker_addr": worker.worker_addr}

    # =========================================================================
    # Capacity management (admin key required)
    # =========================================================================

    @app.post("/grant_capacity")
    async def grant_capacity(request: Request):
        """Increment session capacity by 1.

        Called by the rollout controller (via the gateway) when the current
        weight version is within the allowed staleness window.  Each call
        issues one permit for a future ``/rl/start_session`` request.
        """
        _require_admin_key(request, config.admin_api_key)
        new_capacity = await capacity_manager.grant()
        logger.info("Capacity granted — now %d", new_capacity)
        return {"status": "ok", "capacity": new_capacity}

    @app.post("/acquire_capacity")
    async def acquire_capacity(request: Request):
        """Try to consume one capacity permit.

        Called by the gateway before forwarding ``/rl/start_session`` to a
        data proxy.  Returns 200 on success, 429 when no permits remain.
        """
        _require_admin_key(request, config.admin_api_key)
        acquired = await capacity_manager.try_acquire()
        if not acquired:
            raise HTTPException(
                status_code=429,
                detail="No available capacity to start a new session",
            )
        return {"status": "ok"}

    @app.post("/release_capacity")
    async def release_capacity(request: Request):
        """Return one previously acquired capacity permit.

        Called by the gateway when ``/rl/start_session`` fails after
        ``/acquire_capacity`` succeeded, to avoid leaking permits.
        """
        _require_admin_key(request, config.admin_api_key)
        new_capacity = await capacity_manager.release()
        logger.info("Capacity released — now %d", new_capacity)
        return {"status": "ok", "capacity": new_capacity}

    return app
