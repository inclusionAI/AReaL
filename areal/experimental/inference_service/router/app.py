# SPDX-License-Identifier: Apache-2.0

"""Router service — stateful routing, session pinning, worker registry.

The Router is a separate FastAPI service from the Gateway.
It owns worker health state, session→worker mappings, and routing strategy.
It never proxies traffic — it only answers routing queries.
"""

from __future__ import annotations

import asyncio
import hmac
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from areal.experimental.inference_service.router.config import RouterConfig
from areal.experimental.inference_service.router.state import (
    ModelRegistry,
    SessionRegistry,
    WorkerRegistry,
)
from areal.experimental.inference_service.router.strategies import get_strategy
from areal.utils import logging

logger = logging.getLogger("InferenceRouter")


# =============================================================================
# Auth
# =============================================================================


def _extract_bearer_token(request: Request) -> str:
    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip()
    raise HTTPException(
        status_code=401,
        detail="Missing or malformed Authorization header.",
    )


def _require_admin_key(request: Request, admin_key: str) -> str:
    token = _extract_bearer_token(request)
    if not hmac.compare_digest(token, admin_key):
        raise HTTPException(status_code=403, detail="Invalid admin API key.")
    return token


# =============================================================================
# Request models
# =============================================================================


class RegisterWorkerRequest(BaseModel):
    worker_addr: str


class UnregisterWorkerRequest(BaseModel):
    worker_addr: str | None = None
    worker_id: str | None = None


class RouteRequest(BaseModel):
    api_key: str | None = None
    path: str | None = None
    session_id: str | None = None
    model: str | None = None


class RegisterSessionRequest(BaseModel):
    session_api_key: str
    session_id: str
    worker_addr: str
    model: str | None = None
    url: str | None = None
    provider_api_key: str | None = None


class RemoveSessionRequest(BaseModel):
    session_id: str


class RegisterModelRequest(BaseModel):
    model: str
    url: str = ""
    api_key: str | None = None
    data_proxy_addrs: list[str] = []


class RemoveModelRequest(BaseModel):
    name: str


# =============================================================================
# Response models
# =============================================================================


class StatusResponse(BaseModel):
    status: str


class HealthResponse(BaseModel):
    status: str
    workers: int
    sessions: int
    strategy: str


class RegisterWorkerResponse(BaseModel):
    status: str
    worker_id: str


class UnregisterWorkerResponse(BaseModel):
    status: str
    sessions_revoked: int


class RouteResponse(BaseModel):
    worker_addr: str
    url: str | None = None
    api_key: str | None = None


class RemoveSessionResponse(BaseModel):
    status: str
    removed: bool
    persistent: bool


class WorkerInfoResponse(BaseModel):
    worker_id: str
    addr: str
    healthy: bool
    active_requests: int


class WorkersResponse(BaseModel):
    workers: list[WorkerInfoResponse]


class RegisterModelResponse(BaseModel):
    status: str
    model: str
    data_proxy_addrs: list[str]


class ModelsResponse(BaseModel):
    models: list[str]


class RemoveModelResponse(BaseModel):
    status: str
    name: str


class ResolveWorkerResponse(BaseModel):
    worker_id: str
    worker_addr: str


# =============================================================================
# App factory
# =============================================================================


def create_app(config: RouterConfig) -> FastAPI:
    """Factory that creates the router FastAPI app."""

    worker_registry = WorkerRegistry()
    session_registry = SessionRegistry()
    model_registry = ModelRegistry()
    strategy = get_strategy(config.routing_strategy)

    async def _poll_workers() -> None:
        while True:
            workers = await worker_registry.get_all_workers()

            async def _check(w):
                try:
                    resp = await app.state.http_client.get(f"{w.worker_addr}/health")
                    await worker_registry.update_health(
                        w.worker_addr, resp.status_code == 200
                    )
                except Exception:
                    await worker_registry.update_health(w.worker_addr, False)

            await asyncio.gather(*[_check(w) for w in workers])
            await asyncio.sleep(config.poll_interval)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info(
            "Router starting — strategy=%s, poll_interval=%.1fs",
            config.routing_strategy,
            config.poll_interval,
        )
        app.state.http_client = httpx.AsyncClient(timeout=config.worker_health_timeout)
        poll_task = asyncio.create_task(_poll_workers())
        app.state.worker_registry = worker_registry
        app.state.session_registry = session_registry
        app.state.model_registry = model_registry
        app.state.strategy = strategy
        try:
            yield
        finally:
            poll_task.cancel()
            try:
                await poll_task
            except asyncio.CancelledError:
                pass
            await app.state.http_client.aclose()
            logger.info("Router shutting down")

    app = FastAPI(title="AReaL Router", lifespan=lifespan)

    # Expose registries on app.state for tests that bypass lifespan
    app.state.worker_registry = worker_registry
    app.state.session_registry = session_registry
    app.state.model_registry = model_registry
    app.state.strategy = strategy

    # =========================================================================
    # Health
    # =========================================================================

    @app.get("/health", response_model=HealthResponse)
    async def health():
        all_workers = await worker_registry.get_all_workers()
        session_count = await session_registry.count()
        return HealthResponse(
            status="ok",
            workers=len(all_workers),
            sessions=session_count,
            strategy=config.routing_strategy,
        )

    # =========================================================================
    # Worker management (admin key required)
    # =========================================================================

    @app.post("/register", response_model=RegisterWorkerResponse)
    async def register(body: RegisterWorkerRequest, request: Request):
        _require_admin_key(request, config.admin_api_key)
        worker_id = await worker_registry.register(body.worker_addr)
        logger.info("Worker registered: %s (id=%s)", body.worker_addr, worker_id)
        return RegisterWorkerResponse(status="ok", worker_id=worker_id)

    @app.post("/unregister", response_model=UnregisterWorkerResponse)
    async def unregister(body: UnregisterWorkerRequest, request: Request):
        _require_admin_key(request, config.admin_api_key)
        if body.worker_id is not None:
            worker_addr = await worker_registry.deregister_by_id(body.worker_id)
            if worker_addr is None:
                raise HTTPException(
                    status_code=404, detail=f"Worker ID {body.worker_id} not found"
                )
            revoked = await session_registry.revoke_by_worker(worker_addr)
            logger.info(
                "Worker unregistered by id: %s addr=%s (revoked %d sessions)",
                body.worker_id,
                worker_addr,
                revoked,
            )
            return UnregisterWorkerResponse(status="ok", sessions_revoked=revoked)
        elif body.worker_addr is not None:
            await worker_registry.deregister(body.worker_addr)
            revoked = await session_registry.revoke_by_worker(body.worker_addr)
            logger.info(
                "Worker unregistered: %s (revoked %d sessions)",
                body.worker_addr,
                revoked,
            )
            return UnregisterWorkerResponse(status="ok", sessions_revoked=revoked)
        else:
            raise HTTPException(
                status_code=422,
                detail="Either 'worker_id' or 'worker_addr' must be provided",
            )

    # =========================================================================
    # Routing (admin key required)
    # =========================================================================

    @app.post("/route", response_model=RouteResponse)
    async def route(body: RouteRequest, request: Request):
        _require_admin_key(request, config.admin_api_key)

        # Step 1: Check session pinning (by session_id or api_key)
        pin = None
        if body.session_id is not None:
            pin = await session_registry.lookup_by_id(body.session_id)
        elif body.api_key is not None:
            pin = await session_registry.lookup_by_key(body.api_key)

        if pin is not None:
            return RouteResponse(
                worker_addr=pin.worker_addr,
                url=pin.url,
                api_key=pin.api_key,
            )

        # Step 2: No pinned session — pick a worker via model + strategy
        model_info = await _resolve_model(body.model)

        if body.session_id is not None:
            raise HTTPException(status_code=404, detail="Session not found")

        if body.api_key is None:
            if body.model is not None:
                if model_info is None:
                    raise HTTPException(
                        status_code=404, detail=f"Model '{body.model}' not found"
                    )
                worker = await _pick_worker(model_info)
                return RouteResponse(
                    worker_addr=worker.worker_addr,
                    url=model_info.url,
                    api_key=model_info.api_key,
                )
            raise HTTPException(
                status_code=422,
                detail="Either 'api_key', 'session_id', or 'model' must be provided",
            )

        if not hmac.compare_digest(body.api_key, config.admin_api_key):
            raise HTTPException(status_code=404, detail="Unknown API key")

        worker = await _pick_worker(model_info)

        # Pin admin key for HITL sticky routing
        await session_registry.register_session(
            session_key=body.api_key,
            session_id="__hitl__",
            worker_addr=worker.worker_addr,
            model=model_info.name if model_info else None,
            url=model_info.url if model_info else None,
            api_key=model_info.api_key if model_info else None,
        )

        return RouteResponse(
            worker_addr=worker.worker_addr,
            url=model_info.url if model_info else None,
            api_key=model_info.api_key if model_info else None,
        )

    async def _resolve_model(model_name: str | None):
        """Resolve model name to ModelInfo. Falls back to first registered."""
        if model_name is not None:
            info = await model_registry.get(model_name)
            if info is not None:
                return info
        return await model_registry.first()

    async def _pick_worker(model_info):
        """Pick a worker filtered by model's data_proxy_addrs."""
        all_workers = await worker_registry.get_all_workers()
        if model_info is not None and model_info.data_proxy_addrs:
            addr_set = set(model_info.data_proxy_addrs)
            candidates = [w for w in all_workers if w.worker_addr in addr_set]
        else:
            candidates = all_workers
        if not candidates:
            raise HTTPException(status_code=503, detail="No registered workers")
        worker = strategy.pick(candidates)
        if worker is None:
            raise HTTPException(status_code=503, detail="No registered workers")
        return worker

    # =========================================================================
    # Session registration (admin key required)
    # =========================================================================

    @app.post("/register_session", response_model=StatusResponse)
    async def register_session(body: RegisterSessionRequest, request: Request):
        _require_admin_key(request, config.admin_api_key)
        await session_registry.register_session(
            session_key=body.session_api_key,
            session_id=body.session_id,
            worker_addr=body.worker_addr,
            model=body.model,
            url=body.url,
            api_key=body.provider_api_key,
        )
        return StatusResponse(status="ok")

    # =========================================================================
    # Session cleanup (admin key required)
    # =========================================================================

    @app.post("/remove_session", response_model=RemoveSessionResponse)
    async def remove_session(body: RemoveSessionRequest, request: Request):
        _require_admin_key(request, config.admin_api_key)
        session_key = await session_registry.session_key_for_id(body.session_id)
        is_hitl_persistent = session_key is not None and hmac.compare_digest(
            session_key, config.admin_api_key
        )
        removed = (
            False
            if is_hitl_persistent
            else await session_registry.revoke_session(body.session_id)
        )
        return RemoveSessionResponse(
            status="ok",
            removed=removed,
            persistent=is_hitl_persistent,
        )

    # =========================================================================
    # Worker listing (admin key required)
    # =========================================================================

    @app.get("/workers", response_model=WorkersResponse)
    async def list_workers(request: Request):
        _require_admin_key(request, config.admin_api_key)
        all_workers = await worker_registry.get_all_workers()
        return WorkersResponse(
            workers=[
                WorkerInfoResponse(
                    worker_id=w.worker_id,
                    addr=w.worker_addr,
                    healthy=w.is_healthy,
                    active_requests=w.active_requests,
                )
                for w in all_workers
            ]
        )

    @app.post("/register_model", response_model=RegisterModelResponse)
    async def register_model(body: RegisterModelRequest, request: Request):
        _require_admin_key(request, config.admin_api_key)
        addrs = body.data_proxy_addrs
        if not addrs:
            healthy = await worker_registry.get_healthy_workers()
            if not healthy:
                raise HTTPException(status_code=503, detail="No healthy workers")
            addrs = [w.worker_addr for w in healthy]
        await model_registry.register(
            body.model,
            body.url,
            body.api_key,
            addrs,
        )
        logger.info(
            "Model registered: model=%s url=%s data_proxy_addrs=%s",
            body.model,
            body.url or "(internal)",
            addrs,
        )
        return RegisterModelResponse(
            status="ok",
            model=body.model,
            data_proxy_addrs=addrs,
        )

    @app.get("/models", response_model=ModelsResponse)
    async def list_models(request: Request):
        _require_admin_key(request, config.admin_api_key)
        names = await model_registry.list_names()
        return ModelsResponse(models=names)

    @app.post("/remove_model", response_model=RemoveModelResponse)
    async def remove_model(body: RemoveModelRequest, request: Request):
        _require_admin_key(request, config.admin_api_key)
        removed = await model_registry.remove(body.name)
        if not removed:
            raise HTTPException(
                status_code=404,
                detail=f"External model '{body.name}' not found",
            )
        logger.info("External model removed: name=%s", body.name)
        return RemoveModelResponse(status="ok", name=body.name)

    # =========================================================================
    # Worker resolution by ID (admin key required)
    # =========================================================================

    @app.get("/resolve_worker/{worker_id}", response_model=ResolveWorkerResponse)
    async def resolve_worker(worker_id: str, request: Request):
        _require_admin_key(request, config.admin_api_key)
        worker = await worker_registry.get_by_id(worker_id)
        if worker is None:
            raise HTTPException(
                status_code=404, detail=f"Worker ID {worker_id} not found"
            )
        return ResolveWorkerResponse(
            worker_id=worker.worker_id, worker_addr=worker.worker_addr
        )

    return app
