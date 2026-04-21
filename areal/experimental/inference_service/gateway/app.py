# SPDX-License-Identifier: Apache-2.0

"""Inference Gateway — thin HTTP proxy with auth, routing, and forwarding.

The gateway holds only ``admin_api_key`` and ``router_addr``. All worker state,
session pinning, and routing strategies live in the Router service.
"""

from __future__ import annotations

import json

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from areal.experimental.inference_service.gateway.auth import (
    extract_bearer_token,
    require_admin_key,
)
from areal.experimental.inference_service.gateway.config import GatewayConfig
from areal.experimental.inference_service.gateway.streaming import (
    RouterKeyRejectedError,
    RouterUnreachableError,
    _forwarding_headers,
    broadcast_to_workers,
    forward_request,
    forward_sse_stream,
    grant_capacity_in_router,
    list_models_from_router,
    query_router,
    register_model_in_router,
    register_session_in_router,
    remove_model_from_router,
    resolve_worker_addr,
    revoke_session_in_router,
)
from areal.utils import logging

logger = logging.getLogger("InferenceGateway")


def _router_error_response(exc: Exception) -> JSONResponse:
    """Convert router exceptions to HTTP responses."""
    if isinstance(exc, RouterUnreachableError):
        return JSONResponse({"error": str(exc)}, status_code=502)
    if isinstance(exc, RouterKeyRejectedError):
        status = 401 if exc.status_code == 404 else exc.status_code
        return JSONResponse({"error": exc.detail}, status_code=status)
    return JSONResponse({"error": str(exc)}, status_code=500)


def create_app(config: GatewayConfig) -> FastAPI:
    """Factory that creates the inference gateway FastAPI app."""

    app = FastAPI(title="AReaL Inference Gateway")

    # =========================================================================
    # Health
    # =========================================================================

    @app.get("/health")
    async def health():
        return {"status": "ok", "router_addr": config.router_addr}

    # =========================================================================
    # POST /chat/completions — admin OR session key, streaming or non-streaming
    # =========================================================================

    @app.post("/chat/completions")
    async def chat_completions(request: Request):
        token = extract_bearer_token(request)
        body = await request.body()
        headers = _forwarding_headers(dict(request.headers))

        model_name = None
        is_streaming = False
        try:
            body_json = json.loads(body)
            model_name = body_json.get("model")
            is_streaming = body_json.get("stream", False) or False
        except (json.JSONDecodeError, AttributeError):
            pass

        try:
            worker_addr = await query_router(
                config.router_addr,
                token,
                "/chat/completions",
                config.router_timeout,
                admin_api_key=config.admin_api_key,
                model=model_name,
            )
        except (RouterUnreachableError, RouterKeyRejectedError) as exc:
            return _router_error_response(exc)

        if is_streaming:
            return StreamingResponse(
                forward_sse_stream(
                    f"{worker_addr}/chat/completions",
                    body,
                    headers,
                    config.forward_timeout,
                ),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        resp = await forward_request(
            f"{worker_addr}/chat/completions", body, headers, config.forward_timeout
        )
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type"),
        )

    @app.post("/register_model")
    async def register_model(request: Request):
        require_admin_key(request, config.admin_api_key)
        body = await request.json()
        model = body.get("model")
        url = body.get("url", "")
        api_key = body.get("api_key")
        data_proxy_addrs = body.get("data_proxy_addrs", [])
        if not model:
            return JSONResponse({"error": "model is required"}, status_code=400)
        try:
            result = await register_model_in_router(
                config.router_addr,
                model,
                url,
                api_key,
                data_proxy_addrs,
                config.admin_api_key,
                config.router_timeout,
            )
        except (RouterUnreachableError, RouterKeyRejectedError) as exc:
            return _router_error_response(exc)

        resolved_addrs = result.get("data_proxy_addrs", data_proxy_addrs)
        headers = _forwarding_headers(dict(request.headers))

        for addr in resolved_addrs:
            resp = await forward_request(
                f"{addr}/register_model",
                json.dumps(
                    {
                        "name": model,
                        "url": url,
                        "model": model,
                        "api_key": api_key,
                    }
                ).encode(),
                headers,
                config.forward_timeout,
            )
            if resp.status_code != 200:
                await remove_model_from_router(
                    config.router_addr,
                    model,
                    config.admin_api_key,
                    config.router_timeout,
                )
                return JSONResponse(
                    {"error": f"Data proxy registration failed: {resp.text}"},
                    status_code=502,
                )
        return result

    @app.get("/models")
    async def list_models(request: Request):
        require_admin_key(request, config.admin_api_key)
        try:
            names = await list_models_from_router(
                config.router_addr,
                config.admin_api_key,
                config.router_timeout,
            )
        except (RouterUnreachableError, RouterKeyRejectedError) as exc:
            return _router_error_response(exc)
        return {"models": names}

    # =========================================================================
    # POST /rl/start_session — admin key ONLY, intercept response
    # =========================================================================

    @app.post("/rl/start_session")
    async def start_session(request: Request):
        token = require_admin_key(request, config.admin_api_key)

        try:
            worker_addr = await query_router(
                config.router_addr,
                token,
                "/rl/start_session",
                config.router_timeout,
                admin_api_key=config.admin_api_key,
            )
        except (RouterUnreachableError, RouterKeyRejectedError) as exc:
            return _router_error_response(exc)

        body = await request.body()
        headers = _forwarding_headers(dict(request.headers))

        resp = await forward_request(
            f"{worker_addr}/rl/start_session",
            body,
            headers,
            config.forward_timeout,
        )

        # Intercept: if data proxy returned 201, extract session info and register
        if resp.status_code == 201:
            try:
                resp_data = resp.json()
                session_api_key = resp_data.get("api_key")
                session_id = resp_data.get("session_id")
                if session_api_key and session_id:
                    await register_session_in_router(
                        config.router_addr,
                        session_api_key,
                        session_id,
                        worker_addr,
                        config.router_timeout,
                        admin_api_key=config.admin_api_key,
                    )
            except Exception as exc:
                logger.error("Failed to register session in router: %s", exc)
                return JSONResponse(
                    {
                        "error": f"Session created on worker but router registration failed: {exc}"
                    },
                    status_code=502,
                )

        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type"),
        )

    # =========================================================================
    # POST /rl/set_reward — session key or admin key (HITL)
    # =========================================================================

    @app.post("/rl/set_reward")
    async def set_reward(request: Request):
        token = extract_bearer_token(request)
        body = await request.body()
        headers = _forwarding_headers(dict(request.headers))

        model = None
        try:
            body_json = json.loads(body)
            model = body_json.get("model")
        except (json.JSONDecodeError, AttributeError):
            pass

        try:
            worker_addr = await query_router(
                config.router_addr,
                token,
                "/rl/set_reward",
                config.router_timeout,
                admin_api_key=config.admin_api_key,
                model=model,
            )
        except (RouterUnreachableError, RouterKeyRejectedError) as exc:
            return _router_error_response(exc)

        resp = await forward_request(
            f"{worker_addr}/rl/set_reward", body, headers, config.forward_timeout
        )
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type"),
        )

    # =========================================================================
    # POST /pause_generation/{worker_id} — admin key ONLY, target single worker
    # =========================================================================

    @app.post("/pause_generation/{worker_id}")
    async def pause_generation(worker_id: str, request: Request):
        require_admin_key(request, config.admin_api_key)
        try:
            worker_addr = await resolve_worker_addr(
                config.router_addr,
                config.admin_api_key,
                worker_id,
                config.router_timeout,
            )
        except (RouterUnreachableError, RouterKeyRejectedError) as exc:
            return _router_error_response(exc)

        body = await request.body()
        headers = _forwarding_headers(dict(request.headers))
        results = await broadcast_to_workers(
            [worker_addr], "/pause_generation", body, headers
        )
        return {"results": results}

    # =========================================================================
    # POST /continue_generation/{worker_id} — admin key ONLY, target single worker
    # =========================================================================

    @app.post("/continue_generation/{worker_id}")
    async def continue_generation(worker_id: str, request: Request):
        require_admin_key(request, config.admin_api_key)
        try:
            worker_addr = await resolve_worker_addr(
                config.router_addr,
                config.admin_api_key,
                worker_id,
                config.router_timeout,
            )
        except (RouterUnreachableError, RouterKeyRejectedError) as exc:
            return _router_error_response(exc)

        body = await request.body()
        headers = _forwarding_headers(dict(request.headers))
        results = await broadcast_to_workers(
            [worker_addr], "/continue_generation", body, headers
        )
        return {"results": results}

    # =========================================================================
    # POST /export_trajectories — admin key ONLY, route by session_id or model field
    # =========================================================================

    @app.post("/export_trajectories")
    async def export_trajectories(request: Request):
        require_admin_key(request, config.admin_api_key)
        body = await request.body()

        try:
            body_json = json.loads(body)
        except (json.JSONDecodeError, AttributeError):
            return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

        model = body_json.get("model")
        session_id = body_json.get("session_id")

        if model and not session_id:
            session_id = model
            body_json["session_id"] = session_id
            body = json.dumps(body_json).encode()

        if not session_id:
            return JSONResponse({"error": "session_id is required"}, status_code=400)

        try:
            worker_addr = await query_router(
                config.router_addr,
                timeout=config.router_timeout,
                session_id=session_id,
                admin_api_key=config.admin_api_key,
                model=model,
            )
        except (RouterUnreachableError, RouterKeyRejectedError) as exc:
            return _router_error_response(exc)

        headers = _forwarding_headers(dict(request.headers))
        resp = await forward_request(
            f"{worker_addr}/export_trajectories",
            body,
            headers,
            config.forward_timeout,
        )

        if resp.status_code == 200:
            await revoke_session_in_router(
                config.router_addr,
                config.admin_api_key,
                session_id,
                config.router_timeout,
            )

        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type"),
        )

    # =========================================================================
    # POST /set_version/{worker_id} — admin key ONLY, target single worker
    # =========================================================================

    @app.post("/set_version/{worker_id}")
    async def set_version(worker_id: str, request: Request):
        require_admin_key(request, config.admin_api_key)
        try:
            worker_addr = await resolve_worker_addr(
                config.router_addr,
                config.admin_api_key,
                worker_id,
                config.router_timeout,
            )
        except (RouterUnreachableError, RouterKeyRejectedError) as exc:
            return _router_error_response(exc)

        body = await request.body()
        headers = _forwarding_headers(dict(request.headers))
        resp = await forward_request(
            f"{worker_addr}/set_version", body, headers, config.forward_timeout
        )
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type"),
        )

    # =========================================================================
    # GET /get_version/{worker_id} — admin key ONLY, target single worker
    # =========================================================================

    @app.get("/get_version/{worker_id}")
    async def get_version(worker_id: str, request: Request):
        require_admin_key(request, config.admin_api_key)
        try:
            worker_addr = await resolve_worker_addr(
                config.router_addr,
                config.admin_api_key,
                worker_id,
                config.router_timeout,
            )
        except (RouterUnreachableError, RouterKeyRejectedError) as exc:
            return _router_error_response(exc)

        try:
            async with httpx.AsyncClient(timeout=config.forward_timeout) as client:
                resp = await client.get(
                    f"{worker_addr}/get_version",
                    headers=_forwarding_headers(dict(request.headers)),
                )
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                media_type=resp.headers.get("content-type"),
            )
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=502)

    @app.post("/grant_capacity")
    async def grant_capacity(request: Request):
        """Forward capacity grant to the Router (not data proxies).

        Staleness control lives at the router level — data proxies do not
        track capacity.
        """
        require_admin_key(request, config.admin_api_key)
        try:
            result = await grant_capacity_in_router(
                config.router_addr, config.admin_api_key, config.router_timeout
            )
        except RouterUnreachableError as exc:
            return _router_error_response(exc)
        return result

    # =========================================================================
    # Compatibility aliases for RolloutCallback — map /callback/* to endpoints
    # =========================================================================
    # RolloutCallback uses /callback/* prefixed paths for generation control.
    # Gateway implements the actual handlers at unprefixed paths.  These aliases
    # register the SAME handler functions on both routes.
    # POST /callback/pause_generation/{worker_id} → pause_generation
    app.add_api_route(
        "/callback/pause_generation/{worker_id}",
        pause_generation,
        methods=["POST"],
    )

    # POST /callback/continue_generation/{worker_id} → continue_generation
    app.add_api_route(
        "/callback/continue_generation/{worker_id}",
        continue_generation,
        methods=["POST"],
    )

    # =========================================================================
    # OpenAI / OpenRouter compatibility aliases — /v1/* prefixed routes
    # =========================================================================
    app.add_api_route(
        "/v1/chat/completions",
        chat_completions,
        methods=["POST"],
    )
    app.add_api_route(
        "/v1/models",
        list_models,
        methods=["GET"],
    )

    return app
