"""Inference Gateway — thin HTTP proxy with auth, routing, and forwarding.

The gateway holds only ``admin_api_key`` and ``router_addr``. All worker state,
session pinning, and routing strategies live in the Router service.
"""

from __future__ import annotations

import json
import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from areal.experimental.gateway.gateway.auth import (
    extract_bearer_token,
    require_admin_key,
)
from areal.experimental.gateway.gateway.config import GatewayConfig
from areal.experimental.gateway.gateway.streaming import (
    RouterKeyRejectedError,
    RouterUnreachableError,
    broadcast_to_workers,
    forward_request,
    forward_sse_stream,
    get_all_worker_addrs,
    query_router,
    query_router_by_session_id,
    register_session_in_router,
    _forwarding_headers,
)

logger = logging.getLogger("Gateway")


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
    # POST /generate — admin OR session key, SSE streaming
    # =========================================================================

    @app.post("/generate")
    async def generate(request: Request):
        token = extract_bearer_token(request)
        try:
            worker_addr = await query_router(
                config.router_addr, token, "/generate", config.router_timeout
            )
        except (RouterUnreachableError, RouterKeyRejectedError) as exc:
            return _router_error_response(exc)

        body = await request.body()
        headers = _forwarding_headers(dict(request.headers))
        return StreamingResponse(
            forward_sse_stream(
                f"{worker_addr}/generate", body, headers, config.forward_timeout
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # =========================================================================
    # POST /chat/completions — admin OR session key, streaming or non-streaming
    # =========================================================================

    @app.post("/chat/completions")
    async def chat_completions(request: Request):
        token = extract_bearer_token(request)
        try:
            worker_addr = await query_router(
                config.router_addr, token, "/chat/completions", config.router_timeout
            )
        except (RouterUnreachableError, RouterKeyRejectedError) as exc:
            return _router_error_response(exc)

        body = await request.body()
        headers = _forwarding_headers(dict(request.headers))

        # Detect streaming from request body
        is_streaming = False
        try:
            body_json = json.loads(body)
            is_streaming = body_json.get("stream", False) or False
        except (json.JSONDecodeError, AttributeError):
            pass

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

    # =========================================================================
    # POST /rl/start_session — admin key ONLY, intercept response
    # =========================================================================

    @app.post("/rl/start_session")
    async def start_session(request: Request):
        token = require_admin_key(request, config.admin_api_key)
        try:
            worker_addr = await query_router(
                config.router_addr, token, "/rl/start_session", config.router_timeout
            )
        except (RouterUnreachableError, RouterKeyRejectedError) as exc:
            return _router_error_response(exc)

        body = await request.body()
        headers = _forwarding_headers(dict(request.headers))

        resp = await forward_request(
            f"{worker_addr}/rl/start_session", body, headers, config.forward_timeout
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
    # POST /rl/end_session — session key ONLY
    # =========================================================================

    @app.post("/rl/end_session")
    async def end_session(request: Request):
        token = extract_bearer_token(request)
        try:
            worker_addr = await query_router(
                config.router_addr, token, "/rl/end_session", config.router_timeout
            )
        except (RouterUnreachableError, RouterKeyRejectedError) as exc:
            return _router_error_response(exc)

        body = await request.body()
        headers = _forwarding_headers(dict(request.headers))
        resp = await forward_request(
            f"{worker_addr}/rl/end_session", body, headers, config.forward_timeout
        )
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type"),
        )

    # =========================================================================
    # POST /rl/set_reward — session key ONLY
    # =========================================================================

    @app.post("/rl/set_reward")
    async def set_reward(request: Request):
        token = extract_bearer_token(request)
        try:
            worker_addr = await query_router(
                config.router_addr, token, "/rl/set_reward", config.router_timeout
            )
        except (RouterUnreachableError, RouterKeyRejectedError) as exc:
            return _router_error_response(exc)

        body = await request.body()
        headers = _forwarding_headers(dict(request.headers))
        resp = await forward_request(
            f"{worker_addr}/rl/set_reward", body, headers, config.forward_timeout
        )
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type"),
        )

    # =========================================================================
    # POST /pause_generation — admin key ONLY, broadcast
    # =========================================================================

    @app.post("/pause_generation")
    async def pause_generation(request: Request):
        require_admin_key(request, config.admin_api_key)
        try:
            worker_addrs = await get_all_worker_addrs(
                config.router_addr, config.admin_api_key, config.router_timeout
            )
        except RouterUnreachableError as exc:
            return _router_error_response(exc)

        body = await request.body()
        headers = _forwarding_headers(dict(request.headers))
        results = await broadcast_to_workers(
            worker_addrs, "/pause_generation", body, headers
        )
        return {"results": results}

    # =========================================================================
    # POST /continue_generation — admin key ONLY, broadcast
    # =========================================================================

    @app.post("/continue_generation")
    async def continue_generation(request: Request):
        require_admin_key(request, config.admin_api_key)
        try:
            worker_addrs = await get_all_worker_addrs(
                config.router_addr, config.admin_api_key, config.router_timeout
            )
        except RouterUnreachableError as exc:
            return _router_error_response(exc)

        body = await request.body()
        headers = _forwarding_headers(dict(request.headers))
        results = await broadcast_to_workers(
            worker_addrs, "/continue_generation", body, headers
        )
        return {"results": results}

    # =========================================================================
    # POST /export_trajectories — admin key ONLY, route by session_id
    # =========================================================================

    @app.post("/export_trajectories")
    async def export_trajectories(request: Request):
        require_admin_key(request, config.admin_api_key)

        body = await request.body()

        # Parse body to extract session_id for routing
        try:
            body_json = json.loads(body)
            session_id = body_json.get("session_id")
        except (json.JSONDecodeError, AttributeError):
            return JSONResponse(
                {"error": "Invalid JSON body or missing session_id"},
                status_code=400,
            )

        if not session_id:
            return JSONResponse({"error": "session_id is required"}, status_code=400)

        try:
            worker_addr = await query_router_by_session_id(
                config.router_addr, session_id, config.router_timeout
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
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type"),
        )

    # =========================================================================
    # Weight update broadcasts — admin key ONLY
    # =========================================================================

    @app.post("/update_weights_from_disk")
    async def update_weights_from_disk(request: Request):
        require_admin_key(request, config.admin_api_key)
        try:
            worker_addrs = await get_all_worker_addrs(
                config.router_addr, config.admin_api_key, config.router_timeout
            )
        except RouterUnreachableError as exc:
            return _router_error_response(exc)

        body = await request.body()
        headers = _forwarding_headers(dict(request.headers))
        results = await broadcast_to_workers(
            worker_addrs, "/update_weights_from_disk", body, headers
        )
        return {"results": results}

    @app.post("/update_weights_from_distributed")
    async def update_weights_from_distributed(request: Request):
        require_admin_key(request, config.admin_api_key)
        try:
            worker_addrs = await get_all_worker_addrs(
                config.router_addr, config.admin_api_key, config.router_timeout
            )
        except RouterUnreachableError as exc:
            return _router_error_response(exc)

        body = await request.body()
        headers = _forwarding_headers(dict(request.headers))
        results = await broadcast_to_workers(
            worker_addrs, "/update_weights_from_distributed", body, headers
        )
        return {"results": results}

    @app.post("/init_weights_update_group")
    async def init_weights_update_group(request: Request):
        require_admin_key(request, config.admin_api_key)
        try:
            worker_addrs = await get_all_worker_addrs(
                config.router_addr, config.admin_api_key, config.router_timeout
            )
        except RouterUnreachableError as exc:
            return _router_error_response(exc)

        body = await request.body()
        headers = _forwarding_headers(dict(request.headers))
        results = await broadcast_to_workers(
            worker_addrs, "/init_weights_update_group", body, headers
        )
        return {"results": results}

    @app.post("/set_version")
    async def set_version(request: Request):
        require_admin_key(request, config.admin_api_key)
        try:
            worker_addrs = await get_all_worker_addrs(
                config.router_addr, config.admin_api_key, config.router_timeout
            )
        except RouterUnreachableError as exc:
            return _router_error_response(exc)

        body = await request.body()
        headers = _forwarding_headers(dict(request.headers))
        results = await broadcast_to_workers(
            worker_addrs, "/set_version", body, headers
        )
        return {"results": results}

    @app.post("/grant_capacity")
    async def grant_capacity(request: Request):
        require_admin_key(request, config.admin_api_key)
        try:
            worker_addrs = await get_all_worker_addrs(
                config.router_addr, config.admin_api_key, config.router_timeout
            )
        except RouterUnreachableError as exc:
            return _router_error_response(exc)

        body = await request.body()
        headers = _forwarding_headers(dict(request.headers))
        results = await broadcast_to_workers(
            worker_addrs, "/grant_capacity", body, headers
        )
        return {"results": results}


    # =========================================================================
    # Compatibility aliases for RolloutCallback — map /callback/* to broadcast endpoints
    # =========================================================================
    # RolloutCallback (shared infrastructure in areal/infra/controller/rollout_callback.py)
    # uses /callback/* prefixed paths for weight updates and generation control.
    # Gateway implements the actual handlers at unprefixed paths. These aliases
    # register the SAME handler functions on both routes to bridge the gap without
    # modifying the shared RolloutCallback class.

    # POST /callback/init_weights_group → init_weights_update_group
    app.add_api_route(
        "/callback/init_weights_group",
        init_weights_update_group,
        methods=["POST"],
    )

    # POST /callback/update_weights_xccl → update_weights_from_distributed
    app.add_api_route(
        "/callback/update_weights_xccl",
        update_weights_from_distributed,
        methods=["POST"],
    )

    # POST /callback/update_weights_disk → update_weights_from_disk
    app.add_api_route(
        "/callback/update_weights_disk",
        update_weights_from_disk,
        methods=["POST"],
    )

    # POST /callback/pause_generation → pause_generation
    app.add_api_route(
        "/callback/pause_generation",
        pause_generation,
        methods=["POST"],
    )

    # POST /callback/continue_generation → continue_generation
    app.add_api_route(
        "/callback/continue_generation",
        continue_generation,
        methods=["POST"],
    )
    return app
