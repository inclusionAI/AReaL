from __future__ import annotations

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from areal.experimental.training_service.gateway.auth import extract_bearer_token
from areal.experimental.training_service.gateway.config import GatewayConfig
from areal.experimental.training_service.gateway.engine import register_engine_routes
from areal.experimental.training_service.gateway.streaming import (
    RouterKeyRejectedError,
    RouterUnreachableError,
    _forwarding_headers,
    forward_request,
    query_router,
)
from areal.utils import logging

logger = logging.getLogger("TrainGateway")


def _router_error_response(exc: Exception) -> JSONResponse:
    if isinstance(exc, RouterUnreachableError):
        return JSONResponse({"error": str(exc)}, status_code=502)
    if isinstance(exc, RouterKeyRejectedError):
        status = 401 if exc.status_code == 404 else exc.status_code
        return JSONResponse({"error": exc.detail}, status_code=status)
    return JSONResponse({"error": str(exc)}, status_code=500)


async def _forward_post(
    request: Request,
    path: str,
    config: GatewayConfig,
    *,
    use_admin_auth_for_upstream: bool = False,
) -> Response:
    token = extract_bearer_token(request)
    try:
        model_addr = await query_router(
            config.router_addr,
            token,
            config.router_timeout,
            admin_api_key=config.admin_api_key,
        )
    except (RouterUnreachableError, RouterKeyRejectedError) as exc:
        return _router_error_response(exc)

    body = await request.body()
    headers = _forwarding_headers(dict(request.headers))
    if use_admin_auth_for_upstream:
        for key in list(headers.keys()):
            if key.lower() == "authorization":
                headers.pop(key)
        headers["Authorization"] = f"Bearer {config.admin_api_key}"
    try:
        resp = await forward_request(
            f"{model_addr}{path}",
            body,
            headers,
            config.forward_timeout,
        )
    except Exception as exc:
        logger.error("Forwarding POST failed for %s: %s", path, exc)
        return JSONResponse({"error": str(exc)}, status_code=502)
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type"),
    )


async def _forward_get(request: Request, path: str, config: GatewayConfig) -> Response:
    token = extract_bearer_token(request)
    try:
        model_addr = await query_router(
            config.router_addr,
            token,
            config.router_timeout,
            admin_api_key=config.admin_api_key,
        )
    except (RouterUnreachableError, RouterKeyRejectedError) as exc:
        return _router_error_response(exc)

    try:
        async with httpx.AsyncClient(timeout=config.forward_timeout) as client:
            resp = await client.get(
                f"{model_addr}{path}",
                headers=_forwarding_headers(dict(request.headers)),
            )
    except Exception as exc:
        logger.error("Forwarding GET failed for %s: %s", path, exc)
        return JSONResponse({"error": str(exc)}, status_code=502)
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type"),
    )


def create_app(config: GatewayConfig) -> FastAPI:
    app = FastAPI(title="AReaL Training Gateway")

    register_engine_routes(
        app,
        config,
        _forward_post=_forward_post,
        _forward_get=_forward_get,
    )

    return app
