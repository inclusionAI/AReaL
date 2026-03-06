"""Weight-update forwarding endpoints for the data proxy.

Each endpoint receives an HTTP request from the gateway and forwards it
to the co-located SGLang/vLLM server.  The data proxy itself does NOT
participate in NCCL — it is a pure HTTP-to-HTTP relay.
"""

from __future__ import annotations

import logging
from typing import Any

import aiohttp
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("DataProxy")


async def _forward_to_backend(
    backend_addr: str,
    endpoint: str,
    payload: dict[str, Any],
    timeout: float = 120.0,
) -> JSONResponse:
    """Forward a JSON payload to the co-located backend server."""
    url = f"{backend_addr}{endpoint}"
    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=timeout)
        ) as session:
            async with session.post(url, json=payload) as resp:
                body = await resp.json()
                return JSONResponse(content=body, status_code=resp.status)
    except aiohttp.ClientError as exc:
        logger.error("Failed to forward %s to %s: %s", endpoint, backend_addr, exc)
        raise HTTPException(status_code=502, detail=f"Backend unreachable: {exc}")
    except Exception as exc:
        logger.error("Unexpected error forwarding %s: %s", endpoint, exc)
        raise HTTPException(status_code=500, detail=str(exc))


async def update_weights_from_disk(request: Request) -> JSONResponse:
    """Forward a disk-based weight update request to the SGLang server."""
    backend_addr: str = request.app.state.config.backend_addr
    timeout: float = request.app.state.config.request_timeout
    payload = await request.json()
    return await _forward_to_backend(
        backend_addr, "/update_weights_from_disk", payload, timeout
    )


async def update_weights_from_distributed(request: Request) -> JSONResponse:
    """Forward a distributed (NCCL/XCCL) weight update request."""
    backend_addr: str = request.app.state.config.backend_addr
    timeout: float = request.app.state.config.request_timeout
    payload = await request.json()
    return await _forward_to_backend(
        backend_addr, "/update_weights_from_distributed", payload, timeout
    )


async def init_weights_update_group(request: Request) -> JSONResponse:
    """Forward an NCCL group initialization request."""
    backend_addr: str = request.app.state.config.backend_addr
    timeout: float = request.app.state.config.request_timeout
    payload = await request.json()
    return await _forward_to_backend(
        backend_addr, "/init_weights_update_group", payload, timeout
    )


async def set_version(request: Request) -> JSONResponse:
    """Forward a set_version request and update local backend version tracking."""
    backend_addr: str = request.app.state.config.backend_addr
    timeout: float = request.app.state.config.request_timeout
    payload = await request.json()
    # Also update local backend version if the backend supports it
    return await _forward_to_backend(backend_addr, "/set_version", payload, timeout)
