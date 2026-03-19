"""Router communication and request forwarding utilities for the gateway."""

from __future__ import annotations

import asyncio
from typing import Any, AsyncGenerator

import httpx

from areal.utils import logging

logger = logging.getLogger("RolloutGateway")


class RouterUnreachableError(Exception):
    """Router service is unreachable or returned an error."""

    pass


class RouterKeyRejectedError(Exception):
    """Router rejected the API key (unknown key) or has no healthy workers."""

    def __init__(self, detail: str, status_code: int = 404):
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


async def query_router(
    router_addr: str,
    api_key: str | None = None,
    path: str | None = None,
    timeout: float = 2.0,
    *,
    session_id: str | None = None,
) -> str:
    """Ask the Router for a worker address.

    POST ``{router_addr}/route`` with ``{"api_key": ..., "path": ...}``
    or ``{"session_id": ...}``.
    Returns the ``worker_addr`` string.

    Raises
    ------
    RouterUnreachableError
        Router connection failed.
    RouterKeyRejectedError
        Router returned 404 (unknown key / session) or 503 (no healthy workers).
    """
    payload: dict[str, str] = {}
    if session_id is not None:
        payload["session_id"] = session_id
    else:
        if api_key is not None:
            payload["api_key"] = api_key
        if path is not None:
            payload["path"] = path
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                f"{router_addr}/route",
                json=payload,
            )
        if resp.status_code == 404:
            data = resp.json()
            raise RouterKeyRejectedError(
                data.get("detail", data.get("error", "Not found")), 404
            )
        if resp.status_code == 503:
            data = resp.json()
            raise RouterKeyRejectedError(
                data.get("detail", data.get("error", "No healthy workers")), 503
            )
        resp.raise_for_status()
        return resp.json()["worker_addr"]
    except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
        raise RouterUnreachableError(f"Router unreachable: {exc}") from exc


async def register_session_in_router(
    router_addr: str,
    session_api_key: str,
    session_id: str,
    worker_addr: str,
    timeout: float,
) -> None:
    """Register a session→worker mapping in the Router.

    POST ``{router_addr}/register_session``.
    Called by the gateway after intercepting ``/rl/start_session`` response.
    """
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                f"{router_addr}/register_session",
                json={
                    "session_api_key": session_api_key,
                    "session_id": session_id,
                    "worker_addr": worker_addr,
                },
            )
        resp.raise_for_status()
    except Exception as exc:
        logger.error("Failed to register session in router: %s", exc)
        raise RouterUnreachableError(f"Failed to register session: {exc}") from exc


async def grant_capacity_in_router(
    router_addr: str,
    admin_api_key: str,
    timeout: float,
) -> dict:
    """Forward a grant_capacity request to the Router.

    POST ``{router_addr}/grant_capacity`` with admin key auth.
    Returns the JSON response body from the router.
    """
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                f"{router_addr}/grant_capacity",
                headers={"Authorization": f"Bearer {admin_api_key}"},
            )
        resp.raise_for_status()
        return resp.json()
    except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
        raise RouterUnreachableError(
            f"Router unreachable for grant_capacity: {exc}"
        ) from exc
    except Exception as exc:
        raise RouterUnreachableError(
            f"Failed to grant capacity in router: {exc}"
        ) from exc


class CapacityExhaustedError(Exception):
    """Router has no capacity permits remaining (HTTP 429)."""

    def __init__(self, detail: str = "No available capacity to start a new session"):
        super().__init__(detail)
        self.detail = detail


async def acquire_capacity_in_router(
    router_addr: str,
    admin_api_key: str,
    timeout: float,
) -> None:
    """Acquire one capacity permit from the Router.

    POST ``{router_addr}/acquire_capacity`` with admin key auth.
    Raises ``CapacityExhaustedError`` if the router returns 429.
    """
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                f"{router_addr}/acquire_capacity",
                headers={"Authorization": f"Bearer {admin_api_key}"},
            )
        if resp.status_code == 429:
            data = resp.json()
            raise CapacityExhaustedError(
                data.get("detail", "No available capacity to start a new session")
            )
        resp.raise_for_status()
    except CapacityExhaustedError:
        raise
    except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
        raise RouterUnreachableError(
            f"Router unreachable for acquire_capacity: {exc}"
        ) from exc
    except Exception as exc:
        raise RouterUnreachableError(
            f"Failed to acquire capacity from router: {exc}"
        ) from exc


async def release_capacity_in_router(
    router_addr: str,
    admin_api_key: str,
    timeout: float,
) -> None:
    """Return one capacity permit to the Router.

    POST ``{router_addr}/release_capacity`` with admin key auth.
    Best-effort: logs errors but never raises, since the caller is
    already handling a failure path.
    """
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                f"{router_addr}/release_capacity",
                headers={"Authorization": f"Bearer {admin_api_key}"},
            )
        if resp.status_code != 200:
            logger.warning(
                "release_capacity returned %d: %s", resp.status_code, resp.text
            )
    except Exception as exc:
        logger.warning("Failed to release capacity in router: %s", exc)


async def get_all_worker_addrs(
    router_addr: str,
    admin_api_key: str,
    timeout: float,
) -> list[str]:
    """Fetch all worker addresses from the Router (for broadcast).

    GET ``{router_addr}/workers`` with admin key auth.
    """
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(
                f"{router_addr}/workers",
                headers={"Authorization": f"Bearer {admin_api_key}"},
            )
        resp.raise_for_status()
        data = resp.json()
        return [w["addr"] for w in data.get("workers", [])]
    except Exception as exc:
        raise RouterUnreachableError(f"Failed to get workers: {exc}") from exc


async def resolve_worker_addr(
    router_addr: str,
    admin_api_key: str,
    worker_id: str,
    timeout: float,
) -> str:
    """Resolve a worker_id to its address via the Router.

    GET ``{router_addr}/resolve_worker/{worker_id}`` with admin key auth.

    Raises
    ------
    RouterKeyRejectedError
        Worker ID not found (404).
    RouterUnreachableError
        Router connection failed.
    """
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(
                f"{router_addr}/resolve_worker/{worker_id}",
                headers={"Authorization": f"Bearer {admin_api_key}"},
            )
        if resp.status_code == 404:
            data = resp.json()
            raise RouterKeyRejectedError(
                data.get("detail", f"Worker ID {worker_id} not found"), 404
            )
        resp.raise_for_status()
        return resp.json()["worker_addr"]
    except RouterKeyRejectedError:
        raise
    except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
        raise RouterUnreachableError(
            f"Router unreachable for resolve_worker: {exc}"
        ) from exc
    except Exception as exc:
        raise RouterUnreachableError(
            f"Failed to resolve worker {worker_id}: {exc}"
        ) from exc


def _forwarding_headers(raw_headers: dict[str, str]) -> dict[str, str]:
    """Build headers to forward to data proxy.

    Strips hop-by-hop headers (``host``, ``content-length``,
    ``transfer-encoding``) that are incompatible with proxied requests.
    """
    skip = {"host", "content-length", "transfer-encoding"}
    return {k: v for k, v in raw_headers.items() if k.lower() not in skip}


async def forward_sse_stream(
    upstream_url: str,
    body: bytes,
    headers: dict[str, str],
    timeout: float | None = None,
) -> AsyncGenerator[bytes, None]:
    """True SSE streaming proxy — yields bytes as they arrive from upstream.

    Uses ``httpx.AsyncClient.stream()`` so the client sees tokens
    as soon as the data proxy emits them.

    On upstream HTTP errors or mid-stream failures, an SSE error event
    (``data: {"error": ...}``) is emitted so clients can distinguish a
    clean end-of-stream from a backend failure.
    """
    import json as _json

    fwd_headers = _forwarding_headers(headers)
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
            async with client.stream(
                "POST", upstream_url, content=body, headers=fwd_headers
            ) as resp:
                if resp.status_code != 200:
                    # Upstream returned a non-200 status — read body and emit
                    # an SSE error event so the client sees the failure.
                    error_body = await resp.aread()
                    try:
                        detail = _json.loads(error_body)
                    except Exception:
                        detail = error_body.decode("utf-8", errors="replace")
                    error_event = _json.dumps(
                        {"error": detail, "status_code": resp.status_code}
                    )
                    yield f"data: {error_event}\n\n".encode()
                    return
                async for chunk in resp.aiter_bytes():
                    yield chunk
    except Exception as exc:
        logger.error("SSE stream error from %s: %s", upstream_url, exc)
        error_event = _json.dumps({"error": str(exc)})
        yield f"data: {error_event}\n\n".encode()


async def forward_request(
    upstream_url: str,
    body: bytes,
    headers: dict[str, str],
    timeout: float = 120.0,
) -> httpx.Response:
    """Forward a non-streaming request to upstream, return full response."""
    fwd_headers = _forwarding_headers(headers)
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(upstream_url, content=body, headers=fwd_headers)
    return resp


async def broadcast_to_workers(
    worker_addrs: list[str],
    path: str,
    body: bytes,
    headers: dict[str, str],
    timeout: float = 10.0,
) -> list[dict[str, Any]]:
    """Broadcast a request to all workers (best-effort).

    Returns a list of per-worker result dicts::

        [{"worker_addr": "...", "status": 200, "ok": True}, ...]

    Failed workers get ``ok=False`` with an ``error`` field.
    """

    async def _call(addr: str) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(
                    f"{addr}{path}",
                    content=body,
                    headers=_forwarding_headers(headers),
                )
            return {
                "worker_addr": addr,
                "status": resp.status_code,
                "ok": resp.status_code < 400,
            }
        except Exception as exc:
            return {
                "worker_addr": addr,
                "status": 502,
                "ok": False,
                "error": str(exc),
            }

    tasks = [_call(addr) for addr in worker_addrs]
    results = await asyncio.gather(*tasks)
    return list(results)
