"""Agent Router — independent HTTP service for session-affine routing.

DataProxy instances register with the Router at startup.  The Gateway
calls ``POST /route`` to discover which DataProxy owns a given session.
New sessions are assigned round-robin across registered proxies.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
from fastapi import FastAPI

from areal.utils import logging

logger = logging.getLogger("AgentRouter")


def create_router_app() -> FastAPI:
    """Create the Agent Router HTTP application."""
    app = FastAPI(title="AReaL Agent Router")

    registered_proxies: list[str] = []
    session_map: dict[str, str] = {}
    rr_idx = 0
    lock = asyncio.Lock()

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "registered_proxies": len(registered_proxies),
            "active_sessions": len(session_map),
        }

    @app.post("/register")
    async def register(body: dict[str, Any]):
        addr = body["addr"]
        async with lock:
            if addr not in registered_proxies:
                registered_proxies.append(addr)
                logger.info(
                    "Registered DataProxy: %s (total=%d)", addr, len(registered_proxies)
                )
        return {"status": "ok"}

    @app.post("/unregister")
    async def unregister(body: dict[str, Any]):
        addr = body["addr"]
        async with lock:
            if addr in registered_proxies:
                registered_proxies.remove(addr)
                stale = [k for k, v in session_map.items() if v == addr]
                for k in stale:
                    del session_map[k]
                logger.info(
                    "Unregistered DataProxy: %s (removed %d sessions)", addr, len(stale)
                )
        return {"status": "ok"}

    @app.post("/route")
    async def route(body: dict[str, Any]):
        nonlocal rr_idx
        session_key = body["session_key"]

        async with lock:
            if session_key in session_map:
                return {"data_proxy_addr": session_map[session_key]}

            if not registered_proxies:
                return {"error": "No DataProxy registered"}, 503

            addr = registered_proxies[rr_idx % len(registered_proxies)]
            rr_idx += 1
            session_map[session_key] = addr
            logger.info("Routed session %s → %s", session_key, addr)

        return {"data_proxy_addr": addr}

    @app.post("/remove_session")
    async def remove_session(body: dict[str, Any]):
        session_key = body["session_key"]
        async with lock:
            session_map.pop(session_key, None)
        return {"status": "ok"}

    return app


class RouterClient:
    """HTTP client for calling the Router service."""

    def __init__(self, router_addr: str) -> None:
        self._addr = router_addr
        self._http = httpx.AsyncClient(timeout=30.0)

    async def register(self, addr: str) -> None:
        resp = await self._http.post(f"{self._addr}/register", json={"addr": addr})
        resp.raise_for_status()

    async def unregister(self, addr: str) -> None:
        resp = await self._http.post(f"{self._addr}/unregister", json={"addr": addr})
        resp.raise_for_status()

    async def route(self, session_key: str) -> str:
        resp = await self._http.post(
            f"{self._addr}/route", json={"session_key": session_key}
        )
        resp.raise_for_status()
        return resp.json()["data_proxy_addr"]

    async def remove_session(self, session_key: str) -> None:
        resp = await self._http.post(
            f"{self._addr}/remove_session", json={"session_key": session_key}
        )
        resp.raise_for_status()

    async def close(self) -> None:
        await self._http.aclose()


def main() -> None:
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Agent Router")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8081)
    args = parser.parse_args()

    uvicorn.run(create_router_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
