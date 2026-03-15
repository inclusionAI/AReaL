"""Data Proxy — stateful session proxy between Gateway and Worker.

Each DataProxy is paired 1:1 with an Agent Worker.  It maintains
per-session conversation history and forwards requests to the Worker
with the accumulated context.  The Gateway never talks to Workers
directly — all traffic flows through DataProxy.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
from fastapi import FastAPI

from areal.utils import logging

logger = logging.getLogger("DataProxy")


@dataclass
class _SessionData:
    history: list[dict[str, str]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    last_active: float = field(default_factory=time.monotonic)


def create_data_proxy_app(
    worker_addr: str,
    session_timeout: int = 3600,
) -> FastAPI:
    """Create the DataProxy HTTP application.

    Parameters
    ----------
    worker_addr : str
        HTTP address of the paired Agent Worker (e.g. ``http://localhost:9000``).
    session_timeout : int
        Idle timeout in seconds before a session is reaped.
    """
    app = FastAPI(title="AReaL Data Proxy")
    sessions: dict[str, _SessionData] = {}
    http_client = httpx.AsyncClient(timeout=600.0)

    async def _reap_idle_sessions() -> None:
        while True:
            await asyncio.sleep(60)
            now = time.monotonic()
            stale = [
                k for k, s in sessions.items() if now - s.last_active > session_timeout
            ]
            for k in stale:
                del sessions[k]
            if stale:
                logger.info("Reaped %d idle sessions", len(stale))

    @app.on_event("startup")
    async def startup():
        asyncio.create_task(_reap_idle_sessions())

    @app.on_event("shutdown")
    async def shutdown():
        await http_client.aclose()

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "active_sessions": len(sessions),
            "worker_addr": worker_addr,
        }

    @app.post("/session/{session_key}/turn")
    async def turn(session_key: str, body: dict[str, Any]):
        session = sessions.get(session_key)
        if session is None:
            session = _SessionData()
            sessions[session_key] = session

        message = body.get("message", "")
        run_id = body.get("run_id", "")
        queue_mode = body.get("queue_mode", "collect")
        metadata = body.get("metadata", {})

        worker_request = {
            "message": message,
            "session_key": session_key,
            "run_id": run_id,
            "history": session.history.copy(),
            "queue_mode": queue_mode,
            "metadata": metadata,
        }

        resp = await http_client.post(f"{worker_addr}/run", json=worker_request)
        resp.raise_for_status()
        result = resp.json()

        session.history.append({"role": "user", "content": message})

        for evt in result.get("events", []):
            if evt.get("type") == "tool_call":
                session.history.append(
                    {
                        "role": "assistant",
                        "content": f"[tool_call] {evt.get('name', '')}: {evt.get('args', '')}",
                    }
                )
            elif evt.get("type") == "tool_result":
                session.history.append(
                    {
                        "role": "tool",
                        "content": f"[tool_result] {evt.get('name', '')}: {evt.get('result', '')}",
                    }
                )

        summary = result.get("summary", "")
        if summary:
            session.history.append({"role": "assistant", "content": summary})

        session.last_active = time.monotonic()
        return result

    @app.post("/session/{session_key}/close")
    async def close_session(session_key: str):
        sessions.pop(session_key, None)
        return {"status": "ok"}

    @app.get("/session/{session_key}/history")
    async def get_history(session_key: str):
        session = sessions.get(session_key)
        if session is None:
            return {"history": []}
        return {"history": session.history}

    return app


class DataProxyClient:
    """HTTP client for calling a DataProxy service."""

    def __init__(self, data_proxy_addr: str) -> None:
        self._addr = data_proxy_addr
        self._http = httpx.AsyncClient(timeout=600.0)

    async def turn(
        self,
        session_key: str,
        message: str,
        run_id: str = "",
        queue_mode: str = "collect",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        resp = await self._http.post(
            f"{self._addr}/session/{session_key}/turn",
            json={
                "message": message,
                "run_id": run_id,
                "queue_mode": queue_mode,
                "metadata": metadata or {},
            },
        )
        resp.raise_for_status()
        return resp.json()

    async def close_session(self, session_key: str) -> None:
        resp = await self._http.post(
            f"{self._addr}/session/{session_key}/close",
        )
        resp.raise_for_status()

    async def get_history(self, session_key: str) -> list[dict[str, str]]:
        resp = await self._http.get(
            f"{self._addr}/session/{session_key}/history",
        )
        resp.raise_for_status()
        return resp.json()["history"]

    async def close(self) -> None:
        await self._http.aclose()
