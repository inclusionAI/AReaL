from __future__ import annotations

import httpx


class RouterClient:
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
