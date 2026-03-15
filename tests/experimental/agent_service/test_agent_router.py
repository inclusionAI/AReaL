"""Tests for Agent Router HTTP server."""

from __future__ import annotations

import pytest

from areal.experimental.agent_service.agent_router import create_router_app

httpx = pytest.importorskip("httpx")


def _make_client():
    app = create_router_app()
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://router")


class TestRouterHealth:
    @pytest.mark.asyncio
    async def test_health(self):
        async with _make_client() as client:
            resp = await client.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "ok"
            assert data["registered_proxies"] == 0


class TestRegistration:
    @pytest.mark.asyncio
    async def test_register_and_health(self):
        async with _make_client() as client:
            await client.post("/register", json={"addr": "http://proxy1:9100"})
            resp = await client.get("/health")
            assert resp.json()["registered_proxies"] == 1

    @pytest.mark.asyncio
    async def test_unregister(self):
        async with _make_client() as client:
            await client.post("/register", json={"addr": "http://proxy1:9100"})
            await client.post("/unregister", json={"addr": "http://proxy1:9100"})
            resp = await client.get("/health")
            assert resp.json()["registered_proxies"] == 0


class TestRouting:
    @pytest.mark.asyncio
    async def test_route_new_session(self):
        async with _make_client() as client:
            await client.post("/register", json={"addr": "http://proxy1:9100"})
            resp = await client.post("/route", json={"session_key": "s1"})
            assert resp.json()["data_proxy_addr"] == "http://proxy1:9100"

    @pytest.mark.asyncio
    async def test_route_existing_session_returns_same(self):
        async with _make_client() as client:
            await client.post("/register", json={"addr": "http://proxy1:9100"})
            await client.post("/register", json={"addr": "http://proxy2:9101"})
            r1 = await client.post("/route", json={"session_key": "s1"})
            r2 = await client.post("/route", json={"session_key": "s1"})
            assert r1.json()["data_proxy_addr"] == r2.json()["data_proxy_addr"]

    @pytest.mark.asyncio
    async def test_round_robin(self):
        async with _make_client() as client:
            await client.post("/register", json={"addr": "http://p1"})
            await client.post("/register", json={"addr": "http://p2"})
            r1 = await client.post("/route", json={"session_key": "a"})
            r2 = await client.post("/route", json={"session_key": "b"})
            addrs = {r1.json()["data_proxy_addr"], r2.json()["data_proxy_addr"]}
            assert addrs == {"http://p1", "http://p2"}

    @pytest.mark.asyncio
    async def test_remove_session(self):
        async with _make_client() as client:
            await client.post("/register", json={"addr": "http://p1"})
            await client.post("/route", json={"session_key": "s1"})
            await client.post("/remove_session", json={"session_key": "s1"})
            health = await client.get("/health")
            assert health.json()["active_sessions"] == 0
