"""Unit tests for training-service gateway."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest
import pytest_asyncio

from areal.experimental.training_service.gateway.app import create_app
from areal.experimental.training_service.gateway.config import GatewayConfig
from areal.experimental.training_service.gateway.streaming import (
    RouterKeyRejectedError,
    RouterUnreachableError,
)

MODULE = "areal.experimental.training_service.gateway.app"
ADMIN_KEY = "test-admin-key"
SESSION_KEY = "session-key"
WORKER_ADDR = "http://mock-worker:18082"


@pytest.fixture
def config() -> GatewayConfig:
    return GatewayConfig(
        host="127.0.0.1",
        port=18080,
        router_addr="http://mock-router:18081",
        admin_api_key=ADMIN_KEY,
        router_timeout=2.0,
        forward_timeout=20.0,
    )


@pytest_asyncio.fixture
async def client(config):
    app = create_app(config)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


class TestGatewayHealth:
    @pytest.mark.asyncio
    async def test_health_reports_router(self, client, config):
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["router_addr"] == config.router_addr


class TestGatewayRoutingAndForwarding:
    @pytest.mark.asyncio
    async def test_missing_bearer_token_returns_401(self, client):
        resp = await client.post("/train_batch", json={"args": [], "kwargs": {}})
        assert resp.status_code == 401

    @pytest.mark.asyncio
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    async def test_router_unreachable_maps_to_502(self, mock_query_router, client):
        mock_query_router.side_effect = RouterUnreachableError("router unavailable")

        resp = await client.post(
            "/train_batch",
            json={"args": [], "kwargs": {}},
            headers={"Authorization": f"Bearer {SESSION_KEY}"},
        )
        assert resp.status_code == 502

    @pytest.mark.asyncio
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    async def test_router_404_key_rejected_maps_to_401(self, mock_query_router, client):
        mock_query_router.side_effect = RouterKeyRejectedError("unknown key", 404)

        resp = await client.post(
            "/eval_batch",
            json={"args": [], "kwargs": {}},
            headers={"Authorization": f"Bearer {SESSION_KEY}"},
        )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    @patch(f"{MODULE}.forward_request", new_callable=AsyncMock)
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    async def test_forward_batch_forwards_response(
        self,
        mock_query_router,
        mock_forward_request,
        client,
    ):
        mock_query_router.return_value = WORKER_ADDR
        mock_forward_request.return_value = httpx.Response(
            200,
            json={"status": "success", "result": {"ok": True}},
        )

        resp = await client.post(
            "/forward_batch",
            json={"args": [], "kwargs": {}},
            headers={"Authorization": f"Bearer {SESSION_KEY}"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"

    @pytest.mark.asyncio
    @patch(f"{MODULE}.forward_request", new_callable=AsyncMock)
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    async def test_ppo_actor_compute_logp_forwards_response(
        self,
        mock_query_router,
        mock_forward_request,
        client,
    ):
        mock_query_router.return_value = WORKER_ADDR
        mock_forward_request.return_value = httpx.Response(
            200,
            json={"status": "success", "result": {"ok": True}},
        )

        resp = await client.post(
            "/ppo/actor/compute_logp",
            json={"args": [], "kwargs": {}},
            headers={"Authorization": f"Bearer {SESSION_KEY}"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"

    @pytest.mark.asyncio
    @patch(f"{MODULE}.forward_request", new_callable=AsyncMock)
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    async def test_sft_train_forwards_response(
        self,
        mock_query_router,
        mock_forward_request,
        client,
    ):
        mock_query_router.return_value = WORKER_ADDR
        mock_forward_request.return_value = httpx.Response(
            200,
            json={"status": "success", "result": {"ok": True}},
        )

        resp = await client.post(
            "/sft/train",
            json={"args": [], "kwargs": {}},
            headers={"Authorization": f"Bearer {SESSION_KEY}"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"

    @pytest.mark.asyncio
    @patch(f"{MODULE}.forward_request", new_callable=AsyncMock)
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    async def test_offload_uses_admin_auth_upstream(
        self,
        mock_query_router,
        mock_forward_request,
        client,
    ):
        mock_query_router.return_value = WORKER_ADDR

        async def _check_forward(_url, _body, headers, _timeout):
            assert headers["Authorization"] == f"Bearer {ADMIN_KEY}"
            return httpx.Response(200, json={"status": "success", "result": None})

        mock_forward_request.side_effect = _check_forward

        resp = await client.post(
            "/offload",
            json={"args": [], "kwargs": {}},
            headers={"Authorization": f"Bearer {SESSION_KEY}"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"
