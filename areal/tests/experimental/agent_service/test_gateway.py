"""Unit tests for the Agent Service Gateway."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from areal.experimental.agent_service.config import GatewayConfig
from areal.experimental.agent_service.gateway import Gateway
from areal.experimental.agent_service.schemas import (
    DispatchResponse,
)


def _make_gateway(
    queue_size: int = 100,
    worker_timeout: float = 5.0,
) -> Gateway:
    """Create a Gateway instance with test-friendly defaults."""
    config = GatewayConfig(
        queue_size=queue_size,
        worker_timeout=worker_timeout,
    )
    return Gateway(config=config)


class TestGatewayHealth:
    """Tests for GET /health endpoint."""

    @pytest.mark.asyncio
    async def test_health_returns_ok_when_initialized(self):
        """GET /health should return status='ok' after Gateway starts."""
        gateway = _make_gateway()
        app = gateway.create_app()

        async with app.router.lifespan_context(app):
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                response = await client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "workers" in data
        assert data["workers"]["total"] == 0

    @pytest.mark.asyncio
    async def test_health_reflects_registered_workers(self):
        """GET /health should show correct worker count after registration."""
        gateway = _make_gateway()
        app = gateway.create_app()

        async with app.router.lifespan_context(app):
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                # Register a worker
                await client.post(
                    "/register_worker",
                    json={"worker_id": "w1", "address": "http://localhost:9001"},
                )
                response = await client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["workers"]["total"] == 1
        assert data["workers"]["idle"] == 1


class TestGatewayWorkerRegistration:
    """Tests for /register_worker and /unregister_worker endpoints."""

    @pytest.mark.asyncio
    async def test_register_worker_adds_to_pool(self):
        """POST /register_worker should add worker to pool."""
        gateway = _make_gateway()
        app = gateway.create_app()

        async with app.router.lifespan_context(app):
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                response = await client.post(
                    "/register_worker",
                    json={"worker_id": "w1", "address": "http://localhost:9001"},
                )
                assert response.status_code == 200
                assert response.json()["status"] == "ok"

                # Verify via /workers
                workers_resp = await client.get("/workers")
                workers = workers_resp.json()
                assert len(workers) == 1
                assert workers[0]["worker_id"] == "w1"
                assert workers[0]["address"] == "http://localhost:9001"

    @pytest.mark.asyncio
    async def test_unregister_worker_removes_from_pool(self):
        """POST /unregister_worker should remove worker from pool."""
        gateway = _make_gateway()
        app = gateway.create_app()

        async with app.router.lifespan_context(app):
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                # Register then unregister
                await client.post(
                    "/register_worker",
                    json={"worker_id": "w1", "address": "http://localhost:9001"},
                )
                response = await client.post(
                    "/unregister_worker",
                    json={"worker_id": "w1", "address": "http://localhost:9001"},
                )
                assert response.status_code == 200

                workers_resp = await client.get("/workers")
                assert workers_resp.json() == []

    @pytest.mark.asyncio
    async def test_list_workers_returns_all_registered(self):
        """GET /workers should list all registered workers."""
        gateway = _make_gateway()
        app = gateway.create_app()

        async with app.router.lifespan_context(app):
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                for i in range(3):
                    await client.post(
                        "/register_worker",
                        json={
                            "worker_id": f"w{i}",
                            "address": f"http://localhost:900{i}",
                        },
                    )

                response = await client.get("/workers")
                workers = response.json()
                assert len(workers) == 3
                worker_ids = {w["worker_id"] for w in workers}
                assert worker_ids == {"w0", "w1", "w2"}


class TestGatewayRunEpisode:
    """Tests for POST /run_episode endpoint."""

    @pytest.mark.asyncio
    async def test_run_episode_returns_503_when_queue_full(self):
        """POST /run_episode should return 503 when the request queue is full."""
        gateway = _make_gateway(queue_size=1)
        app = gateway.create_app()

        async with app.router.lifespan_context(app):
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test", timeout=5.0
            ) as client:
                # Patch put_nowait to always raise QueueFull
                import asyncio as _asyncio

                def always_full(item):
                    raise _asyncio.QueueFull()

                gateway._router._queue.put_nowait = always_full

                response = await client.post(
                    "/run_episode",
                    json={
                        "data": {"prompt": "test"},
                        "session_url": "http://proxy/session/1",
                    },
                )
                assert response.status_code == 503
                assert "queue" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_run_episode_dispatches_to_worker(self):
        """POST /run_episode should dispatch to a registered worker and return result."""
        gateway = _make_gateway()
        app = gateway.create_app()

        # Mock the router's _post_to_worker to return a success response
        success_response = DispatchResponse(status="success", result=0.75)

        async with app.router.lifespan_context(app):
            # Register a fake worker
            await gateway._pool.register_worker("w1", "http://localhost:9001")

            # Patch the HTTP call so we don't need a real worker
            with patch.object(
                gateway._router,
                "_post_to_worker",
                new=AsyncMock(return_value=success_response),
            ):
                transport = ASGITransport(app=app)
                async with AsyncClient(
                    transport=transport, base_url="http://test", timeout=5.0
                ) as client:
                    response = await client.post(
                        "/run_episode",
                        json={
                            "data": {"prompt": "hello"},
                            "session_url": "http://proxy/session/abc",
                        },
                    )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["result"] == 0.75
        assert data["error"] is None
