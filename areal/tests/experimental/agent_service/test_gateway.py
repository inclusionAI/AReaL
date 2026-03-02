"""Unit tests for the Agent Service Gateway (reverse proxy)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from areal.experimental.agent_service.config import GatewayConfig
from areal.experimental.agent_service.gateway import Gateway


def _make_gateway(
    worker_timeout: float = 5.0,
    max_retries: int = 3,
) -> Gateway:
    """Create a Gateway instance with test-friendly defaults."""
    config = GatewayConfig(
        worker_timeout=worker_timeout,
        max_retries=max_retries,
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
        assert data["workers"]["healthy"] == 1


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
    """Tests for POST /run_episode endpoint (reverse proxy)."""

    @pytest.mark.asyncio
    async def test_run_episode_returns_503_when_no_workers(self):
        """POST /run_episode should return 503 when no workers are registered."""
        gateway = _make_gateway(max_retries=1)
        app = gateway.create_app()

        async with app.router.lifespan_context(app):
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test", timeout=5.0
            ) as client:
                response = await client.post(
                    "/run_episode",
                    json={
                        "data": {"prompt": "test"},
                        "session_url": "http://proxy/session/1",
                    },
                )
                assert response.status_code == 503
                assert "workers failed" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_run_episode_forwards_to_worker(self):
        """POST /run_episode should forward to a registered worker and return result."""
        gateway = _make_gateway()
        app = gateway.create_app()

        async with app.router.lifespan_context(app):
            # Register a fake worker
            await gateway._pool.register_worker("w1", "http://localhost:9001")

            # Mock the aiohttp session to return a success response
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.json = AsyncMock(
                return_value={"status": "success", "result": 0.75, "error": None}
            )

            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)

            with patch.object(
                gateway._session,
                "post",
                return_value=mock_ctx,
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

    @pytest.mark.asyncio
    async def test_run_episode_retries_on_worker_failure(self):
        """POST /run_episode should retry on next worker when first fails."""
        gateway = _make_gateway(max_retries=3)
        app = gateway.create_app()

        async with app.router.lifespan_context(app):
            await gateway._pool.register_worker("w1", "http://localhost:9001")
            await gateway._pool.register_worker("w2", "http://localhost:9002")

            call_count = 0

            def mock_post(url, **kwargs):
                nonlocal call_count
                call_count += 1

                if call_count == 1:
                    # First call fails
                    raise ConnectionError("Worker w1 unreachable")

                # Second call succeeds
                mock_resp = AsyncMock()
                mock_resp.status = 200
                mock_resp.json = AsyncMock(
                    return_value={"status": "success", "result": 0.9, "error": None}
                )
                mock_ctx = AsyncMock()
                mock_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
                mock_ctx.__aexit__ = AsyncMock(return_value=False)
                return mock_ctx

            with patch.object(gateway._session, "post", side_effect=mock_post):
                transport = ASGITransport(app=app)
                async with AsyncClient(
                    transport=transport, base_url="http://test", timeout=5.0
                ) as client:
                    response = await client.post(
                        "/run_episode",
                        json={
                            "data": {"prompt": "test"},
                            "session_url": "http://proxy/session/1",
                        },
                    )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["result"] == 0.9
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_run_episode_returns_503_after_all_retries_exhausted(self):
        """POST /run_episode should return 503 when all workers fail."""
        gateway = _make_gateway(max_retries=2)
        app = gateway.create_app()

        async with app.router.lifespan_context(app):
            await gateway._pool.register_worker("w1", "http://localhost:9001")
            await gateway._pool.register_worker("w2", "http://localhost:9002")

            def always_fail(url, **kwargs):
                raise ConnectionError("Worker unreachable")

            with patch.object(gateway._session, "post", side_effect=always_fail):
                transport = ASGITransport(app=app)
                async with AsyncClient(
                    transport=transport, base_url="http://test", timeout=5.0
                ) as client:
                    response = await client.post(
                        "/run_episode",
                        json={
                            "data": {"prompt": "test"},
                            "session_url": "http://proxy/session/1",
                        },
                    )

        assert response.status_code == 503
        assert "failed" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_run_episode_marks_failed_worker_unhealthy(self):
        """Worker that fails should be marked unhealthy in the pool."""
        gateway = _make_gateway(max_retries=2)
        app = gateway.create_app()

        async with app.router.lifespan_context(app):
            await gateway._pool.register_worker("w1", "http://localhost:9001")
            await gateway._pool.register_worker("w2", "http://localhost:9002")

            def always_fail(url, **kwargs):
                raise ConnectionError("Unreachable")

            with patch.object(gateway._session, "post", side_effect=always_fail):
                transport = ASGITransport(app=app)
                async with AsyncClient(
                    transport=transport, base_url="http://test", timeout=5.0
                ) as client:
                    await client.post(
                        "/run_episode",
                        json={
                            "data": {"prompt": "test"},
                            "session_url": "http://proxy/session/1",
                        },
                    )

        stats = await gateway._pool.get_stats()
        assert stats["unhealthy"] >= 1


class TestGatewayMetrics:
    """Tests for GET /metrics endpoint."""

    @pytest.mark.asyncio
    async def test_metrics_initially_zero(self):
        """GET /metrics should return zeros when no requests have been made."""
        gateway = _make_gateway()
        app = gateway.create_app()

        async with app.router.lifespan_context(app):
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                response = await client.get("/metrics")

        assert response.status_code == 200
        data = response.json()
        assert data["total_requests"] == 0
        assert data["total_success"] == 0
        assert data["total_errors"] == 0

    @pytest.mark.asyncio
    async def test_metrics_tracks_successful_requests(self):
        """GET /metrics should count successful requests."""
        gateway = _make_gateway()
        app = gateway.create_app()

        async with app.router.lifespan_context(app):
            await gateway._pool.register_worker("w1", "http://localhost:9001")

            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.json = AsyncMock(
                return_value={"status": "success", "result": 1.0, "error": None}
            )
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)

            with patch.object(gateway._session, "post", return_value=mock_ctx):
                transport = ASGITransport(app=app)
                async with AsyncClient(
                    transport=transport, base_url="http://test", timeout=5.0
                ) as client:
                    await client.post(
                        "/run_episode",
                        json={
                            "data": {"prompt": "test"},
                            "session_url": "http://proxy/session/1",
                        },
                    )
                    metrics_resp = await client.get("/metrics")

        data = metrics_resp.json()
        assert data["total_requests"] == 1
        assert data["total_success"] == 1
        assert data["total_errors"] == 0


class TestGatewayConfigure:
    """Tests for POST /configure endpoint."""

    @pytest.mark.asyncio
    async def test_configure_returns_success(self):
        """POST /configure should return success (no-op for Gateway)."""
        gateway = _make_gateway()
        app = gateway.create_app()

        async with app.router.lifespan_context(app):
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                response = await client.post("/configure")

        assert response.status_code == 200
        assert response.json()["status"] == "success"
