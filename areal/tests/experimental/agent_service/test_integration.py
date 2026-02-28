"""Integration tests for the full Gateway + Worker flow.

Tests the complete request path:
  Caller → Gateway (/run_episode) → Router → Worker (/run_episode) → Agent → result

All components run in-process using ASGI test transport (no real network ports).
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from areal.experimental.agent_service.config import GatewayConfig
from areal.experimental.agent_service.gateway import Gateway
from areal.experimental.agent_service.schemas import DispatchResponse

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_gateway(
    queue_size: int = 100,
    worker_timeout: float = 5.0,
) -> Gateway:
    """Create a Gateway with test-friendly defaults."""
    config = GatewayConfig(
        queue_size=queue_size,
        worker_timeout=worker_timeout,
    )
    return Gateway(config=config)


def _make_worker_env(
    gateway_addr: str = "http://gateway",
    agent_import_path: str = "areal.tests.experimental.agent_service.conftest.MockAgent",
    agent_reuse: bool = False,
) -> dict[str, str]:
    """Build env vars for a Worker process."""
    from areal.experimental.agent_service.worker_server import (
        ENV_AGENT_GATEWAY_ADDR,
        ENV_AGENT_HOST,
        ENV_AGENT_IMPORT_PATH_INTERNAL,
        ENV_AGENT_INIT_KWARGS_INTERNAL,
        ENV_AGENT_PORT,
        ENV_AGENT_REUSE_INTERNAL,
    )

    return {
        ENV_AGENT_HOST: "127.0.0.1",
        ENV_AGENT_PORT: "8301",
        ENV_AGENT_IMPORT_PATH_INTERNAL: agent_import_path,
        ENV_AGENT_REUSE_INTERNAL: "true" if agent_reuse else "false",
        ENV_AGENT_INIT_KWARGS_INTERNAL: "",
        ENV_AGENT_GATEWAY_ADDR: gateway_addr,
    }


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestGatewayWorkerIntegration:
    """End-to-end tests for Gateway + Worker flow."""

    @pytest.mark.asyncio
    async def test_full_flow_gateway_dispatches_to_worker(self):
        """Full flow: Gateway receives request, dispatches to Worker, returns result."""
        import areal.experimental.agent_service.worker_server as rpc_module

        gateway = _make_gateway()
        gateway_app = gateway.create_app()

        # Start Gateway
        async with gateway_app.router.lifespan_context(gateway_app):
            # Start Worker in-process (no actual Gateway registration since
            # we're using ASGI transport — we register manually)
            env_vars = _make_worker_env()
            with patch.dict(os.environ, env_vars):
                from areal.experimental.agent_service.worker_server import create_app

                worker_app = create_app()
                async with worker_app.router.lifespan_context(worker_app):
                    # Manually register the worker with the Gateway pool
                    # (simulates what the Worker's lifespan would do via HTTP)
                    await gateway._pool.register_worker("worker-0", "http://worker-0")

                    # Patch the Router's HTTP dispatch to use the Worker's ASGI app
                    worker_transport = ASGITransport(app=worker_app)

                    async def asgi_post_to_worker(worker, request):
                        async with AsyncClient(
                            transport=worker_transport,
                            base_url="http://worker-0",
                            timeout=5.0,
                        ) as worker_client:
                            resp = await worker_client.post(
                                "/run_episode",
                                json=request.model_dump(),
                            )
                            data = resp.json()
                            return DispatchResponse(**data)

                    with patch.object(
                        gateway._router, "_post_to_worker", new=asgi_post_to_worker
                    ):
                        # Send request to Gateway
                        gateway_transport = ASGITransport(app=gateway_app)
                        async with AsyncClient(
                            transport=gateway_transport,
                            base_url="http://gateway",
                            timeout=5.0,
                        ) as gateway_client:
                            response = await gateway_client.post(
                                "/run_episode",
                                json={
                                    "data": {"prompt": "hello"},
                                    "session_url": "http://proxy/session/abc",
                                },
                            )

        rpc_module._service = None

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["result"] == 1.0  # MockAgent returns 1.0
        assert data["error"] is None

    @pytest.mark.asyncio
    async def test_multiple_workers_round_robin_dispatch(self):
        """Multiple workers should receive requests in round-robin order."""

        gateway = _make_gateway()
        gateway_app = gateway.create_app()

        # Track which worker handled each request
        handled_by: list[str] = []

        async def tracking_post(worker, request):
            handled_by.append(worker.worker_id)
            return DispatchResponse(status="success", result=1.0)

        async with gateway_app.router.lifespan_context(gateway_app):
            # Register 3 workers
            for i in range(3):
                await gateway._pool.register_worker(f"worker-{i}", f"http://worker-{i}")

            with patch.object(gateway._router, "_post_to_worker", new=tracking_post):
                gateway_transport = ASGITransport(app=gateway_app)
                async with AsyncClient(
                    transport=gateway_transport,
                    base_url="http://gateway",
                    timeout=5.0,
                ) as client:
                    # Send 3 requests
                    for _ in range(3):
                        resp = await client.post(
                            "/run_episode",
                            json={
                                "data": {"prompt": "test"},
                                "session_url": "http://proxy/session/1",
                            },
                        )
                        assert resp.status_code == 200

        # All 3 workers should have been used (round-robin)
        assert len(handled_by) == 3
        assert set(handled_by) == {"worker-0", "worker-1", "worker-2"}

    @pytest.mark.asyncio
    async def test_worker_failure_handled_gracefully(self):
        """Gateway should handle worker failure gracefully and return error response."""
        gateway = _make_gateway()
        gateway_app = gateway.create_app()

        async def always_fail(worker, request):
            raise ConnectionError("Worker unreachable")

        async with gateway_app.router.lifespan_context(gateway_app):
            # Register 2 workers (both will fail)
            await gateway._pool.register_worker("worker-0", "http://worker-0")
            await gateway._pool.register_worker("worker-1", "http://worker-1")

            with patch.object(gateway._router, "_post_to_worker", new=always_fail):
                gateway_transport = ASGITransport(app=gateway_app)
                async with AsyncClient(
                    transport=gateway_transport,
                    base_url="http://gateway",
                    timeout=5.0,
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
        assert data["status"] == "error"
        assert data["error"] is not None

    @pytest.mark.asyncio
    async def test_worker_registration_visible_in_gateway(self):
        """Worker registration should be visible via Gateway's /workers endpoint."""
        gateway = _make_gateway()
        gateway_app = gateway.create_app()

        async with gateway_app.router.lifespan_context(gateway_app):
            # Register workers directly (simulating Worker startup)
            await gateway._pool.register_worker("worker-0", "http://worker-0:8301")
            await gateway._pool.register_worker("worker-1", "http://worker-1:8302")

            gateway_transport = ASGITransport(app=gateway_app)
            async with AsyncClient(
                transport=gateway_transport, base_url="http://gateway"
            ) as client:
                response = await client.get("/workers")

        assert response.status_code == 200
        workers = response.json()
        assert len(workers) == 2
        worker_ids = {w["worker_id"] for w in workers}
        assert worker_ids == {"worker-0", "worker-1"}

    @pytest.mark.asyncio
    async def test_multiple_requests_use_all_workers(self):
        """Multiple sequential requests should be dispatched across all workers."""
        gateway = _make_gateway()
        gateway_app = gateway.create_app()

        # Track which workers handled requests
        handled_by: list[str] = []

        async def tracking_post(worker, request):
            handled_by.append(worker.worker_id)
            return DispatchResponse(status="success", result=1.0)

        async with gateway_app.router.lifespan_context(gateway_app):
            # Register 3 workers
            for i in range(3):
                await gateway._pool.register_worker(f"worker-{i}", f"http://worker-{i}")

            with patch.object(gateway._router, "_post_to_worker", new=tracking_post):
                gateway_transport = ASGITransport(app=gateway_app)
                async with AsyncClient(
                    transport=gateway_transport,
                    base_url="http://gateway",
                    timeout=5.0,
                ) as client:
                    # Send 3 requests sequentially
                    for i in range(3):
                        resp = await client.post(
                            "/run_episode",
                            json={
                                "data": {"prompt": f"test-{i}"},
                                "session_url": f"http://proxy/session/{i}",
                            },
                        )
                        assert resp.status_code == 200
                        assert resp.json()["status"] == "success"

        # All 3 workers should have been used (round-robin)
        assert len(handled_by) == 3
        assert set(handled_by) == {"worker-0", "worker-1", "worker-2"}
