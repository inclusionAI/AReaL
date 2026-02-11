"""Unit tests for Agent RPC Server endpoints."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from areal.experimental.agent_service.rpc_server import (
    ENV_AGENT_HOST,
    ENV_AGENT_IMPORT_PATH_INTERNAL,
    ENV_AGENT_INIT_KWARGS_INTERNAL,
    ENV_AGENT_PORT,
    ENV_AGENT_REUSE_INTERNAL,
    ENV_AGENT_WORKERS,
    create_app,
)
from areal.tests.experimental.agent_service.conftest import CountingAgent


def _make_env_vars(
    agent_import_path: str,
    agent_reuse: bool = False,
    agent_init_kwargs: str = "",
) -> dict[str, str]:
    """Build env var dict for create_app lifespan."""
    return {
        ENV_AGENT_HOST: "127.0.0.1",
        ENV_AGENT_PORT: "8300",
        ENV_AGENT_WORKERS: "1",
        ENV_AGENT_IMPORT_PATH_INTERNAL: agent_import_path,
        ENV_AGENT_REUSE_INTERNAL: "true" if agent_reuse else "false",
        ENV_AGENT_INIT_KWARGS_INTERNAL: agent_init_kwargs,
    }


class TestAgentRPCServerEndpoints:
    """Tests for HTTP endpoints."""

    @pytest.mark.asyncio
    async def test_health_endpoint(self, agent_config, mock_agent_import_path):
        """GET /health should return correct status."""
        import areal.experimental.agent_service.rpc_server as rpc_module

        env_vars = _make_env_vars(mock_agent_import_path)

        with patch.dict(os.environ, env_vars):
            app = create_app()
            async with app.router.lifespan_context(app):
                transport = ASGITransport(app=app)
                async with AsyncClient(
                    transport=transport, base_url="http://test"
                ) as client:
                    response = await client.get("/health")

                    assert response.status_code == 200
                    data = response.json()
                    assert data["status"] == "ok"
                    assert data["running"] is True
                    assert data["agent_import_path"] == mock_agent_import_path
                    assert data["agent_reuse"] is False

        rpc_module._service = None

    @pytest.mark.asyncio
    async def test_run_episode_success(self, agent_config, mock_agent_import_path):
        """POST /run_episode should return success with result."""
        import areal.experimental.agent_service.rpc_server as rpc_module

        env_vars = _make_env_vars(mock_agent_import_path)

        with patch.dict(os.environ, env_vars):
            app = create_app()
            async with app.router.lifespan_context(app):
                transport = ASGITransport(app=app)
                async with AsyncClient(
                    transport=transport, base_url="http://test"
                ) as client:
                    response = await client.post(
                        "/run_episode",
                        json={
                            "data": {"test": "data"},
                            "session_url": "http://localhost:8000/session/abc",
                            "agent_kwargs": {"param": "value"},
                        },
                    )

                    assert response.status_code == 200
                    data = response.json()
                    assert data["status"] == "success"
                    assert data["result"] == 1.0  # MockAgent returns 1.0
                    assert data["error"] is None

        rpc_module._service = None

    @pytest.mark.asyncio
    async def test_run_episode_without_agent_kwargs(
        self, agent_config, mock_agent_import_path
    ):
        """POST /run_episode should work without agent_kwargs."""
        import areal.experimental.agent_service.rpc_server as rpc_module

        env_vars = _make_env_vars(mock_agent_import_path)

        with patch.dict(os.environ, env_vars):
            app = create_app()
            async with app.router.lifespan_context(app):
                transport = ASGITransport(app=app)
                async with AsyncClient(
                    transport=transport, base_url="http://test"
                ) as client:
                    response = await client.post(
                        "/run_episode",
                        json={
                            "data": {"test": "data"},
                            "session_url": "http://localhost:8000/session/abc",
                        },
                    )

                    assert response.status_code == 200
                    data = response.json()
                    assert data["status"] == "success"
                    assert data["result"] == 1.0

        rpc_module._service = None

    @pytest.mark.asyncio
    async def test_run_episode_service_not_running(
        self, agent_config, mock_agent_import_path
    ):
        """POST /run_episode should return 503 if service not initialized."""
        import areal.experimental.agent_service.rpc_server as rpc_module

        # create_app WITHOUT triggering lifespan — _service stays None
        old_service = rpc_module._service
        rpc_module._service = None

        try:
            env_vars = _make_env_vars(mock_agent_import_path)
            with patch.dict(os.environ, env_vars):
                app = create_app()

                # Don't enter lifespan — _service remains None
                transport = ASGITransport(app=app)
                async with AsyncClient(
                    transport=transport, base_url="http://test"
                ) as client:
                    response = await client.post(
                        "/run_episode",
                        json={
                            "data": {"test": "data"},
                            "session_url": "http://localhost:8000/session/abc",
                        },
                    )

                    assert response.status_code == 503
                    assert "not initialized" in response.json()["detail"]
        finally:
            rpc_module._service = old_service

    @pytest.mark.asyncio
    async def test_run_episode_agent_error(
        self, agent_config, failing_agent_import_path
    ):
        """POST /run_episode should return error status when agent fails."""
        import areal.experimental.agent_service.rpc_server as rpc_module

        env_vars = _make_env_vars(failing_agent_import_path)

        with patch.dict(os.environ, env_vars):
            app = create_app()
            async with app.router.lifespan_context(app):
                transport = ASGITransport(app=app)
                async with AsyncClient(
                    transport=transport, base_url="http://test"
                ) as client:
                    response = await client.post(
                        "/run_episode",
                        json={
                            "data": {"test": "data"},
                            "session_url": "http://localhost:8000/session/abc",
                        },
                    )

                    assert response.status_code == 200
                    data = response.json()
                    assert data["status"] == "error"
                    assert data["result"] is None
                    assert "Agent failed intentionally" in data["error"]

        rpc_module._service = None


class TestAgentRPCServerSharedMode:
    """Tests for RPC server with shared mode service."""

    @pytest.mark.asyncio
    async def test_shared_mode_health(self, agent_config, mock_agent_import_path):
        """Health endpoint should report agent_reuse correctly."""
        import areal.experimental.agent_service.rpc_server as rpc_module

        env_vars = _make_env_vars(
            mock_agent_import_path,
            agent_reuse=True,
            agent_init_kwargs='{"model": "shared-model"}',
        )

        with patch.dict(os.environ, env_vars):
            app = create_app()
            async with app.router.lifespan_context(app):
                transport = ASGITransport(app=app)
                async with AsyncClient(
                    transport=transport, base_url="http://test"
                ) as client:
                    response = await client.get("/health")

                    assert response.status_code == 200
                    data = response.json()
                    assert data["agent_reuse"] is True

        rpc_module._service = None

    @pytest.mark.asyncio
    async def test_shared_mode_multiple_requests(
        self, agent_config, counting_agent_import_path
    ):
        """Multiple requests in shared mode should use same agent instance."""
        import areal.experimental.agent_service.rpc_server as rpc_module

        count_before = CountingAgent.instance_count

        env_vars = _make_env_vars(counting_agent_import_path, agent_reuse=True)

        with patch.dict(os.environ, env_vars):
            app = create_app()
            async with app.router.lifespan_context(app):
                transport = ASGITransport(app=app)
                async with AsyncClient(
                    transport=transport, base_url="http://test"
                ) as client:
                    # Make multiple requests
                    results = []
                    for i in range(3):
                        response = await client.post(
                            "/run_episode",
                            json={
                                "data": {"test": i},
                                "session_url": f"http://localhost:8000/session/{i}",
                            },
                        )
                        assert response.status_code == 200
                        results.append(response.json()["result"])

                    # All results should be the same (same instance)
                    assert results[0] == results[1] == results[2]
                    # Only one instance should have been created during this test
                    assert CountingAgent.instance_count - count_before == 1

        rpc_module._service = None
