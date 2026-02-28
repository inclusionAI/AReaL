"""Unit tests for Agent Service multi-worker support."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from areal.experimental.agent_service.worker_server import (
    ENV_AGENT_HOST,
    ENV_AGENT_IMPORT_PATH_INTERNAL,
    ENV_AGENT_INIT_KWARGS_INTERNAL,
    ENV_AGENT_PORT,
    ENV_AGENT_REUSE_INTERNAL,
    _get_agent_config_from_env,
    create_app,
)


class TestGetAgentConfigFromEnv:
    """Tests for _get_agent_config_from_env function."""

    def test_returns_defaults_when_no_env_vars(self):
        """Should return defaults when env vars are not set."""
        with patch.dict(os.environ, {}, clear=True):
            agent_import_path, agent_reuse, agent_init_kwargs = (
                _get_agent_config_from_env()
            )

            assert agent_import_path is None
            assert agent_reuse is False
            assert agent_init_kwargs == {}

    def test_reads_agent_import_path(self):
        """Should read AREAL_AGENT_IMPORT_PATH_INTERNAL env var."""
        with patch.dict(
            os.environ, {ENV_AGENT_IMPORT_PATH_INTERNAL: "mymodule.MyAgent"}
        ):
            agent_import_path, _, _ = _get_agent_config_from_env()
            assert agent_import_path == "mymodule.MyAgent"

    def test_empty_import_path_returns_none(self):
        """Empty import path should return None."""
        with patch.dict(os.environ, {ENV_AGENT_IMPORT_PATH_INTERNAL: ""}):
            agent_import_path, _, _ = _get_agent_config_from_env()
            assert agent_import_path is None

    def test_reads_agent_reuse_true(self):
        """Should read AREAL_AGENT_REUSE_INTERNAL=true."""
        with patch.dict(os.environ, {ENV_AGENT_REUSE_INTERNAL: "true"}):
            _, agent_reuse, _ = _get_agent_config_from_env()
            assert agent_reuse is True

    def test_reads_agent_reuse_one(self):
        """Should read AREAL_AGENT_REUSE_INTERNAL=1 as True."""
        with patch.dict(os.environ, {ENV_AGENT_REUSE_INTERNAL: "1"}):
            _, agent_reuse, _ = _get_agent_config_from_env()
            assert agent_reuse is True

    def test_reads_agent_reuse_false(self):
        """Should read AREAL_AGENT_REUSE_INTERNAL=false."""
        with patch.dict(os.environ, {ENV_AGENT_REUSE_INTERNAL: "false"}):
            _, agent_reuse, _ = _get_agent_config_from_env()
            assert agent_reuse is False

    def test_reads_agent_init_kwargs(self):
        """Should parse JSON from AREAL_AGENT_INIT_KWARGS_INTERNAL."""
        with patch.dict(
            os.environ,
            {ENV_AGENT_INIT_KWARGS_INTERNAL: '{"model": "gpt-4", "temperature": 0.7}'},
        ):
            _, _, agent_init_kwargs = _get_agent_config_from_env()
            assert agent_init_kwargs == {"model": "gpt-4", "temperature": 0.7}

    def test_invalid_json_returns_empty_dict(self):
        """Invalid JSON in init kwargs should return empty dict."""
        with patch.dict(os.environ, {ENV_AGENT_INIT_KWARGS_INTERNAL: "not valid json"}):
            _, _, agent_init_kwargs = _get_agent_config_from_env()
            assert agent_init_kwargs == {}


class TestCreateAppFactory:
    """Tests for create_app factory function."""

    @pytest.fixture
    def mock_env_vars(self, mock_agent_import_path):
        """Set up environment variables for create_app tests."""
        env_vars = {
            ENV_AGENT_HOST: "127.0.0.1",
            ENV_AGENT_PORT: "8300",
            ENV_AGENT_IMPORT_PATH_INTERNAL: mock_agent_import_path,
            ENV_AGENT_REUSE_INTERNAL: "false",
            ENV_AGENT_INIT_KWARGS_INTERNAL: "",
        }
        with patch.dict(os.environ, env_vars):
            yield

    @pytest.mark.asyncio
    async def test_create_app_returns_fastapi(self, mock_env_vars):
        """create_app should return a FastAPI application."""
        from fastapi import FastAPI

        app = create_app()
        assert isinstance(app, FastAPI)
        assert app.title == "Agent Service RPC Server"

    @pytest.mark.asyncio
    async def test_create_app_has_health_route(self, mock_env_vars):
        """create_app should include /health route."""
        app = create_app()
        routes = [route.path for route in app.routes]
        assert "/health" in routes

    @pytest.mark.asyncio
    async def test_create_app_has_run_episode_route(self, mock_env_vars):
        """create_app should include /run_episode route."""
        app = create_app()
        routes = [route.path for route in app.routes]
        assert "/run_episode" in routes

    @pytest.mark.asyncio
    async def test_create_app_with_lifespan(self, mock_agent_import_path):
        """Test create_app with lifespan context properly initialized."""
        import areal.experimental.agent_service.worker_server as rpc_module

        env_vars = {
            ENV_AGENT_HOST: "127.0.0.1",
            ENV_AGENT_PORT: "8300",
            ENV_AGENT_IMPORT_PATH_INTERNAL: mock_agent_import_path,
            ENV_AGENT_REUSE_INTERNAL: "false",
            ENV_AGENT_INIT_KWARGS_INTERNAL: "",
        }

        with patch.dict(os.environ, env_vars):
            app = create_app()

            # Manually trigger lifespan
            async with app.router.lifespan_context(app):
                # Now _service should be initialized
                assert rpc_module._service is not None
                assert rpc_module._service.is_running

                transport = ASGITransport(app=app)
                async with AsyncClient(
                    transport=transport, base_url="http://test"
                ) as client:
                    response = await client.get("/health")
                    assert response.status_code == 200
                    data = response.json()
                    assert data["status"] == "ok"
                    assert data["running"] is True

            # After lifespan exits, service should be stopped
            assert rpc_module._service is None or not rpc_module._service.is_running

        # Clean up global state
        rpc_module._service = None

    @pytest.mark.asyncio
    async def test_create_app_run_episode_with_lifespan(self, mock_agent_import_path):
        """Test run_episode endpoint with lifespan properly initialized."""
        import areal.experimental.agent_service.worker_server as rpc_module

        env_vars = {
            ENV_AGENT_HOST: "127.0.0.1",
            ENV_AGENT_PORT: "8300",
            ENV_AGENT_IMPORT_PATH_INTERNAL: mock_agent_import_path,
            ENV_AGENT_REUSE_INTERNAL: "false",
            ENV_AGENT_INIT_KWARGS_INTERNAL: "",
        }

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
                            "session_url": "http://localhost:8000/session/test",
                        },
                    )
                    assert response.status_code == 200
                    data = response.json()
                    assert data["status"] == "success"
                    assert data["result"] == 1.0

        # Clean up global state
        rpc_module._service = None

    @pytest.mark.asyncio
    async def test_create_app_shared_mode_with_lifespan(self, mock_agent_import_path):
        """Test create_app in shared mode with lifespan."""
        import areal.experimental.agent_service.worker_server as rpc_module

        env_vars = {
            ENV_AGENT_HOST: "127.0.0.1",
            ENV_AGENT_PORT: "8300",
            ENV_AGENT_IMPORT_PATH_INTERNAL: mock_agent_import_path,
            ENV_AGENT_REUSE_INTERNAL: "true",
            ENV_AGENT_INIT_KWARGS_INTERNAL: '{"shared": true}',
        }

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

        # Clean up global state
        rpc_module._service = None


class TestParseArgsWorkers:
    """Tests for --workers CLI argument parsing."""

    def test_default_workers_is_one(self):
        """Default workers should be 1."""
        from areal.experimental.agent_service.__main__ import _parse_args

        with patch("sys.argv", ["worker_server"]):
            args = _parse_args()
            assert args.workers == 1

    def test_workers_argument_parsed(self):
        """--workers argument should be parsed correctly."""
        from areal.experimental.agent_service.__main__ import _parse_args

        with patch("sys.argv", ["worker_server", "--workers", "4"]):
            args = _parse_args()
            assert args.workers == 4

    def test_workers_with_other_args(self):
        """--workers should work with other arguments."""
        from areal.experimental.agent_service.__main__ import _parse_args

        with patch(
            "sys.argv",
            [
                "worker_server",
                "--host",
                "0.0.0.0",
                "--port",
                "9000",
                "--workers",
                "8",
                "--agent-reuse",
            ],
        ):
            args = _parse_args()
            assert args.workers == 8
            assert args.host == "0.0.0.0"
            assert args.port == 9000
            assert args.agent_reuse is True
