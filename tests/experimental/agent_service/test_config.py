"""Unit tests for the Agent Service config module."""

from __future__ import annotations

import pytest

from areal.experimental.agent_service.config import AgentServiceConfig


class TestAgentServiceConfigValidation:
    def test_valid_config(self):
        cfg = AgentServiceConfig(agent_import_path="myapp.agent.MyAgent")
        assert cfg.agent_import_path == "myapp.agent.MyAgent"
        assert cfg.num_workers == 1
        assert cfg.max_concurrent_sessions == 8
        assert cfg.session_timeout_seconds == 3600
        assert cfg.admin_api_key == "areal-admin-key"

    def test_empty_import_path_raises(self):
        with pytest.raises(ValueError, match="agent_import_path must be set"):
            AgentServiceConfig(agent_import_path="")

    def test_default_import_path_raises(self):
        with pytest.raises(ValueError, match="agent_import_path must be set"):
            AgentServiceConfig()

    def test_zero_workers_raises(self):
        with pytest.raises(ValueError, match="num_workers must be positive"):
            AgentServiceConfig(agent_import_path="a.B", num_workers=0)

    def test_negative_workers_raises(self):
        with pytest.raises(ValueError, match="num_workers must be positive"):
            AgentServiceConfig(agent_import_path="a.B", num_workers=-1)

    def test_zero_concurrent_sessions_raises(self):
        with pytest.raises(
            ValueError, match="max_concurrent_sessions must be positive"
        ):
            AgentServiceConfig(agent_import_path="a.B", max_concurrent_sessions=0)

    def test_empty_api_key_raises(self):
        with pytest.raises(ValueError, match="admin_api_key must not be empty"):
            AgentServiceConfig(agent_import_path="a.B", admin_api_key="")

    def test_whitespace_api_key_raises(self):
        with pytest.raises(ValueError, match="admin_api_key must not be empty"):
            AgentServiceConfig(agent_import_path="a.B", admin_api_key="   ")

    def test_custom_values(self):
        cfg = AgentServiceConfig(
            agent_import_path="pkg.mod.Cls",
            num_workers=4,
            max_concurrent_sessions=16,
            session_timeout_seconds=7200,
            admin_api_key="custom-key",
        )
        assert cfg.num_workers == 4
        assert cfg.max_concurrent_sessions == 16
        assert cfg.session_timeout_seconds == 7200
        assert cfg.admin_api_key == "custom-key"
