"""Unit tests for AgentService."""

from __future__ import annotations

import pytest

from areal.experimental.agent_service.service import AgentService
from areal.tests.experimental.agent_service.conftest import CountingAgent


class TestAgentServicePerRequestMode:
    """Tests for per-request mode (agent_reuse=False)."""

    @pytest.mark.asyncio
    async def test_get_agent_class_imports_once(
        self, agent_config, mock_agent_import_path
    ):
        """Agent class should only be imported once and cached."""
        service = AgentService(
            agent_import_path=mock_agent_import_path,
            config=agent_config,
            agent_reuse=False,
        )

        # First call imports the class
        class1 = service._get_agent_class()
        # Second call returns cached class
        class2 = service._get_agent_class()

        assert class1 is class2
        assert mock_agent_import_path in service._agent_classes

    @pytest.mark.asyncio
    async def test_per_request_creates_new_agent(
        self, agent_config, counting_agent_import_path
    ):
        """Each request should create a new agent instance."""
        count_before = CountingAgent.instance_count

        service = AgentService(
            agent_import_path=counting_agent_import_path,
            config=agent_config,
            agent_reuse=False,
        )

        await service.start()
        try:
            # First request
            result1 = await service.run_episode(
                data={"test": 1},
                session_url="http://localhost:8000/session/1",
            )
            # Second request
            result2 = await service.run_episode(
                data={"test": 2},
                session_url="http://localhost:8000/session/2",
            )

            # Each request creates a new instance, so results differ
            assert result1 != result2
            assert result2 > result1  # Instance ID increments
            assert CountingAgent.instance_count - count_before == 2
        finally:
            await service.stop()

    @pytest.mark.asyncio
    async def test_per_request_uses_agent_kwargs(
        self, agent_config, mock_agent_import_path
    ):
        """Per-request mode should use agent_kwargs from the request."""
        service = AgentService(
            agent_import_path=mock_agent_import_path,
            config=agent_config,
            agent_reuse=False,
        )

        await service.start()
        try:
            # Call with agent_kwargs
            agent = service._get_or_create_agent({"model": "test-model", "temp": 0.5})
            assert agent.init_kwargs == {"model": "test-model", "temp": 0.5}

            # Call with different kwargs creates different agent
            agent2 = service._get_or_create_agent({"model": "other-model"})
            assert agent2.init_kwargs == {"model": "other-model"}
            assert agent is not agent2
        finally:
            await service.stop()

    @pytest.mark.asyncio
    async def test_run_episode_not_running_raises(
        self, agent_config, mock_agent_import_path
    ):
        """run_episode should raise RuntimeError if service not started."""
        service = AgentService(
            agent_import_path=mock_agent_import_path,
            config=agent_config,
            agent_reuse=False,
        )

        with pytest.raises(RuntimeError, match="not running"):
            await service.run_episode(
                data={"test": 1},
                session_url="http://localhost:8000/session/1",
            )


class TestAgentServiceSharedMode:
    """Tests for shared mode (agent_reuse=True)."""

    @pytest.mark.asyncio
    async def test_shared_mode_creates_once(
        self, agent_config, counting_agent_import_path
    ):
        """Shared mode should create only one agent instance."""
        # Record count before test
        count_before = CountingAgent.instance_count

        service = AgentService(
            agent_import_path=counting_agent_import_path,
            config=agent_config,
            agent_reuse=True,
            agent_init_kwargs={"shared": True},
        )

        await service.start()
        try:
            # Multiple requests
            result1 = await service.run_episode(
                data={"test": 1},
                session_url="http://localhost:8000/session/1",
            )
            result2 = await service.run_episode(
                data={"test": 2},
                session_url="http://localhost:8000/session/2",
            )
            result3 = await service.run_episode(
                data={"test": 3},
                session_url="http://localhost:8000/session/3",
            )

            # All requests use the same instance (results are equal)
            assert result1 == result2 == result3
            # Only one instance was created during this test
            assert CountingAgent.instance_count - count_before == 1
        finally:
            await service.stop()

    @pytest.mark.asyncio
    async def test_shared_mode_ignores_agent_kwargs(
        self, agent_config, mock_agent_import_path
    ):
        """Shared mode should ignore agent_kwargs from request."""
        service = AgentService(
            agent_import_path=mock_agent_import_path,
            config=agent_config,
            agent_reuse=True,
            agent_init_kwargs={"init_param": "from_init"},
        )

        await service.start()
        try:
            # Request provides different kwargs, but they should be ignored
            agent1 = service._get_or_create_agent({"request_param": "ignored"})
            agent2 = service._get_or_create_agent({"other_param": "also_ignored"})

            # Both should be the same instance with init kwargs
            assert agent1 is agent2
            assert agent1.init_kwargs == {"init_param": "from_init"}
        finally:
            await service.stop()

    @pytest.mark.asyncio
    async def test_shared_mode_precreates_on_start(
        self, agent_config, counting_agent_import_path
    ):
        """Shared mode should pre-create agent instance on start()."""
        # Record count before test
        count_before = CountingAgent.instance_count

        service = AgentService(
            agent_import_path=counting_agent_import_path,
            config=agent_config,
            agent_reuse=True,
        )

        assert service._shared_agent is None

        await service.start()
        try:
            # Agent should be created during start()
            assert service._shared_agent is not None
            # One instance was created during start()
            assert CountingAgent.instance_count - count_before == 1
        finally:
            await service.stop()


class TestAgentServiceLifecycle:
    """Tests for service lifecycle management."""

    @pytest.mark.asyncio
    async def test_start_sets_running(self, agent_config, mock_agent_import_path):
        """start() should set is_running to True."""
        service = AgentService(
            agent_import_path=mock_agent_import_path,
            config=agent_config,
        )

        assert not service.is_running

        await service.start()
        try:
            assert service.is_running
        finally:
            await service.stop()

    @pytest.mark.asyncio
    async def test_start_idempotent(self, agent_config, mock_agent_import_path):
        """Multiple start() calls should be safe (idempotent)."""
        service = AgentService(
            agent_import_path=mock_agent_import_path,
            config=agent_config,
        )

        await service.start()
        try:
            # Second start should not raise
            await service.start()
            assert service.is_running
        finally:
            await service.stop()

    @pytest.mark.asyncio
    async def test_stop_clears_state(self, agent_config, mock_agent_import_path):
        """stop() should clear running state and shared agent."""
        service = AgentService(
            agent_import_path=mock_agent_import_path,
            config=agent_config,
            agent_reuse=True,
        )

        await service.start()
        assert service.is_running
        assert service._shared_agent is not None

        await service.stop()
        assert not service.is_running
        assert service._shared_agent is None

    @pytest.mark.asyncio
    async def test_stop_idempotent(self, agent_config, mock_agent_import_path):
        """Multiple stop() calls should be safe (idempotent)."""
        service = AgentService(
            agent_import_path=mock_agent_import_path,
            config=agent_config,
        )

        await service.start()
        await service.stop()

        # Second stop should not raise
        await service.stop()
        assert not service.is_running

    @pytest.mark.asyncio
    async def test_context_manager(self, agent_config, mock_agent_import_path):
        """async with syntax should work correctly."""
        service = AgentService(
            agent_import_path=mock_agent_import_path,
            config=agent_config,
        )

        assert not service.is_running

        async with service:
            assert service.is_running
            result = await service.run_episode(
                data={"test": 1},
                session_url="http://localhost:8000/session/1",
            )
            assert result == 1.0

        assert not service.is_running

    @pytest.mark.asyncio
    async def test_health_check(self, agent_config, mock_agent_import_path):
        """health_check should return correct status information."""
        service = AgentService(
            agent_import_path=mock_agent_import_path,
            config=agent_config,
            agent_reuse=True,
        )

        # Before start
        health = await service.health_check()
        assert health["status"] == "stopped"
        assert health["running"] is False
        assert health["agent_import_path"] == mock_agent_import_path
        assert health["agent_reuse"] is True

        await service.start()
        try:
            # After start
            health = await service.health_check()
            assert health["status"] == "ok"
            assert health["running"] is True
        finally:
            await service.stop()


class TestAgentServiceErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_invalid_import_path_fails_fast(self, agent_config):
        """Invalid import path should fail during start()."""
        service = AgentService(
            agent_import_path="nonexistent.module.FakeAgent",
            config=agent_config,
        )

        with pytest.raises(ImportError):
            await service.start()

    @pytest.mark.asyncio
    async def test_agent_run_exception_propagates(
        self, agent_config, failing_agent_import_path
    ):
        """Exception from agent.run() should propagate to caller."""
        service = AgentService(
            agent_import_path=failing_agent_import_path,
            config=agent_config,
        )

        await service.start()
        try:
            with pytest.raises(ValueError, match="Agent failed intentionally"):
                await service.run_episode(
                    data={"test": 1},
                    session_url="http://localhost:8000/session/1",
                )
        finally:
            await service.stop()
