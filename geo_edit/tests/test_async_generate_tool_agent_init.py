"""Tests for ToolRouter agent initialization in async_generate_with_tool_call_api.py

Validates that:
1. ToolRouter automatically initializes agents when enabled in config.yaml
2. Agents are properly scheduled on nodes with specified resources
3. Agents are properly shutdown via ToolRouter.shutdown_agents()
"""

import pytest
from unittest.mock import MagicMock, patch
from PIL import Image


class TestToolRouterAgentInitialization:
    """Test that ToolRouter initializes agents on construction."""

    @patch("geo_edit.tool_definitions.router.get_manager")
    def test_agents_created_when_enabled(self, mock_get_manager):
        """Verify agents are created when agent tools are enabled."""
        from geo_edit.tool_definitions.router import ToolRouter, _TOOL_CONFIG

        mock_manager = MagicMock()
        mock_manager.create_agents.return_value = {"chartmoe": MagicMock()}
        mock_get_manager.return_value = mock_manager

        # Check if any agents are enabled
        has_agents = any(
            _TOOL_CONFIG.get(name, False)
            for name in ["chartmoe", "multimath", "gllava"]
        )

        router = ToolRouter(tool_mode="auto")

        if has_agents:
            mock_manager.create_agents.assert_called_once()
            assert router.is_agent_enabled()

    @patch("geo_edit.tool_definitions.router.get_manager")
    def test_no_agents_created_in_direct_mode(self, mock_get_manager):
        """Verify no agents are created in direct mode."""
        from geo_edit.tool_definitions.router import ToolRouter

        mock_manager = MagicMock()
        mock_get_manager.return_value = mock_manager

        router = ToolRouter(tool_mode="direct")

        mock_manager.create_agents.assert_not_called()
        assert not router.is_agent_enabled()

    @patch("geo_edit.tool_definitions.router.get_manager")
    def test_node_resource_added_to_configs(self, mock_get_manager):
        """Verify node_resource is added to agent configs."""
        from geo_edit.tool_definitions.router import ToolRouter, _TOOL_CONFIG

        mock_manager = MagicMock()
        mock_manager.create_agents.return_value = {}
        mock_get_manager.return_value = mock_manager

        # Skip if no agents enabled
        has_agents = any(
            _TOOL_CONFIG.get(name, False)
            for name in ["chartmoe", "multimath", "gllava"]
        )
        if not has_agents:
            pytest.skip("No agent tools enabled in config.yaml")

        router = ToolRouter(tool_mode="auto", node_resource="tool_agent")

        call_args = mock_manager.create_agents.call_args
        if call_args:
            configs = call_args[0][0]
            for name, cfg in configs.items():
                assert "resources" in cfg
                assert cfg["resources"] == {"tool_agent": 1}


class TestToolRouterAgentShutdown:
    """Test that ToolRouter properly shuts down agents."""

    @patch("geo_edit.tool_definitions.router.get_manager")
    def test_shutdown_agents_calls_manager(self, mock_get_manager):
        """Verify shutdown_agents delegates to manager."""
        from geo_edit.tool_definitions.router import ToolRouter

        mock_manager = MagicMock()
        mock_manager.create_agents.return_value = {"chartmoe": MagicMock()}
        mock_get_manager.return_value = mock_manager

        router = ToolRouter(tool_mode="auto")
        # Manually set _agents to simulate agents were created
        router._agents = {"chartmoe": MagicMock()}

        router.shutdown_agents()

        mock_manager.shutdown.assert_called_once_with(None)
        assert router._agents == {}

    @patch("geo_edit.tool_definitions.router.get_manager")
    def test_shutdown_agents_noop_when_no_agents(self, mock_get_manager):
        """Verify shutdown_agents does nothing when no agents exist."""
        from geo_edit.tool_definitions.router import ToolRouter

        mock_manager = MagicMock()
        mock_get_manager.return_value = mock_manager

        router = ToolRouter(tool_mode="direct")
        router.shutdown_agents()

        mock_manager.shutdown.assert_not_called()


class TestToolRouterConfigIntegration:
    """Test ToolRouter config-based agent initialization."""

    def test_get_enabled_agent_configs_returns_configs(self):
        """Verify get_enabled_agent_configs returns correct configs."""
        from geo_edit.tool_definitions.router import ToolRouter, _TOOL_CONFIG

        router = ToolRouter.__new__(ToolRouter)
        router.tool_mode = "auto"
        router._agents = {}

        configs = router.get_enabled_agent_configs()

        # Verify structure for enabled agents
        for name, cfg in configs.items():
            assert _TOOL_CONFIG.get(name, False), f"{name} should be enabled"
            assert "model_name_or_path" in cfg
            assert "max_model_len" in cfg
            assert "gpu_memory_utilization" in cfg

    def test_get_enabled_agent_configs_empty_in_direct_mode(self):
        """Verify get_enabled_agent_configs returns empty in direct mode."""
        from geo_edit.tool_definitions.router import ToolRouter

        router = ToolRouter.__new__(ToolRouter)
        router.tool_mode = "direct"
        router._agents = {}

        configs = router.get_enabled_agent_configs()
        assert configs == {}


class TestToolAgentCallFlow:
    """Test the complete call flow from agent tool to Ray Actor."""

    @patch("geo_edit.environment.tool_agents.manager.get_manager")
    def test_call_agent_delegates_to_manager(self, mock_get_manager):
        """Verify call_agent delegates to ToolAgentManager."""
        from geo_edit.environment.tool_agents import call_agent

        mock_manager = MagicMock()
        mock_manager.call.return_value = "Analysis result"
        mock_get_manager.return_value = mock_manager

        test_image = Image.new("RGB", (100, 100), "white")
        result = call_agent("chartmoe", [test_image], 0, "Analyze this")

        mock_manager.call.assert_called_once_with(
            "chartmoe", [test_image], 0, "Analyze this"
        )
        assert result == "Analysis result"

    def test_chartmoe_execute_calls_call_agent(self):
        """Verify chartmoe.execute calls call_agent with correct parameters."""
        with patch("geo_edit.tool_definitions.agents.chartmoe.call_agent") as mock_call:
            mock_call.return_value = "Chart analysis result"

            from geo_edit.tool_definitions.agents.chartmoe import execute

            test_image = Image.new("RGB", (100, 100), "white")
            result = execute([test_image], 0, "What does this chart show?")

            mock_call.assert_called_once_with(
                "chartmoe", [test_image], 0, "What does this chart show?"
            )
            assert result == "Chart analysis result"

    def test_tool_functions_include_chartmoe_when_enabled(self):
        """Verify chartmoe is in available tools when enabled."""
        from geo_edit.tool_definitions.router import ToolRouter, _TOOL_CONFIG

        router = ToolRouter.__new__(ToolRouter)
        router.tool_mode = "auto"
        router._agents = {}

        functions = router.get_available_tools()

        chartmoe_enabled = _TOOL_CONFIG.get("chartmoe", False)
        if chartmoe_enabled:
            assert "chartmoe" in functions
            assert callable(functions["chartmoe"])


class TestAgentErrorHandling:
    """Test error handling when agent is not initialized."""

    def test_manager_returns_error_when_actor_not_found(self):
        """Verify graceful error when calling uninitialized agent."""
        from geo_edit.environment.tool_agents.manager import ToolAgentManager

        manager = ToolAgentManager()
        test_image = Image.new("RGB", (100, 100), "white")

        result = manager.call("nonexistent_agent", [test_image], 0, "Question")

        assert "Error" in result
        assert "not found" in result

    @patch("geo_edit.environment.tool_agents.manager.ray")
    def test_manager_reconnects_to_existing_actor(self, mock_ray):
        """Verify manager can reconnect to detached actors."""
        from geo_edit.environment.tool_agents.manager import ToolAgentManager

        mock_actor = MagicMock()
        mock_ray.get_actor.return_value = mock_actor

        manager = ToolAgentManager()
        actor = manager.get_actor("chartmoe")

        mock_ray.get_actor.assert_called_once_with("tool_agent_chartmoe")
        assert actor is mock_actor
        assert manager._actors["chartmoe"] is mock_actor
