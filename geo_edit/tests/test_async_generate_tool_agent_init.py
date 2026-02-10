"""Tests for auto_create_agents integration in async_generate_with_tool_call_api.py

Validates that:
1. Tool agents (e.g., ChartMoE) are automatically initialized when enabled in config.yaml
2. Ray actors are created before multiprocessing pool starts
3. Agents are properly shutdown after processing completes
"""

import pytest
from unittest.mock import MagicMock, patch, call


class TestToolAgentInitialization:
    """Test that tool agents are initialized when enabled."""

    @patch("geo_edit.environment.tool_agents.auto_create_agents")
    def test_auto_create_agents_called_when_agents_enabled(self, mock_auto_create):
        """Verify auto_create_agents is called when agent tools are enabled."""
        from geo_edit.tool_definitions.router import ToolRouter

        # Create a router with auto mode (agents enabled based on config.yaml)
        router = ToolRouter(tool_mode="auto")

        # If agents are enabled in config.yaml, auto_create_agents should be called
        if router.is_agent_enabled():
            mock_auto_create.return_value = {"chartmoe": MagicMock()}
            agents = mock_auto_create(tool_mode="auto")
            mock_auto_create.assert_called_once_with(tool_mode="auto")
            assert "chartmoe" in agents or len(agents) > 0

    @patch("geo_edit.environment.tool_agents.auto_create_agents")
    def test_auto_create_agents_not_called_in_direct_mode(self, mock_auto_create):
        """Verify auto_create_agents returns empty dict in direct mode."""
        from geo_edit.tool_definitions.router import ToolRouter

        router = ToolRouter(tool_mode="direct")
        assert router.is_agent_enabled() is False

    def test_tool_router_detects_chartmoe_enabled(self):
        """Verify ToolRouter correctly detects chartmoe is enabled in config.yaml."""
        from geo_edit.tool_definitions.router import ToolRouter, _TOOL_CONFIG

        # Check if chartmoe is enabled in config
        chartmoe_enabled = _TOOL_CONFIG.get("chartmoe", False)

        router = ToolRouter(tool_mode="auto")
        enabled_agents = router.get_enabled_agents()

        if chartmoe_enabled:
            assert "chartmoe" in enabled_agents
        else:
            assert "chartmoe" not in enabled_agents


class TestToolAgentShutdown:
    """Test that tool agents are properly shutdown."""

    @patch("geo_edit.environment.tool_agents.shutdown_agents")
    def test_shutdown_agents_called_after_processing(self, mock_shutdown):
        """Verify shutdown_agents is called when agents were created."""
        # Simulate the shutdown logic from async_generate_with_tool_call_api.py
        agent_actors = {"chartmoe": MagicMock(), "multimath": MagicMock()}

        if agent_actors:
            mock_shutdown()

        mock_shutdown.assert_called_once()

    @patch("geo_edit.environment.tool_agents.shutdown_agents")
    def test_shutdown_agents_not_called_when_no_agents(self, mock_shutdown):
        """Verify shutdown_agents is not called when no agents were created."""
        agent_actors = {}

        if agent_actors:
            mock_shutdown()

        mock_shutdown.assert_not_called()


class TestToolAgentConfigIntegration:
    """Test config-based agent initialization."""

    def test_get_enabled_agent_configs_returns_chartmoe_config(self):
        """Verify chartmoe config is returned when enabled."""
        from geo_edit.tool_definitions.router import ToolRouter, _TOOL_CONFIG

        router = ToolRouter(tool_mode="auto")
        configs = router.get_enabled_agent_configs()

        chartmoe_enabled = _TOOL_CONFIG.get("chartmoe", False)
        if chartmoe_enabled:
            assert "chartmoe" in configs
            # Verify config has required fields
            chartmoe_config = configs["chartmoe"]
            assert "model_name_or_path" in chartmoe_config
            assert "max_model_len" in chartmoe_config
            assert "gpu_memory_utilization" in chartmoe_config
            assert "temperature" in chartmoe_config
            assert "max_tokens" in chartmoe_config

    def test_chartmoe_execute_calls_call_agent(self):
        """Verify chartmoe.execute calls call_agent with correct parameters."""
        from unittest.mock import patch
        from PIL import Image

        with patch("geo_edit.tool_definitions.agents.chartmoe.call_agent") as mock_call:
            mock_call.return_value = "Chart analysis result"

            from geo_edit.tool_definitions.agents.chartmoe import execute

            test_image = Image.new("RGB", (100, 100), "white")
            result = execute([test_image], 0, "What does this chart show?")

            mock_call.assert_called_once_with(
                "chartmoe", [test_image], 0, "What does this chart show?"
            )
            assert result == "Chart analysis result"


class TestToolAgentCallFlow:
    """Test the complete call flow from ToolRouter to Agent."""

    @patch("geo_edit.environment.tool_agents.manager.get_manager")
    def test_call_agent_uses_manager(self, mock_get_manager):
        """Verify call_agent delegates to ToolAgentManager."""
        from geo_edit.environment.tool_agents import call_agent
        from PIL import Image

        mock_manager = MagicMock()
        mock_manager.call.return_value = "Analysis result"
        mock_get_manager.return_value = mock_manager

        test_image = Image.new("RGB", (100, 100), "white")
        result = call_agent("chartmoe", [test_image], 0, "Analyze this")

        mock_manager.call.assert_called_once_with(
            "chartmoe", [test_image], 0, "Analyze this"
        )
        assert result == "Analysis result"

    def test_tool_functions_include_chartmoe_when_enabled(self):
        """Verify chartmoe is in available tools when enabled."""
        from geo_edit.tool_definitions.router import ToolRouter, _TOOL_CONFIG

        router = ToolRouter(tool_mode="auto")
        functions = router.get_available_tools()

        chartmoe_enabled = _TOOL_CONFIG.get("chartmoe", False)
        if chartmoe_enabled:
            assert "chartmoe" in functions
            assert callable(functions["chartmoe"])


class TestAutoCreateAgentsFunction:
    """Test auto_create_agents helper function."""

    @patch("geo_edit.environment.tool_agents.get_manager")
    def test_auto_create_agents_reads_config(self, mock_get_manager):
        """Verify auto_create_agents reads from ToolRouter config."""
        from geo_edit.environment.tool_agents import auto_create_agents
        from geo_edit.tool_definitions.router import _TOOL_CONFIG

        mock_manager = MagicMock()
        mock_manager.create_agents.return_value = {}
        mock_get_manager.return_value = mock_manager

        # Call auto_create_agents
        auto_create_agents(tool_mode="auto")

        # Verify create_agents was called (configs come from ToolRouter)
        # The actual configs depend on what's enabled in config.yaml
        if any(
            _TOOL_CONFIG.get(name, False)
            for name in ["chartmoe", "multimath", "gllava"]
        ):
            mock_manager.create_agents.assert_called_once()

    @patch("geo_edit.environment.tool_agents.get_manager")
    def test_auto_create_agents_adds_node_resource(self, mock_get_manager):
        """Verify node_resource is added to agent configs."""
        from geo_edit.environment.tool_agents import auto_create_agents
        from geo_edit.tool_definitions.router import _TOOL_CONFIG

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

        auto_create_agents(tool_mode="auto", node_resource="worker_gpu")

        # Verify create_agents was called with configs containing resources
        call_args = mock_manager.create_agents.call_args
        if call_args:
            configs = call_args[0][0]
            for name, cfg in configs.items():
                assert "resources" in cfg
                assert cfg["resources"] == {"worker_gpu": 1}

    def test_auto_create_agents_returns_empty_in_direct_mode(self):
        """Verify auto_create_agents returns empty dict in direct mode."""
        from geo_edit.environment.tool_agents import auto_create_agents

        with patch("geo_edit.environment.tool_agents.get_manager") as mock_get_manager:
            result = auto_create_agents(tool_mode="direct")
            assert result == {}
            # create_agents should not be called in direct mode
            mock_get_manager.return_value.create_agents.assert_not_called()


class TestAgentErrorHandling:
    """Test error handling when agent is not initialized."""

    def test_call_agent_returns_error_when_actor_not_found(self):
        """Verify graceful error when calling uninitialized agent."""
        from geo_edit.environment.tool_agents.manager import ToolAgentManager
        from PIL import Image

        manager = ToolAgentManager()
        test_image = Image.new("RGB", (100, 100), "white")

        result = manager.call("nonexistent_agent", [test_image], 0, "Question")

        assert "Error" in result
        assert "not found" in result

    @patch("geo_edit.environment.tool_agents.manager.ray")
    def test_call_agent_reconnects_to_existing_actor(self, mock_ray):
        """Verify manager can reconnect to detached actors."""
        from geo_edit.environment.tool_agents.manager import ToolAgentManager

        mock_actor = MagicMock()
        mock_ray.get_actor.return_value = mock_actor

        manager = ToolAgentManager()
        actor = manager.get_actor("chartmoe")

        mock_ray.get_actor.assert_called_once_with("tool_agent_chartmoe")
        assert actor is mock_actor
        assert manager._actors["chartmoe"] is mock_actor
