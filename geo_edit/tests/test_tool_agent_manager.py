"""Tests for ToolAgentManager - environment/tool_agents/manager.py"""

import pytest
import ray
from unittest.mock import MagicMock, patch, PropertyMock
from PIL import Image


@pytest.fixture(scope="module", autouse=True)
def connect_to_ray():
    """Connect to existing Ray cluster before tests."""
    ray.init(address="auto")
    yield


class TestToolAgentManagerInit:
    """Test ToolAgentManager initialization."""

    def test_manager_initializes_empty_actors_dict(self):
        from geo_edit.environment.tool_agents.manager import ToolAgentManager

        manager = ToolAgentManager()
        assert manager._actors == {}
        assert manager._configs == {}

    @patch("geo_edit.environment.tool_agents.manager._MANAGER", None)
    def test_get_manager_returns_singleton(self):
        from geo_edit.environment.tool_agents.manager import get_manager

        manager1 = get_manager()
        manager2 = get_manager()
        assert manager1 is manager2


class TestCreateAgents:
    """Test create_agents method."""

    @patch("geo_edit.environment.tool_agents.manager.ray")
    @patch("geo_edit.environment.tool_agents.manager.get_actor_class")
    def test_create_agents_initializes_ray_when_not_initialized(self, mock_get_actor_class, mock_ray):
        from geo_edit.environment.tool_agents.manager import ToolAgentManager

        mock_ray.is_initialized.return_value = False
        mock_actor_class = MagicMock()
        mock_get_actor_class.return_value = mock_actor_class
        mock_ray.remote.return_value = MagicMock(options=MagicMock(return_value=MagicMock(remote=MagicMock(return_value=MagicMock()))))

        manager = ToolAgentManager()
        configs = {
            "test_agent": {
                "model_name_or_path": "/path/to/model",
                "max_model_len": 8192,
                "gpu_memory_utilization": 0.8,
                "temperature": 0.0,
                "max_tokens": 1024,
            }
        }

        manager.create_agents(configs, ray_address="auto")

        mock_ray.init.assert_called_once_with(address="auto", ignore_reinit_error=True)

    @patch("geo_edit.environment.tool_agents.manager.ray")
    @patch("geo_edit.environment.tool_agents.manager.get_actor_class")
    def test_create_agents_skips_ray_init_when_already_initialized(self, mock_get_actor_class, mock_ray):
        from geo_edit.environment.tool_agents.manager import ToolAgentManager

        mock_ray.is_initialized.return_value = True
        mock_actor_class = MagicMock()
        mock_get_actor_class.return_value = mock_actor_class
        mock_ray.remote.return_value = MagicMock(options=MagicMock(return_value=MagicMock(remote=MagicMock(return_value=MagicMock()))))

        manager = ToolAgentManager()
        configs = {
            "test_agent": {
                "model_name_or_path": "/path/to/model",
                "max_model_len": 8192,
                "gpu_memory_utilization": 0.8,
                "temperature": 0.0,
                "max_tokens": 1024,
            }
        }

        manager.create_agents(configs, ray_address="auto")

        mock_ray.init.assert_not_called()

    @patch("geo_edit.environment.tool_agents.manager.ray")
    @patch("geo_edit.environment.tool_agents.manager.get_actor_class")
    def test_create_agents_builds_actor_with_num_gpus(self, mock_get_actor_class, mock_ray):
        from geo_edit.environment.tool_agents.manager import ToolAgentManager

        mock_ray.is_initialized.return_value = True
        mock_actor_class = MagicMock()
        mock_get_actor_class.return_value = mock_actor_class
        mock_remote_class = MagicMock()
        mock_ray.remote.return_value = mock_remote_class

        manager = ToolAgentManager()
        configs = {
            "test_agent": {
                "model_name_or_path": "/path/to/model",
                "max_model_len": 8192,
                "gpu_memory_utilization": 0.8,
                "temperature": 0.0,
                "max_tokens": 1024,
                "num_gpus": 2,
            }
        }

        manager.create_agents(configs)

        # Verify ray.remote was called with num_gpus=2
        mock_ray.remote.assert_called_once_with(num_gpus=2)

    @patch("geo_edit.environment.tool_agents.manager.ray")
    @patch("geo_edit.environment.tool_agents.manager.get_actor_class")
    def test_create_agents_includes_resources_when_specified(self, mock_get_actor_class, mock_ray):
        from geo_edit.environment.tool_agents.manager import ToolAgentManager

        mock_ray.is_initialized.return_value = True
        mock_actor_class = MagicMock()
        mock_get_actor_class.return_value = mock_actor_class
        mock_remote_class = MagicMock()
        mock_ray.remote.return_value = mock_remote_class

        manager = ToolAgentManager()
        configs = {
            "test_agent": {
                "model_name_or_path": "/path/to/model",
                "max_model_len": 8192,
                "gpu_memory_utilization": 0.8,
                "temperature": 0.0,
                "max_tokens": 1024,
                "resources": {"node1_gpu": 1},
            }
        }

        manager.create_agents(configs)

        # Verify options was called with resources
        call_kwargs = mock_remote_class.options.call_args[1]
        assert call_kwargs["resources"] == {"node1_gpu": 1}

    @patch("geo_edit.environment.tool_agents.manager.ray")
    @patch("geo_edit.environment.tool_agents.manager.get_actor_class")
    def test_create_agents_sets_detached_lifetime(self, mock_get_actor_class, mock_ray):
        from geo_edit.environment.tool_agents.manager import ToolAgentManager

        mock_ray.is_initialized.return_value = True
        mock_actor_class = MagicMock()
        mock_get_actor_class.return_value = mock_actor_class
        mock_remote_class = MagicMock()
        mock_ray.remote.return_value = mock_remote_class

        manager = ToolAgentManager()
        configs = {
            "test_agent": {
                "model_name_or_path": "/path/to/model",
                "max_model_len": 8192,
                "gpu_memory_utilization": 0.8,
                "temperature": 0.0,
                "max_tokens": 1024,
            }
        }

        manager.create_agents(configs)

        call_kwargs = mock_remote_class.options.call_args[1]
        assert call_kwargs["lifetime"] == "detached"

    @patch("geo_edit.environment.tool_agents.manager.ray")
    @patch("geo_edit.environment.tool_agents.manager.get_actor_class")
    def test_create_agents_skips_existing_agent(self, mock_get_actor_class, mock_ray):
        from geo_edit.environment.tool_agents.manager import ToolAgentManager

        mock_ray.is_initialized.return_value = True

        manager = ToolAgentManager()
        manager._actors["existing_agent"] = MagicMock()

        configs = {
            "existing_agent": {
                "model_name_or_path": "/path/to/model",
                "max_model_len": 8192,
                "gpu_memory_utilization": 0.8,
                "temperature": 0.0,
                "max_tokens": 1024,
            }
        }

        manager.create_agents(configs)

        # get_actor_class should not be called for existing agent
        mock_get_actor_class.assert_not_called()


class TestGetActor:
    """Test get_actor method."""

    def test_get_actor_returns_existing_actor(self):
        from geo_edit.environment.tool_agents.manager import ToolAgentManager

        manager = ToolAgentManager()
        mock_actor = MagicMock()
        manager._actors["test_agent"] = mock_actor

        result = manager.get_actor("test_agent")

        assert result is mock_actor

    @patch("geo_edit.environment.tool_agents.manager.ray")
    def test_get_actor_reconnects_by_name(self, mock_ray):
        from geo_edit.environment.tool_agents.manager import ToolAgentManager

        mock_actor = MagicMock()
        mock_ray.get_actor.return_value = mock_actor

        manager = ToolAgentManager()
        result = manager.get_actor("test_agent")

        mock_ray.get_actor.assert_called_once_with("tool_agent_test_agent")
        assert result is mock_actor
        assert manager._actors["test_agent"] is mock_actor

    @patch("geo_edit.environment.tool_agents.manager.ray")
    def test_get_actor_returns_none_when_not_found(self, mock_ray):
        from geo_edit.environment.tool_agents.manager import ToolAgentManager

        mock_ray.get_actor.side_effect = ValueError("Actor not found")

        manager = ToolAgentManager()
        result = manager.get_actor("nonexistent_agent")

        assert result is None


class TestCall:
    """Test call method."""

    def test_call_returns_error_for_missing_agent(self):
        from geo_edit.environment.tool_agents.manager import ToolAgentManager

        manager = ToolAgentManager()
        img = Image.new("RGB", (100, 100), "white")

        result = manager.call("nonexistent", [img], 0, "question")

        assert "Error" in result
        assert "not found" in result

    def test_call_returns_error_for_invalid_image_index(self):
        from geo_edit.environment.tool_agents.manager import ToolAgentManager

        manager = ToolAgentManager()
        manager._actors["test_agent"] = MagicMock()
        manager._configs["test_agent"] = {"temperature": 0.0, "max_tokens": 1024}
        img = Image.new("RGB", (100, 100), "white")

        result = manager.call("test_agent", [img], 5, "question")

        assert "Error" in result
        assert "Invalid image index" in result

    def test_call_returns_error_for_negative_image_index(self):
        from geo_edit.environment.tool_agents.manager import ToolAgentManager

        manager = ToolAgentManager()
        manager._actors["test_agent"] = MagicMock()
        manager._configs["test_agent"] = {"temperature": 0.0, "max_tokens": 1024}
        img = Image.new("RGB", (100, 100), "white")

        result = manager.call("test_agent", [img], -1, "question")

        assert "Error" in result
        assert "Invalid image index" in result

    @patch("geo_edit.environment.tool_agents.manager.ray")
    def test_call_converts_image_to_base64(self, mock_ray):
        from geo_edit.environment.tool_agents.manager import ToolAgentManager

        mock_actor = MagicMock()
        mock_ray.get.return_value = "analysis result"

        manager = ToolAgentManager()
        manager._actors["test_agent"] = mock_actor
        manager._configs["test_agent"] = {"temperature": 0.5, "max_tokens": 2048}

        img = Image.new("RGB", (100, 100), "red")
        result = manager.call("test_agent", [img], 0, "What color?")

        # Verify analyze.remote was called
        mock_actor.analyze.remote.assert_called_once()
        call_args = mock_actor.analyze.remote.call_args[0]
        # First arg should be base64 string
        assert isinstance(call_args[0], str)
        # Second arg should be question
        assert call_args[1] == "What color?"
        # Third and fourth args should be temperature and max_tokens
        assert call_args[2] == 0.5
        assert call_args[3] == 2048

    @patch("geo_edit.environment.tool_agents.manager.ray")
    def test_call_returns_result_from_actor(self, mock_ray):
        from geo_edit.environment.tool_agents.manager import ToolAgentManager

        mock_actor = MagicMock()
        mock_ray.get.return_value = "The image shows a red square"

        manager = ToolAgentManager()
        manager._actors["test_agent"] = mock_actor
        manager._configs["test_agent"] = {"temperature": 0.0, "max_tokens": 1024}

        img = Image.new("RGB", (100, 100), "red")
        result = manager.call("test_agent", [img], 0, "Describe")

        assert result == "The image shows a red square"


class TestShutdown:
    """Test shutdown method."""

    @patch("geo_edit.environment.tool_agents.manager.ray")
    def test_shutdown_kills_specified_agents(self, mock_ray):
        from geo_edit.environment.tool_agents.manager import ToolAgentManager

        mock_actor1 = MagicMock()
        mock_actor2 = MagicMock()

        manager = ToolAgentManager()
        manager._actors = {"agent1": mock_actor1, "agent2": mock_actor2}
        manager._configs = {"agent1": {}, "agent2": {}}

        manager.shutdown(["agent1"])

        mock_ray.kill.assert_called_once_with(mock_actor1)
        assert "agent1" not in manager._actors
        assert "agent2" in manager._actors

    @patch("geo_edit.environment.tool_agents.manager.ray")
    def test_shutdown_all_when_no_names_provided(self, mock_ray):
        from geo_edit.environment.tool_agents.manager import ToolAgentManager

        mock_actor1 = MagicMock()
        mock_actor2 = MagicMock()

        manager = ToolAgentManager()
        manager._actors = {"agent1": mock_actor1, "agent2": mock_actor2}
        manager._configs = {"agent1": {}, "agent2": {}}

        manager.shutdown()

        assert mock_ray.kill.call_count == 2
        assert manager._actors == {}
        assert manager._configs == {}

    @patch("geo_edit.environment.tool_agents.manager.ray")
    def test_shutdown_removes_from_internal_dicts(self, mock_ray):
        from geo_edit.environment.tool_agents.manager import ToolAgentManager

        mock_actor = MagicMock()

        manager = ToolAgentManager()
        manager._actors = {"agent1": mock_actor}
        manager._configs = {"agent1": {"some": "config"}}

        manager.shutdown(["agent1"])

        assert "agent1" not in manager._actors
        assert "agent1" not in manager._configs


class TestStatus:
    """Test status method."""

    @patch("geo_edit.environment.tool_agents.manager.ray")
    def test_status_returns_healthy_for_working_actors(self, mock_ray):
        from geo_edit.environment.tool_agents.manager import ToolAgentManager

        mock_actor = MagicMock()
        mock_ray.get.return_value = {"model": "test_model", "initialized": True}

        manager = ToolAgentManager()
        manager._actors = {"test_agent": mock_actor}

        result = manager.status()

        assert "test_agent" in result
        assert result["test_agent"]["status"] == "healthy"
        assert result["test_agent"]["model"] == "test_model"

    @patch("geo_edit.environment.tool_agents.manager.ray")
    def test_status_returns_error_for_failed_actors(self, mock_ray):
        from geo_edit.environment.tool_agents.manager import ToolAgentManager

        mock_actor = MagicMock()
        mock_ray.get.side_effect = Exception("Actor died")

        manager = ToolAgentManager()
        manager._actors = {"test_agent": mock_actor}

        result = manager.status()

        assert "test_agent" in result
        assert result["test_agent"]["status"] == "error"
        assert "Actor died" in result["test_agent"]["error"]
