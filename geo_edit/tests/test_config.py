"""Tests for config.py - build_google_agent_configs and build_api_agent_configs."""

import pytest
from unittest.mock import patch, MagicMock


class TestAPIAgentConfigsDataclass:
    """Test APIAgentConfigs dataclass."""

    def test_api_agent_configs_fields(self):
        from geo_edit.config import APIAgentConfigs

        config = APIAgentConfigs(
            tools=[{"name": "test"}],
            tool_choice="auto",
            generate_config={"temperature": 1.0},
            force_final_generate_config={"temperature": 1.0, "tool_choice": "none"},
            system_prompt="Test prompt",
        )
        assert config.tools == [{"name": "test"}]
        assert config.tool_choice == "auto"
        assert config.generate_config == {"temperature": 1.0}
        assert config.system_prompt == "Test prompt"


class TestBuildAPIAgentConfigs:
    """Test build_api_agent_configs function."""

    @patch("geo_edit.config.get_tool_declarations")
    def test_responses_api_mode_tool_format(self, mock_get_decls):
        from geo_edit.config import build_api_agent_configs

        mock_get_decls.return_value = [
            {"name": "test_tool", "description": "A test tool", "parameters": {"type": "object"}}
        ]

        config = build_api_agent_configs(api_mode="responses", tool_mode="auto")

        # Responses API uses flat format
        assert config.tools is not None
        assert len(config.tools) == 1
        assert config.tools[0]["type"] == "function"
        assert config.tools[0]["name"] == "test_tool"
        assert "function" not in config.tools[0]  # No nested "function" key

    @patch("geo_edit.config.get_tool_declarations")
    def test_chat_completions_api_mode_tool_format(self, mock_get_decls):
        from geo_edit.config import build_api_agent_configs

        mock_get_decls.return_value = [
            {"name": "test_tool", "description": "A test tool", "parameters": {"type": "object"}}
        ]

        config = build_api_agent_configs(api_mode="chat_completions", tool_mode="auto")

        # Chat completions API uses nested format
        assert config.tools is not None
        assert len(config.tools) == 1
        assert config.tools[0]["type"] == "function"
        assert "function" in config.tools[0]  # Has nested "function" key
        assert config.tools[0]["function"]["name"] == "test_tool"

    @patch("geo_edit.config.get_tool_declarations")
    def test_direct_mode_disables_tools(self, mock_get_decls):
        from geo_edit.config import build_api_agent_configs

        mock_get_decls.return_value = [
            {"name": "test_tool", "description": "A test tool", "parameters": {"type": "object"}}
        ]

        config = build_api_agent_configs(api_mode="responses", tool_mode="direct")

        assert config.tools is None
        assert "tools" not in config.generate_config

    @patch("geo_edit.config.get_tool_declarations")
    def test_force_mode_sets_tool_choice_required(self, mock_get_decls):
        from geo_edit.config import build_api_agent_configs

        mock_get_decls.return_value = [
            {"name": "test_tool", "description": "A test tool", "parameters": {"type": "object"}}
        ]

        config = build_api_agent_configs(api_mode="responses", tool_mode="force")

        assert config.tool_choice == "required"
        assert config.generate_config["tool_choice"] == "required"

    @patch("geo_edit.config.get_tool_declarations")
    def test_auto_mode_sets_tool_choice_auto(self, mock_get_decls):
        from geo_edit.config import build_api_agent_configs

        mock_get_decls.return_value = [
            {"name": "test_tool", "description": "A test tool", "parameters": {"type": "object"}}
        ]

        config = build_api_agent_configs(api_mode="responses", tool_mode="auto")

        assert config.tool_choice == "auto"
        assert config.generate_config["tool_choice"] == "auto"

    @patch("geo_edit.config.get_tool_declarations")
    def test_force_final_config_sets_tool_choice_none(self, mock_get_decls):
        from geo_edit.config import build_api_agent_configs

        mock_get_decls.return_value = [
            {"name": "test_tool", "description": "A test tool", "parameters": {"type": "object"}}
        ]

        config = build_api_agent_configs(api_mode="responses", tool_mode="auto")

        assert config.force_final_generate_config["tool_choice"] == "none"

    @patch("geo_edit.config.get_tool_declarations")
    def test_responses_api_uses_max_output_tokens_key(self, mock_get_decls):
        from geo_edit.config import build_api_agent_configs

        mock_get_decls.return_value = []

        config = build_api_agent_configs(
            api_mode="responses", tool_mode="direct", max_output_tokens=1000
        )

        assert "max_output_tokens" in config.generate_config
        assert config.generate_config["max_output_tokens"] == 1000

    @patch("geo_edit.config.get_tool_declarations")
    def test_chat_completions_api_uses_max_tokens_key(self, mock_get_decls):
        from geo_edit.config import build_api_agent_configs

        mock_get_decls.return_value = []

        config = build_api_agent_configs(
            api_mode="chat_completions", tool_mode="direct", max_output_tokens=1000
        )

        assert "max_tokens" in config.generate_config
        assert config.generate_config["max_tokens"] == 1000

    @patch("geo_edit.config.get_tool_declarations")
    def test_responses_api_includes_reasoning(self, mock_get_decls):
        from geo_edit.config import build_api_agent_configs

        mock_get_decls.return_value = []

        config = build_api_agent_configs(
            api_mode="responses", tool_mode="direct", reasoning_level="low"
        )

        assert "reasoning" in config.generate_config
        assert config.generate_config["reasoning"]["effort"] == "low"

    @patch("geo_edit.config.get_tool_declarations")
    def test_responses_api_includes_instructions(self, mock_get_decls):
        from geo_edit.config import build_api_agent_configs

        mock_get_decls.return_value = []

        config = build_api_agent_configs(
            api_mode="responses", tool_mode="direct", system_prompt="Test system prompt"
        )

        assert "instructions" in config.generate_config
        assert config.generate_config["instructions"] == "Test system prompt"

    def test_invalid_api_mode_raises_error(self):
        from geo_edit.config import build_api_agent_configs

        with pytest.raises(ValueError, match="Invalid api_mode"):
            build_api_agent_configs(api_mode="invalid", tool_mode="auto")

    @patch("geo_edit.config.get_tool_declarations")
    def test_invalid_tool_mode_raises_error(self, mock_get_decls):
        from geo_edit.config import build_api_agent_configs

        mock_get_decls.return_value = []

        with pytest.raises(ValueError, match="Invalid tool_mode"):
            build_api_agent_configs(api_mode="responses", tool_mode="invalid")

    @patch("geo_edit.config.get_tool_declarations")
    def test_temperature_is_set(self, mock_get_decls):
        from geo_edit.config import build_api_agent_configs

        mock_get_decls.return_value = []

        config = build_api_agent_configs(
            api_mode="responses", tool_mode="direct", temperature=0.5
        )

        assert config.generate_config["temperature"] == 0.5


class TestBuildGoogleAgentConfigs:
    """Test build_google_agent_configs function."""

    @patch("geo_edit.config.get_tool_declarations")
    @patch("geo_edit.config.get_tool_functions")
    def test_auto_mode_sets_function_calling_mode(self, mock_get_funcs, mock_get_decls):
        from geo_edit.config import build_google_agent_configs

        mock_get_decls.return_value = [
            {"name": "test_tool", "description": "A test tool", "parameters": {"type": "object"}}
        ]
        mock_get_funcs.return_value = {"test_tool": lambda: None}

        config = build_google_agent_configs(tool_mode="auto")

        assert config.tools is not None
        assert config.tool_config is not None

    @patch("geo_edit.config.get_tool_declarations")
    @patch("geo_edit.config.get_tool_functions")
    def test_direct_mode_disables_tools(self, mock_get_funcs, mock_get_decls):
        from geo_edit.config import build_google_agent_configs

        mock_get_decls.return_value = []
        mock_get_funcs.return_value = {}

        config = build_google_agent_configs(tool_mode="direct")

        assert config.tools is None
        assert config.tool_config is None

    @patch("geo_edit.config.get_tool_declarations")
    @patch("geo_edit.config.get_tool_functions")
    def test_force_mode_sets_any_mode(self, mock_get_funcs, mock_get_decls):
        from geo_edit.config import build_google_agent_configs

        mock_get_decls.return_value = [
            {"name": "test_tool", "description": "A test tool", "parameters": {"type": "object"}}
        ]
        mock_get_funcs.return_value = {"test_tool": lambda: None}

        config = build_google_agent_configs(tool_mode="force")

        assert config.tools is not None
        assert config.tool_config is not None

    @patch("geo_edit.config.get_tool_declarations")
    @patch("geo_edit.config.get_tool_functions")
    def test_invalid_tool_mode_raises_error(self, mock_get_funcs, mock_get_decls):
        from geo_edit.config import build_google_agent_configs

        mock_get_decls.return_value = []
        mock_get_funcs.return_value = {}

        with pytest.raises(ValueError, match="Invalid tool_mode"):
            build_google_agent_configs(tool_mode="invalid")

    @patch("geo_edit.config.get_tool_declarations")
    @patch("geo_edit.config.get_tool_functions")
    def test_thinking_config_is_set(self, mock_get_funcs, mock_get_decls):
        from geo_edit.config import build_google_agent_configs

        mock_get_decls.return_value = []
        mock_get_funcs.return_value = {}

        config = build_google_agent_configs(
            tool_mode="direct", thinking_level="high", include_thoughts=True
        )

        # Config should be created successfully with thinking options
        assert config.generate_config is not None


class TestConfigExports:
    """Test module exports."""

    def test_all_exports_defined(self):
        from geo_edit import config

        assert hasattr(config, "GoogleAgentConfigs")
        assert hasattr(config, "APIAgentConfigs")
        assert hasattr(config, "build_google_agent_configs")
        assert hasattr(config, "build_api_agent_configs")
