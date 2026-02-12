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

    def _create_mock_router(self, tool_mode: str, declarations: list, tools: dict):
        """Create a mock ToolRouter."""
        router = MagicMock()
        router.tool_mode = tool_mode
        router.get_available_declarations.return_value = declarations
        router.get_available_tools.return_value = tools
        return router

    def test_responses_api_mode_tool_format(self):
        from geo_edit.config import build_api_agent_configs

        router = self._create_mock_router(
            tool_mode="auto",
            declarations=[{"name": "test_tool", "description": "A test tool", "parameters": {"type": "object"}}],
            tools={"test_tool": lambda: None},
        )

        config = build_api_agent_configs(router, api_mode="responses")

        # Responses API uses flat format
        assert config.tools is not None
        assert len(config.tools) == 1
        assert config.tools[0]["type"] == "function"
        assert config.tools[0]["name"] == "test_tool"
        assert "function" not in config.tools[0]  # No nested "function" key

    def test_chat_completions_api_mode_tool_format(self):
        from geo_edit.config import build_api_agent_configs

        router = self._create_mock_router(
            tool_mode="auto",
            declarations=[{"name": "test_tool", "description": "A test tool", "parameters": {"type": "object"}}],
            tools={"test_tool": lambda: None},
        )

        config = build_api_agent_configs(router, api_mode="chat_completions")

        # Chat completions API uses nested format
        assert config.tools is not None
        assert len(config.tools) == 1
        assert config.tools[0]["type"] == "function"
        assert "function" in config.tools[0]  # Has nested "function" key
        assert config.tools[0]["function"]["name"] == "test_tool"

    def test_direct_mode_disables_tools(self):
        from geo_edit.config import build_api_agent_configs

        router = self._create_mock_router(
            tool_mode="direct",
            declarations=[],
            tools={},
        )

        config = build_api_agent_configs(router, api_mode="responses")

        assert config.tools is None
        assert "tools" not in config.generate_config

    def test_force_mode_sets_tool_choice_required(self):
        from geo_edit.config import build_api_agent_configs

        router = self._create_mock_router(
            tool_mode="force",
            declarations=[{"name": "test_tool", "description": "A test tool", "parameters": {"type": "object"}}],
            tools={"test_tool": lambda: None},
        )

        config = build_api_agent_configs(router, api_mode="responses")

        assert config.tool_choice == "required"
        assert config.generate_config["tool_choice"] == "required"

    def test_auto_mode_sets_tool_choice_auto(self):
        from geo_edit.config import build_api_agent_configs

        router = self._create_mock_router(
            tool_mode="auto",
            declarations=[{"name": "test_tool", "description": "A test tool", "parameters": {"type": "object"}}],
            tools={"test_tool": lambda: None},
        )

        config = build_api_agent_configs(router, api_mode="responses")

        assert config.tool_choice == "auto"
        assert config.generate_config["tool_choice"] == "auto"

    def test_force_final_config_sets_tool_choice_none(self):
        from geo_edit.config import build_api_agent_configs

        router = self._create_mock_router(
            tool_mode="auto",
            declarations=[{"name": "test_tool", "description": "A test tool", "parameters": {"type": "object"}}],
            tools={"test_tool": lambda: None},
        )

        config = build_api_agent_configs(router, api_mode="responses")

        assert config.force_final_generate_config["tool_choice"] == "none"

    def test_responses_api_uses_max_output_tokens_key(self):
        from geo_edit.config import build_api_agent_configs

        router = self._create_mock_router(tool_mode="direct", declarations=[], tools={})

        config = build_api_agent_configs(router, api_mode="responses", max_output_tokens=1000)

        assert "max_output_tokens" in config.generate_config
        assert config.generate_config["max_output_tokens"] == 1000

    def test_chat_completions_api_uses_max_tokens_key(self):
        from geo_edit.config import build_api_agent_configs

        router = self._create_mock_router(tool_mode="direct", declarations=[], tools={})

        config = build_api_agent_configs(router, api_mode="chat_completions", max_output_tokens=1000)

        assert "max_tokens" in config.generate_config
        assert config.generate_config["max_tokens"] == 1000

    def test_responses_api_includes_reasoning(self):
        from geo_edit.config import build_api_agent_configs

        router = self._create_mock_router(tool_mode="direct", declarations=[], tools={})

        config = build_api_agent_configs(router, api_mode="responses", reasoning_level="low")

        assert "reasoning" in config.generate_config
        assert config.generate_config["reasoning"]["effort"] == "low"

    def test_responses_api_includes_instructions(self):
        from geo_edit.config import build_api_agent_configs

        router = self._create_mock_router(tool_mode="direct", declarations=[], tools={})

        config = build_api_agent_configs(router, api_mode="responses", system_prompt="Test system prompt")

        assert "instructions" in config.generate_config
        assert config.generate_config["instructions"] == "Test system prompt"

    def test_invalid_api_mode_raises_error(self):
        from geo_edit.config import build_api_agent_configs

        router = self._create_mock_router(tool_mode="auto", declarations=[], tools={})

        with pytest.raises(ValueError, match="Invalid api_mode"):
            build_api_agent_configs(router, api_mode="invalid")

    def test_invalid_tool_mode_raises_error(self):
        from geo_edit.config import build_api_agent_configs

        router = MagicMock()
        router.tool_mode = "invalid"
        router.get_available_declarations.return_value = []
        router.get_available_tools.return_value = {}

        with pytest.raises(ValueError, match="Invalid tool_mode"):
            build_api_agent_configs(router, api_mode="responses")

    def test_temperature_is_set(self):
        from geo_edit.config import build_api_agent_configs

        router = self._create_mock_router(tool_mode="direct", declarations=[], tools={})

        config = build_api_agent_configs(router, api_mode="responses", temperature=0.5)

        assert config.generate_config["temperature"] == 0.5


class TestBuildGoogleAgentConfigs:
    """Test build_google_agent_configs function."""

    def _create_mock_router(self, tool_mode: str, declarations: list, tools: dict):
        """Create a mock ToolRouter."""
        router = MagicMock()
        router.tool_mode = tool_mode
        router.get_available_declarations.return_value = declarations
        router.get_available_tools.return_value = tools
        return router

    def test_auto_mode_sets_function_calling_mode(self):
        from geo_edit.config import build_google_agent_configs

        router = self._create_mock_router(
            tool_mode="auto",
            declarations=[{"name": "test_tool", "description": "A test tool", "parameters": {"type": "object"}}],
            tools={"test_tool": lambda: None},
        )

        config = build_google_agent_configs(router)

        assert config.tools is not None
        assert config.tool_config is not None

    def test_direct_mode_disables_tools(self):
        from geo_edit.config import build_google_agent_configs

        router = self._create_mock_router(tool_mode="direct", declarations=[], tools={})

        config = build_google_agent_configs(router)

        assert config.tools is None
        assert config.tool_config is None

    def test_force_mode_sets_any_mode(self):
        from geo_edit.config import build_google_agent_configs

        router = self._create_mock_router(
            tool_mode="force",
            declarations=[{"name": "test_tool", "description": "A test tool", "parameters": {"type": "object"}}],
            tools={"test_tool": lambda: None},
        )

        config = build_google_agent_configs(router)

        assert config.tools is not None
        assert config.tool_config is not None

    def test_thinking_config_is_set(self):
        from geo_edit.config import build_google_agent_configs

        router = self._create_mock_router(tool_mode="direct", declarations=[], tools={})

        config = build_google_agent_configs(router, thinking_level="high", include_thoughts=True)

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
