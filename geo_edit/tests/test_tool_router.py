"""Tests for ToolRouter - tool_definitions/router.py"""

import pytest
from unittest.mock import patch, MagicMock


class TestToolRouterInit:
    """Test ToolRouter initialization."""

    @patch("geo_edit.tool_definitions.router.get_manager")
    def test_init_with_auto_mode(self, mock_get_manager):
        from geo_edit.tool_definitions.router import ToolRouter

        mock_get_manager.return_value.create_agents.return_value = {}
        router = ToolRouter(tool_mode="auto")
        assert router.tool_mode == "auto"

    @patch("geo_edit.tool_definitions.router.get_manager")
    def test_init_with_force_mode(self, mock_get_manager):
        from geo_edit.tool_definitions.router import ToolRouter

        mock_get_manager.return_value.create_agents.return_value = {}
        router = ToolRouter(tool_mode="force")
        assert router.tool_mode == "force"

    def test_init_with_direct_mode(self):
        from geo_edit.tool_definitions.router import ToolRouter

        # direct mode doesn't create agents, no mock needed
        router = ToolRouter(tool_mode="direct")
        assert router.tool_mode == "direct"

    @patch("geo_edit.tool_definitions.router.get_manager")
    def test_init_default_mode_is_auto(self, mock_get_manager):
        from geo_edit.tool_definitions.router import ToolRouter

        mock_get_manager.return_value.create_agents.return_value = {}
        router = ToolRouter()
        assert router.tool_mode == "auto"


class TestToolRouterDirectMode:
    """Test ToolRouter behavior in direct mode (no tools)."""

    def test_get_available_declarations_returns_empty_in_direct_mode(self):
        from geo_edit.tool_definitions.router import ToolRouter

        router = ToolRouter(tool_mode="direct")
        declarations = router.get_available_declarations()
        assert declarations == []

    def test_get_available_tools_returns_empty_in_direct_mode(self):
        from geo_edit.tool_definitions.router import ToolRouter

        router = ToolRouter(tool_mode="direct")
        functions = router.get_available_tools()
        assert functions == {}

    def test_get_enabled_agents_returns_empty_in_direct_mode(self):
        from geo_edit.tool_definitions.router import ToolRouter

        router = ToolRouter(tool_mode="direct")
        agents = router.get_enabled_agents()
        assert agents == []

    def test_is_tool_enabled_returns_false_in_direct_mode(self):
        from geo_edit.tool_definitions.router import ToolRouter

        router = ToolRouter(tool_mode="direct")
        assert router.is_tool_enabled() is False


class TestToolRouterAutoMode:
    """Test ToolRouter behavior in auto mode."""

    @patch("geo_edit.tool_definitions.router.get_manager")
    def test_get_available_declarations_returns_list(self, mock_get_manager):
        from geo_edit.tool_definitions.router import ToolRouter

        mock_get_manager.return_value.create_agents.return_value = {}
        router = ToolRouter(tool_mode="auto")
        declarations = router.get_available_declarations()
        assert isinstance(declarations, list)

    @patch("geo_edit.tool_definitions.router.get_manager")
    def test_declarations_have_required_fields(self, mock_get_manager):
        from geo_edit.tool_definitions.router import ToolRouter

        mock_get_manager.return_value.create_agents.return_value = {}
        router = ToolRouter(tool_mode="auto")
        declarations = router.get_available_declarations()
        for decl in declarations:
            assert "name" in decl
            assert "description" in decl
            assert "parameters" in decl

    @patch("geo_edit.tool_definitions.router.get_manager")
    def test_get_available_tools_returns_dict(self, mock_get_manager):
        from geo_edit.tool_definitions.router import ToolRouter

        mock_get_manager.return_value.create_agents.return_value = {}
        router = ToolRouter(tool_mode="auto")
        functions = router.get_available_tools()
        assert isinstance(functions, dict)

    @patch("geo_edit.tool_definitions.router.get_manager")
    def test_tools_are_callable(self, mock_get_manager):
        from geo_edit.tool_definitions.router import ToolRouter

        mock_get_manager.return_value.create_agents.return_value = {}
        router = ToolRouter(tool_mode="auto")
        functions = router.get_available_tools()
        for name, func in functions.items():
            assert callable(func), f"Function {name} is not callable"


class TestToolRouterToolRegistry:
    """Test tool registry content."""

    def test_function_tools_registered(self):
        from geo_edit.tool_definitions.functions import FUNCTION_TOOLS

        expected_tools = ["image_crop", "image_label", "draw_line", "bounding_box", "image_highlight"]
        for tool_name in expected_tools:
            assert tool_name in FUNCTION_TOOLS, f"Missing function tool: {tool_name}"

    def test_agent_tools_registered(self):
        from geo_edit.tool_definitions.agents import AGENT_TOOLS

        expected_tools = ["multimath", "gllava", "chartmoe"]
        for tool_name in expected_tools:
            assert tool_name in AGENT_TOOLS, f"Missing agent tool: {tool_name}"

    def test_tool_registry_structure(self):
        from geo_edit.tool_definitions.functions import FUNCTION_TOOLS
        from geo_edit.tool_definitions.agents import AGENT_TOOLS

        all_tools = {**FUNCTION_TOOLS, **AGENT_TOOLS}
        for name, entry in all_tools.items():
            assert isinstance(entry, tuple), f"Tool {name} entry is not a tuple"
            assert len(entry) == 4, f"Tool {name} entry should have 4 elements"
            declaration, func, tool_type, return_type = entry
            assert isinstance(declaration, dict), f"Tool {name} declaration is not a dict"
            assert callable(func), f"Tool {name} function is not callable"
            assert tool_type in ("function", "agent"), f"Tool {name} has invalid type"
            assert return_type in ("image", "text"), f"Tool {name} has invalid return type"


class TestToolRouterReturnTypes:
    """Test tool return type methods."""

    @patch("geo_edit.tool_definitions.router.get_manager")
    def test_get_tool_return_type_for_image_tools(self, mock_get_manager):
        from geo_edit.tool_definitions.router import ToolRouter

        mock_get_manager.return_value.create_agents.return_value = {}
        router = ToolRouter(tool_mode="auto")
        image_tools = ["image_crop", "image_label", "draw_line", "bounding_box", "image_highlight"]
        for tool_name in image_tools:
            assert router.get_tool_return_type(tool_name) == "image"

    @patch("geo_edit.tool_definitions.router.get_manager")
    def test_get_tool_return_type_for_agent_tools(self, mock_get_manager):
        from geo_edit.tool_definitions.router import ToolRouter

        mock_get_manager.return_value.create_agents.return_value = {}
        router = ToolRouter(tool_mode="auto")
        agent_tools = ["multimath", "gllava", "chartmoe"]
        for tool_name in agent_tools:
            assert router.get_tool_return_type(tool_name) == "text"


class TestToolRouterAgentMethods:
    """Test agent-specific methods."""

    @patch("geo_edit.tool_definitions.router.get_manager")
    def test_get_enabled_agents_returns_agent_type_only(self, mock_get_manager):
        from geo_edit.tool_definitions.router import ToolRouter

        mock_get_manager.return_value.create_agents.return_value = {}
        router = ToolRouter(tool_mode="auto")
        agents = router.get_enabled_agents()
        # Should only contain agent-type tools, not function-type
        function_tools = ["image_crop", "image_label", "draw_line", "bounding_box", "image_highlight"]
        for func_tool in function_tools:
            assert func_tool not in agents

    @patch("geo_edit.tool_definitions.router.get_manager")
    def test_is_agent_enabled_returns_bool(self, mock_get_manager):
        from geo_edit.tool_definitions.router import ToolRouter

        mock_get_manager.return_value.create_agents.return_value = {"chartmoe": MagicMock()}
        router = ToolRouter(tool_mode="auto")
        result = router.is_agent_enabled()
        assert isinstance(result, bool)
