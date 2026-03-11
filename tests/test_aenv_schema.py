"""Unit tests for AEnvironment schema helpers."""

import pytest

from areal.infra.aenv.schema import normalize_openai_tools, parse_tool_arguments


def test_normalize_openai_tools_maps_input_schema():
    """Test inputSchema is mapped to OpenAI function parameters."""
    tools = [
        {
            "name": "math/calculate",
            "description": "Calculate expression",
            "inputSchema": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        }
    ]

    normalized = normalize_openai_tools(tools)

    assert normalized == [
        {
            "type": "function",
            "function": {
                "name": "math/calculate",
                "description": "Calculate expression",
                "parameters": {
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                    "required": ["expression"],
                },
            },
        }
    ]


def test_normalize_openai_tools_falls_back_to_parameters():
    """Test schema fallback for mixed tool metadata."""
    tools = [
        {
            "name": "legacy/tool",
            "description": "Legacy schema",
            "parameters": {"type": "object", "properties": {}},
        }
    ]

    normalized = normalize_openai_tools(tools)

    assert normalized[0]["function"]["name"] == "legacy/tool"
    assert normalized[0]["function"]["parameters"] == {
        "type": "object",
        "properties": {},
    }


def test_normalize_openai_tools_raises_for_invalid_name():
    """Test invalid tool names are rejected."""
    with pytest.raises(ValueError, match="Invalid tool name"):
        normalize_openai_tools([{"name": "", "inputSchema": {}}])


def test_parse_tool_arguments_accepts_dict_and_none():
    """Test parser supports dict and None inputs."""
    assert parse_tool_arguments({"a": 1}) == {"a": 1}
    assert parse_tool_arguments(None) == {}


def test_parse_tool_arguments_parses_json_string():
    """Test parser decodes JSON strings into dictionaries."""
    parsed = parse_tool_arguments('{"city": "Shanghai", "unit": "celsius"}')

    assert parsed == {"city": "Shanghai", "unit": "celsius"}


def test_parse_tool_arguments_rejects_invalid_json():
    """Test invalid JSON arguments raise ValueError."""
    with pytest.raises(ValueError, match="Invalid tool arguments JSON"):
        parse_tool_arguments("{bad-json}")


def test_parse_tool_arguments_rejects_non_dict_json():
    """Test non-dictionary JSON payloads are rejected."""
    with pytest.raises(ValueError, match="must decode to a dict"):
        parse_tool_arguments('["not", "a", "dict"]')
