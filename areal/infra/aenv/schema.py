"""Schema helpers for AEnvironment tool integration."""

from __future__ import annotations

import json
from typing import Any


def normalize_openai_tools(aenv_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert AEnvironment tool metadata into OpenAI function-tool schema.

    Args:
        aenv_tools: Tool metadata returned by ``Environment.list_tools()``.

    Returns:
        OpenAI-compatible tool list suitable for ``chat.completions.create(..., tools=...)``.

    Raises:
        ValueError: If a tool definition is missing required fields.
    """
    normalized: list[dict[str, Any]] = []

    for tool in aenv_tools:
        name = tool.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"Invalid tool name: {name!r}")

        description = tool.get("description")
        if not isinstance(description, str):
            description = ""

        # AEnvironment uses `inputSchema`; keep a fallback for mixed responses.
        parameters = tool.get("inputSchema")
        if parameters is None:
            parameters = tool.get("parameters")
        if not isinstance(parameters, dict):
            parameters = {"type": "object", "properties": {}}

        normalized.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                },
            }
        )

    return normalized


def parse_tool_arguments(raw_arguments: str | dict[str, Any] | None) -> dict[str, Any]:
    """Parse tool arguments safely without ``eval``.

    Args:
        raw_arguments: Tool call arguments from model output.

    Returns:
        Parsed argument dictionary.

    Raises:
        ValueError: If argument payload cannot be parsed into a dictionary.
        TypeError: If argument payload type is unsupported.
    """
    if raw_arguments is None:
        return {}

    if isinstance(raw_arguments, dict):
        return raw_arguments

    if isinstance(raw_arguments, str):
        text = raw_arguments.strip()
        if not text:
            return {}

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid tool arguments JSON: {text}") from exc

        if not isinstance(parsed, dict):
            raise ValueError(
                f"Tool arguments must decode to a dict, got {type(parsed).__name__}"
            )
        return parsed

    raise TypeError(f"Unsupported tool argument type: {type(raw_arguments).__name__}")
