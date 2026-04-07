from __future__ import annotations

"""
Lightweight MCP client helpers for tool discovery and invocation.

This module now ONLY supports the MCP‑Universe SSE gateway path that
exposes multiple servers under ``{MCP_GATEWAY_ADDRESS}/{server}/sse``.

Usage:
- Set ``MCP_GATEWAY_ADDRESS`` to the gateway base URL (e.g. http://localhost:8002).
- Always pass a ``server`` name (e.g. "yfinance") so we connect to
  ``{MCP_GATEWAY_ADDRESS}/{server}/sse`` via ``mcp.client.sse``.

Note: The legacy single-server ``MCP_SERVER_URL`` path is removed.
"""

import os
import json
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional, Tuple


class MCPClientError(Exception):
    pass


def _require_gateway() -> str:
    url = os.getenv("MCP_GATEWAY_ADDRESS", "").rstrip("/")
    if not url:
        raise MCPClientError(
            "MCP_GATEWAY_ADDRESS is not set. It should point to an MCP gateway base URL."
        )
    return url


@asynccontextmanager
async def mcp_sse_session(server: str):
    """
    Connect to server over SSE using a gateway base address and a server name.
    The final URL is ``{MCP_GATEWAY_ADDRESS}/{server}/sse``.
    """
    try:
        from mcp.client.sse import sse_client  # type: ignore
        from mcp.client.session import ClientSession  # type: ignore
    except Exception as e:
        raise MCPClientError(
            "mcp client library is not installed or import failed."
        ) from e

    base = _require_gateway()
    url = f"{base}/{server}/sse"

    async with sse_client(url=url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def list_tools_and_resources(server: Optional[str] = None) -> Tuple[Any, Any]:
    """
    Return (tools_result, resources_result) from the MCP session.
    Requires a ``server`` name; connects via the SSE gateway.
    """
    if not server:
        raise MCPClientError(
            "'server' is required in <tool_call> when using MCP gateway. "
            "Set MCP_GATEWAY_ADDRESS and include a server field (e.g. 'yfinance')."
        )
    context = mcp_sse_session(server)
    async with context as session:
        tools = await session.list_tools()
        try:
            resources = await session.list_resources()
        except Exception:
            class _Empty:
                resources = []

            resources = _Empty()
        return tools, resources


async def call_mcp_tool(tool_name: str, arguments: Optional[Dict[str, Any]] = None, *, server: Optional[str] = None) -> Any:
    """
    Invoke a tool on a remote MCP server and return the CallToolResult.
    Requires a ``server`` name; connects via the SSE gateway.
    """
    if not server:
        raise MCPClientError(
            "'server' is required in <tool_call> when using MCP gateway. "
            "Set MCP_GATEWAY_ADDRESS and include a server field (e.g. 'yfinance')."
        )
    context = mcp_sse_session(server)
    async with context as session:
        return await session.call_tool(tool_name, arguments or {})


def get_content_text(result: Any) -> str:
    """
    Extract human-readable text from an MCP CallToolResult according to the content schema.
    Fallback to JSON serialization of structured or str(result).
    """
    try:
        import mcp.types as types  # type: ignore
    except Exception:
        types = None  # best-effort extraction without types

    if isinstance(result, str):
        return result

    # content array case
    content = getattr(result, "content", None)
    if content:
        out_parts = []
        for item in content:
            if types is not None and isinstance(item, getattr(types, "TextContent", object)):
                out_parts.append(item.text)
            else:
                out_parts.append(str(item))
        return "".join(out_parts)

    # structured case
    structured = getattr(result, "structured", None)
    if structured is not None:
        try:
            return json.dumps(structured)
        except Exception:
            return str(structured)

    return str(result)

