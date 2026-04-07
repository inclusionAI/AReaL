#!/usr/bin/env python
"""
Lightweight dispatch test for MCPInterfaceTool with monkeypatching.

We do not require a running MCP server; instead we monkeypatch the
`list_tools_and_resources` and `call_mcp_tool` functions inside the
tool module to simulate discovery and invocation.

Run:
  python -m verl_tool.servers.tests.test_mcp_interface_dispatch run_all
"""
from __future__ import annotations

import json
import types
import fire

import verl_tool.servers.tools.mcp_interface as mi
from verl_tool.servers.tools.mcp_interface import MCPInterfaceTool


class _DummyTool:
    def __init__(self, name: str, description: str = "", schema: dict | None = None):
        self.name = name
        self.description = description
        self.inputSchema = schema or {"type": "object", "properties": {}}


class _DummyToolsObj:
    def __init__(self, tools: list[_DummyTool]):
        self.tools = tools


async def _fake_list_tools_and_resources(server: str | None = None):
    # Pretend discovery returns one tool named 'demo'
    return _DummyToolsObj([_DummyTool("demo", "demo tool")]), types.SimpleNamespace(resources=[])


async def _fake_call_mcp_tool(name: str, arguments: dict | None = None, *, server: str | None = None):
    # Return an object that stringifies nicely
    class _R:
        def __str__(self):
            return json.dumps({"echo": {"name": name, "server": server, "args": arguments or {}}})

    return _R()


def test_dispatch_success():
    # Monkeypatch in module namespace
    mi.list_tools_and_resources = _fake_list_tools_and_resources  # type: ignore[attr-defined]
    mi.call_mcp_tool = _fake_call_mcp_tool  # type: ignore[attr-defined]

    tool = MCPInterfaceTool(num_workers=1)
    action = "<mcp_call>{}</mcp_call>".format(json.dumps({
        "server": "dummy",  # triggers server-specific discovery
        "name": "demo",
        "arguments": {"x": 1}
    }))

    obs, done, valid = tool.conduct_action("traj-1", action, {})
    assert valid is True
    assert done is False
    # observation is wrapped into a mcp_response block
    assert isinstance(obs, str) or isinstance(obs, dict)
    if isinstance(obs, dict):
        text = obs.get("obs", "")
    else:
        text = obs
    assert "mcp_response" in text
    assert "demo" in text
    print("test_dispatch_success: OK")


def test_dispatch_invalid_tool():
    mi.list_tools_and_resources = _fake_list_tools_and_resources  # type: ignore[attr-defined]
    mi.call_mcp_tool = _fake_call_mcp_tool  # type: ignore[attr-defined]

    tool = MCPInterfaceTool(num_workers=1)
    # Ensure discovery for the target server so parse_action can validate
    tool._ensure_discovered("dummy")  # noqa: SLF001 (accessing internal for test)
    action = "<mcp_call>{}</mcp_call>".format(json.dumps({
        "server": "dummy",
        "name": "not_exist",
        "arguments": {}
    }))

    # parse_action should reject unknown tool after discovery
    parsed, ok = tool.parse_action(action)
    assert ok is False
    print("test_dispatch_invalid_tool: OK")


def run_all():
    test_dispatch_success()
    test_dispatch_invalid_tool()
    print("All dispatch tests passed.")


if __name__ == "__main__":
    fire.Fire({
        "run_all": run_all,
        "success": test_dispatch_success,
        "invalid": test_dispatch_invalid_tool,
    })
