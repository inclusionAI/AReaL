#!/usr/bin/env python3
import os
import sys
import json
import argparse
from pathlib import Path


def ensure_project_on_path() -> None:
    # Add repo root to sys.path so `verl_tool` can be imported when running directly
    repo_root = Path(__file__).resolve().parents[4]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def main() -> None:
    ensure_project_on_path()

    # Require gateway address (e.g., http://localhost:8000) per MCP-Universe SSE client
    mcp_addr = os.environ.get("MCP_GATEWAY_ADDRESS", "").rstrip("/")
    if not mcp_addr:
        print("[ERROR] MCP_GATEWAY_ADDRESS not set. Start the gateway and export MCP_GATEWAY_ADDRESS first.")
        print("Example: export MCP_GATEWAY_ADDRESS=http://localhost:8000")
        sys.exit(2)

    # Action payload: read from argv or use a default example
    parser = argparse.ArgumentParser(description="Minimal MCP interface connectivity test.")
    parser.add_argument(
        "action",
        nargs="?",
        default='<tool_call>{"name":"add","arguments":{"a":1,"b":1}}</tool_call>',
        help="Raw tool_call payload (default: add 1+1 on calculator).",
    )
    parser.add_argument(
        "--server",
        default="calculator",
        help="Preferred MCP server name (default: calculator).",
    )
    args = parser.parse_args()

    raw_action = args.action
    server_name = args.server

    from verl_tool.servers.tools.mcp_interface import MCPInterfaceTool

    tool = MCPInterfaceTool(num_workers=1)
    trajectory_id = "t1"
    # Provide per-task metadata equivalent to dataset extra_info so the interface
    # can infer the MCP server when the action omits it.
    extra_field = {
        "use_specified_server": True,
        "mcp_servers": [{"name": server_name}],
    }

    observation, done, valid = tool.conduct_action(
        trajectory_id=trajectory_id,
        action=raw_action,
        extra_field=extra_field,
    )

    # Pretty-print result
    print("valid:", valid)
    print("done:", done)
    if isinstance(observation, (dict, list)):
        print(json.dumps(observation, ensure_ascii=False, indent=2))
    else:
        print(str(observation))


if __name__ == "__main__":
    main()
