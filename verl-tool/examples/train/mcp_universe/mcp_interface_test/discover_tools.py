#!/usr/bin/env python3
from __future__ import annotations

import os
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List

from mcpuniverse.mcp.manager import MCPManager


def to_dict_tool(t: Any) -> Dict[str, Any]:
    d: Dict[str, Any] = {}
    for key in ("name", "description", "inputSchema"):
        try:
            val = getattr(t, key, None)
        except Exception:
            val = None
        if hasattr(val, "model_dump"):
            try:
                val = val.model_dump(mode="json")
            except Exception:
                val = None
        d[key] = val
    return d


async def list_tools_for_server(manager: MCPManager, server: str) -> List[Dict[str, Any]]:
    client = await manager.build_client(server_name=server, transport="sse")
    try:
        tools = await client.list_tools()
        return [to_dict_tool(t) for t in tools]
    finally:
        await client.cleanup()


async def main_async(args):
    mcp_addr = os.environ.get("MCP_GATEWAY_ADDRESS", "")
    if not mcp_addr:
        raise SystemExit("MCP_GATEWAY_ADDRESS is not set; please start gateway and export it.")
    manager = MCPManager()
    servers = [s.strip() for s in args.servers.split(",") if s.strip()] if args.servers else []
    if not servers:
        # fallback: infer from discovered server list in gateway config
        servers = [
            "google-search", "fetch", "notion", "playwright",
            "weather", "date", "yfinance", "calculator"
        ]

    out = {"servers": []}
    for s in servers:
        try:
            tools = await list_tools_for_server(manager, s)
            out["servers"].append({"name": s, "tools": tools})
            print(f"[OK] listed tools for {s}: {len(tools)}")
        except Exception as e:
            print(f"[FAIL] list tools for {s}: {e}")
            out["servers"].append({"name": s, "tools": [], "error": str(e)})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote discovery to {args.out}")


def main():
    ap = argparse.ArgumentParser(description="Discover tools via gateway and dump to JSON")
    ap.add_argument("--servers", default="", help="Comma-separated server names; empty to use defaults")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "discovered_tools.json"))
    args = ap.parse_args()

    import asyncio
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()

