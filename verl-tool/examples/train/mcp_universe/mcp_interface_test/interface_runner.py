#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from mcpuniverse.mcp.manager import MCPManager


def load_cases(root: Path, server_filter: Optional[Set[str]] = None) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    for server_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        if server_filter and server_dir.name not in server_filter:
            continue
        case_file = server_dir / "cases.json"
        if not case_file.exists():
            continue
        try:
            data = json.loads(case_file.read_text(encoding="utf-8"))
            for idx, item in enumerate(data):
                item = dict(item)
                item.setdefault("server", server_dir.name)
                item.setdefault("id", f"{server_dir.name}#{idx}")
                cases.append(item)
        except Exception as e:
            print(f"[WARN] Failed to load {case_file}: {e}")
    return cases


async def _extract_message(resp: Any) -> Optional[str]:
    msg = None
    try:
        content = getattr(resp, "content", None)
        if isinstance(content, list) and content:
            texts = []
            for c in content:
                t = getattr(c, "text", None) if hasattr(c, "text") else None
                if t:
                    texts.append(str(t))
            if texts:
                msg = "\n".join(texts)
    except Exception:
        pass
    return msg


async def run_case(manager: MCPManager, case: Dict[str, Any], client: Any = None) -> Dict[str, Any]:
    server = case.get("server")
    ctype = case.get("type", "call")  # call | list | sequence
    result: Dict[str, Any] = {"id": case.get("id"), "server": server, "type": ctype}
    owns_client = client is None
    try:
        if client is None:
            client = await manager.build_client(server_name=server, transport="sse")
        try:
            if ctype == "list":
                tools = await client.list_tools()
                result.update({"ok": True, "tools_count": len(tools), "name": "<list_tools>"})
            elif ctype == "sequence":
                steps: List[Dict[str, Any]] = case.get("steps", []) or []
                step_results: List[Dict[str, Any]] = []
                all_ok = True
                for idx, st in enumerate(steps):
                    name = st.get("name")
                    args = st.get("arguments", {})
                    try:
                        resp = await client.execute_tool(tool_name=name, arguments=args)
                        is_error = bool(getattr(resp, "isError", False))
                        msg = await _extract_message(resp)
                        ok = not is_error
                        step_results.append({
                            "step": idx,
                            "name": name,
                            "ok": ok,
                            "isError": is_error,
                            "message": msg,
                            "response_type": resp.__class__.__name__,
                        })
                        if not ok:
                            all_ok = False
                            break
                    except Exception as se:
                        step_results.append({
                            "step": idx,
                            "name": name,
                            "ok": False,
                            "error": str(se),
                        })
                        all_ok = False
                        break
                result.update({
                    "ok": all_ok,
                    "name": "<sequence>",
                    "steps": step_results,
                })
            else:
                name = case.get("name")
                args = case.get("arguments", {})
                resp = await client.execute_tool(tool_name=name, arguments=args)
                is_error = bool(getattr(resp, "isError", False))
                msg = await _extract_message(resp)
                result.update({
                    "ok": not is_error,
                    "isError": is_error,
                    "response_type": resp.__class__.__name__,
                    "name": name,
                    "message": msg,
                })
        finally:
            if owns_client and client is not None:
                await client.cleanup()
    except Exception as e:
        if ctype != "list" and "name" not in result:
            result["name"] = case.get("name")
        result.update({"ok": False, "error": str(e)})
    return result


async def main_async(args):
    # Require MCP_GATEWAY_ADDRESS
    mcp_addr = os.environ.get("MCP_GATEWAY_ADDRESS", "")
    if not mcp_addr:
        print("[ERROR] MCP_GATEWAY_ADDRESS not set. Please start gateway and export it.")
        sys.exit(2)
    manager = MCPManager()
    server_filter: Optional[Set[str]] = None
    if args.servers:
        server_filter = {s.strip() for s in args.servers.split(',') if s.strip()}
    cases = load_cases(Path(args.root), server_filter=server_filter)
    if not cases:
        print(f"[WARN] No cases found under {args.root}")
        return

    passed, total = 0, 0
    failures_by_server: Dict[str, List[Dict[str, Any]]] = {}
    all_results: List[Dict[str, Any]] = []
    # Per-server stats (for call cases only)
    per_server = {}
    # Track listed tools count if available from list_tools case
    server_tools_count: Dict[str, int] = {}
    # Prepare stateful server set
    stateful_servers: Set[str] = set()
    ss_arg = (args.stateful_servers or os.environ.get("STATEFUL_SERVERS", "")).strip()
    if ss_arg:
        stateful_servers = {s.strip() for s in ss_arg.split(',') if s.strip()}
    # Cache clients for stateful servers
    stateful_clients: Dict[str, Any] = {}
    try:
        for case in cases:
            total += 1
            srv = case.get("server")
            client = None
            if srv in stateful_servers:
                client = stateful_clients.get(srv)
                if client is None:
                    try:
                        client = await manager.build_client(server_name=srv, transport="sse")
                        stateful_clients[srv] = client
                    except Exception as e:
                        res = {"id": case.get("id"), "server": srv, "type": case.get("type"), "ok": False, "error": str(e)}
                        status = "OK" if res.get("ok") else "FAIL"
                        print(f"[{status}] {case.get('id')} ({case.get('server')}:{case.get('type')}) -> {res}")
                        passed += int(bool(res.get("ok")))
                        all_results.append(res)
                        fsrv = res.get("server", "<unknown>")
                        failures_by_server.setdefault(fsrv, []).append({
                            "type": res.get("type"),
                            "name": res.get("name"),
                            "error": res.get("error"),
                            "isError": res.get("isError"),
                        })
                        # Still update per_server attempted for non-list
                        ps = per_server.setdefault(fsrv, {"calls_attempted": 0, "calls_passed": 0})
                        if res.get("type") != "list":
                            ps["calls_attempted"] += 1
                        continue
            res = await run_case(manager, case, client=client)
            status = "OK" if res.get("ok") else "FAIL"
            print(f"[{status}] {case.get('id')} ({case.get('server')}:{case.get('type')}) -> {res}")
            passed += int(bool(res.get("ok")))
            all_results.append(res)
            if not res.get("ok"):
                srv = res.get("server", "<unknown>")
                failures_by_server.setdefault(srv, []).append({
                    "type": res.get("type"),
                    "name": res.get("name"),
                    "error": res.get("error"),
                    "isError": res.get("isError"),
                })
            # Aggregate per-server stats
            srv = res.get("server", "<unknown>")
            per = per_server.setdefault(srv, {"calls_attempted": 0, "calls_passed": 0})
            if res.get("type") == "list":
                # capture tools_count if present
                tc = res.get("tools_count")
                if isinstance(tc, int):
                    server_tools_count[srv] = tc
            else:
                per["calls_attempted"] += 1
                per["calls_passed"] += int(bool(res.get("ok")))
    finally:
        # Cleanup any stateful clients
        for c in stateful_clients.values():
            try:
                await c.cleanup()
            except Exception:
                pass
    print(f"Summary: {passed}/{total} passed")
    if failures_by_server:
        print("Failures by server/tool:")
        for srv, items in failures_by_server.items():
            print(f"  - {srv}:")
            for it in items:
                tname = it.get("name") or "<unknown>"
                tt = it.get("type")
                err = it.get("error") or it.get("message")
                iserr = it.get("isError")
                extra = f" isError={iserr}" if iserr is not None else ""
                if err:
                    print(f"      * {tt}:{tname} -> ERROR: {err}{extra}")
                else:
                    print(f"      * {tt}:{tname} -> FAILED{extra}")

    # Print per-server pass rate using tool counts when available
    print("Per-server pass rate (calls vs total tools):")
    # Optionally fetch tools_count for servers without list case
    missing_tool_counts = [s for s in per_server.keys() if s not in server_tools_count]
    if missing_tool_counts:
        # Try listing tools quickly
        for s in missing_tool_counts:
            try:
                client = await manager.build_client(server_name=s, transport="sse")
                try:
                    tools = await client.list_tools()
                    server_tools_count[s] = len(tools)
                finally:
                    await client.cleanup()
            except Exception:
                server_tools_count.setdefault(s, 0)
    per_server_report: List[Dict[str, Any]] = []
    for s, st in per_server.items():
        calls = st["calls_attempted"]
        ok = st["calls_passed"]
        tools_total = server_tools_count.get(s, calls)
        ratio = f"{ok}/{tools_total}"
        pct = (ok / tools_total * 100.0) if tools_total > 0 else 0.0
        print(f"  - {s}: {ratio} ({pct:.1f}%), calls_attempted={calls}")
        per_server_report.append({
            "server": s,
            "tools_total": tools_total,
            "calls_attempted": calls,
            "calls_passed": ok,
            "pass_rate": pct,
        })

    # Write logs if requested
    if args.log_json or args.log_text:
        import datetime as _dt
        payload = {
            "summary": {"passed": passed, "total": total, "timestamp": _dt.datetime.utcnow().isoformat()+"Z"},
            "per_server": per_server_report,
            "failures": failures_by_server,
            "results": all_results,
        }
        if args.log_json:
            Path(args.log_json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.log_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Wrote JSON log to {args.log_json}")
        if args.log_text:
            lines: List[str] = []
            lines.append(f"Summary: {passed}/{total} passed")
            lines.append("Per-server pass rate (calls vs total tools):")
            for p in per_server_report:
                lines.append(f"  - {p['server']}: {p['calls_passed']}/{p['tools_total']} ({p['pass_rate']:.1f}%), calls_attempted={p['calls_attempted']}")
            if failures_by_server:
                lines.append("Failures by server/tool:")
                for srv, items in failures_by_server.items():
                    lines.append(f"  - {srv}:")
                    for it in items:
                        tname = (it.get('name') or '<unknown>')
                        tt = it.get('type')
                        err = it.get('error')
                        iserr = it.get('isError')
                        extra = f" isError={iserr}" if iserr is not None else ""
                        if err:
                            lines.append(f"      * {tt}:{tname} -> ERROR: {err}{extra}")
                        else:
                            lines.append(f"      * {tt}:{tname} -> FAILED{extra}")
            Path(args.log_text).parent.mkdir(parents=True, exist_ok=True)
            Path(args.log_text).write_text("\n".join(lines), encoding="utf-8")
            print(f"Wrote text log to {args.log_text}")


def main():
    ap = argparse.ArgumentParser(description="MCP interface runner (via gateway)")
    ap.add_argument("--root", default=os.path.join(os.path.dirname(__file__), "cases"), help="Cases root folder")
    ap.add_argument("--servers", default="", help="Comma-separated servers to run (empty = all in cases root)")
    ap.add_argument("--log_json", default=os.path.join(os.path.dirname(__file__), "interface_results.json"), help="Path to write JSON log (empty to disable)")
    ap.add_argument("--log_text", default=os.path.join(os.path.dirname(__file__), "interface_results.log"), help="Path to write text log (empty to disable)")
    ap.add_argument("--stateful_servers", default="", help="Comma-separated servers to reuse a single client across cases. Fallback to env STATEFUL_SERVERS if empty.")
    args = ap.parse_args()

    import asyncio
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
