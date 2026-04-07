#!/usr/bin/env python3
from __future__ import annotations

import os
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List


def _sample_value(prop_name: str, schema: Dict[str, Any], server_name: str | None = None) -> Any:
    t = schema.get("type")
    if t == "string":
        name = prop_name.lower()
        # Special-case calculator expression to a valid safe math expr
        if name == "expression" and (server_name == "calculator" or server_name is None):
            return "1 + 2"
        if "url" in name:
            return "https://example.com"
        if name in ("query", "q", "search"):
            return "OpenAI"
        if name in ("ticker", "symbol"):
            return "AAPL"
        if "start" in name and "date" in name:
            return "2024-01-01"
        if "end" in name and "date" in name:
            return "2024-02-01"
        if name == "state":
            return "CA"
        return "test"
    if t == "number":
        if prop_name.lower() == "latitude":
            return 37.7749
        if prop_name.lower() == "longitude":
            return -122.4194
        return 1.0
    if t == "integer":
        return 1
    if t == "boolean":
        return True
    if t == "array":
        items = schema.get("items", {"type": "string"})
        return [_sample_value(prop_name, items, server_name)]
    if t == "object":
        return {}
    return None


def _args_from_schema(schema: Dict[str, Any], server_name: str | None = None) -> Dict[str, Any]:
    if not isinstance(schema, dict):
        return {}
    if schema.get("type") != "object":
        return {}
    props: Dict[str, Any] = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    out: Dict[str, Any] = {}
    for name, ps in props.items():
        if name in required:
            out[name] = _sample_value(name, ps, server_name)
    # Optionally include non-required with samples
    return out


def main():
    ap = argparse.ArgumentParser(description="Generate cases from discovered tools JSON")
    ap.add_argument("--infile", default=os.path.join(os.path.dirname(__file__), "discovered_tools.json"))
    ap.add_argument("--out_root", default=os.path.join(os.path.dirname(__file__), "cases_auto"))
    args = ap.parse_args()

    data = json.loads(Path(args.infile).read_text(encoding="utf-8"))
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    for srv in data.get("servers", []):
        name = srv.get("name")
        tools = srv.get("tools", [])
        if not name:
            continue
        cases: List[Dict[str, Any]] = [{"type": "list"}]
        # For stateful servers like playwright, default to list-only unless explicitly curated
        if name == "playwright":
            # Provide a minimal curated sequence that can run in a single session
            # Steps: navigate -> snapshot -> take_screenshot
            seq_steps: List[Dict[str, Any]] = [
                {"name": "browser_navigate", "arguments": {"url": "https://example.com"}},
                {"name": "browser_snapshot", "arguments": {}},
                {"name": "browser_take_screenshot", "arguments": {}}
            ]
            cases.append({"type": "sequence", "steps": seq_steps})
        else:
            for t in tools:
                tname = t.get("name")
                schema = t.get("inputSchema") or {}
                args_map = _args_from_schema(schema, name)
                cases.append({"type": "call", "name": tname, "arguments": args_map})
        out_dir = out_root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote cases for {name}: {out_dir / 'cases.json'}")


if __name__ == "__main__":
    from pathlib import Path
    main()
