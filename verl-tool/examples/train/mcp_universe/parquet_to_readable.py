#!/usr/bin/env python3
"""
Parquet -> Human‑Readable Markdown preview for MCP‑Universe datasets.

Usage:
  python examples/train/mcp-universe/parquet_to_readable.py \
    data/mcp_universe/web_search/test.parquet \
    --out readable.md \
    --max_rows 20 \
    --max_chars 2000

If --out is omitted, writes `readable.md` next to the Parquet file.
"""
from __future__ import annotations

import os
import argparse
from typing import List, Dict


def _try_load_with_datasets(path: str) -> List[Dict]:
    from datasets import load_dataset  # type: ignore
    ds = load_dataset("parquet", data_files={"data": [path]})["data"]
    return list(ds)


def _try_load_with_pandas(path: str) -> List[Dict]:
    import pandas as pd  # type: ignore
    df = pd.read_parquet(path)
    # normalize to list[dict]
    rows = []
    for _, r in df.iterrows():
        rows.append({k: r[k] for k in df.columns})
    return rows


def load_rows(path: str) -> List[Dict]:
    try:
        return _try_load_with_datasets(path)
    except Exception:
        return _try_load_with_pandas(path)


def main():
    ap = argparse.ArgumentParser(description="Dump a human‑readable preview for a verl‑tool Parquet dataset")
    ap.add_argument("parquet", type=str, help="Path to the Parquet file")
    ap.add_argument("--out", type=str, default=None, help="Output markdown file path (default: readable.md beside Parquet)")
    ap.add_argument("--max_rows", type=int, default=20, help="Max rows to include in the dump")
    ap.add_argument("--max_chars", type=int, default=2000, help="Max chars per prompt block")
    args = ap.parse_args()

    parquet_path = os.path.abspath(args.parquet)
    if not os.path.isfile(parquet_path):
        raise FileNotFoundError(f"Parquet not found: {parquet_path}")

    out_path = args.out
    if not out_path:
        out_dir = os.path.dirname(parquet_path)
        out_path = os.path.join(out_dir, "readable.md")

    rows = load_rows(parquet_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def trunc(s: str, n: int) -> str:
        return s if len(s) <= n else s[:n] + "...\n"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"File: {parquet_path}\n")
        f.write(f"Items: {len(rows)}\n")
        # union servers
        srv = []
        for r in rows:
            extra = (r.get("extra_info") or {})
            for s in extra.get("mcp_servers", []) or []:
                if isinstance(s, dict) and s.get("name"):
                    srv.append(s["name"])
                elif isinstance(s, str):
                    srv.append(s)
        f.write("Servers (union): " + ", ".join(sorted(set(srv))) + "\n\n")

        upto = min(args.max_rows, len(rows))
        for i in range(upto):
            r = rows[i]
            prompt = r.get("prompt") or []
            sys_txt = (prompt[0].get("content") if len(prompt) > 0 else "") or ""
            user_txt = (prompt[1].get("content") if len(prompt) > 1 else "") or ""
            extra = r.get("extra_info") or {}
            mcp_srvs = extra.get("mcp_servers", []) or []
            mcp_srvs_str = ", ".join(
                [s.get("name") for s in mcp_srvs if isinstance(s, dict) and s.get("name")] +
                [s for s in mcp_srvs if isinstance(s, str)]
            )
            has_tools_desc = "### Tools Description ###" in sys_txt
            f.write(f"## Item {i} | idx={extra.get('index')}\n")
            f.write(f"Task: {extra.get('task','')}\n")
            f.write(f"Servers (per item): {mcp_srvs_str}\n")
            f.write(f"ToolsDescription: {has_tools_desc}\n")
            f.write("Question:\n")
            f.write(trunc(str(user_txt), args.max_chars))
            f.write("SystemPrompt (truncated):\n")
            f.write(trunc(str(sys_txt), args.max_chars))
            f.write("\n")

    print(f"Wrote human-readable preview to {out_path}")


if __name__ == "__main__":
    main()

