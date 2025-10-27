#!/usr/bin/env python3
"""
Patch JSONL files by updating the `main_thread` field in each line:
- Replace every occurrence of "<launch_threads>" with
  "<parallel_processing>\n<launch_threads>"
- Replace every occurrence of "</step_resolution>" with
  "</step_resolution></parallel_processing>"

By default, edits are applied in-place and a .bak backup is created.

Usage:
  python3 patch_main_thread_parallel_processing.py /path/to/file.jsonl
  # Optional flags:
  python3 patch_main_thread_parallel_processing.py /path/to/file.jsonl --no-backup
  python3 patch_main_thread_parallel_processing.py /path/to/file.jsonl --output /path/to/output.jsonl

Notes:
- Non-JSON lines or lines without a string `main_thread` are copied unchanged.
- Output preserves one-line-per-JSON-object format.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys
from typing import Iterable

REPL_A_FROM = "<launch_threads>"
REPL_A_TO = "<parallel_processing>\n<launch_threads>"
REPL_B_FROM = "</step_resolution>"
REPL_B_TO = "</step_resolution></parallel_processing>"


def process_line(line: str) -> str:
    line_stripped = line.strip()
    if not line_stripped:
        return line  # preserve blank lines as-is
    try:
        obj = json.loads(line_stripped)
    except Exception:
        # Not JSON; return original line unchanged
        return line

    if isinstance(obj, dict) and isinstance(obj.get("main_thread"), str):
        s = obj["main_thread"]
        s = s.replace(REPL_A_FROM, REPL_A_TO)
        s = s.replace(REPL_B_FROM, REPL_B_TO)
        obj["main_thread"] = s
        return json.dumps(obj, ensure_ascii=False) + "\n"

    # No changes needed
    return line


def process_stream(in_fp: io.TextIOBase, out_fp: io.TextIOBase) -> None:
    for line in in_fp:
        out_fp.write(process_line(line))


def patch_file(path: str, output: str | None, backup: bool) -> None:
    if output and os.path.abspath(output) == os.path.abspath(path):
        output = None  # treat as in-place

    if output is None:
        tmp_path = path + ".tmp"
        with open(path, "r", encoding="utf-8") as in_fp, open(tmp_path, "w", encoding="utf-8") as out_fp:
            process_stream(in_fp, out_fp)
        if backup:
            bak_path = path + ".bak"
            # If a previous backup exists, overwrite it to avoid errors
            try:
                shutil.copy2(path, bak_path)
            except Exception:
                # Best-effort backup; continue even if it fails
                pass
        os.replace(tmp_path, path)
    else:
        os.makedirs(os.path.dirname(os.path.abspath(output)) or ".", exist_ok=True)
        with open(path, "r", encoding="utf-8") as in_fp, open(output, "w", encoding="utf-8") as out_fp:
            process_stream(in_fp, out_fp)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Patch main_thread with parallel_processing markers in JSONL lines.")
    parser.add_argument("input", help="Path to input .jsonl file")
    parser.add_argument("--output", "-o", help="Write to this file instead of in-place")
    parser.add_argument("--no-backup", action="store_true", help="Do not create .bak when writing in-place")

    args = parser.parse_args(argv)

    in_path = args.input
    if not os.path.isfile(in_path):
        print(f"Error: input file not found: {in_path}", file=sys.stderr)
        return 2

    try:
        patch_file(in_path, args.output, backup=not args.no_backup)
    except KeyboardInterrupt:
        print("Aborted", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Failed to patch {in_path}: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
