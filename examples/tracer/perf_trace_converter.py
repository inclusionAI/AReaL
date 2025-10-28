from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path


def _load_events(path: Path) -> list[dict]:
    events: list[dict] = []
    with path.open("r", encoding="utf-8") as fin:
        for lineno, raw_line in enumerate(fin, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as exc:  # pragma: no cover - invalid payload
                raise ValueError(
                    f"Failed to parse JSONL at line {lineno} in {path}: {exc}"
                ) from exc
    return events


def convert_jsonl_to_chrome_trace(
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str] | None = None,
    *,
    display_time_unit: str = "ms",
) -> dict:
    """Convert newline-delimited trace events into Chrome Trace JSON."""

    source = Path(input_path)
    if not source.is_file():  # pragma: no cover - defensive guard
        raise FileNotFoundError(f"Input trace file not found: {source}")

    events = _load_events(source)
    chrome_trace = {
        "traceEvents": events,
        "displayTimeUnit": display_time_unit,
    }

    if output_path is not None:
        destination = Path(output_path)
        if destination.parent and not destination.parent.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as fout:
            json.dump(chrome_trace, fout, ensure_ascii=False)
    return chrome_trace


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert PerfTracer JSONL output into Chrome Trace JSON format.",
    )
    parser.add_argument("input", type=str, help="Path to the PerfTracer JSONL file")
    parser.add_argument(
        "output",
        type=str,
        nargs="?",
        help="Optional output path for the Chrome Trace JSON file (stdout if omitted)",
    )
    parser.add_argument(
        "--display-time-unit",
        type=str,
        default="ms",
        help="Value for the displayTimeUnit field in the Chrome trace output",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    chrome_trace = convert_jsonl_to_chrome_trace(
        args.input,
        args.output,
        display_time_unit=args.display_time_unit,
    )
    if args.output is None:
        json.dump(chrome_trace, sys.stdout, ensure_ascii=False)
        sys.stdout.write("\n")
        sys.stdout.flush()
        return 0
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
