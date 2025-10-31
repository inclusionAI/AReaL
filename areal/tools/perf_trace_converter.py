from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from glob import glob
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


def _resolve_trace_files(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if source.is_dir():
        return sorted(p for p in source.glob("*.jsonl") if p.is_file())
    matches = [Path(p) for p in glob(str(source), recursive=True)]
    files = [p for p in matches if p.is_file()]
    return sorted(files)


def convert_jsonl_to_chrome_trace(
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str] | None = None,
    *,
    display_time_unit: str = "ms",
) -> dict:
    """Convert newline-delimited trace events into Chrome Trace JSON.

    The ``input_path`` may point to a single JSONL file, a directory containing
    per-rank JSONL files, or a glob pattern. All matching files are concatenated
    in lexical order before emitting the Chrome trace payload.
    """

    sources = _resolve_trace_files(Path(input_path))
    if not sources:
        raise FileNotFoundError(f"No trace files matched input path: {input_path}")

    events: list[dict] = []
    for path in sources:
        events.extend(_load_events(path))

    events.sort(
        key=lambda event: (event.get("ts", 0), event.get("pid", 0), event.get("tid", 0))
    )

    chrome_trace = {
        "traceEvents": events,
        "displayTimeUnit": display_time_unit,
    }

    if output_path is not None:
        destination = Path(output_path)
        if destination.parent != Path(".") and not destination.parent.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as fout:
            json.dump(chrome_trace, fout, ensure_ascii=False)
    return chrome_trace


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert PerfTracer JSONL output into Chrome Trace JSON format.",
    )
    parser.add_argument(
        "input",
        type=str,
        help=(
            "Path, directory, or glob pattern for PerfTracer JSONL files "
            "(per-rank outputs allowed)"
        ),
    )
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


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
