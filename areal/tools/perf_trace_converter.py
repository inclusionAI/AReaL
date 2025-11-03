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


def _extract_rank(event: dict) -> str | int | None:
    """Best-effort extraction of the rank identifier from a trace event."""

    args = event.get("args")
    if not isinstance(args, dict):
        return None
    rank = args.get("rank")
    if rank is None:
        return None
    if isinstance(rank, bool):  # guard against bool subclassing int
        return None
    if isinstance(rank, (int, float)):
        try:
            return int(rank)
        except (TypeError, ValueError):
            return None
    if isinstance(rank, str):
        text = rank.strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            return text
    return str(rank)


def _format_rank(rank: str | int) -> str:
    return str(rank)


def _rank_sort_key(rank: str | int | None) -> tuple[int, object]:
    if rank is None:
        return (2, 0)
    if isinstance(rank, int):
        return (0, rank)
    return (1, str(rank))


def _value_sort_key(value: object) -> tuple[int, object]:
    if isinstance(value, bool):
        return (0, int(value))
    if isinstance(value, int):
        return (1, value)
    if isinstance(value, float):
        return (2, value)
    if isinstance(value, str):
        return (3, value)
    return (4, repr(value))


def _remap_process_and_thread_ids(events: list[dict]) -> list[dict]:
    """Ensure unique pid/tid pairs across hosts and emit naming metadata."""

    pid_keys: set[tuple[str | int, object]] = set()
    tid_keys: set[tuple[str | int, object, object]] = set()

    for event in events:
        rank = _extract_rank(event)
        if rank is None:
            continue

        original_pid = event.get("pid")
        if original_pid is None:
            continue
        pid_keys.add((rank, original_pid))

        original_tid = event.get("tid")
        if original_tid is not None:
            tid_keys.add((rank, original_pid, original_tid))

    sorted_pid_keys = sorted(
        pid_keys,
        key=lambda item: (_rank_sort_key(item[0]), _value_sort_key(item[1])),
    )

    pid_map: dict[tuple[str | int, object], int] = {}
    pid_labels: dict[int, tuple[str | int, object]] = {}
    for new_pid, key in enumerate(sorted_pid_keys):
        pid_map[key] = new_pid
        pid_labels[new_pid] = key

    tid_counters: dict[int, int] = {}
    tid_map: dict[tuple[str | int, object, object], int] = {}
    tid_labels: dict[tuple[int, int], tuple[str | int, object]] = {}

    sorted_tid_keys = sorted(
        tid_keys,
        key=lambda item: (
            _rank_sort_key(item[0]),
            _value_sort_key(item[1]),
            _value_sort_key(item[2]),
        ),
    )

    for key in sorted_tid_keys:
        rank, original_pid, original_tid = key
        new_pid = pid_map[(rank, original_pid)]
        next_tid = tid_counters.get(new_pid, 0)
        tid_counters[new_pid] = next_tid + 1
        tid_map[key] = next_tid
        tid_labels[(new_pid, next_tid)] = (rank, original_tid)

    for event in events:
        rank = _extract_rank(event)
        if rank is None:
            continue

        original_pid = event.get("pid")
        if original_pid is None:
            continue
        new_pid = pid_map[(rank, original_pid)]
        event["pid"] = new_pid

        original_tid = event.get("tid")
        if original_tid is not None:
            event["tid"] = tid_map[(rank, original_pid, original_tid)]

    metadata_events: list[dict] = []
    for pid, (rank, original_pid) in pid_labels.items():
        rank_text = _format_rank(rank)
        metadata_events.append(
            {
                "name": "process_name",
                "ph": "M",
                "pid": pid,
                "tid": 0,
                "args": {
                    "name": f"[Rank {rank_text}, Process {original_pid}]",
                    "rank": rank,
                },
            }
        )
        metadata_events.append(
            {
                "name": "process_sort_index",
                "ph": "M",
                "pid": pid,
                "tid": 0,
                "args": {"sort_index": pid, "rank": rank},
            }
        )

    for (pid, tid), (rank, original_tid) in tid_labels.items():
        rank_text = _format_rank(rank)
        metadata_events.append(
            {
                "name": "thread_name",
                "ph": "M",
                "pid": pid,
                "tid": tid,
                "args": {
                    "name": f"[Rank {rank_text}, Thread {original_tid}]",
                    "rank": rank,
                },
            }
        )
        metadata_events.append(
            {
                "name": "thread_sort_index",
                "ph": "M",
                "pid": pid,
                "tid": tid,
                "args": {"sort_index": tid, "rank": rank},
            }
        )

    return metadata_events


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

    filtered_events: list[dict] = []
    ignored_metadata = {
        "process_name",
        "thread_name",
        "process_sort_index",
        "thread_sort_index",
    }
    for event in events:
        if event.get("ph") == "M" and event.get("name") in ignored_metadata:
            continue
        filtered_events.append(event)

    events = filtered_events

    metadata_events = _remap_process_and_thread_ids(events)

    metadata_events.sort(
        key=lambda event: (
            _rank_sort_key(event.get("args", {}).get("rank")),
            0
            if event.get("name") == "process_name"
            else 1
            if event.get("name") == "process_sort_index"
            else 2
            if event.get("name") == "thread_name"
            else 3,
            _value_sort_key(event.get("pid")),
            _value_sort_key(event.get("tid")),
        )
    )

    events.sort(
        key=lambda event: (
            event.get("ts", 0),
            _value_sort_key(event.get("pid")),
            _value_sort_key(event.get("tid")),
        )
    )

    events = metadata_events + events

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
        help=(
            "Optional output path for the Chrome Trace JSON file "
            "(defaults to ./traces.json; pass '-' for stdout)"
        ),
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
    emit_stdout = args.output == "-"
    if args.output is None:
        destination: str | os.PathLike[str] | None = Path.cwd() / "traces.json"
    elif emit_stdout:
        destination = None
    else:
        destination = args.output
    chrome_trace = convert_jsonl_to_chrome_trace(
        args.input,
        destination,
        display_time_unit=args.display_time_unit,
    )
    if emit_stdout:
        json.dump(chrome_trace, sys.stdout, ensure_ascii=False)
        sys.stdout.write("\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
