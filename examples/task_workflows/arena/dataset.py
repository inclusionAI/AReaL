# SPDX-License-Identifier: Apache-2.0
"""Prepare exact task inputs from an existing frozen DAPO split."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from examples.task_workflows.arena.runner import write_json
from examples.task_workflows.benchmarks.dapo_math import load_tasks


def export_task(task_file: Path, index: int, output: Path) -> dict:
    tasks, manifest = load_tasks(task_file)
    task = tasks[index]
    output.mkdir(parents=True, exist_ok=False)
    public, oracle = output / "public", output / "reward_only"
    public.mkdir()
    oracle.mkdir(mode=0o700)
    (public / "query.txt").write_text(task.prompt())
    write_json(
        oracle / "oracle.json",
        {"task_id": task.task_id, "gold_answer": task.gold_answer},
    )
    record = {
        "task_id": task.task_id,
        "source": manifest["source"],
        "index": index,
        "split_sha256": hashlib.sha256(task_file.read_bytes()).hexdigest(),
        "prompt_sha256": hashlib.sha256(task.prompt().encode()).hexdigest(),
    }
    write_json(output / "selection.json", record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-file", type=Path, required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export_task(args.task_file, args.index, args.output)


if __name__ == "__main__":
    main()
