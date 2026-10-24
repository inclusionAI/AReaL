# SPDX-License-Identifier: Apache-2.0
"""Grade the Arena answer in a separate sandbox with the frozen DAPO verifier."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

from examples.task_workflows.arena.runner import write_json
from examples.task_workflows.benchmarks.dapo_math import score_answer


def grade(answer_path: Path, oracle_path: Path, output: Path) -> dict:
    answer = json.loads(answer_path.read_text())
    if oracle_path.is_dir():
        oracle_path = oracle_path / "oracle.json"
    oracle = json.loads(oracle_path.read_text())
    score = score_answer(
        answer["answer"], oracle["gold_answer"], completed=answer["completed"]
    )
    result = {
        "status": "OK",
        "score": score.reward,
        "raw": {
            **asdict(score),
            "model_mode": answer["model_mode"],
            "training_eligible": answer["training_eligible"],
        },
    }
    write_json(output, result)
    return result


def main() -> None:
    grade(
        Path(os.environ["ARENA_AGENT_OUTPUT_DIR"]) / "answer.json",
        Path(os.environ["ARENA_ORACLE_PATH"]),
        Path(os.environ["ARENA_REWARD_OUTPUT_FILE"]),
    )


if __name__ == "__main__":
    main()
