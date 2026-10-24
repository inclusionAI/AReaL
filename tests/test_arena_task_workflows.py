# SPDX-License-Identifier: Apache-2.0
"""Normal-path checks without a training stack or a live Arena account."""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from examples.task_workflows.arena.dataset import export_task
from examples.task_workflows.arena.package import build
from examples.task_workflows.arena.reward import grade
from examples.task_workflows.benchmarks.dapo_math import MathTask, question_id


def test_arena_export_preserves_frozen_prompt_and_separate_oracle(tmp_path):
    """The public task bytes and source digest survive the Arena export."""
    task = MathTask(question_id("Compute 6 times 7."), "Compute 6 times 7.", "42")
    payload = json.dumps(asdict(task)) + "\n"
    source = tmp_path / "frozen"
    source.mkdir()
    task_file = source / "smoke.jsonl"
    task_file.write_text(payload)
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "source": {"revision": "test-fixture"},
                "splits": {
                    task_file.name: {
                        "count": 1,
                        "sha256": hashlib.sha256(payload.encode()).hexdigest(),
                    }
                },
            }
        )
    )
    target = tmp_path / "exported"
    result = export_task(task_file, 0, target)
    assert (target / "public/query.txt").read_text() == task.prompt()
    assert (
        json.loads((target / "reward_only/oracle.json").read_text())["gold_answer"]
        == "42"
    )
    assert "gold_answer" not in (target / "public/query.txt").read_text()
    assert result["source"] == {"revision": "test-fixture"}


@pytest.mark.parametrize("oracle_directory", [False, True])
def test_arena_reward_uses_existing_final_answer_semantics(tmp_path, oracle_directory):
    """A completed answer receives the original DAPO score and mock provenance."""
    answer = tmp_path / "answer.json"
    oracle = tmp_path / "oracle.json"
    answer.write_text(
        json.dumps(
            {
                "answer": r"\boxed{42}",
                "completed": True,
                "model_mode": "mock",
                "training_eligible": False,
            }
        )
    )
    oracle.write_text(json.dumps({"gold_answer": "42"}))
    if oracle_directory:
        directory = tmp_path / "oracle"
        directory.mkdir()
        oracle.rename(directory / "oracle.json")
        oracle = directory
    result = grade(answer, oracle, tmp_path / "reward.json")
    assert result["status"] == "OK" and result["score"] == 1
    assert result["raw"]["reason"] == "correct"
    assert result["raw"]["training_eligible"] is False


def test_arena_package_is_repeatable_and_runs_reward_without_training_imports(tmp_path):
    """The actual self-extracting executable can grade using a clean interpreter."""
    one, two = tmp_path / "one", tmp_path / "two"
    build(one, mode="reward")
    build(two, mode="reward")
    assert one.read_bytes() == two.read_bytes()
    agent = tmp_path / "agent"
    agent.mkdir()
    (agent / "answer.json").write_text(
        json.dumps(
            {
                "answer": r"\boxed{42}",
                "completed": True,
                "model_mode": "mock",
                "training_eligible": False,
            }
        )
    )
    oracle = tmp_path / "oracle.json"
    oracle.write_text(json.dumps({"gold_answer": "42"}))
    output = tmp_path / "reward.json"
    env = {
        **os.environ,
        "ARENA_AGENT_OUTPUT_DIR": str(agent),
        "ARENA_ORACLE_PATH": str(oracle),
        "ARENA_REWARD_OUTPUT_FILE": str(output),
    }
    # This venv provides the already-declared math-verify dependency. The runner
    # uses no AReaL package initializer, torch, Ray or training engine imports.
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env["PATH"]
    subprocess.run(
        ["sh", str(one)], env=env, check=True, capture_output=True, text=True
    )
    assert json.loads(output.read_text())["score"] == 1


@pytest.mark.skipif(
    importlib.util.find_spec("run_agent") is None,
    reason="hermes-agent==0.19.0 is optional",
)
def test_arena_native_hermes_calls_tool_and_preserves_episode_identity(tmp_path):
    """Native Hermes and its SDK make real HTTP calls; local tool transport is explicit."""
    prompt = "Compute 6 times 7."
    recipe = {
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "code": "print(6 * 7)",
    }
    output = tmp_path / "out"
    query, recipe_file = tmp_path / "query.txt", tmp_path / "recipe.json"
    query.write_text(prompt)
    recipe_file.write_text(json.dumps(recipe))
    # Hermes has top-level modules named utils/tools. Run its real import in a
    # clean process so pytest's tests/utils.py and scripted-agent doubles cannot
    # shadow the upstream package.
    subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.task_workflows.arena.smoke",
            "--query",
            str(query),
            "--recipe",
            str(recipe_file),
            "--output",
            str(output),
            "--local-trusted-fixture",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "HERMES_HOME": str(tmp_path / "hermes-home")},
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    result = json.loads((output / "answer.json").read_text())
    requests = json.loads((output / "mock_requests.json").read_text())
    trace = json.loads((output / "call_trace.json").read_text())
    assert result["answer"] == r"\boxed{42}" and result["completed"]
    assert not result["training_eligible"]
    assert len(requests) == 2
    assert all(row["areal_episode"] == result["areal_episode"] for row in requests)
    assert trace["dependencies_complete"] and trace["lengths_complete"]
    assert [node["kind"] for node in trace["nodes"]] == [
        "request",
        "external",
        "request",
    ]
    assert trace["nodes"][2]["predecessors"] == [trace["nodes"][1]["name"]]
