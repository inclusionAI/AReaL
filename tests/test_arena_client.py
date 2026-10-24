# SPDX-License-Identifier: Apache-2.0
"""Normal-path protocol checks; Arena and AReaL services are explicit doubles."""

import hashlib
import json
import sqlite3
import subprocess
import sys
from dataclasses import asdict

import httpx
import pytest

from examples.task_workflows.arena.backend import (
    ArenaCLI,
    ArenaExecutor,
    check_combination,
)
from examples.task_workflows.arena.config import ArenaConfig, digest
from examples.task_workflows.benchmarks.dapo_math import (
    DapoBenchmark,
    MathTask,
    question_id,
)
from examples.task_workflows.client import ClientSettings, EpisodeClient


@pytest.fixture
def arena_fixture(tmp_path, monkeypatch):
    """A real subprocess CLI double, with immutable registry and task artifacts."""
    tasks = [
        MathTask(question_id(q), q, "42")
        for q in ("Compute six times seven.", "Compute seven times six.")
    ]
    source = {
        "dataset": "fixture/dapo",
        "revision": "fixed",
        "config": "en",
        "split": "train",
    }
    task_file = tmp_path / "tasks.jsonl"
    task_file.write_text("".join(json.dumps(asdict(t)) + "\n" for t in tasks))
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "source": source,
                "splits": {
                    task_file.name: {
                        "count": 2,
                        "sha256": hashlib.sha256(task_file.read_bytes()).hexdigest(),
                    }
                },
            }
        )
    )
    rows = [
        {
            "task_id": t.task_id,
            "data_id": f"env:case-{i}@1.0.0",
            "prompt_sha256": hashlib.sha256(t.prompt().encode()).hexdigest(),
        }
        for i, t in enumerate(tasks)
    ]
    index = tmp_path / "task-index.jsonl"
    index.write_text("".join(json.dumps(row) + "\n" for row in rows))
    harness, reward = (
        {"key": "hermes-dapo", "version": "0.1.3"},
        {"key": "dapo-answer", "version": "0.1.3"},
    )
    specs = {
        "harness": {"agent": {"env": {"AUTOPILOT_ENTRYPOINT": "runner"}}},
        "reward": {"accepted_oracle_schema": "dapo.final-answer.v1"},
    }
    fixture_file = tmp_path / "registry.json"
    fixture_file.write_text(
        json.dumps(
            {
                "rows": rows,
                "prompts": [t.prompt() for t in tasks],
                "harness": harness,
                "reward": reward,
                "specs": specs,
            }
        )
    )
    executable = tmp_path / "arena"
    executable.write_text(
        "#!"
        + sys.executable
        + "\n"
        + r"""
import json, os, pathlib, sys
root = pathlib.Path(os.environ["ARENA_FIXTURE_ROOT"])
fixture = json.loads((root / "registry.json").read_text())
args = sys.argv[sys.argv.index("--json") + 1:]
payload = sys.stdin.read()
body = json.loads(payload) if payload else None
with (root / "calls.jsonl").open("a") as stream:
    stream.write(json.dumps({"args": args, "body": body}) + "\n")
kind, action = args[:2]
if kind in ("harness", "reward", "env") and action == "version-get":
    if kind == "env":
        i = int(args[2].split("-")[-1])
        spec = {"query": {"uri": "data:text/plain," + fixture["prompts"][i]},
                "oracle_schema": "dapo.final-answer.v1"}
    else:
        spec = fixture["specs"][kind]
    data = {"status": "PUBLISHED", "spec": spec}
elif kind == "stream" and action == "get":
    data = {"stream": {"stream_id": args[2], "status": "ACTIVE",
            "default_harness_ref": fixture["harness"], "default_reward_ref": fixture["reward"]}}
elif kind == "api":
    offset = int(args[-1].split("=")[-1])
    data = {"data_ids": [fixture["rows"][offset]["data_id"]], "total": 2}
elif kind == "stream" and action == "launch":
    env_key = body["data_id"][4:].split("@")[0]
    data = {"job_id": body["job_id"], "task_id": body["job_id"] + ":" + env_key + ":0"}
    (root / "launch.json").write_text(json.dumps(body))
elif kind == "task":
    launch = json.loads((root / "launch.json").read_text())
    episode = json.loads(launch["envs"].get("AUTOPILOT_AREAL_EPISODE_JSON", "null"))
    artifacts = {
        "answer.json": {"answer": r"\boxed{42}", "completed": True, "model_mode": "external",
                        "training_eligible": episode is not None, "areal_episode": episode,
                        "sampling": json.loads(launch["envs"]["AUTOPILOT_SAMPLING_JSON"]), "model_calls": 2},
        "environment.json": {"calls": [{"status": "completed"}], "timing_calls": [{"started": 3, "finished": 4}]},
        "call_trace.json": {"nodes": [
            {"name": "request-1", "kind": "request", "seconds": 2, "predecessors": [], "input_tokens": 10, "output_tokens": 3},
            {"name": "tool-2", "kind": "external", "seconds": 1, "predecessors": ["request-1"]},
            {"name": "request-2", "kind": "request", "seconds": 3, "predecessors": ["tool-2"], "input_tokens": 20, "output_tokens": 5}]},
        "tools.jsonl": {"code": "print(6 * 7)", "stdout": "42\n"}}
    if action == "get":
        key, version = launch["data_id"][4:].split("@")
        data = {"status": "DONE", "env_ref": {"key": key, "version": version},
                "harness_ref": fixture["harness"], "reward_ref": fixture["reward"]}
    elif action == "result":
        data = {"status": "OK", "score": float(os.environ.get("ARENA_FIXTURE_REWARD", "1")),
                "raw": {"reason": "fixture_score", "model_mode": "external", "training_eligible": episode is not None}}
    elif action == "artifacts":
        data = [{"name": name} for name in artifacts]
    elif action == "download":
        pathlib.Path(args[-1]).write_text(json.dumps(artifacts[args[3]]))
        data = {"path": args[-1]}
    else:
        raise AssertionError(args)
else:
    raise AssertionError(args)
print(json.dumps({"ok": True, "data": data}))
"""
    )
    executable.chmod(0o755)
    monkeypatch.setenv("ARENA_FIXTURE_ROOT", str(tmp_path))
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "name": "fixture",
                "benchmark": "dapo_math",
                "model_mode": "external",
                "arena": {
                    "binary": str(executable),
                    "base_url": "https://arena.invalid/api",
                    "profile": "fixture",
                    "stream_id": "dapo",
                },
                "harness_ref": harness,
                "reward_ref": reward,
                "component_spec_sha256": {
                    f"{ref['key']}@{ref['version']}": digest(specs[kind])
                    for kind, ref in (("harness", harness), ("reward", reward))
                },
                "oracle_schema": "dapo.final-answer.v1",
                "sampling": {"max_tokens": 128, "temperature": 1.0, "top_p": 1.0},
                "dataset": {
                    "source": source,
                    "count": 2,
                    "task_index": index.name,
                    "task_index_sha256": hashlib.sha256(index.read_bytes()).hexdigest(),
                },
            }
        )
    )
    return ArenaConfig.load(config_path), tasks, task_file


@pytest.mark.asyncio
async def test_pinned_combination_checks_registry_pages_and_exact_selected_prompts(
    arena_fixture,
):
    config, tasks, _ = arena_fixture
    report = await check_combination(
        config, ArenaCLI(config), [DapoBenchmark().prepare(t) for t in tasks]
    )
    assert report["status"] == "OK"
    assert report["dataset_count"] == report["selected_prompts_checked"] == 2
    calls = [
        json.loads(line)
        for line in (config.path.parent / "calls.jsonl").read_text().splitlines()
    ]
    pages = [call for call in calls if call["args"][:2] == ["api", "get"]]
    assert [page["args"][-1] for page in pages] == ["offset=0", "offset=1"]


def inbox_fixture(state_root):
    """Only the existing client's read-only SQLite view; no trainer is simulated."""
    directory = state_root / "inbox"
    directory.mkdir(parents=True)
    database = directory / "queue.sqlite3"
    with sqlite3.connect(database) as db:
        db.execute(
            "CREATE TABLE online_episodes (episode_id TEXT,source_id TEXT,session_id TEXT,trajectory_id INTEGER,status TEXT,digest TEXT,source_acked INTEGER,min_version INTEGER,max_version INTEGER,batch_id TEXT,rejection TEXT)"
        )
        db.execute(
            "CREATE TABLE online_progress (admission_open INTEGER,max_staleness INTEGER,batch_size INTEGER)"
        )
        db.execute("INSERT INTO online_progress VALUES (1,2,1)")
    return database


@pytest.mark.parametrize("mode", ["eval", "resident", "on_demand"])
@pytest.mark.asyncio
async def test_arena_client_preserves_episode_reward_trace_and_completed_resume(
    tmp_path, monkeypatch, arena_fixture, mode
):
    config, tasks, _ = arena_fixture
    output = tmp_path / "output"
    output.mkdir()
    state_root = tmp_path / "state"
    database = inbox_fixture(state_root) if mode == "on_demand" else None
    episode = {"incarnation_id": "worker-one", "trajectory_id": 0}
    observed = []
    # The client must use the separate Arena reward even when it disagrees
    # with local final-answer grading. This also exercises an ordinary zero score.
    reward = 0.0 if mode == "eval" else 1.0
    monkeypatch.setenv("ARENA_FIXTURE_REWARD", str(reward))

    def respond(request):
        body = json.loads(request.content)
        observed.append(request.url.path)
        if request.url.path == "/rl/start_session":
            assert request.headers["authorization"] == "Bearer admin-test-secret"
            assert body == {"task_id": "math-stable"}
            return httpx.Response(
                201,
                json={
                    "sessions": [
                        {
                            "session_id": "session-one",
                            "session_api_key": "session-test-secret",
                            "areal_episode": episode,
                        }
                    ]
                },
            )
        assert request.url.path == "/rl/set_reward"
        assert request.headers["authorization"] == "Bearer session-test-secret"
        assert body == {"reward": reward, "areal_episode": episode}
        if database:
            with sqlite3.connect(database) as db:
                db.execute(
                    "INSERT INTO online_episodes VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        "episode-one",
                        "worker:worker-one",
                        "session-one",
                        0,
                        "received",
                        "fixture-digest",
                        1,
                        0,
                        0,
                        None,
                        None,
                    ),
                )
        return httpx.Response(
            200,
            json={
                "session_id": "session-one",
                "trajectory_id": 0,
                "trajectory_ready": True,
            },
        )

    benchmark = DapoBenchmark()
    monkeypatch.setattr(
        benchmark, "grade", lambda *args: pytest.fail("Arena owns scoring")
    )
    settings = ClientSettings(
        "",
        "http://inference",
        "",
        "admin-test-secret",
        "small-model",
        mode,
        output,
        state_root=state_root if database else None,
        backend="arena",
    )
    executor = ArenaExecutor(
        config,
        ArenaCLI(config),
        output,
        identity={"mode": mode, "model": "small-model"},
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        client = EpisodeClient(settings, http, benchmark, executor=executor)
        result = await client.run_task(tasks[0], request_id="math-stable")
        calls_before = (tmp_path / "calls.jsonl").read_text()
        again = await client.run_task(tasks[0], request_id="math-stable")
    assert result == again
    assert (tmp_path / "calls.jsonl").read_text() == calls_before
    assert observed == (
        [] if mode == "eval" else ["/rl/start_session", "/rl/set_reward"]
    )
    assert result["status"] == "completed" and result["score"]["reward"] == reward
    assert result["metadata"]["api_calls"] == 2
    assert (
        result["timing"]["inference_seconds"] == 5
        and result["timing"]["tool_seconds"] == 1
    )
    assert result.get("handoff", {}).get("verified", False) == (mode == "on_demand")
    assert "test-secret" not in json.dumps(result)
    launch = json.loads((tmp_path / "launch.json").read_text())
    assert launch["api_key"] == (
        "admin-test-secret" if mode == "eval" else "session-test-secret"
    )
    assert launch["base_url"] == "http://inference/v1"
    assert launch["harness_ref"] == config.value["harness_ref"]
    assert "gold_answer" not in json.dumps(launch) and "question" not in launch
    if mode != "eval":
        assert json.loads(launch["envs"]["AUTOPILOT_AREAL_EPISODE_JSON"]) == episode
    saved = output / "math-stable/state.json"
    assert saved.stat().st_mode & 0o777 == 0o600
    assert json.loads(saved.read_text())["completed_result"] == result
    trace = json.loads((saved.parent / "artifacts/call_trace.json").read_text())
    assert trace["nodes"][-1]["predecessors"] == ["tool-2"]
    calls = [json.loads(line) for line in calls_before.splitlines()]
    assert all("test-secret" not in json.dumps(c["args"]) for c in calls)
    assert sum(c["args"][:2] == ["stream", "launch"] for c in calls) == 1


@pytest.mark.asyncio
async def test_arena_client_resumes_a_saved_completed_task_before_reward(
    tmp_path, monkeypatch, arena_fixture
):
    """A normal checkpoint between task execution and reward keeps both identities."""
    config, tasks, _ = arena_fixture
    output = tmp_path / "resumed"
    output.mkdir()
    task = tasks[0]
    episode = {"incarnation_id": "worker-one", "trajectory_id": 0}
    credentials = {
        "session_id": "session-one",
        "session_api_key": "session-test-secret",
        "areal_episode": episode,
    }
    state_root = tmp_path / "state"
    database = inbox_fixture(state_root)
    identity = {"mode": "on_demand", "model": "small-model"}
    executor = ArenaExecutor(config, ArenaCLI(config), output, identity=identity)
    await executor.save(
        "math-planned", task_id=task.task_id, credentials=credentials, admitted=True
    )
    execution = await executor.execute(
        DapoBenchmark().prepare(task),
        request_id="math-planned",
        base_url="http://inference/v1",
        api_key=credentials["session_api_key"],
        model="small-model",
        episode=episode,
    )
    assert execution.score.reward == 1.0
    observed = []

    def respond(request):
        observed.append(request.url.path)
        assert request.url.path == "/rl/set_reward"
        assert json.loads(request.content) == {"reward": 1.0, "areal_episode": episode}
        with sqlite3.connect(database) as db:
            db.execute(
                "INSERT INTO online_episodes VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    "episode-one",
                    "worker:worker-one",
                    "session-one",
                    0,
                    "received",
                    "fixture-digest",
                    1,
                    0,
                    0,
                    None,
                    None,
                ),
            )
        return httpx.Response(
            200,
            json={
                "session_id": "session-one",
                "trajectory_id": 0,
                "trajectory_ready": True,
            },
        )

    settings = ClientSettings(
        "",
        "http://inference",
        "",
        "admin-test-secret",
        "small-model",
        "on_demand",
        output,
        state_root=state_root,
        backend="arena",
    )
    resumed = ArenaExecutor(config, ArenaCLI(config), output, identity=identity)
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        client = EpisodeClient(settings, http, executor=resumed)
        execute = resumed.execute

        async def observe_reservation(*args, **kwargs):
            assert client._starting == {credentials["session_id"]: credentials}
            return await execute(*args, **kwargs)

        monkeypatch.setattr(resumed, "execute", observe_reservation)
        summary = await client.run([task], request_ids=["math-planned"])
    assert summary["completed"] == summary["durable_handoffs"] == 1
    assert observed == ["/rl/set_reward"]
    calls = [
        json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()
    ]
    assert sum(c["args"][:2] == ["stream", "launch"] for c in calls) == 1


def test_arena_cli_run_snapshots_inputs_and_resume_reuses_existing_result(
    tmp_path, monkeypatch, arena_fixture
):
    config, _, task_file = arena_fixture
    monkeypatch.setenv("AREAL_INFERENCE_ADMIN_KEY", "eval-test-secret")
    output = tmp_path / "run"
    command = [sys.executable, "-m", "examples.task_workflows.arena.cli"]
    result = subprocess.run(
        command
        + [
            "run",
            "--config",
            str(config.path),
            "--tasks",
            str(task_file),
            "--limit",
            "1",
            "--output",
            str(output),
            "--inference-url",
            "http://inference",
            "--model",
            "small-model",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout)["completed"] == 1
    assert "test-secret" not in result.stdout + result.stderr
    plan = json.loads((output / "run.json").read_text())
    assert len(plan["request_ids"]) == 1
    assert "test-secret" not in json.dumps(plan)
    # The saved run no longer depends on the original source files.
    task_file.unlink()
    config.path.unlink()
    result = subprocess.run(
        command + ["resume", "--run-dir", str(output)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout)["completed"] == 1
    calls = [
        json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()
    ]
    assert sum(c["args"][:2] == ["stream", "launch"] for c in calls) == 1
    assert (output / "config.json").stat().st_mode & 0o777 == 0o600
