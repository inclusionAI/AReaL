# SPDX-License-Identifier: Apache-2.0
"""Normal paths with real Unix RPC/git/pytest; only the Apptainer CLI is simulated."""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

from examples.task_workflows.environments.python import (
    PythonSandbox,
    ToolSettings,
    resolve_image,
)
from examples.task_workflows.environments.singularity import (
    Host,
    HostServer,
    create_environment,
    rpc,
)
from tests import test_task_workflows as task_fixtures
from tests.test_cc_task_workflows import (
    test_cc_native_sdk_uses_shared_environment_grading_and_reward as run_cc_case,
)
from tests.test_task_workflows import (
    test_task_service_runs_multi_call_tools_with_both_environments as run_hermes_case,
)

repository_case = task_fixtures.repository_case


@pytest.fixture(params=["default", "none"], ids=["default-proc", "no-proc"])
def sif_transport(tmp_path, monkeypatch, repository_case, request):
    """Keep the executable host-only: the agent never locates or invokes it."""
    sif = tmp_path / "fixture.sif"
    sif.write_bytes(b"offline SIF transport fixture, not a real filesystem image")
    manifest = tmp_path / "images.json"
    manifest.write_text(
        json.dumps(
            {
                "offline-image": {
                    "sif": str(sif),
                    "sha256": hashlib.sha256(sif.read_bytes()).hexdigest(),
                }
            }
        )
    )
    executable = tmp_path / "host-only-singularity"
    executable.write_text(
        "#!"
        + sys.executable
        + "\n"
        + r"""
import json, os, pathlib, shutil, subprocess, sys
args = sys.argv[1:]
log = pathlib.Path(os.environ["OFFLINE_SIF_LOG"])
with log.open("a") as stream:
    stream.write(json.dumps(args) + "\n")
if args[:2] == ["overlay", "create"]:
    overlay = pathlib.Path(args[-1])
    overlay.write_bytes(b"overlay")
    shutil.copytree(os.environ["OFFLINE_SIF_IMAGE"], overlay.parent / "scratch/image")
    sys.exit(0)
assert args[0] == "exec"
for flag in ["--fakeroot", "--containall", "--cleanenv", "--no-home", "--net"]:
    assert flag in args
assert args[args.index("--network") + 1] == "none"
expected = "hostfs,bind-paths,cwd"
if os.environ["OFFLINE_SIF_PROC_MODE"] == "none":
    expected += ",proc"
assert args[args.index("--no-mount") + 1] == expected
root = pathlib.Path(args[args.index("--overlay") + 1]).parent / "scratch/image"
image_index = next(i for i, value in enumerate(args) if value.endswith(".sif"))
command = args[image_index + 1:]
cwd = root / args[args.index("--pwd") + 1].lstrip("/")
if command == ["mkdir", "-p", "/work"]:
    (root / "work").mkdir(exist_ok=True)
    sys.exit(0)
if command == ["bash", "-lc", "command -v timeout >/dev/null"]:
    sys.exit(0)
if "--bind" in args:
    source = args[args.index("--bind") + 1].split(":")[0]
    command = [value.replace("/patches", source) for value in command]
if command[0] == "timeout":
    command = command[3:]
command = ["-c" if value == "-lc" else value for value in command]
if command[0] == "python":
    print("SIF diagnostic", file=sys.stderr, flush=True)
    command[0] = sys.executable
env = {"PATH": os.environ["PATH"], "HOME": str(root), "LANG": "C.UTF-8"}
sys.exit(subprocess.run(command, cwd=cwd, env=env).returncode)
"""
    )
    executable.chmod(0o755)
    monkeypatch.setenv("OFFLINE_SIF_IMAGE", repository_case["_test_image_directory"])
    monkeypatch.setenv("OFFLINE_SIF_LOG", str(tmp_path / "host-commands.jsonl"))
    monkeypatch.setenv("OFFLINE_SIF_PROC_MODE", request.param)
    monkeypatch.setenv("AREAL_TASK_RUNTIME", "singularity")
    empty = tmp_path / "unused-docker"
    empty.mkdir()
    with tempfile.TemporaryDirectory(prefix="areal-sif-", dir="/tmp") as socket_dir:
        path = Path(socket_dir) / "tools.sock"
        monkeypatch.setenv("AREAL_TASK_HOST_SOCKET", str(path))
        with _host_process(tmp_path, manifest, executable, path, request.param) as host:
            yield host, empty


@contextmanager
def _host_process(tmp_path, manifest, executable, path, proc_mode):
    """Optional real legacy interpreter; the agent/client remains on modern Python."""
    root = tmp_path / "host-state"
    host_python = os.environ.get("AREAL_TEST_HOST_PYTHON")
    if host_python:
        with (tmp_path / "host.log").open("w+") as log:
            process = subprocess.Popen(
                [
                    host_python,
                    "-S",
                    "-m",
                    "examples.task_workflows.environments.singularity_host",
                    "serve",
                    "--socket",
                    str(path),
                    "--root",
                    str(root),
                    "--images",
                    str(manifest),
                    "--executable",
                    str(executable),
                    "--workers",
                    "2",
                    "--proc-mode",
                    proc_mode,
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=Path(__file__).resolve().parents[1],
            )

            class ObservedHost:
                @property
                def tasks(self):
                    return rpc("health")["active_tasks"]

            observed = ObservedHost()
            observed.root = root
            try:
                deadline = time.monotonic() + 15
                while (
                    not path.exists()
                    and process.poll() is None
                    and time.monotonic() < deadline
                ):
                    time.sleep(0.05)
                log.seek(0)
                assert path.exists(), log.read()
                yield observed
            finally:
                process.terminate()
                assert process.wait(timeout=15) == 0
        return
    host = Host(
        root, manifest, executable=str(executable), workers=2, proc_mode=proc_mode
    )
    with HostServer(path, host) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield host
        finally:
            server.shutdown()
            thread.join()
            host.close()


def test_sif_host_preserves_task_files_without_exposing_credentials(
    tmp_path, monkeypatch, sif_transport
):
    host, _ = sif_transport
    monkeypatch.setenv("AREAL_AGENT_ADMIN_KEY", "private-training-key")
    settings = ToolSettings("offline-image", tmp_path / "client-tools")
    runtime = create_environment(
        PythonSandbox, settings, "normal-task", image_id=resolve_image(settings)
    )
    try:
        first = json.loads(
            runtime.execute("from pathlib import Path; Path('kept').write_text('42')")
        )
        second = json.loads(
            runtime.execute(
                "import os; print(open('kept').read()); print(os.getenv('AREAL_AGENT_ADMIN_KEY'))"
            )
        )
        assert first["status"] == second["status"] == "completed"
        assert second["output"] == "42\nNone\n"
        assert second["stdout"] == second["output"]
        assert second["stderr"] == "SIF diagnostic\n"
        assert runtime.metadata()["calls"][-1]["stderr"] == second["stderr"]
        assert runtime.metadata()["backend"] == "singularity"
        expected_mode = os.environ["OFFLINE_SIF_PROC_MODE"]
        assert runtime.metadata()["proc_mode"] == expected_mode
        assert rpc("health")["proc_mode"] == expected_mode
        assert rpc("health")["images"] == 1
    finally:
        runtime.close()
    assert not host.tasks
    assert not list(host.root.glob("*/overlay.img"))
    assert not list(host.root.glob("*/scratch"))
    assert runtime.metadata()["backend"] == "singularity"
    assert runtime.metadata()["proc_mode"] == expected_mode


@pytest.mark.parametrize("kind", ["python", "repository"])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.asyncio
async def test_sif_hermes_uses_shared_task_trace_and_exports(
    tmp_path, monkeypatch, repository_case, sif_transport, kind, streaming
):
    host, empty = sif_transport
    await run_hermes_case(
        tmp_path, monkeypatch, empty, repository_case, kind, streaming
    )
    assert not host.tasks


@pytest.mark.parametrize("benchmark_name", ["dapo_math", "swe_rebench"])
@pytest.mark.asyncio
async def test_sif_cc_uses_native_sdk_tools_and_grading(
    tmp_path, monkeypatch, repository_case, sif_transport, benchmark_name
):
    host, empty = sif_transport
    await run_cc_case(tmp_path, monkeypatch, empty, repository_case, benchmark_name)
    assert not host.tasks
