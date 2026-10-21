# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest

from examples.autopilot.elastic.fixtures.inference_service import InferenceService
from examples.autopilot.elastic.swe import rollout_to_queue

from areal.autopilot.elastic.runtime.processes import (
    OWNER_ENV,
    owned_processes,
    stop_owned_processes,
)


def test_orphaned_session_child_is_stopped_without_touching_other_owner(tmp_path):
    """Leader exit and setsid must not leave a compute process behind."""
    owner = uuid.uuid4().hex
    code = (
        "import subprocess,sys; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)'], "
        "start_new_session=True, stdout=subprocess.DEVNULL); "
        "print(p.pid, flush=True)"
    )
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    try:
        leader = subprocess.run(
            [sys.executable, "-c", code],
            env={**os.environ, OWNER_ENV: owner},
            stdout=subprocess.PIPE,
            text=True,
            check=True,
            timeout=10,
        )
        child = int(leader.stdout)
        deadline = time.monotonic() + 5
        while child not in [p.pid for p in owned_processes(owner)]:
            if time.monotonic() > deadline:
                pytest.fail("Owned child did not start")
            time.sleep(0.05)
        stop_owned_processes(owner, grace=1, force_timeout=2)
        assert not owned_processes(owner)
        assert unrelated.poll() is None
    finally:
        stop_owned_processes(owner, grace=0.1, force_timeout=2)
        unrelated.terminate()
        unrelated.wait(timeout=5)


def test_close_waits_for_entire_start_before_releasing_resources():
    """Cancellation of to_thread must not permit close to race late startup."""
    service = object.__new__(InferenceService)
    service._lifecycle_lock = threading.Lock()
    service._closed = False
    started, finish = threading.Event(), threading.Event()
    order = []

    def start(timeout):
        started.set()
        assert finish.wait(timeout=5)
        order.append("created")
        return "gateway"

    def close():
        assert order == ["created"]
        order.append("closed")

    service._start, service._close = start, close
    with ThreadPoolExecutor(max_workers=2) as executor:
        startup = executor.submit(service.start)
        assert started.wait(timeout=5)
        cleanup = executor.submit(service.close)
        try:
            assert not cleanup.done()
        finally:
            finish.set()
        assert startup.result(timeout=5) == "gateway"
        cleanup.result(timeout=5)
    assert order == ["created", "closed"]


@pytest.mark.asyncio
async def test_report_disk_failure_still_closes_inference(tmp_path, monkeypatch):
    """ENOSPC while writing diagnostics must not bypass GPU/service cleanup."""
    import examples.autopilot.elastic.fixtures.inference_service as inference

    import areal.autopilot.elastic.runtime.state.checkpoint as checkpoint

    closed = []

    class Service:
        def __init__(self, **kwargs):
            pass

        def start(self):
            raise RuntimeError("startup failed")

        def close(self):
            closed.append(True)
            return {"gpu_free": True}

    def no_space(*args):
        raise OSError("ENOSPC")

    async def sandbox_cleanup(args):
        return False

    monkeypatch.setattr(inference, "InferenceService", Service)
    monkeypatch.setattr(checkpoint, "atomic_json", no_space)
    monkeypatch.setattr(rollout_to_queue, "verify_sandbox_cleanup", sandbox_cleanup)
    args = argparse.Namespace(
        output=tmp_path / "run",
        queue=tmp_path / "queue",
        model=tmp_path,
        model_name="test",
        model_revision="pinned",
        device=0,
    )
    with pytest.raises(OSError, match="ENOSPC"):
        await rollout_to_queue.run(args)
    assert closed == [True]


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_on", [None, "delete", "get"])
async def test_sandbox_cleanup_deletes_only_owned_pod_and_requires_successful_query(
    tmp_path, fail_on
):
    """An SDK exit or an API failure alone is not proof that the sandbox is gone."""
    calls = tmp_path / "calls.jsonl"
    kubectl = tmp_path / "kubectl"
    kubectl.write_text(
        f"#!{sys.executable}\n"
        "import json,sys\n"
        f"with open({str(calls)!r},'a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\n"
        "operation='delete' if 'delete' in sys.argv else 'get'\n"
        f"if operation == {fail_on!r}: sys.exit(1)\n"
        "print(json.dumps({'items':[]}))\n"
    )
    kubectl.chmod(0o755)
    (tmp_path / "agent-report.sandbox.json").write_text(
        json.dumps({"instance_id": "owned-episode"})
    )
    args = argparse.Namespace(
        output=tmp_path,
        kubectl=kubectl,
        kubeconfig=tmp_path / "config",
        sandbox_namespace="sandbox-test",
    )
    if fail_on:
        with pytest.raises(RuntimeError, match="Could not"):
            await rollout_to_queue.verify_sandbox_cleanup(args)
    else:
        assert await rollout_to_queue.verify_sandbox_cleanup(args)
    recorded = [json.loads(line) for line in calls.read_text().splitlines()]
    assert "pod/owned-episode" in recorded[0]
    assert recorded[0][recorded[0].index("-n") + 1] == "sandbox-test"
    if fail_on != "delete":
        assert "metadata.name=owned-episode" in recorded[1]


def test_resume_requires_every_proxy_to_leave_pause(monkeypatch):
    from unittest.mock import MagicMock

    service = object.__new__(InferenceService)
    service.controller = MagicMock()
    service.controller._inf_addrs = ["http://backend-a", "http://backend-b"]
    service.controller._data_proxy_addrs = ["http://proxy-a", "http://proxy-b"]
    client = MagicMock()
    client.__enter__.return_value = client

    def response(url):
        result = MagicMock()
        result.json.return_value = {"paused": url.startswith("http://proxy-b")}
        return result

    client.get.side_effect = response
    monkeypatch.setattr(
        "examples.autopilot.elastic.fixtures.inference_service.httpx.Client",
        lambda **kwargs: client,
    )
    with pytest.raises(RuntimeError, match="still paused"):
        service.resume()
    assert client.get.call_count == 4
