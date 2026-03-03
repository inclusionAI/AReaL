"""Tests for the ZMQ DEALER Worker and helper functions.

Verifies: _get_agent_config_from_env() env var parsing, WorkerZMQ lifecycle
and task processing with real ZMQ sockets, and _run_http_server health/configure
endpoints.

Uses real ZMQ sockets (ROUTER + PULL) to avoid mocking transport, matching
the test patterns in test_router.py and test_gateway.py.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Any
from unittest.mock import patch

import httpx
import pytest
import zmq

from areal.experimental.agent_service.worker_server import (
    ENV_AGENT_IMPORT_PATH_INTERNAL,
    ENV_AGENT_INIT_KWARGS_INTERNAL,
    ENV_AGENT_REUSE_INTERNAL,
    WorkerZMQ,
    _get_agent_config_from_env,
    _run_http_server,
)
from areal.utils.network import find_free_ports

# ---------------------------------------------------------------------------
# Mock agent service
# ---------------------------------------------------------------------------


class _MockService:
    """Stub AgentService for testing WorkerZMQ."""

    def __init__(self, return_value: Any = 1.0, raise_error: Exception | None = None):
        self.return_value = return_value
        self.raise_error = raise_error
        self.call_count = 0

    async def run_episode(
        self,
        data: dict,
        session_url: str,
        agent_kwargs: dict | None = None,
        agent_import_path: str | None = None,
    ) -> Any:
        self.call_count += 1
        if self.raise_error:
            raise self.raise_error
        return self.return_value

    async def health_check(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "running": True,
            "agent_import_path": "mock.Agent",
            "agent_reuse": False,
        }

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def zmq_ctx():
    """Create a ZMQ context for test sockets, cleaned up after use."""
    ctx = zmq.Context()
    yield ctx
    ctx.term()


@pytest.fixture
def worker_addrs():
    """Allocate free ports for Router-side sockets that WorkerZMQ connects to."""
    ports = find_free_ports(2)
    return {
        "task_addr": f"tcp://127.0.0.1:{ports[0]}",
        "result_addr": f"tcp://127.0.0.1:{ports[1]}",
    }


@pytest.fixture
def router_pull_sockets(zmq_ctx, worker_addrs):
    """Bind ROUTER (task dispatch) and PULL (result collection) sockets.

    * router_sock (ROUTER) — binds to task_addr; simulates Router's req_backend.
    * pull_sock   (PULL)   — binds to result_addr; simulates Router's res_frontend.
    """
    router_sock = zmq_ctx.socket(zmq.ROUTER)
    router_sock.setsockopt(zmq.LINGER, 0)
    router_sock.setsockopt(zmq.RCVTIMEO, 3000)
    router_sock.bind(worker_addrs["task_addr"])

    pull_sock = zmq_ctx.socket(zmq.PULL)
    pull_sock.setsockopt(zmq.LINGER, 0)
    pull_sock.setsockopt(zmq.RCVTIMEO, 3000)
    pull_sock.bind(worker_addrs["result_addr"])

    yield router_sock, pull_sock

    router_sock.close()
    pull_sock.close()


def _start_worker_thread(worker: WorkerZMQ) -> threading.Thread:
    """Start a WorkerZMQ in a daemon thread (start() is blocking)."""
    t = threading.Thread(target=worker.start, daemon=True)
    t.start()
    return t


def _recv_ready(router_sock: zmq.Socket) -> tuple[bytes, dict]:
    """Receive the READY handshake from a DEALER via ROUTER.

    ROUTER receives: [identity, empty, payload].
    Returns (identity, parsed_payload).
    """
    frames = router_sock.recv_multipart()
    assert len(frames) == 3, (
        f"Expected [identity, empty, payload], got {len(frames)} frames"
    )
    identity = frames[0]
    payload = json.loads(frames[2])
    return identity, payload


def _send_task(router_sock: zmq.Socket, identity: bytes, task: dict) -> None:
    """Send a task TO a DEALER via ROUTER: [identity, empty, payload]."""
    router_sock.send_multipart([identity, b"", json.dumps(task).encode()])


# ---------------------------------------------------------------------------
# TestGetAgentConfigFromEnv
# ---------------------------------------------------------------------------


class TestGetAgentConfigFromEnv:
    """Tests for _get_agent_config_from_env() environment variable parsing."""

    def test_defaults_when_no_env_vars_set(self, monkeypatch):
        """Test that defaults are (None, False, {}) when no env vars are set."""
        monkeypatch.delenv(ENV_AGENT_IMPORT_PATH_INTERNAL, raising=False)
        monkeypatch.delenv(ENV_AGENT_REUSE_INTERNAL, raising=False)
        monkeypatch.delenv(ENV_AGENT_INIT_KWARGS_INTERNAL, raising=False)

        path, reuse, kwargs = _get_agent_config_from_env()

        assert path is None
        assert reuse is False
        assert kwargs == {}

    def test_import_path_parsed_from_env(self, monkeypatch):
        """Test that AREAL_AGENT_IMPORT_PATH_INTERNAL is returned as agent_import_path."""
        monkeypatch.setenv(ENV_AGENT_IMPORT_PATH_INTERNAL, "my_module.MyAgent")

        path, _reuse, _kwargs = _get_agent_config_from_env()

        assert path == "my_module.MyAgent"

    def test_empty_import_path_returns_none(self, monkeypatch):
        """Test that empty string for import path returns None."""
        monkeypatch.setenv(ENV_AGENT_IMPORT_PATH_INTERNAL, "")

        path, _reuse, _kwargs = _get_agent_config_from_env()

        assert path is None

    def test_agent_reuse_true_variants(self, monkeypatch):
        """Test that 'true' and '1' both parse to agent_reuse=True."""
        for value in ("true", "True", "TRUE", "1"):
            monkeypatch.setenv(ENV_AGENT_REUSE_INTERNAL, value)

            _path, reuse, _kwargs = _get_agent_config_from_env()

            assert reuse is True, f"Expected True for {value!r}"

    def test_agent_reuse_false_variants(self, monkeypatch):
        """Test that 'false', '0', and random strings parse to agent_reuse=False."""
        for value in ("false", "False", "0", "no", "random"):
            monkeypatch.setenv(ENV_AGENT_REUSE_INTERNAL, value)

            _path, reuse, _kwargs = _get_agent_config_from_env()

            assert reuse is False, f"Expected False for {value!r}"

    def test_agent_init_kwargs_valid_json(self, monkeypatch):
        """Test that valid JSON for init kwargs is parsed correctly."""
        monkeypatch.setenv(
            ENV_AGENT_INIT_KWARGS_INTERNAL,
            '{"model": "gpt-4", "temperature": 0.7}',
        )

        _path, _reuse, kwargs = _get_agent_config_from_env()

        assert kwargs == {"model": "gpt-4", "temperature": 0.7}

    def test_agent_init_kwargs_invalid_json_returns_empty_dict(self, monkeypatch):
        """Test that invalid JSON for init kwargs falls back to empty dict."""
        monkeypatch.setenv(ENV_AGENT_INIT_KWARGS_INTERNAL, "not-json{")

        _path, _reuse, kwargs = _get_agent_config_from_env()

        assert kwargs == {}

    def test_all_env_vars_set_together(self, monkeypatch):
        """Test that all three env vars are parsed correctly together."""
        monkeypatch.setenv(ENV_AGENT_IMPORT_PATH_INTERNAL, "pkg.Agent")
        monkeypatch.setenv(ENV_AGENT_REUSE_INTERNAL, "1")
        monkeypatch.setenv(ENV_AGENT_INIT_KWARGS_INTERNAL, '{"k": "v"}')

        path, reuse, kwargs = _get_agent_config_from_env()

        assert path == "pkg.Agent"
        assert reuse is True
        assert kwargs == {"k": "v"}


# ---------------------------------------------------------------------------
# TestWorkerZMQ
# ---------------------------------------------------------------------------


class TestWorkerZMQ:
    """Tests for WorkerZMQ lifecycle and task processing with real ZMQ sockets."""

    def test_worker_sends_ready_on_startup(self, worker_addrs, router_pull_sockets):
        """Test that WorkerZMQ sends a READY message to the ROUTER on startup."""
        router_sock, _pull_sock = router_pull_sockets
        service = _MockService()
        worker = WorkerZMQ(
            task_addr=worker_addrs["task_addr"],
            result_addr=worker_addrs["result_addr"],
            service=service,
            worker_id="test-ready",
        )

        _start_worker_thread(worker)
        try:
            identity, payload = _recv_ready(router_sock)

            assert identity == b"test-ready"
            assert payload["type"] == "READY"
            assert payload["worker_id"] == "test-ready"
        finally:
            worker.stop()

    def test_worker_auto_generates_id(self, worker_addrs, router_pull_sockets):
        """Test that WorkerZMQ auto-generates a worker_id when none is provided."""
        router_sock, _pull_sock = router_pull_sockets
        service = _MockService()
        worker = WorkerZMQ(
            task_addr=worker_addrs["task_addr"],
            result_addr=worker_addrs["result_addr"],
            service=service,
            worker_id=None,
        )

        assert worker._worker_id.startswith("worker-")
        _start_worker_thread(worker)
        try:
            identity, payload = _recv_ready(router_sock)

            assert identity == worker._worker_id.encode()
            assert payload["type"] == "READY"
        finally:
            worker.stop()

    def test_worker_processes_task_success(self, worker_addrs, router_pull_sockets):
        """Test that WorkerZMQ processes a task and pushes success result via PUSH."""
        router_sock, pull_sock = router_pull_sockets
        service = _MockService(return_value=42.0)
        worker = WorkerZMQ(
            task_addr=worker_addrs["task_addr"],
            result_addr=worker_addrs["result_addr"],
            service=service,
            worker_id="test-success",
        )

        _start_worker_thread(worker)
        try:
            # Receive READY
            identity, _ = _recv_ready(router_sock)

            # Send a task via ROUTER -> DEALER: [identity, empty, payload]
            task = {
                "task_id": "task-1",
                "data": {"question": "2+2"},
                "session_url": "http://localhost:8000",
            }
            _send_task(router_sock, identity, task)

            # Receive result via PULL
            result = pull_sock.recv_json()

            assert result["task_id"] == "task-1"
            assert result["status"] == "success"
            assert result["result"] == 42.0
            assert service.call_count == 1
        finally:
            worker.stop()

    def test_worker_processes_task_error(self, worker_addrs, router_pull_sockets):
        """Test that WorkerZMQ sends error result when run_episode raises."""
        router_sock, pull_sock = router_pull_sockets
        service = _MockService(raise_error=ValueError("agent exploded"))
        worker = WorkerZMQ(
            task_addr=worker_addrs["task_addr"],
            result_addr=worker_addrs["result_addr"],
            service=service,
            worker_id="test-error",
        )

        _start_worker_thread(worker)
        try:
            identity, _ = _recv_ready(router_sock)

            task = {
                "task_id": "task-err",
                "data": {},
                "session_url": "http://localhost:8000",
            }
            _send_task(router_sock, identity, task)

            result = pull_sock.recv_json()

            assert result["task_id"] == "task-err"
            assert result["status"] == "error"
            assert "agent exploded" in result["error"]
        finally:
            worker.stop()

    def test_worker_handles_multiple_tasks(self, worker_addrs, router_pull_sockets):
        """Test that WorkerZMQ processes multiple tasks sequentially."""
        router_sock, pull_sock = router_pull_sockets
        service = _MockService(return_value=1.0)
        worker = WorkerZMQ(
            task_addr=worker_addrs["task_addr"],
            result_addr=worker_addrs["result_addr"],
            service=service,
            worker_id="test-multi",
        )

        _start_worker_thread(worker)
        try:
            identity, _ = _recv_ready(router_sock)

            # Send 3 tasks
            for i in range(3):
                task = {
                    "task_id": f"task-{i}",
                    "data": {"idx": i},
                    "session_url": "http://localhost:8000",
                }
                _send_task(router_sock, identity, task)
                # Small delay between sends to avoid overwhelming the poller
                time.sleep(0.05)

            # Collect results
            results = []
            for _ in range(3):
                results.append(pull_sock.recv_json())

            task_ids = sorted(r["task_id"] for r in results)
            assert task_ids == ["task-0", "task-1", "task-2"]
            assert all(r["status"] == "success" for r in results)
            assert service.call_count == 3
        finally:
            worker.stop()

    def test_worker_stop_terminates_loop(self, worker_addrs, router_pull_sockets):
        """Test that calling stop() causes the worker thread to exit."""
        _router_sock, _pull_sock = router_pull_sockets
        service = _MockService()
        worker = WorkerZMQ(
            task_addr=worker_addrs["task_addr"],
            result_addr=worker_addrs["result_addr"],
            service=service,
            worker_id="test-stop",
        )

        thread = _start_worker_thread(worker)
        # Wait for READY to be sent (worker is running)
        time.sleep(0.3)

        worker.stop()
        thread.join(timeout=3.0)

        assert not thread.is_alive(), "Worker thread did not exit after stop()"

    def test_worker_skips_malformed_payload(self, worker_addrs, router_pull_sockets):
        """Test that WorkerZMQ skips malformed JSON payloads without crashing."""
        router_sock, pull_sock = router_pull_sockets
        service = _MockService(return_value=99.0)
        worker = WorkerZMQ(
            task_addr=worker_addrs["task_addr"],
            result_addr=worker_addrs["result_addr"],
            service=service,
            worker_id="test-malformed",
        )

        _start_worker_thread(worker)
        try:
            identity, _ = _recv_ready(router_sock)

            # Send malformed payload (not valid JSON)
            router_sock.send_multipart([identity, b"", b"not-json{{{"])
            time.sleep(0.2)

            # Send a valid task after the malformed one
            valid_task = {
                "task_id": "task-after",
                "data": {},
                "session_url": "http://localhost:8000",
            }
            _send_task(router_sock, identity, valid_task)

            result = pull_sock.recv_json()
            assert result["task_id"] == "task-after"
            assert result["status"] == "success"
            assert result["result"] == 99.0
        finally:
            worker.stop()

    def test_worker_task_missing_fields_uses_defaults(
        self, worker_addrs, router_pull_sockets
    ):
        """Test that WorkerZMQ handles tasks with missing optional fields."""
        router_sock, pull_sock = router_pull_sockets
        service = _MockService(return_value=7.0)
        worker = WorkerZMQ(
            task_addr=worker_addrs["task_addr"],
            result_addr=worker_addrs["result_addr"],
            service=service,
            worker_id="test-defaults",
        )

        _start_worker_thread(worker)
        try:
            identity, _ = _recv_ready(router_sock)

            # Send minimal task — no data, session_url, agent_kwargs, or agent_import_path
            minimal_task = {"task_id": "task-minimal"}
            _send_task(router_sock, identity, minimal_task)

            result = pull_sock.recv_json()
            assert result["task_id"] == "task-minimal"
            assert result["status"] == "success"
            assert result["result"] == 7.0
        finally:
            worker.stop()

    def test_task_timeout(self, worker_addrs, router_pull_sockets):
        """Test that WorkerZMQ sends error result when run_episode exceeds task_timeout."""
        router_sock, pull_sock = router_pull_sockets

        class _SlowService(_MockService):
            async def run_episode(self, **kwargs) -> Any:
                await asyncio.sleep(10)  # simulate hung agent
                return 1.0

        service = _SlowService()
        worker = WorkerZMQ(
            task_addr=worker_addrs["task_addr"],
            result_addr=worker_addrs["result_addr"],
            service=service,
            worker_id="test-timeout",
            task_timeout=1.0,
        )

        _start_worker_thread(worker)
        t0 = time.monotonic()
        try:
            identity, _ = _recv_ready(router_sock)

            task = {
                "task_id": "task-slow",
                "data": {},
                "session_url": "http://localhost:8000",
            }
            _send_task(router_sock, identity, task)

            # Increase receive timeout for this test
            pull_sock.setsockopt(zmq.RCVTIMEO, 5000)
            result = pull_sock.recv_json()

            elapsed = time.monotonic() - t0
            assert result["task_id"] == "task-slow"
            assert result["status"] == "error"
            assert "timed out" in result["error"]
            assert elapsed < 5.0, f"Timeout took too long: {elapsed:.1f}s"
        finally:
            worker.stop()


# ---------------------------------------------------------------------------
# TestWorkerHttpEndpoints
# ---------------------------------------------------------------------------


class TestWorkerHttpEndpoints:
    """Tests for _run_http_server health and configure endpoints."""

    def test_health_endpoint_returns_ok(self):
        """Test that GET /health returns status=ok with agent info."""
        service = _MockService()
        port = find_free_ports(1)[0]

        thread = threading.Thread(
            target=_run_http_server, args=(service, port), daemon=True
        )
        thread.start()
        time.sleep(0.5)  # let uvicorn start

        resp = httpx.get(f"http://127.0.0.1:{port}/health", timeout=5.0)

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["running"] is True

    def test_configure_endpoint_returns_success(self):
        """Test that POST /configure sets seed and returns status=success."""
        from unittest.mock import MagicMock

        service = _MockService()
        port = find_free_ports(1)[0]

        config_obj = MagicMock()
        config_obj.seed = 42

        # Patch deserialize_value BEFORE starting the server thread so the
        # local import inside _run_http_server picks up the mock.
        with (
            patch(
                "areal.infra.rpc.serialization.deserialize_value",
                return_value=config_obj,
            ),
            patch("areal.utils.seeding.set_random_seed") as mock_seed,
        ):
            thread = threading.Thread(
                target=_run_http_server, args=(service, port), daemon=True
            )
            thread.start()
            time.sleep(0.5)

            resp = httpx.post(
                f"http://127.0.0.1:{port}/configure",
                json={"config": {"seed": 42}, "rank": 0},
                timeout=5.0,
            )

            assert resp.status_code == 200, f"Got {resp.status_code}: {resp.text}"
            body = resp.json()
            assert body["status"] == "success"
            mock_seed.assert_called_once_with(42, key="agent0")
