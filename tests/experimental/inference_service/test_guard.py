"""Unit tests for RPCGuard Flask app (areal.experimental.inference_service.guard.app).

Tests all 4 endpoints (/health, /alloc_ports, /fork, /kill_forked_worker)
and the cleanup_forked_children() function using Flask test client with
mocked subprocess spawning.
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from areal.experimental.inference_service.guard import app as guard_module
from areal.experimental.inference_service.guard.app import app, cleanup_forked_children


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def _reset_guard_globals():
    """Reset module-level global state before each test."""
    guard_module._allocated_ports = set()
    guard_module._forked_children = []
    guard_module._forked_children_map = {}
    guard_module._server_host = "10.0.0.1"
    guard_module._experiment_name = "test-exp"
    guard_module._trial_name = "test-trial"
    guard_module._fileroot = None
    yield
    # Cleanup after test
    guard_module._allocated_ports = set()
    guard_module._forked_children = []
    guard_module._forked_children_map = {}


@pytest.fixture()
def client():
    """Flask test client for RPCGuard app."""
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def _make_mock_process(pid: int = 12345, running: bool = True) -> MagicMock:
    """Create a mock subprocess.Popen with controllable poll()."""
    proc = MagicMock(spec=subprocess.Popen)
    proc.pid = pid
    proc.poll.return_value = None if running else 0
    return proc


# =============================================================================
# TestHealth
# =============================================================================


class TestHealth:
    """GET /health returns healthy status with child count."""

    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "healthy"
        assert data["forked_children"] == 0

    def test_health_counts_forked_children(self, client):
        # Add mock children to the global list
        guard_module._forked_children = [MagicMock(), MagicMock(), MagicMock()]
        resp = client.get("/health")
        data = resp.get_json()
        assert data["forked_children"] == 3


# =============================================================================
# TestAllocPorts
# =============================================================================


class TestAllocPorts:
    """POST /alloc_ports allocates unique ports and tracks exclusions."""

    @patch("areal.experimental.inference_service.guard.app.find_free_ports")
    def test_alloc_ports_success(self, mock_find, client):
        mock_find.return_value = [9001, 9002, 9003]
        resp = client.post("/alloc_ports", json={"count": 3})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "success"
        assert data["ports"] == [9001, 9002, 9003]
        assert data["host"] == "10.0.0.1"
        # Ports tracked in exclusion set
        assert guard_module._allocated_ports == {9001, 9002, 9003}

    @patch("areal.experimental.inference_service.guard.app.find_free_ports")
    def test_alloc_ports_excludes_previous(self, mock_find, client):
        """Second allocation excludes ports from the first."""
        mock_find.return_value = [9001, 9002, 9003]
        client.post("/alloc_ports", json={"count": 3})

        mock_find.return_value = [9004, 9005]
        resp = client.post("/alloc_ports", json={"count": 2})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ports"] == [9004, 9005]

        # find_free_ports called with prior exclusions
        _, kwargs = mock_find.call_args
        assert 9001 in kwargs.get("exclude_ports", set())
        assert 9002 in kwargs.get("exclude_ports", set())
        assert 9003 in kwargs.get("exclude_ports", set())

        # All 5 ports tracked
        assert guard_module._allocated_ports == {9001, 9002, 9003, 9004, 9005}

    def test_alloc_ports_missing_count(self, client):
        resp = client.post("/alloc_ports", json={})
        assert resp.status_code == 400
        assert "count" in resp.get_json()["error"].lower()

    def test_alloc_ports_invalid_count_zero(self, client):
        resp = client.post("/alloc_ports", json={"count": 0})
        assert resp.status_code == 400

    def test_alloc_ports_invalid_count_negative(self, client):
        resp = client.post("/alloc_ports", json={"count": -1})
        assert resp.status_code == 400

    def test_alloc_ports_invalid_count_string(self, client):
        resp = client.post("/alloc_ports", json={"count": "three"})
        assert resp.status_code == 400

    def test_alloc_ports_no_json_body(self, client):
        resp = client.post("/alloc_ports", data="not json", content_type="text/plain")
        assert resp.status_code == 400


# =============================================================================
# TestForkModulePath
# =============================================================================


class TestForkModulePath:
    """POST /fork with command field — module-path mode."""

    @patch(
        "areal.experimental.inference_service.guard.app._wait_for_worker_ready", return_value=True
    )
    @patch("areal.experimental.inference_service.guard.app.run_with_streaming_logs")
    @patch("areal.experimental.inference_service.guard.app.find_free_ports")
    def test_fork_module_path_builds_correct_cmd(
        self, mock_find, mock_run, mock_wait, client
    ):
        mock_find.return_value = [8001]
        mock_proc = _make_mock_process(pid=42)
        mock_run.return_value = mock_proc

        resp = client.post(
            "/fork",
            json={"role": "test-worker", "worker_index": 0, "command": "some.module"},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "success"
        assert data["host"] == "10.0.0.1"
        assert data["port"] == 8001
        assert data["pid"] == 42

        # Verify command was built correctly
        call_args = mock_run.call_args
        cmd = call_args[0][0]
        assert "-m" in cmd
        assert "some.module" in cmd
        assert "--role" in cmd
        assert "test-worker" in cmd
        assert "--worker-index" in cmd
        assert "0" in cmd
        assert "--experiment-name" in cmd
        assert "test-exp" in cmd
        assert "--trial-name" in cmd
        assert "test-trial" in cmd
        assert "--port" in cmd
        assert "8001" in cmd

    @patch(
        "areal.experimental.inference_service.guard.app._wait_for_worker_ready", return_value=True
    )
    @patch("areal.experimental.inference_service.guard.app.run_with_streaming_logs")
    @patch("areal.experimental.inference_service.guard.app.find_free_ports")
    def test_fork_module_path_tracks_child(
        self, mock_find, mock_run, mock_wait, client
    ):
        mock_find.return_value = [8001]
        mock_proc = _make_mock_process(pid=42)
        mock_run.return_value = mock_proc

        client.post(
            "/fork",
            json={"role": "test", "worker_index": 0, "command": "some.module"},
        )

        assert mock_proc in guard_module._forked_children
        assert ("test", 0) in guard_module._forked_children_map
        assert guard_module._forked_children_map[("test", 0)] is mock_proc

    @patch(
        "areal.experimental.inference_service.guard.app._wait_for_worker_ready", return_value=True
    )
    @patch("areal.experimental.inference_service.guard.app.run_with_streaming_logs")
    @patch("areal.experimental.inference_service.guard.app.find_free_ports")
    def test_fork_module_path_waits_for_ready(
        self, mock_find, mock_run, mock_wait, client
    ):
        """Module-path mode polls health before returning."""
        mock_find.return_value = [8001]
        mock_run.return_value = _make_mock_process()

        client.post(
            "/fork",
            json={"role": "test", "worker_index": 0, "command": "some.module"},
        )

        mock_wait.assert_called_once_with("10.0.0.1", 8001)

    @patch("areal.experimental.inference_service.guard.app.kill_process_tree")
    @patch(
        "areal.experimental.inference_service.guard.app._wait_for_worker_ready",
        return_value=False,
    )
    @patch("areal.experimental.inference_service.guard.app.run_with_streaming_logs")
    @patch("areal.experimental.inference_service.guard.app.find_free_ports")
    def test_fork_module_path_cleanup_on_ready_timeout(
        self, mock_find, mock_run, mock_wait, mock_kill, client
    ):
        """If readiness polling fails, child is killed and cleaned up."""
        mock_find.return_value = [8001]
        mock_proc = _make_mock_process(pid=99)
        mock_run.return_value = mock_proc

        resp = client.post(
            "/fork",
            json={"role": "test", "worker_index": 0, "command": "some.module"},
        )
        assert resp.status_code == 500
        assert "failed to start" in resp.get_json()["error"].lower()

        # Child cleaned up
        assert mock_proc not in guard_module._forked_children
        assert ("test", 0) not in guard_module._forked_children_map
        # Port freed
        assert 8001 not in guard_module._allocated_ports
        # kill_process_tree called
        mock_kill.assert_called_once_with(99, timeout=3, graceful=True)


# =============================================================================
# TestForkRawCommand
# =============================================================================


class TestForkRawCommand:
    """POST /fork with raw_cmd field — raw-command mode."""

    @patch("areal.experimental.inference_service.guard.app._wait_for_worker_ready")
    @patch("areal.experimental.inference_service.guard.app.run_with_streaming_logs")
    @patch("areal.experimental.inference_service.guard.app.find_free_ports")
    def test_fork_raw_cmd_passes_command_as_is(
        self, mock_find, mock_run, mock_wait, client
    ):
        mock_find.return_value = [8001]
        mock_proc = _make_mock_process(pid=55)
        mock_run.return_value = mock_proc

        raw = ["python", "-m", "sglang.launch_server", "--model", "test-model"]
        resp = client.post(
            "/fork",
            json={"role": "sglang", "worker_index": 0, "raw_cmd": raw},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "success"
        assert data["pid"] == 55

        # Command passed as-is — NO scheduler args injected
        call_args = mock_run.call_args
        cmd = call_args[0][0]
        assert cmd == raw
        assert "--experiment-name" not in cmd
        assert "--role" not in cmd

    @patch("areal.experimental.inference_service.guard.app._wait_for_worker_ready")
    @patch("areal.experimental.inference_service.guard.app.run_with_streaming_logs")
    @patch("areal.experimental.inference_service.guard.app.find_free_ports")
    def test_fork_raw_cmd_skips_readiness_polling(
        self, mock_find, mock_run, mock_wait, client
    ):
        """Raw-command mode returns immediately without polling health."""
        mock_find.return_value = [8001]
        mock_run.return_value = _make_mock_process()

        client.post(
            "/fork",
            json={
                "role": "sglang",
                "worker_index": 0,
                "raw_cmd": ["python", "-m", "sglang.launch_server"],
            },
        )

        mock_wait.assert_not_called()

    @patch("areal.experimental.inference_service.guard.app.run_with_streaming_logs")
    @patch("areal.experimental.inference_service.guard.app.find_free_ports")
    def test_fork_raw_cmd_allocates_port_but_not_injected(
        self, mock_find, mock_run, client
    ):
        """A port is allocated for tracking but NOT injected into raw_cmd."""
        mock_find.return_value = [9999]
        mock_run.return_value = _make_mock_process()

        resp = client.post(
            "/fork",
            json={
                "role": "sglang",
                "worker_index": 0,
                "raw_cmd": ["echo", "hello"],
            },
        )
        data = resp.get_json()
        assert data["port"] == 9999
        assert 9999 in guard_module._allocated_ports


# =============================================================================
# TestForkErrorHandling
# =============================================================================


class TestForkErrorHandling:
    """POST /fork error cases — missing fields and validation."""

    def test_fork_missing_role(self, client):
        resp = client.post("/fork", json={"worker_index": 0, "command": "some.module"})
        assert resp.status_code == 400
        assert "role" in resp.get_json()["error"].lower()

    def test_fork_missing_worker_index(self, client):
        resp = client.post("/fork", json={"role": "test", "command": "some.module"})
        assert resp.status_code == 400
        assert "worker_index" in resp.get_json()["error"].lower()

    def test_fork_missing_command_and_raw_cmd(self, client):
        resp = client.post("/fork", json={"role": "test", "worker_index": 0})
        assert resp.status_code == 400
        assert "command" in resp.get_json()["error"].lower()

    def test_fork_no_json_body(self, client):
        resp = client.post("/fork", data="not json", content_type="text/plain")
        assert resp.status_code == 400

    def test_fork_invalid_count(self, client):
        """Verify /alloc_ports validates count type."""
        resp = client.post("/alloc_ports", json={"count": 1.5})
        assert resp.status_code == 400


# =============================================================================
# TestKillForkedWorker
# =============================================================================


class TestKillForkedWorker:
    """POST /kill_forked_worker kills correct child."""

    @patch("areal.experimental.inference_service.guard.app.kill_process_tree")
    def test_kill_known_worker(self, mock_kill, client):
        """Kill a tracked worker — removes from tracking, calls kill_process_tree."""
        mock_proc = _make_mock_process(pid=123)
        guard_module._forked_children.append(mock_proc)
        guard_module._forked_children_map[("test", 0)] = mock_proc

        resp = client.post(
            "/kill_forked_worker", json={"role": "test", "worker_index": 0}
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "success"
        assert "123" in data["message"]

        # Removed from tracking
        assert mock_proc not in guard_module._forked_children
        assert ("test", 0) not in guard_module._forked_children_map

        # kill_process_tree called
        mock_kill.assert_called_once_with(123, timeout=3, graceful=True)

    def test_kill_unknown_worker_returns_404(self, client):
        """Killing a non-existent worker returns 404."""
        resp = client.post(
            "/kill_forked_worker", json={"role": "ghost", "worker_index": 99}
        )
        assert resp.status_code == 404
        assert "not found" in resp.get_json()["error"].lower()

    @patch("areal.experimental.inference_service.guard.app.kill_process_tree")
    def test_kill_already_exited_worker(self, mock_kill, client):
        """Worker that already exited (poll() != None) — no kill needed."""
        mock_proc = _make_mock_process(pid=456, running=False)
        guard_module._forked_children.append(mock_proc)
        guard_module._forked_children_map[("done", 0)] = mock_proc

        resp = client.post(
            "/kill_forked_worker", json={"role": "done", "worker_index": 0}
        )
        assert resp.status_code == 200
        # kill_process_tree NOT called because poll() returned non-None
        mock_kill.assert_not_called()

    def test_kill_missing_role(self, client):
        resp = client.post("/kill_forked_worker", json={"worker_index": 0})
        assert resp.status_code == 400
        assert "role" in resp.get_json()["error"].lower()

    def test_kill_missing_worker_index(self, client):
        resp = client.post("/kill_forked_worker", json={"role": "test"})
        assert resp.status_code == 400
        assert "worker_index" in resp.get_json()["error"].lower()

    @patch("areal.experimental.inference_service.guard.app.kill_process_tree")
    def test_kill_then_kill_again_returns_404(self, mock_kill, client):
        """Killing the same worker twice — second attempt gets 404."""
        mock_proc = _make_mock_process(pid=789)
        guard_module._forked_children.append(mock_proc)
        guard_module._forked_children_map[("test", 0)] = mock_proc

        resp1 = client.post(
            "/kill_forked_worker", json={"role": "test", "worker_index": 0}
        )
        assert resp1.status_code == 200

        resp2 = client.post(
            "/kill_forked_worker", json={"role": "test", "worker_index": 0}
        )
        assert resp2.status_code == 404


# =============================================================================
# TestCleanup
# =============================================================================


class TestCleanup:
    """cleanup_forked_children() kills all tracked children."""

    @patch("areal.experimental.inference_service.guard.app.kill_process_tree")
    def test_cleanup_kills_all_running_children(self, mock_kill):
        proc1 = _make_mock_process(pid=100)
        proc2 = _make_mock_process(pid=200)
        guard_module._forked_children = [proc1, proc2]
        guard_module._forked_children_map = {("a", 0): proc1, ("b", 0): proc2}

        cleanup_forked_children()

        assert mock_kill.call_count == 2
        pids_killed = {call.args[0] for call in mock_kill.call_args_list}
        assert pids_killed == {100, 200}

        # Tracking cleared
        assert guard_module._forked_children == []
        assert guard_module._forked_children_map == {}

    @patch("areal.experimental.inference_service.guard.app.kill_process_tree")
    def test_cleanup_skips_already_exited(self, mock_kill):
        """Already-exited children (poll() != None) are not killed."""
        running = _make_mock_process(pid=100, running=True)
        exited = _make_mock_process(pid=200, running=False)
        guard_module._forked_children = [running, exited]
        guard_module._forked_children_map = {("a", 0): running, ("b", 0): exited}

        cleanup_forked_children()

        # Only the running child gets killed
        mock_kill.assert_called_once_with(100, timeout=3, graceful=True)

        # Tracking still fully cleared
        assert guard_module._forked_children == []
        assert guard_module._forked_children_map == {}

    @patch("areal.experimental.inference_service.guard.app.kill_process_tree")
    def test_cleanup_no_children_is_noop(self, mock_kill):
        """Cleanup with no children does nothing."""
        cleanup_forked_children()
        mock_kill.assert_not_called()

    @patch("areal.experimental.inference_service.guard.app.kill_process_tree")
    def test_cleanup_tolerates_kill_exception(self, mock_kill):
        """If kill_process_tree raises, other children are still cleaned."""
        proc1 = _make_mock_process(pid=100)
        proc2 = _make_mock_process(pid=200)
        guard_module._forked_children = [proc1, proc2]
        guard_module._forked_children_map = {("a", 0): proc1, ("b", 0): proc2}

        mock_kill.side_effect = [OSError("boom"), None]

        cleanup_forked_children()

        # Both attempted despite first raising
        assert mock_kill.call_count == 2
        # Tracking cleared even on error
        assert guard_module._forked_children == []
        assert guard_module._forked_children_map == {}
