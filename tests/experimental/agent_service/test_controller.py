# SPDX-License-Identifier: Apache-2.0

"""Unit tests for AgentServiceController.

All Guard HTTP interactions are mocked — no real processes or servers.
Tests cover: initialize, destroy, scale_up, scale_down, and error handling.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from areal.experimental.agent_service.controller.config import (
    AgentServiceControllerConfig,
)
from areal.experimental.agent_service.controller.controller import (
    AgentServiceController,
)

CTRL = "areal.experimental.agent_service.controller.controller"


@dataclass
class _FakeWorker:
    id: str
    ip: str
    worker_ports: list[str]
    engine_ports: list[str]


def _make_scheduler(*guard_specs: tuple[str, str]) -> MagicMock:
    """Return a mock Scheduler whose get_workers returns _FakeWorkers."""
    workers = [
        _FakeWorker(id=f"agent-guard/{i}", ip=ip, worker_ports=[port], engine_ports=[])
        for i, (ip, port) in enumerate(guard_specs)
    ]
    scheduler = MagicMock()
    scheduler.get_workers.return_value = workers
    return scheduler


def _mock_alloc_ports_response(host: str, ports: list[int]) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"status": "success", "host": host, "ports": ports}
    resp.raise_for_status = MagicMock()
    return resp


def _mock_fork_response(host: str, pid: int) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"status": "success", "host": host, "pid": pid}
    resp.raise_for_status = MagicMock()
    return resp


def _mock_kill_response() -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"status": "success"}
    resp.text = '{"status": "success"}'
    return resp


def _mock_register_response() -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    return resp


def _mock_health_response(active_sessions: int = 0) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"status": "ok", "active_sessions": active_sessions}
    return resp


@pytest.fixture()
def config():
    return AgentServiceControllerConfig(
        agent_cls_path="my.Agent",
        admin_api_key="test-key",
        num_pairs=2,
        setup_timeout=1.0,
        health_poll_interval=0,
    )


def _setup_mock_requests(mock_requests, port_start=9001):
    port_counter = iter(range(port_start, port_start + 100))

    def mock_post(url, **kwargs):
        if "/alloc_ports" in url:
            return _mock_alloc_ports_response("10.0.0.1", [next(port_counter)])
        if "/fork" in url:
            return _mock_fork_response("10.0.0.1", 100)
        if "/register" in url:
            return _mock_register_response()
        if "/kill_forked_worker" in url:
            return _mock_kill_response()
        if "/unregister" in url:
            return _mock_register_response()
        return MagicMock(status_code=404)

    mock_requests.post = mock_post
    mock_requests.get = lambda url, **kw: _mock_health_response()
    mock_requests.RequestException = Exception


class TestConstruction:
    def test_construction(self, config):
        scheduler = _make_scheduler(("10.0.0.1", "8090"))
        ctrl = AgentServiceController(config=config, scheduler=scheduler)
        assert ctrl.router_addr == ""
        assert ctrl.gateway_addr == ""
        assert ctrl.pairs == {}


class TestInitialize:
    @patch(f"{CTRL}.requests")
    def test_initialize_forks_router_pairs_gateway(self, mock_requests, config):
        """Initialize should create guards via scheduler, then fork services."""
        _setup_mock_requests(mock_requests)

        scheduler = _make_scheduler(("10.0.0.1", "8090"), ("10.0.0.2", "8090"))
        ctrl = AgentServiceController(config=config, scheduler=scheduler)
        ctrl.initialize()

        scheduler.create_workers.assert_called_once()
        scheduler.get_workers.assert_called_once()

        assert "http://" in ctrl.router_addr
        assert "http://" in ctrl.gateway_addr
        assert len(ctrl.pairs) == 2
        assert len(ctrl._forked_services) == 6


class TestScaleUp:
    @patch(f"{CTRL}.requests")
    def test_scale_up_adds_pairs(self, mock_requests, config):
        config.num_pairs = 0
        _setup_mock_requests(mock_requests)

        scheduler = _make_scheduler(("10.0.0.1", "8090"))
        ctrl = AgentServiceController(config=config, scheduler=scheduler)
        ctrl.initialize()
        assert len(ctrl.pairs) == 0

        created = ctrl.scale_up(3)
        assert created == [0, 1, 2]
        assert len(ctrl.pairs) == 3

    @patch(f"{CTRL}.requests")
    def test_scale_up_round_robins_guards(self, mock_requests, config):
        config.num_pairs = 0
        guards_called: list[str] = []

        def mock_post(url, **kwargs):
            if "/alloc_ports" in url:
                guards_called.append(url.split("/alloc_ports")[0])
                return _mock_alloc_ports_response("10.0.0.1", [9001])
            if "/fork" in url:
                return _mock_fork_response("10.0.0.1", 100)
            if "/register" in url:
                return _mock_register_response()
            if "/kill_forked_worker" in url:
                return _mock_kill_response()
            return MagicMock(status_code=404)

        mock_requests.post = mock_post
        mock_requests.get = lambda url, **kw: _mock_health_response()
        mock_requests.RequestException = Exception

        scheduler = _make_scheduler(("g0", "8090"), ("g1", "8091"))
        ctrl = AgentServiceController(config=config, scheduler=scheduler)
        ctrl.initialize()
        guards_called.clear()

        ctrl.scale_up(4)

        g0_calls = [g for g in guards_called if "g0" in g]
        g1_calls = [g for g in guards_called if "g1" in g]
        assert len(g0_calls) == 4
        assert len(g1_calls) == 4


class TestScaleDown:
    @patch(f"{CTRL}.requests")
    def test_scale_down_removes_newest_first(self, mock_requests, config):
        config.num_pairs = 3
        _setup_mock_requests(mock_requests)

        scheduler = _make_scheduler(("10.0.0.1", "8090"))
        ctrl = AgentServiceController(config=config, scheduler=scheduler)
        ctrl.initialize()
        assert len(ctrl.pairs) == 3

        removed = ctrl.scale_down(2)
        assert set(removed) == {2, 1}
        assert len(ctrl.pairs) == 1
        assert 0 in ctrl.pairs


class TestDestroy:
    @patch(f"{CTRL}.requests")
    def test_destroy_clears_everything(self, mock_requests, config):
        config.num_pairs = 1
        _setup_mock_requests(mock_requests)

        scheduler = _make_scheduler(("10.0.0.1", "8090"))
        ctrl = AgentServiceController(config=config, scheduler=scheduler)
        ctrl.initialize()
        assert len(ctrl._forked_services) > 0

        ctrl.destroy()
        assert ctrl.router_addr == ""
        assert ctrl.gateway_addr == ""
        assert ctrl.pairs == {}
        assert ctrl._forked_services == []
        scheduler.delete_workers.assert_called()

    @patch(f"{CTRL}.requests")
    def test_destroy_tolerates_kill_errors(self, mock_requests, config):
        config.num_pairs = 0
        kill_count = 0

        def mock_post(url, **kwargs):
            nonlocal kill_count
            if "/alloc_ports" in url:
                return _mock_alloc_ports_response("10.0.0.1", [9001])
            if "/fork" in url:
                return _mock_fork_response("10.0.0.1", 100)
            if "/kill_forked_worker" in url:
                kill_count += 1
                raise ConnectionError("Guard down")
            return MagicMock(status_code=404)

        mock_requests.post = mock_post
        mock_requests.get = lambda url, **kw: _mock_health_response()
        mock_requests.RequestException = Exception

        scheduler = _make_scheduler(("10.0.0.1", "8090"))
        ctrl = AgentServiceController(config=config, scheduler=scheduler)
        ctrl.initialize()

        ctrl.destroy()
        assert kill_count == 2
        assert ctrl._forked_services == []


class TestDrain:
    @patch(f"{CTRL}.requests")
    def test_scale_down_waits_for_drain(self, mock_requests, config):
        """scale_down should poll DataProxy health until active_sessions reaches 0."""
        config.num_pairs = 1
        config.drain_timeout = 5.0

        _setup_mock_requests(mock_requests)
        health_call_count = 0

        def mock_get(url, **kwargs):
            nonlocal health_call_count
            health_call_count += 1
            if "/health" in url and health_call_count <= 5:
                return _mock_health_response(active_sessions=2)
            return _mock_health_response(active_sessions=0)

        mock_requests.get = mock_get

        scheduler = _make_scheduler(("10.0.0.1", "8090"))
        ctrl = AgentServiceController(config=config, scheduler=scheduler)
        ctrl.initialize()

        health_call_count = 0
        with patch(f"{CTRL}.time") as mock_time:
            mock_time.monotonic = time.monotonic
            mock_time.sleep = MagicMock()
            ctrl.scale_down(1)

        assert len(ctrl.pairs) == 0
        assert health_call_count > 1

    @patch(f"{CTRL}.requests")
    def test_drain_skipped_when_timeout_zero(self, mock_requests, config):
        config.num_pairs = 1
        config.drain_timeout = 0
        _setup_mock_requests(mock_requests)
        get_count = 0

        def counting_get(url, **kwargs):
            nonlocal get_count
            get_count += 1
            return _mock_health_response(active_sessions=5)

        mock_requests.get = counting_get

        scheduler = _make_scheduler(("10.0.0.1", "8090"))
        ctrl = AgentServiceController(config=config, scheduler=scheduler)
        ctrl.initialize()

        pre_get_count = get_count
        ctrl.scale_down(1)
        drain_gets = get_count - pre_get_count
        assert drain_gets == 0


class TestHealthMonitor:
    @patch(f"{CTRL}.requests")
    def test_health_monitor_starts_and_stops(self, mock_requests, config):
        config.num_pairs = 0
        config.health_poll_interval = 0.1
        _setup_mock_requests(mock_requests)

        scheduler = _make_scheduler(("10.0.0.1", "8090"))
        ctrl = AgentServiceController(config=config, scheduler=scheduler)
        ctrl.initialize()
        assert ctrl._health_thread is not None
        assert ctrl._health_thread.is_alive()

        ctrl.destroy()
        assert ctrl._health_thread is None

    @patch(f"{CTRL}.requests")
    def test_health_monitor_disabled_when_interval_zero(self, mock_requests, config):
        config.num_pairs = 0
        config.health_poll_interval = 0
        _setup_mock_requests(mock_requests)

        scheduler = _make_scheduler(("10.0.0.1", "8090"))
        ctrl = AgentServiceController(config=config, scheduler=scheduler)
        ctrl.initialize()
        assert ctrl._health_thread is None

        ctrl.destroy()


@pytest.fixture()
def dc_config():
    return AgentServiceControllerConfig(
        agent_cls_path="my.Agent",
        admin_api_key="test-key",
        num_pairs=0,
        setup_timeout=1.0,
        health_poll_interval=0,
        inference_addr="http://inf-gw:8080",
        inference_model="Qwen/Qwen3-0.6B",
        inference_api_key="inf-admin-key",
    )


def _make_dc_controller(mock_requests, dc_config) -> AgentServiceController:
    _setup_mock_requests(mock_requests)
    scheduler = _make_scheduler(("10.0.0.1", "8090"))
    ctrl = AgentServiceController(config=dc_config, scheduler=scheduler)
    ctrl.initialize()
    return ctrl


class TestNewSession:
    @patch(f"{CTRL}.requests")
    def test_new_session_calls_inference_start_session(self, mock_requests, dc_config):
        post_calls: list[tuple[str, dict]] = []

        ctrl = _make_dc_controller(mock_requests, dc_config)

        def mock_post(url, **kwargs):
            if "/rl/start_session" in url:
                post_calls.append((url, kwargs))
                resp = MagicMock()
                resp.raise_for_status = MagicMock()
                resp.json.return_value = {
                    "session_id": "inf-sess-1",
                    "api_key": "sk-sess-abc123",
                }
                return resp
            return MagicMock(status_code=404)

        mock_requests.post = mock_post

        result = ctrl.new_session(task_id="my-task")

        assert result["inference_session_id"] == "inf-sess-1"
        assert result["inference_api_key"] == "sk-sess-abc123"
        assert result["session_id"].startswith("agent-sess-")

        assert len(post_calls) == 1
        call_url, call_kwargs = post_calls[0]
        assert "/rl/start_session" in call_url
        assert call_kwargs["json"]["task_id"] == "my-task"
        assert "inf-admin-key" in call_kwargs["headers"]["Authorization"]

    @patch(f"{CTRL}.requests")
    def test_new_session_stores_session_and_sets_latest(self, mock_requests, dc_config):
        ctrl = _make_dc_controller(mock_requests, dc_config)

        def mock_post(url, **kwargs):
            if "/rl/start_session" in url:
                resp = MagicMock()
                resp.raise_for_status = MagicMock()
                resp.json.return_value = {
                    "session_id": "inf-sess-1",
                    "api_key": "sk-sess-abc123",
                }
                return resp
            return MagicMock(status_code=404)

        mock_requests.post = mock_post

        result = ctrl.new_session()
        sid = result["session_id"]

        assert ctrl._sessions[sid] == result

    @patch(f"{CTRL}.requests")
    def test_new_session_defaults_task_id_to_session_id(self, mock_requests, dc_config):
        captured_task_id: list[str] = []

        ctrl = _make_dc_controller(mock_requests, dc_config)

        def mock_post(url, **kwargs):
            if "/rl/start_session" in url:
                captured_task_id.append(kwargs["json"]["task_id"])
                resp = MagicMock()
                resp.raise_for_status = MagicMock()
                resp.json.return_value = {
                    "session_id": "inf-sess-1",
                    "api_key": "sk-sess-abc",
                }
                return resp
            return MagicMock(status_code=404)

        mock_requests.post = mock_post

        result = ctrl.new_session()
        assert captured_task_id[0] == result["session_id"]

    def test_new_session_raises_without_inference_addr(self, config):
        scheduler = _make_scheduler(("10.0.0.1", "8090"))
        ctrl = AgentServiceController(config=config, scheduler=scheduler)

        with pytest.raises(RuntimeError, match="inference_addr must be set"):
            ctrl.new_session()


def _mock_start_session_post(url, **kwargs):
    if "/rl/start_session" in url:
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {
            "session_id": "inf-sess-1",
            "api_key": "sk-sess-abc",
        }
        return resp
    return MagicMock(status_code=404)


class TestStep:
    @patch(f"{CTRL}.requests")
    def test_step_sends_string_input_to_gateway(self, mock_requests, dc_config):
        v1_calls: list[tuple[str, dict]] = []

        ctrl = _make_dc_controller(mock_requests, dc_config)
        mock_requests.post = _mock_start_session_post
        session = ctrl.new_session()

        def mock_post(url, **kwargs):
            if "/v1/responses" in url:
                v1_calls.append((url, kwargs))
                resp = MagicMock()
                resp.raise_for_status = MagicMock()
                resp.json.return_value = {
                    "id": "resp-1",
                    "status": "completed",
                    "output": [],
                }
                return resp
            return MagicMock(status_code=404)

        mock_requests.post = mock_post

        result = ctrl.step("Hello agent", session["session_id"])

        assert result["status"] == "completed"
        assert len(v1_calls) == 1
        _, call_kwargs = v1_calls[0]
        body = call_kwargs["json"]
        assert body["input"] == [{"type": "message", "content": "Hello agent"}]
        assert body["user"] == session["session_id"]
        assert body["model"] == "Qwen--Qwen3-0.6B"
        assert body["metadata"]["inference_base_url"] == "http://inf-gw:8080"
        assert body["metadata"]["inference_model"] == "Qwen/Qwen3-0.6B"
        assert body["metadata"]["inference_api_key"] == "sk-sess-abc"

    @patch(f"{CTRL}.requests")
    def test_step_sends_list_input_unchanged(self, mock_requests, dc_config):
        v1_calls: list[tuple[str, dict]] = []

        ctrl = _make_dc_controller(mock_requests, dc_config)
        mock_requests.post = _mock_start_session_post
        session = ctrl.new_session()

        def mock_post(url, **kwargs):
            if "/v1/responses" in url:
                v1_calls.append((url, kwargs))
                resp = MagicMock()
                resp.raise_for_status = MagicMock()
                resp.json.return_value = {"id": "resp-1", "status": "completed"}
                return resp
            return MagicMock(status_code=404)

        mock_requests.post = mock_post

        custom_input = [
            {"type": "message", "content": "first"},
            {"type": "function_call_output", "output": "42"},
        ]
        ctrl.step(custom_input, session["session_id"])

        _, call_kwargs = v1_calls[0]
        assert call_kwargs["json"]["input"] == custom_input

    @patch(f"{CTRL}.requests")
    def test_step_requires_explicit_session_id(self, mock_requests, dc_config):
        v1_calls: list[tuple[str, dict]] = []

        ctrl = _make_dc_controller(mock_requests, dc_config)
        mock_requests.post = _mock_start_session_post
        session = ctrl.new_session()

        def mock_post(url, **kwargs):
            if "/v1/responses" in url:
                v1_calls.append((url, kwargs))
                resp = MagicMock()
                resp.raise_for_status = MagicMock()
                resp.json.return_value = {"id": "r", "status": "completed"}
                return resp
            return MagicMock(status_code=404)

        mock_requests.post = mock_post

        ctrl.step("hi", session["session_id"])

        _, call_kwargs = v1_calls[0]
        assert call_kwargs["json"]["user"] == session["session_id"]


class TestSetReward:
    @patch(f"{CTRL}.requests")
    def test_set_reward_calls_inference_gateway(self, mock_requests, dc_config):
        reward_calls: list[tuple[str, dict]] = []

        ctrl = _make_dc_controller(mock_requests, dc_config)
        mock_requests.post = _mock_start_session_post
        session = ctrl.new_session()

        def mock_post(url, **kwargs):
            if "/rl/set_reward" in url:
                reward_calls.append((url, kwargs))
                resp = MagicMock()
                resp.raise_for_status = MagicMock()
                resp.json.return_value = {"trajectory_id": 42}
                return resp
            return MagicMock(status_code=404)

        mock_requests.post = mock_post

        result = ctrl.set_reward(1.0, session["session_id"])

        assert result["trajectory_id"] == 42
        assert len(reward_calls) == 1
        call_url, call_kwargs = reward_calls[0]
        assert "inf-gw:8080/rl/set_reward" in call_url
        assert call_kwargs["json"]["reward"] == 1.0
        assert call_kwargs["json"]["interaction_id"] is None
        assert "sk-sess-abc" in call_kwargs["headers"]["Authorization"]

    @patch(f"{CTRL}.requests")
    def test_set_reward_requires_explicit_session_id(self, mock_requests, dc_config):
        reward_calls: list[tuple[str, dict]] = []

        ctrl = _make_dc_controller(mock_requests, dc_config)
        mock_requests.post = _mock_start_session_post
        session = ctrl.new_session()

        def mock_post(url, **kwargs):
            if "/rl/set_reward" in url:
                reward_calls.append((url, kwargs))
                resp = MagicMock()
                resp.raise_for_status = MagicMock()
                resp.json.return_value = {"trajectory_id": 1}
                return resp
            return MagicMock(status_code=404)

        mock_requests.post = mock_post

        ctrl.set_reward(0.5, session["session_id"])

        _, call_kwargs = reward_calls[0]
        assert "sk-sess-abc" in call_kwargs["headers"]["Authorization"]


class TestResolveSession:
    def test_resolve_raises_on_unknown_id(self, dc_config):
        scheduler = _make_scheduler(("10.0.0.1", "8090"))
        ctrl = AgentServiceController(config=dc_config, scheduler=scheduler)

        with pytest.raises(KeyError, match="Unknown session_id"):
            ctrl._resolve_session("nonexistent")
