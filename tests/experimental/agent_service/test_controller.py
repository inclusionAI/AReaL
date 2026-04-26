# SPDX-License-Identifier: Apache-2.0

"""Unit tests for AgentController.

All Guard HTTP interactions are mocked — no real processes or servers.
Tests cover: initialize, destroy, scale_up, scale_down, and error handling.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from areal.api.cli_args import AgentConfig
from areal.experimental.agent_service.controller.controller import (
    AgentController,
    _RuntimeSession,
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
    return AgentConfig(
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
        ctrl = AgentController(config=config, scheduler=scheduler)
        assert ctrl.router_addr == ""
        assert ctrl.gateway_addr == ""
        assert ctrl.pairs == {}


class TestInitialize:
    @patch(f"{CTRL}.requests")
    def test_initialize_forks_router_pairs_gateway(self, mock_requests, config):
        """Initialize should create guards via scheduler, then fork services."""
        _setup_mock_requests(mock_requests)

        scheduler = _make_scheduler(("10.0.0.1", "8090"), ("10.0.0.2", "8090"))
        ctrl = AgentController(config=config, scheduler=scheduler)
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
        ctrl = AgentController(config=config, scheduler=scheduler)
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
        ctrl = AgentController(config=config, scheduler=scheduler)
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
        ctrl = AgentController(config=config, scheduler=scheduler)
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
        ctrl = AgentController(config=config, scheduler=scheduler)
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
        ctrl = AgentController(config=config, scheduler=scheduler)
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
        ctrl = AgentController(config=config, scheduler=scheduler)
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
        ctrl = AgentController(config=config, scheduler=scheduler)
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
        ctrl = AgentController(config=config, scheduler=scheduler)
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
        ctrl = AgentController(config=config, scheduler=scheduler)
        ctrl.initialize()
        assert ctrl._health_thread is None

        ctrl.destroy()


class TestRuntimeAPIs:
    @pytest.mark.asyncio
    async def test_start_session_grants_capacity_and_stores_session(self, config):
        scheduler = _make_scheduler(("10.0.0.1", "8090"))
        ctrl = AgentController(config=config, scheduler=scheduler)
        ctrl._grant_capacity = AsyncMock()
        ctrl._post_json = AsyncMock(
            return_value={"session_id": "inf-sess-1", "api_key": "sess-key"}
        )

        session = await ctrl.start_session(
            "task-1",
            inference_gateway_addr="http://inference",
            inference_admin_api_key="rollout-admin",
            inference_model="Qwen/Test",
        )

        assert session["session_id"].startswith("agent-sess-")
        assert session["inference_session_id"] == "inf-sess-1"
        assert session["api_key"] == "sess-key"
        ctrl._grant_capacity.assert_awaited_once_with(
            "http://inference", "rollout-admin"
        )
        ctrl._post_json.assert_awaited_once_with(
            "http://inference/rl/start_session",
            payload={"task_id": "task-1"},
            headers={"Authorization": "Bearer rollout-admin"},
        )

        stored = ctrl._resolve_session(session["session_id"])
        assert stored.inference_session_id == "inf-sess-1"
        assert stored.inference_session_api_key == "sess-key"
        assert stored.inference_model == "Qwen/Test"

    @pytest.mark.asyncio
    async def test_step_posts_async_gateway_request(self, config):
        scheduler = _make_scheduler(("10.0.0.1", "8090"))
        ctrl = AgentController(config=config, scheduler=scheduler)
        ctrl._gateway_addr = "http://agent-gateway"
        ctrl._post_json = AsyncMock(return_value={"status": "completed"})
        ctrl._sessions["agent-sess-1"] = _RuntimeSession(
            agent_session_id="agent-sess-1",
            inference_gateway_addr="http://inference",
            inference_admin_api_key="rollout-admin",
            inference_session_id="inf-sess-1",
            inference_session_api_key="sess-key",
            inference_model="Qwen/Test",
        )

        result = await ctrl.step(
            "hello",
            "agent-sess-1",
            metadata={"extra": "value"},
        )

        assert result == {"status": "completed"}
        ctrl._post_json.assert_awaited_once_with(
            "http://agent-gateway/v1/responses",
            payload={
                "input": [{"type": "message", "content": "hello"}],
                "model": "Qwen--Test",
                "user": "agent-sess-1",
                "metadata": {
                    "inference_base_url": "http://inference",
                    "inference_api_key": "sess-key",
                    "inference_model": "Qwen/Test",
                    "extra": "value",
                },
            },
            headers={"Authorization": "Bearer test-key"},
        )

    @pytest.mark.asyncio
    async def test_set_reward_uses_session_api_key(self, config):
        scheduler = _make_scheduler(("10.0.0.1", "8090"))
        ctrl = AgentController(config=config, scheduler=scheduler)
        ctrl._post_json = AsyncMock(return_value={"trajectory_id": 7})
        ctrl._sessions["agent-sess-1"] = _RuntimeSession(
            agent_session_id="agent-sess-1",
            inference_gateway_addr="http://inference",
            inference_admin_api_key="rollout-admin",
            inference_session_id="inf-sess-1",
            inference_session_api_key="sess-key",
        )

        result = await ctrl.set_reward(1.0, "agent-sess-1", interaction_id="resp-1")

        assert result == {"trajectory_id": 7}
        ctrl._post_json.assert_awaited_once_with(
            "http://inference/rl/set_reward",
            payload={"interaction_id": "resp-1", "reward": 1.0},
            headers={"Authorization": "Bearer sess-key"},
        )

    @pytest.mark.asyncio
    @patch(f"{CTRL}.deserialize_interactions")
    async def test_export_trajectory_deserializes_response(
        self, mock_deserialize, config
    ):
        scheduler = _make_scheduler(("10.0.0.1", "8090"))
        ctrl = AgentController(config=config, scheduler=scheduler)
        ctrl._post_json = AsyncMock(return_value={"interactions": {"k": "v"}})
        ctrl._sessions["agent-sess-1"] = _RuntimeSession(
            agent_session_id="agent-sess-1",
            inference_gateway_addr="http://inference",
            inference_admin_api_key="rollout-admin",
            inference_session_id="inf-sess-1",
            inference_session_api_key="sess-key",
        )
        mock_deserialize.return_value = {"interaction-1": MagicMock(reward=1.0)}

        result = await ctrl.export_trajectory(
            "agent-sess-1",
            trajectory_id=5,
            discount=0.9,
            style="individual",
        )

        assert "interaction-1" in result
        ctrl._post_json.assert_awaited_once_with(
            "http://inference/export_trajectories",
            payload={
                "session_id": "inf-sess-1",
                "trajectory_id": 5,
                "discount": 0.9,
                "style": "individual",
            },
            headers={"Authorization": "Bearer rollout-admin"},
        )
        mock_deserialize.assert_called_once_with({"k": "v"})
