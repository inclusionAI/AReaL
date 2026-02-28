"""Unit tests for AgentController unified scheduler mode.

Tests cover the unified scheduler-based controller (replaces the old
standalone mode). All tests use a mocked Scheduler.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from areal.api.cli_args import AgentServiceSpec
from areal.experimental.agent_service.agent_controller import (
    _GATEWAY_ROLE,
    _WORKER_ROLE,
    AgentController,
)
from areal.experimental.agent_service.config import GatewayConfig


def _make_mock_scheduler(
    gateway_ip="127.0.0.1",
    gateway_port=8300,
    worker_ip="127.0.0.1",
    worker_port=8301,
):
    """Create a mock Scheduler for testing."""
    scheduler = MagicMock()
    gateway_worker = MagicMock()
    gateway_worker.ip = gateway_ip
    gateway_worker.worker_ports = [gateway_port]
    agent_worker = MagicMock()
    agent_worker.ip = worker_ip
    agent_worker.worker_ports = [worker_port]

    def get_workers(role, timeout=None):
        if role == _GATEWAY_ROLE:
            return [gateway_worker]
        elif role == _WORKER_ROLE:
            return [agent_worker]
        return []

    scheduler.get_workers.side_effect = get_workers
    scheduler.create_workers.return_value = ["worker-id-1"]
    return scheduler


class TestSchedulerRequired:
    """Tests that scheduler is required (not optional)."""

    def test_scheduler_none_raises_type_error(self):
        """AgentController(config=..., scheduler=None) should raise TypeError."""
        config = GatewayConfig()
        with pytest.raises(TypeError, match="scheduler must be a Scheduler instance"):
            AgentController(config=config, scheduler=None)

    def test_scheduler_required_in_init(self):
        """AgentController requires a Scheduler instance."""
        config = GatewayConfig()
        scheduler = _make_mock_scheduler()
        controller = AgentController(config=config, scheduler=scheduler)
        assert controller._scheduler is scheduler


class TestUnifiedControllerStart:
    """Tests for unified start() method."""

    def test_start_calls_start_gateway_and_start_workers(self):
        """start() should call start_gateway() then start_workers()."""
        config = GatewayConfig()
        scheduler = _make_mock_scheduler()
        controller = AgentController(config=config, scheduler=scheduler)

        spec = AgentServiceSpec(
            agent_import_path="mymodule.MyAgent",
            agent_reuse=False,
            agent_init_kwargs=None,
            workers=1,
        )
        with (
            patch.object(
                controller, "start_gateway", return_value="http://127.0.0.1:8300"
            ) as mock_gw,
            patch.object(
                controller,
                "start_workers",
                return_value=["http://127.0.0.1:8301"],
            ) as mock_wk,
        ):
            controller._gateway_started = True
            controller._gateway_addr = "http://127.0.0.1:8300"
            controller.start(spec)

        mock_gw.assert_called_once()
        mock_wk.assert_called_once_with(spec)

    def test_start_returns_gateway_addr(self):
        """start() should return the gateway address."""
        config = GatewayConfig()
        scheduler = _make_mock_scheduler()
        controller = AgentController(config=config, scheduler=scheduler)

        spec = AgentServiceSpec(
            agent_import_path="test.Agent",
            agent_reuse=False,
            workers=1,
        )
        with patch("time.sleep"):
            addr = controller.start(spec)

        assert addr == "http://127.0.0.1:8300"
        assert controller.gateway_started is True


class TestStopUnified:
    """Tests for stop() in unified mode."""

    def test_stop_calls_delete_workers_for_both_roles(self):
        """stop() should call delete_workers for agent_worker then agent_gateway."""
        config = GatewayConfig()
        scheduler = _make_mock_scheduler()
        controller = AgentController(config=config, scheduler=scheduler)

        controller.stop()

        calls = [call[0][0] for call in scheduler.delete_workers.call_args_list]
        assert "agent_worker" in calls
        assert "agent_gateway" in calls

    def test_stop_is_idempotent(self):
        """Calling stop() twice should not raise."""
        config = GatewayConfig()
        scheduler = _make_mock_scheduler()
        controller = AgentController(config=config, scheduler=scheduler)

        controller.stop()
        controller.stop()  # Should not raise


class TestControllerModeCLI:
    """Tests for CLI argument parsing (controller-only entry point)."""

    def test_parse_args_no_mode_flag(self):
        """_parse_args should NOT have a --mode argument."""
        from areal.experimental.agent_service.__main__ import _parse_args

        with patch("sys.argv", ["agent_service"]):
            args = _parse_args()
            assert not hasattr(args, "mode")

    def test_scheduler_type_default_is_local(self):
        """--scheduler-type should default to 'local'."""
        from areal.experimental.agent_service.__main__ import _parse_args

        with patch("sys.argv", ["agent_service"]):
            args = _parse_args()
            assert args.scheduler_type == "local"
