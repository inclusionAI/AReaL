"""Unit tests for AgentController."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from areal.api.cli_args import AgentServiceSpec
from areal.experimental.agent_service.agent_controller import (
    _GATEWAY_ROLE,
    _WORKER_ROLE,
    AgentController,
)
from areal.experimental.agent_service.config import GatewayConfig


def _make_mock_scheduler(
    gateway_ip: str = "10.0.0.1",
    gateway_port: int = 8300,
    worker_ip: str = "10.0.0.2",
    worker_port: int = 8301,
):
    """Create a mock Scheduler that returns fake Worker objects."""
    scheduler = MagicMock()

    # Mock gateway worker
    gateway_worker = MagicMock()
    gateway_worker.ip = gateway_ip
    gateway_worker.worker_ports = [gateway_port]

    # Mock agent worker
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
    scheduler.experiment_name = "test-experiment"
    scheduler.trial_name = "test-trial"

    return scheduler


class TestAgentControllerInterface:
    """Tests for AgentController public interface."""

    def test_has_required_methods(self):
        """AgentController should have all required public methods."""
        assert hasattr(AgentController, "start_gateway")
        assert hasattr(AgentController, "start_workers")
        assert hasattr(AgentController, "stop")
        assert hasattr(AgentController, "scale_workers")
        assert hasattr(AgentController, "get_gateway_addr")
        assert hasattr(AgentController, "get_worker_count")
        assert hasattr(AgentController, "gateway_started")

    def test_init_sets_defaults(self):
        """AgentController.__init__ should set correct defaults."""
        config = GatewayConfig()
        scheduler = _make_mock_scheduler()
        controller = AgentController(config=config, scheduler=scheduler)

        assert controller._gateway_started is False
        assert controller._gateway_addr is None
        assert controller._agent_workers == []

    def test_gateway_started_property_initially_false(self):
        """gateway_started property should be False before start_gateway()."""
        config = GatewayConfig()
        scheduler = _make_mock_scheduler()
        controller = AgentController(config=config, scheduler=scheduler)

        assert controller.gateway_started is False

    def test_get_gateway_addr_raises_before_start(self):
        """get_gateway_addr() should raise RuntimeError before start_gateway()."""
        config = GatewayConfig()
        scheduler = _make_mock_scheduler()
        controller = AgentController(config=config, scheduler=scheduler)

        with pytest.raises(RuntimeError, match="not started"):
            controller.get_gateway_addr()

    def test_get_worker_count_initially_zero(self):
        """get_worker_count() should return 0 before start_workers()."""
        config = GatewayConfig()
        scheduler = _make_mock_scheduler()
        controller = AgentController(config=config, scheduler=scheduler)

        assert controller.get_worker_count() == 0


class TestAgentControllerStartGateway:
    """Tests for start_gateway() method."""

    def test_start_gateway_creates_correct_job(self):
        """start_gateway() should create a Job with correct SchedulingSpec."""
        config = GatewayConfig(queue_size=500)
        scheduler = _make_mock_scheduler()
        controller = AgentController(config=config, scheduler=scheduler)

        controller.start_gateway()

        # Verify create_workers was called
        scheduler.create_workers.assert_called_once()
        job = scheduler.create_workers.call_args.kwargs["job"]

        # Verify job properties
        assert job.role == _GATEWAY_ROLE
        assert job.replicas == 1
        assert len(job.tasks) == 1

        spec = job.tasks[0]
        assert "areal.experimental.agent_service.gateway" in spec.cmd
        assert "--queue-size 500" in spec.cmd
        assert "--queue-size 500" in spec.cmd
        assert spec.gpu == 0  # CPU-only

    def test_start_gateway_returns_correct_address(self):
        """start_gateway() should return the Gateway's HTTP address."""
        config = GatewayConfig()
        scheduler = _make_mock_scheduler(gateway_ip="192.168.1.1", gateway_port=9000)
        controller = AgentController(config=config, scheduler=scheduler)

        addr = controller.start_gateway()

        assert addr == "http://192.168.1.1:9000"
        assert controller._gateway_addr == "http://192.168.1.1:9000"
        assert controller.gateway_started is True

    def test_start_gateway_idempotent(self):
        """Calling start_gateway() twice should return same address without re-creating."""
        config = GatewayConfig()
        scheduler = _make_mock_scheduler()
        controller = AgentController(config=config, scheduler=scheduler)

        addr1 = controller.start_gateway()
        addr2 = controller.start_gateway()

        assert addr1 == addr2
        # create_workers should only be called once
        assert scheduler.create_workers.call_count == 1


class TestAgentControllerStartWorkers:
    """Tests for start_workers() method."""

    def test_start_workers_raises_without_gateway(self):
        """start_workers() should raise RuntimeError if gateway not started."""
        config = GatewayConfig()
        scheduler = _make_mock_scheduler()
        controller = AgentController(config=config, scheduler=scheduler)

        with pytest.raises(RuntimeError, match="Gateway must be started"):
            controller.start_workers(
                AgentServiceSpec(
                    agent_import_path="mymodule.MyAgent",
                    agent_reuse=False,
                    workers=2,
                )
            )

    def test_start_workers_creates_correct_job(self):
        """start_workers() should create a Job with correct SchedulingSpec for each worker."""
        config = GatewayConfig()
        scheduler = _make_mock_scheduler()
        controller = AgentController(config=config, scheduler=scheduler)

        # Start gateway first
        controller.start_gateway()
        scheduler.create_workers.reset_mock()

        # Start 3 workers
        controller.start_workers(
            AgentServiceSpec(
                agent_import_path="mymodule.MyAgent",
                agent_reuse=True,
                agent_init_kwargs={"model": "gpt-4"},
                workers=3,
            )
        )

        scheduler.create_workers.assert_called_once()
        job = scheduler.create_workers.call_args.kwargs["job"]

        assert job.role == _WORKER_ROLE
        assert job.replicas == 3
        assert len(job.tasks) == 1

        spec = job.tasks[0]
        assert "areal.experimental.agent_service.worker_server" in spec.cmd
        assert spec.gpu == 0  # CPU-only
        assert "AREAL_AGENT_GATEWAY_ADDR" in spec.env_vars
        assert spec.env_vars["AREAL_AGENT_IMPORT_PATH_INTERNAL"] == "mymodule.MyAgent"
        assert spec.env_vars["AREAL_AGENT_REUSE_INTERNAL"] == "true"
        assert "gpt-4" in spec.env_vars["AREAL_AGENT_INIT_KWARGS_INTERNAL"]

    def test_start_workers_passes_gateway_addr_to_workers(self):
        """start_workers() should pass Gateway address to each worker via env vars."""
        config = GatewayConfig()
        scheduler = _make_mock_scheduler(gateway_ip="10.0.0.1", gateway_port=8300)
        controller = AgentController(config=config, scheduler=scheduler)

        controller.start_gateway()
        scheduler.create_workers.reset_mock()

        controller.start_workers(
            AgentServiceSpec(
                agent_import_path="test.Agent",
                agent_reuse=False,
                workers=2,
            )
        )

        job = scheduler.create_workers.call_args.kwargs["job"]
        spec = job.tasks[0]
        assert spec.env_vars["AREAL_AGENT_GATEWAY_ADDR"] == "http://10.0.0.1:8300"


class TestAgentControllerStop:
    """Tests for stop() method."""

    def test_stop_resets_state(self):
        """stop() should reset all state."""
        config = GatewayConfig()
        scheduler = _make_mock_scheduler()
        controller = AgentController(config=config, scheduler=scheduler)

        controller.start_gateway()
        controller.stop()

        assert controller._gateway_started is False
        assert controller._gateway_addr is None
        assert controller._agent_workers == []


class TestAgentControllerScaleWorkers:
    """Tests for scale_workers() method."""

    def test_scale_up_adds_workers(self):
        """scale_workers() with target > current should add workers."""
        config = GatewayConfig()
        scheduler = _make_mock_scheduler()
        controller = AgentController(config=config, scheduler=scheduler)

        controller.start_gateway()
        # Start 1 worker (mock returns 1 worker)
        controller.start_workers(
            AgentServiceSpec(
                agent_import_path="test.Agent",
                agent_reuse=False,
                workers=1,
            )
        )
        scheduler.create_workers.reset_mock()

        # Scale up to 3 (add 2 more)
        controller.scale_workers(
            target=3,
            spec=AgentServiceSpec(
                agent_import_path="test.Agent",
                agent_reuse=False,
                workers=3,
            ),
        )

        scheduler.create_workers.assert_called_once()
        job = scheduler.create_workers.call_args.kwargs["job"]
        assert job.replicas == 2  # delta = 3 - 1

    def test_scale_same_count_is_noop(self):
        """scale_workers() with target == current should not call create_workers."""
        config = GatewayConfig()
        scheduler = _make_mock_scheduler()
        controller = AgentController(config=config, scheduler=scheduler)

        controller.start_gateway()
        # Start 1 worker (mock returns 1 worker)
        controller.start_workers(
            AgentServiceSpec(
                agent_import_path="test.Agent",
                agent_reuse=False,
                workers=1,
            )
        )
        scheduler.create_workers.reset_mock()

        # Scale to same count (1) — should be a no-op
        controller.scale_workers(target=1)

        scheduler.create_workers.assert_not_called()
