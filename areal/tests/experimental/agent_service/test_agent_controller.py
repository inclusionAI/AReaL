"""Unit tests for AgentController."""

from __future__ import annotations

import pytest

from areal.api.cli_args import AgentServiceSpec
from areal.experimental.agent_service.agent_controller import (
    _GATEWAY_ROLE,
    _WORKER_ROLE,
    AgentController,
)
from areal.experimental.agent_service.config import GatewayConfig

# Import shared mock helper from conftest.py
from areal.tests.experimental.agent_service.conftest import make_mock_scheduler


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
        scheduler = make_mock_scheduler()
        controller = AgentController(config=config, scheduler=scheduler)

        assert controller._gateway_started is False
        assert controller._gateway_addr is None
        assert controller._agent_workers == []

    def test_gateway_started_property_initially_false(self):
        """gateway_started property should be False before start_gateway()."""
        config = GatewayConfig()
        scheduler = make_mock_scheduler()
        controller = AgentController(config=config, scheduler=scheduler)

        assert controller.gateway_started is False

    def test_get_gateway_addr_raises_before_start(self):
        """get_gateway_addr() should raise RuntimeError before start_gateway()."""
        config = GatewayConfig()
        scheduler = make_mock_scheduler()
        controller = AgentController(config=config, scheduler=scheduler)

        with pytest.raises(RuntimeError, match="not started"):
            controller.get_gateway_addr()

    def test_get_worker_count_initially_zero(self):
        """get_worker_count() should return 0 before start_workers()."""
        config = GatewayConfig()
        scheduler = make_mock_scheduler()
        controller = AgentController(config=config, scheduler=scheduler)

        assert controller.get_worker_count() == 0


class TestAgentControllerStartGateway:
    """Tests for start_gateway() method."""

    def test_start_gateway_creates_correct_job(self):
        """start_gateway() should create a Job with correct SchedulingSpec."""
        config = GatewayConfig()
        scheduler = make_mock_scheduler()
        controller = AgentController(config=config, scheduler=scheduler)

        controller.start_router()
        controller.start_gateway()

        # Verify create_workers was called
        # Second call should be for gateway (first call is router)
        assert scheduler.create_workers.call_count == 2
        job = scheduler.create_workers.call_args_list[1].kwargs["job"]

        # Verify job properties
        assert job.role == _GATEWAY_ROLE
        assert job.replicas == 1
        assert len(job.tasks) == 1

        # Verify the gateway job is the second call (after router)
        job = scheduler.create_workers.call_args_list[1].kwargs["job"]
        assert job.role == _GATEWAY_ROLE

        spec = job.tasks[0]
        assert "agent_service.gateway" in spec.cmd
        assert spec.cmd.strip() != ""  # Should have a valid command
        assert spec.gpu == 0  # CPU-only

        # Verify the gateway job is the second call (after router)
        job = scheduler.create_workers.call_args_list[1].kwargs["job"]
        assert job.role == _GATEWAY_ROLE

    def test_start_gateway_returns_correct_address(self):
        """start_gateway() should return the Gateway's HTTP address."""
        config = GatewayConfig()
        scheduler = make_mock_scheduler(gateway_ip="192.168.1.1", gateway_port=9000)
        controller = AgentController(config=config, scheduler=scheduler)

        controller.start_router()
        addr = controller.start_gateway()

        assert addr == "http://192.168.1.1:9000"
        assert controller._gateway_addr == "http://192.168.1.1:9000"
        assert controller.gateway_started is True

    def test_start_gateway_idempotent(self):
        """Calling start_gateway() twice should return same address without re-creating."""
        config = GatewayConfig()
        scheduler = make_mock_scheduler()
        controller = AgentController(config=config, scheduler=scheduler)

        controller.start_router()
        addr1 = controller.start_gateway()
        addr2 = controller.start_gateway()

        assert addr1 == addr2
        # create_workers should be called twice total (router + gateway), but only once for gateway
        assert (
            scheduler.create_workers.call_count == 2
        )  # router once, gateway once (not twice)


class TestAgentControllerStartWorkers:
    """Tests for start_workers() method."""

    def test_start_workers_raises_without_gateway(self):
        """start_workers() should raise RuntimeError if gateway not started."""
        config = GatewayConfig()
        scheduler = make_mock_scheduler()
        controller = AgentController(config=config, scheduler=scheduler)

        controller.start_router()  # Router required, but no Gateway
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
        scheduler = make_mock_scheduler()
        controller = AgentController(config=config, scheduler=scheduler)

        # Start router first
        controller.start_router()
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
        assert "AREAL_ZMQ_TASK_ADDR" in spec.env_vars
        assert "AREAL_ZMQ_RESULT_ADDR" in spec.env_vars
        assert "AREAL_AGENT_IMPORT_PATH_INTERNAL" in spec.env_vars
        assert spec.env_vars["AREAL_AGENT_IMPORT_PATH_INTERNAL"] == "mymodule.MyAgent"
        assert spec.env_vars["AREAL_AGENT_REUSE_INTERNAL"] == "true"
        assert "gpt-4" in spec.env_vars["AREAL_AGENT_INIT_KWARGS_INTERNAL"]

    def test_start_workers_passes_gateway_addr_to_workers(self):
        """start_workers() should pass Gateway address to each worker via env vars."""
        config = GatewayConfig()
        scheduler = make_mock_scheduler(gateway_ip="10.0.0.1", gateway_port=8300)
        controller = AgentController(config=config, scheduler=scheduler)

        controller.start_router()
        controller.start_gateway()
        scheduler.create_workers.reset_mock()

        controller.start_workers(
            AgentServiceSpec(
                agent_import_path="test.Agent",
                agent_reuse=False,
                workers=2,
            )
        )

        job = scheduler.create_workers.call_args.kwargs[
            "job"
        ]  # First call after reset is workers
        spec = job.tasks[0]
        assert "AREAL_ZMQ_TASK_ADDR" in spec.env_vars
        assert "AREAL_ZMQ_RESULT_ADDR" in spec.env_vars


class TestAgentControllerStop:
    """Tests for stop() method."""

    def test_stop_resets_state(self):
        """stop() should reset all state."""
        config = GatewayConfig()
        scheduler = make_mock_scheduler()
        controller = AgentController(config=config, scheduler=scheduler)

        controller.start_router()
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
        scheduler = make_mock_scheduler()
        controller = AgentController(config=config, scheduler=scheduler)

        controller.start_router()
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
        scheduler = make_mock_scheduler()
        controller = AgentController(config=config, scheduler=scheduler)

        controller.start_router()
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
