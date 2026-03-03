"""Pytest fixtures and mock agents for Agent Service tests."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from areal.experimental.agent_service import AgentServiceConfig
from areal.experimental.agent_service.agent_controller import (
    _GATEWAY_ROLE,
    _ROUTER_ROLE,
    _WORKER_ROLE,
)


class MockAgent:
    """Test mock agent that records call information."""

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.call_count = 0
        self.last_data = None
        self.last_extra_kwargs = None

    async def run(self, data: dict, **extra_kwargs) -> float:
        self.call_count += 1
        self.last_data = data
        self.last_extra_kwargs = extra_kwargs
        return 1.0


class FailingAgent:
    """Test mock agent that always raises an exception."""

    async def run(self, data: dict, **extra_kwargs) -> float:
        raise ValueError("Agent failed intentionally")


class CountingAgent:
    """Test mock agent with instance-level counter.

    FIXED: Changed from class-level to instance-level tracking
    to avoid race conditions in parallel test execution.
    """

    _global_counter = 0

    def __init__(self, **kwargs):
        # Use a unique instance ID without modifying class state
        CountingAgent._global_counter += 1
        self.instance_id = CountingAgent._global_counter
        self.init_kwargs = kwargs

    async def run(self, data: dict, **extra_kwargs) -> float:
        return float(self.instance_id)

    @classmethod
    def reset_count(cls):
        """Reset the global counter (for test isolation)."""
        cls._global_counter = 0


@pytest.fixture
def agent_config():
    """Create a default AgentServiceConfig for tests."""
    return AgentServiceConfig(host="127.0.0.1", port=8300)


@pytest.fixture
def mock_agent_import_path():
    """Import path for MockAgent."""
    return "areal.tests.experimental.agent_service.conftest.MockAgent"


@pytest.fixture
def failing_agent_import_path():
    """Import path for FailingAgent."""
    return "areal.tests.experimental.agent_service.conftest.FailingAgent"


@pytest.fixture
def counting_agent_import_path():
    """Import path for CountingAgent."""
    # Reset count before each test that uses this fixture
    CountingAgent.reset_count()
    return "areal.tests.experimental.agent_service.conftest.CountingAgent"


def make_mock_scheduler(
    gateway_ip: str = "10.0.0.1",
    gateway_port: int = 8300,
    router_ip: str = "10.0.0.0",
    router_port: int = 9000,
    worker_ip: str = "10.0.0.2",
    worker_port: int = 8301,
):
    """Create a mock Scheduler that returns fake Worker objects.

    This helper consolidates the _make_mock_scheduler() function that was
    duplicated across test files. It creates a MagicMock Scheduler with
    fake Worker objects for gateway, router, and worker roles.

    Args:
        gateway_ip: IP address for gateway worker
        gateway_port: Port for gateway worker
        router_ip: IP address for router worker
        router_port: Port for router worker
        worker_ip: IP address for agent worker
        worker_port: Port for agent worker

    Returns:
        MagicMock Scheduler instance
    """
    scheduler = MagicMock()

    # Mock router worker
    router_worker = MagicMock()
    router_worker.ip = router_ip
    router_worker.worker_ports = [router_port]

    # Mock gateway worker
    gateway_worker = MagicMock()
    gateway_worker.ip = gateway_ip
    gateway_worker.worker_ports = [gateway_port]

    # Mock agent worker
    agent_worker = MagicMock()
    agent_worker.ip = worker_ip
    agent_worker.worker_ports = [worker_port]

    def get_workers(role, timeout=None):
        if role == _ROUTER_ROLE:
            return [router_worker]
        elif role == _GATEWAY_ROLE:
            return [gateway_worker]
        elif role == _WORKER_ROLE:
            return [agent_worker]
        return []

    scheduler.get_workers.side_effect = get_workers
    scheduler.create_workers.return_value = ["worker-id-1"]
    scheduler.experiment_name = "test-experiment"
    scheduler.trial_name = "test-trial"

    return scheduler
