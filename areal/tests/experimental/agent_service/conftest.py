"""Pytest fixtures and mock agents for Agent Service tests."""

from __future__ import annotations

import pytest

from areal.experimental.agent_service import AgentServiceConfig


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
    """Test mock agent with class-level counter to verify instance creation.

    Note: Use reset_count() before tests that depend on instance_count.
    """

    instance_count = 0

    def __init__(self, **kwargs):
        CountingAgent.instance_count += 1
        self.instance_id = CountingAgent.instance_count
        self.init_kwargs = kwargs

    async def run(self, data: dict, **extra_kwargs) -> float:
        return float(self.instance_id)

    @classmethod
    def reset_count(cls):
        cls.instance_count = 0


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
