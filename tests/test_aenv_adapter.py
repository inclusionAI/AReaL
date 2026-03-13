"""Unit tests for AEnvironment adapter."""

from __future__ import annotations

import pytest

from areal.infra.aenv.adapter import AenvEnvironmentAdapter
from areal.infra.aenv.config import AenvConfig


class _FakeToolResult:
    def __init__(self, content, is_error: bool = False):
        self.content = content
        self.is_error = is_error


class _FakeEnvironment:
    def __init__(
        self,
        env_name,
        datasource,
        ttl,
        environment_variables,
        arguments,
        aenv_url,
        timeout,
        startup_timeout,
        max_retries,
    ):
        self.constructor_kwargs = {
            "env_name": env_name,
            "datasource": datasource,
            "ttl": ttl,
            "environment_variables": environment_variables,
            "arguments": arguments,
            "aenv_url": aenv_url,
            "timeout": timeout,
            "startup_timeout": startup_timeout,
            "max_retries": max_retries,
        }
        self.initialize_calls = 0
        self.release_calls = 0
        self.list_tools_calls = 0
        self.call_tool_calls = 0
        self.tools_result = [{"name": "math/calc", "inputSchema": {"type": "object"}}]
        self.call_plan = [_FakeToolResult(content=[{"type": "text", "text": "ok"}])]

    async def initialize(self):
        self.initialize_calls += 1

    async def release(self):
        self.release_calls += 1

    async def list_tools(self):
        self.list_tools_calls += 1
        return self.tools_result

    async def call_tool(self, tool_name, arguments, timeout=None):
        self.call_tool_calls += 1
        assert isinstance(tool_name, str)
        assert isinstance(arguments, dict)
        if self.call_plan:
            outcome = self.call_plan.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        return _FakeToolResult(content=[{"type": "text", "text": "default"}])


@pytest.fixture
def patched_import(monkeypatch):
    """Patch adapter import path and expose latest fake environment instance."""
    import areal.infra.aenv.adapter as adapter_module

    holder: dict[str, _FakeEnvironment] = {}

    def _import_aenv_environment():
        class _Factory(_FakeEnvironment):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                holder["env"] = self

        return _Factory

    monkeypatch.setattr(
        adapter_module, "_import_aenv_environment", _import_aenv_environment
    )
    return holder


@pytest.mark.asyncio
async def test_adapter_initialize_passes_expected_environment_arguments(patched_import):
    """Test adapter passes config fields into Environment constructor."""
    config = AenvConfig(
        aenv_url="http://aenv-host",
        env_name="math-tools",
        datasource="dataset://math",
        ttl="10m",
        environment_variables={"K": "V"},
        arguments=["--flag"],
        timeout=12.0,
        startup_timeout=45.0,
        max_retries=3,
    )
    adapter = AenvEnvironmentAdapter(config)

    await adapter.initialize()

    env = patched_import["env"]
    assert env.initialize_calls == 1
    assert env.constructor_kwargs["aenv_url"] == "http://aenv-host"
    assert env.constructor_kwargs["env_name"] == "math-tools"
    assert env.constructor_kwargs["datasource"] == "dataset://math"
    assert env.constructor_kwargs["max_retries"] == 3


@pytest.mark.asyncio
async def test_adapter_list_tools_uses_cache_by_default(patched_import):
    """Test list_tools caches the environment result when use_cache=True."""
    adapter = AenvEnvironmentAdapter(AenvConfig())
    await adapter.initialize()

    tools_first = await adapter.list_tools()
    tools_second = await adapter.list_tools()

    env = patched_import["env"]
    assert env.list_tools_calls == 1
    assert tools_first == tools_second


@pytest.mark.asyncio
async def test_adapter_call_tool_retries_on_retriable_error(patched_import):
    """Test call_tool retries when transient failures occur."""
    adapter = AenvEnvironmentAdapter(AenvConfig(max_retries=2, retry_delay=0.0))
    await adapter.initialize()

    env = patched_import["env"]
    env.call_plan = [
        TimeoutError("temporary timeout"),
        _FakeToolResult(content=[{"type": "text", "text": "42"}], is_error=False),
    ]

    result = await adapter.call_tool("math/calc", {"expr": "40+2"})

    assert env.call_tool_calls == 2
    assert result.content == [{"type": "text", "text": "42"}]
    assert result.is_error is False


@pytest.mark.asyncio
async def test_adapter_call_tool_raises_without_retry_for_non_retriable_error(
    patched_import,
):
    """Test call_tool fails fast for non-retriable exceptions."""
    adapter = AenvEnvironmentAdapter(AenvConfig(max_retries=3, retry_delay=0.0))
    await adapter.initialize()

    env = patched_import["env"]
    env.call_plan = [ValueError("bad request")]

    with pytest.raises(RuntimeError, match="Tool call failed"):
        await adapter.call_tool("math/calc", {"expr": "bad"})

    assert env.call_tool_calls == 1


@pytest.mark.asyncio
async def test_adapter_cleanup_respects_auto_release_flag(patched_import):
    """Test cleanup does not release when auto_release is disabled."""
    adapter = AenvEnvironmentAdapter(AenvConfig(auto_release=False))
    await adapter.initialize()

    env = patched_import["env"]
    await adapter.cleanup()

    assert env.release_calls == 0
    assert adapter._env is env


@pytest.mark.asyncio
async def test_adapter_context_manager_forces_release_even_if_auto_release_disabled(
    patched_import,
):
    """Test context manager always releases resources on exit."""
    adapter = AenvEnvironmentAdapter(AenvConfig(auto_release=False))

    async with adapter:
        pass

    env = patched_import["env"]
    assert env.release_calls == 1
    assert adapter._env is None
