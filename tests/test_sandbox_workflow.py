# SPDX-License-Identifier: Apache-2.0

"""Tests for sandbox API and infra components.

These tests use the LocalSandboxExecutor (no external services required).
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# 1. API layer tests
# ---------------------------------------------------------------------------
class TestExecutionResult:
    """Tests for ExecutionResult dataclass."""

    def test_success_result(self):
        from areal.api.sandbox_api import ExecutionResult

        r = ExecutionResult(stdout="hello", exit_code=0)
        assert r.success is True
        assert r.text == "hello"

    def test_error_result(self):
        from areal.api.sandbox_api import ExecutionResult

        r = ExecutionResult(error="fail", exit_code=1)
        assert r.success is False
        assert r.text == ""

    def test_output_text_priority(self):
        from areal.api.sandbox_api import ExecutionResult

        r = ExecutionResult(stdout="raw", output_text="formatted")
        assert r.text == "formatted"


class TestSandboxConfig:
    """Tests for SandboxConfig validation."""

    def test_default_config(self):
        from areal.api.sandbox_api import SandboxConfig

        config = SandboxConfig()
        assert config.enabled is False
        assert config.backend == "e2b"
        assert config.timeout == 30.0

    def test_invalid_timeout(self):
        from areal.api.sandbox_api import SandboxConfig

        with pytest.raises(ValueError, match="timeout must be positive"):
            SandboxConfig(timeout=-1)

    def test_env_var_fallback(self, monkeypatch):
        from areal.api.sandbox_api import SandboxConfig

        monkeypatch.setenv("SANDBOX_API_URL", "http://test:3000")
        monkeypatch.setenv("SANDBOX_API_KEY", "test-key")
        config = SandboxConfig()
        assert config.api_url == "http://test:3000"
        assert config.api_key == "test-key"

    def test_cube_backend_rejected(self):
        from areal.api.sandbox_api import SandboxConfig

        with pytest.raises(ValueError, match="backend='cube' has been removed"):
            SandboxConfig(backend="cube")

    def test_protocol_isinstance_check(self):
        from areal.api.sandbox_api import SandboxExecutor
        from areal.infra.sandbox.local_sandbox import LocalSandboxExecutor

        executor = LocalSandboxExecutor()
        assert isinstance(executor, SandboxExecutor)


# ---------------------------------------------------------------------------
# 2. LocalSandboxExecutor tests
# ---------------------------------------------------------------------------
class TestLocalSandboxExecutor:
    """Tests for LocalSandboxExecutor."""

    @pytest.fixture
    def executor(self):
        from areal.infra.sandbox.local_sandbox import LocalSandboxExecutor

        return LocalSandboxExecutor()

    @pytest.mark.asyncio
    async def test_run_code_success(self, executor):
        result = await executor.run_code("print('hello')")
        assert result.success
        assert "hello" in result.stdout

    @pytest.mark.asyncio
    async def test_run_code_error(self, executor):
        result = await executor.run_code("raise ValueError('test error')")
        assert not result.success
        assert "ValueError" in result.error

    @pytest.mark.asyncio
    async def test_run_code_timeout(self, executor):
        result = await executor.run_code("import time; time.sleep(10)", timeout=0.1)
        assert not result.success
        assert "timed out" in result.error.lower()

    @pytest.mark.asyncio
    async def test_run_code_unsupported_language(self, executor):
        result = await executor.run_code("console.log('hi')", language="javascript")
        assert not result.success
        assert "only supports Python" in result.error

    @pytest.mark.asyncio
    async def test_run_command_success(self, executor):
        result = await executor.run_command("echo hello")
        assert result.success
        assert "hello" in result.stdout

    @pytest.mark.asyncio
    async def test_run_command_failure(self, executor):
        result = await executor.run_command("false")
        assert not result.success

    @pytest.mark.asyncio
    async def test_close_and_reuse(self, executor):
        await executor.close()
        assert executor.is_closed
        with pytest.raises(RuntimeError, match="closed"):
            await executor.run_code("print(1)")


# ---------------------------------------------------------------------------
# 3. Factory tests
# ---------------------------------------------------------------------------
class TestFactory:
    """Tests for create_sandbox factory function."""

    @pytest.mark.asyncio
    async def test_create_local(self):
        from areal.api.sandbox_api import SandboxConfig
        from areal.infra.sandbox.factory import create_sandbox

        config = SandboxConfig(backend="local")
        executor = await create_sandbox(config)
        result = await executor.run_code("print(42)")
        assert result.success
        assert "42" in result.stdout
        await executor.close()

    @pytest.mark.asyncio
    async def test_create_invalid_backend(self):
        from areal.api.sandbox_api import SandboxConfig
        from areal.infra.sandbox.factory import create_sandbox

        config = SandboxConfig(backend="nonexistent")
        with pytest.raises(ValueError, match="Unsupported sandbox backend"):
            await create_sandbox(config)

