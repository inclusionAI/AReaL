# SPDX-License-Identifier: Apache-2.0

"""Tests for sandbox API, infra, and workflow components.

These tests use the LocalSandboxExecutor (no external services required)
and mock the InferenceEngine for workflow testing.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

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
        assert config.max_tool_turns == 5
        assert config.pool_size == 0

    def test_invalid_timeout(self):
        from areal.api.sandbox_api import SandboxConfig

        with pytest.raises(ValueError, match="timeout must be positive"):
            SandboxConfig(timeout=-1)

    def test_invalid_max_tool_turns(self):
        from areal.api.sandbox_api import SandboxConfig

        with pytest.raises(ValueError, match="max_tool_turns must be >= 1"):
            SandboxConfig(max_tool_turns=0)

    def test_invalid_pool_size(self):
        from areal.api.sandbox_api import SandboxConfig

        with pytest.raises(ValueError, match="pool_size must be >= 0"):
            SandboxConfig(pool_size=-1)

    def test_env_var_fallback(self, monkeypatch):
        from areal.api.sandbox_api import SandboxConfig

        monkeypatch.setenv("SANDBOX_API_URL", "http://test:3000")
        monkeypatch.setenv("SANDBOX_API_KEY", "test-key")
        config = SandboxConfig()
        assert config.api_url == "http://test:3000"
        assert config.api_key == "test-key"

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


# ---------------------------------------------------------------------------
# 4. SandboxManager tests
# ---------------------------------------------------------------------------
class TestSandboxManager:
    """Tests for SandboxManager pool and lifecycle."""

    @pytest.mark.asyncio
    async def test_on_demand_checkout_checkin(self):
        from areal.api.sandbox_api import SandboxConfig
        from areal.infra.sandbox.manager import SandboxManager

        config = SandboxConfig(backend="local", pool_size=0)
        manager = SandboxManager(config)

        sandbox = await manager.checkout()
        assert manager.active_count == 1

        result = await sandbox.run_code("print('test')")
        assert result.success

        await manager.checkin(sandbox)
        assert manager.active_count == 0
        # pool_size=0 → sandbox is destroyed, not pooled
        assert manager.pool_size == 0

        await manager.cleanup()

    @pytest.mark.asyncio
    async def test_pooled_checkout_checkin(self):
        from areal.api.sandbox_api import SandboxConfig
        from areal.infra.sandbox.manager import SandboxManager

        config = SandboxConfig(backend="local", pool_size=2)
        manager = SandboxManager(config)

        s1 = await manager.checkout()
        await manager.checkin(s1)
        assert manager.pool_size == 1  # returned to pool

        s2 = await manager.checkout()
        assert manager.pool_size == 0  # taken from pool
        assert manager.active_count == 1

        await manager.checkin(s2)
        await manager.cleanup()

    @pytest.mark.asyncio
    async def test_cleanup_closes_all(self):
        from areal.api.sandbox_api import SandboxConfig
        from areal.infra.sandbox.manager import SandboxManager

        config = SandboxConfig(backend="local", pool_size=3)
        manager = SandboxManager(config)

        # Create and return 3 sandboxes
        sandboxes = [await manager.checkout() for _ in range(3)]
        for s in sandboxes:
            await manager.checkin(s)
        assert manager.pool_size == 3

        await manager.cleanup()
        assert manager.pool_size == 0

    @pytest.mark.asyncio
    async def test_checkout_after_close_raises(self):
        from areal.api.sandbox_api import SandboxConfig
        from areal.infra.sandbox.manager import SandboxManager

        config = SandboxConfig(backend="local")
        manager = SandboxManager(config)
        await manager.cleanup()

        with pytest.raises(RuntimeError, match="closed"):
            await manager.checkout()


# ---------------------------------------------------------------------------
# 5. SandboxToolWorkflow tests (with mock engine)
# ---------------------------------------------------------------------------
class TestSandboxToolWorkflow:
    """Tests for SandboxToolWorkflow with mock inference engine."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = list(range(10))
        tokenizer.decode.side_effect = lambda ids, **kw: f"decoded_{len(ids)}"
        tokenizer.encode.return_value = [100, 101, 102]
        tokenizer.pad_token_id = 0
        tokenizer.eos_token_id = 2
        tokenizer.stop_token_ids = []
        return tokenizer

    @pytest.fixture
    def mock_engine(self):
        """Create a mock inference engine."""
        engine = AsyncMock()
        # Model response that doesn't trigger tool calls (simple completion)
        resp = MagicMock()
        resp.output_tokens = [10, 11, 12, 2]  # ends with EOS
        resp.output_logprobs = [-0.5, -0.3, -0.1, 0.0]
        resp.output_len = 4
        resp.output_versions = [1, 1, 1, 1]
        resp.input_tokens = list(range(10))
        resp.input_len = 10
        resp.stop_reason = "stop"
        engine.agenerate.return_value = resp
        return engine

    @pytest.fixture
    def simple_reward_fn(self):
        """Simple reward function that returns 1.0."""

        def reward_fn(prompt_str, completions_str, *args, **kwargs):
            return 1.0

        return reward_fn

    @pytest.mark.asyncio
    async def test_workflow_basic_no_tool_call(
        self, mock_tokenizer, mock_engine, simple_reward_fn
    ):
        """Test workflow where model doesn't make tool calls."""
        from areal.api.sandbox_api import SandboxConfig
        from areal.workflow.sandbox_tool import SandboxToolWorkflow

        workflow = SandboxToolWorkflow(
            reward_fn=simple_reward_fn,
            gconfig=MagicMock(
                new_with_stop_and_pad_token_ids=lambda t: MagicMock(
                    new=lambda **kw: MagicMock(),
                    max_new_tokens=100,
                ),
                max_new_tokens=100,
            ),
            tokenizer=mock_tokenizer,
            sandbox_config=SandboxConfig(enabled=True, backend="local"),
        )

        data = {"messages": [{"role": "user", "content": "What is 1+1?"}]}
        result = await workflow.arun_episode(mock_engine, data)

        assert result is not None
        assert "input_ids" in result
        assert "rewards" in result
        assert result["input_ids"].dim() == 2
        assert result["rewards"].item() == 1.0

    @pytest.mark.asyncio
    async def test_workflow_with_tool_call(self, mock_tokenizer, simple_reward_fn):
        """Test workflow where model makes a tool call."""
        from areal.api.sandbox_api import SandboxConfig
        from areal.workflow.sandbox_tool import SandboxToolWorkflow

        # Setup: first response triggers code block, second ends normally
        engine = AsyncMock()
        call_count = 0

        async def mock_generate(req):
            nonlocal call_count
            call_count += 1
            resp = MagicMock()

            if call_count == 1:
                # First call: model outputs code start marker
                resp.output_tokens = [20, 21, 22]
                resp.output_logprobs = [-0.1, -0.2, -0.3]
                resp.output_len = 3
                resp.output_versions = [1, 1, 1]
                resp.stop_reason = "stop"
                # Make decode return text ending with code marker
                mock_tokenizer.decode.side_effect = lambda ids, **kw: (
                    "```python\n" if len(ids) == 3 else f"decoded_{len(ids)}"
                )
            elif call_count == 2:
                # Second call: code content ending at end marker
                resp.output_tokens = [30, 31]
                resp.output_logprobs = [-0.1, -0.2]
                resp.output_len = 2
                resp.output_versions = [1, 1]
                resp.stop_reason = "stop"
                mock_tokenizer.decode.side_effect = lambda ids, **kw: (
                    "print(42)\n```" if len(ids) == 2 else f"decoded_{len(ids)}"
                )
            else:
                # Final call: normal completion with EOS
                resp.output_tokens = [2]
                resp.output_logprobs = [0.0]
                resp.output_len = 1
                resp.output_versions = [1]
                resp.stop_reason = "stop"
                mock_tokenizer.decode.side_effect = lambda ids, **kw: (
                    f"decoded_{len(ids)}"
                )

            return resp

        engine.agenerate.side_effect = mock_generate

        workflow = SandboxToolWorkflow(
            reward_fn=simple_reward_fn,
            gconfig=MagicMock(
                new_with_stop_and_pad_token_ids=lambda t: MagicMock(
                    new=lambda **kw: MagicMock(),
                    max_new_tokens=200,
                ),
                max_new_tokens=200,
            ),
            tokenizer=mock_tokenizer,
            sandbox_config=SandboxConfig(enabled=True, backend="local"),
        )

        data = {"messages": [{"role": "user", "content": "Compute 6*7"}]}
        result = await workflow.arun_episode(engine, data)

        assert result is not None
        assert "input_ids" in result


# ---------------------------------------------------------------------------
# 6. Code extraction utility tests
# ---------------------------------------------------------------------------
class TestCodeExtraction:
    """Tests for _extract_code utility."""

    def test_extract_python_fenced(self):
        from areal.workflow.sandbox_tool import _extract_code

        text = "Here is code:\n```python\nprint(42)\n```\nDone."
        assert _extract_code(text) == "print(42)"

    def test_extract_python_xml(self):
        from areal.workflow.sandbox_tool import _extract_code

        text = "Code: <python>x = 1\nprint(x)</python>"
        assert _extract_code(text) == "x = 1\nprint(x)"

    def test_extract_no_code(self):
        from areal.workflow.sandbox_tool import _extract_code

        assert _extract_code("no code here") == ""

    def test_extract_last_block(self):
        from areal.workflow.sandbox_tool import _extract_code

        text = "```python\nfirst\n```\ntext\n```python\nsecond\n```"
        assert _extract_code(text) == "second"
