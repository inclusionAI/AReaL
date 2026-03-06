"""Unit tests for data proxy pause/resume and abort/resubmit loop (Plan 3c)."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import pytest_asyncio

from areal.experimental.gateway.data_proxy.app import create_app
from areal.experimental.gateway.data_proxy.backend import (
    GenerationResult,
    SGLangBackendWithResubmit,
)
from areal.experimental.gateway.data_proxy.config import DataProxyConfig
from areal.experimental.gateway.data_proxy.pause import PauseState
from areal.experimental.gateway.data_proxy.session import SessionStore


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def config():
    return DataProxyConfig(
        host="127.0.0.1",
        port=18083,
        backend_addr="http://mock-sglang:30000",
        tokenizer_path="mock-tokenizer",
        request_timeout=10.0,
        max_resubmit_retries=5,
        resubmit_wait=0.01,  # fast for tests
    )


@pytest.fixture
def mock_tokenizer():
    tok = MagicMock()
    tok.tokenize = AsyncMock(return_value=[101, 102, 103])
    tok.decode_token = MagicMock(side_effect=lambda tid: f"tok_{tid}")
    tok.decode_tokens = MagicMock(return_value="hello world")
    tok.apply_chat_template = AsyncMock(return_value=[100, 200, 300])
    tok.eos_token_id = 2
    tok.pad_token_id = 0
    tok._tok = MagicMock()
    tok._tok.eos_token_id = 2
    tok._tok.pad_token_id = 0
    return tok


@pytest.fixture
def mock_backend():
    """A mock SGLangBackend (Plan 3a base) that returns "stop" by default."""
    backend = MagicMock()
    backend.generate = AsyncMock(
        return_value=GenerationResult(
            output_tokens=[1234, 5678, 2],
            output_logprobs=[-0.5, -0.3, -0.1],
            stop_reason="stop",
        )
    )
    return backend


@pytest.fixture
def mock_chat_handler():
    """Mock ChatCompletionHandler that returns a valid ChatCompletion."""
    from openai.types.chat import ChatCompletion, ChatCompletionMessage
    from openai.types.chat.chat_completion import Choice
    from openai.types.completion_usage import CompletionUsage

    handler = MagicMock()

    async def _mock_create(*, areal_cache=None, **kwargs):
        import torch

        completion = ChatCompletion(
            id="chatcmpl-mock",
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    logprobs=None,
                    message=ChatCompletionMessage(
                        content="mocked response", role="assistant"
                    ),
                )
            ],
            created=1700000000,
            model="sglang",
            object="chat.completion",
            usage=CompletionUsage(completion_tokens=3, prompt_tokens=5, total_tokens=8),
        )
        if areal_cache is not None:
            from areal.experimental.gateway.data_proxy.types import (
                InteractionWithTokenLogpReward,
            )

            interaction = InteractionWithTokenLogpReward(
                messages=[{"role": "user", "content": "test"}],
            )
            interaction._cache = {
                "input_ids": torch.tensor([100, 200, 300]),
                "output_tokens": torch.tensor([1234, 5678]),
            }
            interaction.completion = completion
            cid = completion.id
            areal_cache[cid] = interaction
        return completion

    handler.create = _mock_create
    return handler


@pytest_asyncio.fixture
async def app_client(config, mock_tokenizer, mock_backend, mock_chat_handler):
    """Create an ASGI test client with all app.state attributes injected."""
    app = create_app(config)

    pause_state = PauseState()
    backend = SGLangBackendWithResubmit(
        backend_addr=config.backend_addr,
        pause_state=pause_state,
        request_timeout=config.request_timeout,
        max_resubmit_retries=config.max_resubmit_retries,
        resubmit_wait=config.resubmit_wait,
    )
    backend._call_sglang = mock_backend.generate

    app.state.tokenizer = mock_tokenizer
    app.state.backend = backend
    app.state.pause_state = pause_state
    app.state.config = config
    app.state.session_store = SessionStore()
    app.state.chat_handler = mock_chat_handler

    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    # No auth header needed — /generate has no auth at data proxy level
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, app, pause_state


def _parse_sse_events(content: bytes) -> list[dict]:
    """Parse ``data: {...}`` lines from an SSE byte-stream."""
    events = []
    for line in content.decode().strip().split("\n"):
        line = line.strip()
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))
    return events


# =============================================================================
# PauseState unit tests
# =============================================================================


class TestPauseState:
    """Test PauseState flag transitions."""

    @pytest.mark.asyncio
    async def test_default_not_paused(self):
        state = PauseState()
        assert not await state.is_paused()

    @pytest.mark.asyncio
    async def test_set_paused_true(self):
        state = PauseState()
        await state.set_paused(True)
        assert await state.is_paused()

    @pytest.mark.asyncio
    async def test_set_paused_false(self):
        state = PauseState()
        await state.set_paused(True)
        assert await state.is_paused()
        await state.set_paused(False)
        assert not await state.is_paused()

    @pytest.mark.asyncio
    async def test_multiple_transitions(self):
        state = PauseState()
        for _ in range(3):
            await state.set_paused(True)
            assert await state.is_paused()
            await state.set_paused(False)
            assert not await state.is_paused()


# =============================================================================
# SGLangBackendWithResubmit unit tests
# =============================================================================


class TestSGLangBackendWithResubmit:
    """Test the abort/resubmit loop in isolation."""

    @pytest.mark.asyncio
    async def test_no_abort_pass_through(self):
        """Normal stop — no resubmit needed."""
        pause_state = PauseState()
        backend = SGLangBackendWithResubmit(
            backend_addr="http://mock", pause_state=pause_state
        )
        backend._call_sglang = AsyncMock(
            return_value=GenerationResult([100, 101], [-0.5, -0.3], "stop")
        )

        result = await backend.generate([1, 2, 3], {"max_new_tokens": 20})

        assert result.output_tokens == [100, 101]
        assert result.output_logprobs == [-0.5, -0.3]
        assert result.stop_reason == "stop"
        backend._call_sglang.assert_called_once()

    @pytest.mark.asyncio
    async def test_single_abort_then_stop(self):
        """One abort, then stop — verify token accumulation and resubmit."""
        call_count = 0

        async def mock_call_sglang(input_ids, params):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return GenerationResult([100, 101], [-0.5, -0.3], "abort")
            return GenerationResult([200, 201], [-0.4, -0.2], "stop")

        pause_state = PauseState()
        backend = SGLangBackendWithResubmit(
            backend_addr="http://mock", pause_state=pause_state, resubmit_wait=0.01
        )
        backend._call_sglang = mock_call_sglang

        result = await backend.generate([1, 2, 3], {"max_new_tokens": 20})

        assert call_count == 2
        assert result.output_tokens == [100, 101, 200, 201]
        assert result.output_logprobs == [-0.5, -0.3, -0.4, -0.2]
        assert result.stop_reason == "stop"

    @pytest.mark.asyncio
    async def test_resubmit_input_ids_extended(self):
        """Verify resubmit passes input_ids + accumulated_output as new input."""
        calls = []

        async def mock_call_sglang(input_ids, params):
            calls.append({"input_ids": list(input_ids), "params": dict(params)})
            if len(calls) == 1:
                return GenerationResult([100, 101], [-0.5, -0.3], "abort")
            return GenerationResult([200], [-0.4], "stop")

        pause_state = PauseState()
        backend = SGLangBackendWithResubmit(
            backend_addr="http://mock", pause_state=pause_state, resubmit_wait=0.01
        )
        backend._call_sglang = mock_call_sglang

        await backend.generate([1, 2, 3], {"max_new_tokens": 20})

        # First call: original input_ids
        assert calls[0]["input_ids"] == [1, 2, 3]
        assert calls[0]["params"]["max_new_tokens"] == 20

        # Second call: original + accumulated output tokens
        assert calls[1]["input_ids"] == [1, 2, 3, 100, 101]
        assert calls[1]["params"]["max_new_tokens"] == 18  # 20 - 2

    @pytest.mark.asyncio
    async def test_max_new_tokens_exhausted_becomes_length(self):
        """When accumulated tokens reach max_new_tokens, stop_reason='length'."""
        call_count = 0

        async def mock_call_sglang(input_ids, params):
            nonlocal call_count
            call_count += 1
            # Always return 5 tokens with abort
            return GenerationResult([10, 11, 12, 13, 14], [-0.1] * 5, "abort")

        pause_state = PauseState()
        backend = SGLangBackendWithResubmit(
            backend_addr="http://mock",
            pause_state=pause_state,
            max_resubmit_retries=10,
            resubmit_wait=0.01,
        )
        backend._call_sglang = mock_call_sglang

        result = await backend.generate([1, 2], {"max_new_tokens": 10})

        # First call returns 5 tokens (abort), second call max_new_tokens=5,
        # returns 5 more tokens (abort), third call max_new_tokens=0 -> break as length
        assert call_count == 2
        assert len(result.output_tokens) == 10
        assert result.stop_reason == "length"

    @pytest.mark.asyncio
    async def test_max_retries_final_abort_becomes_length(self):
        """After max retries, final abort is converted to 'length'."""
        pause_state = PauseState()
        backend = SGLangBackendWithResubmit(
            backend_addr="http://mock",
            pause_state=pause_state,
            max_resubmit_retries=3,
            resubmit_wait=0.01,
        )
        backend._call_sglang = AsyncMock(
            return_value=GenerationResult([10], [-0.1], "abort")
        )

        result = await backend.generate([1, 2], {"max_new_tokens": 100})

        assert backend._call_sglang.call_count == 3
        assert result.stop_reason == "length"
        assert len(result.output_tokens) == 3  # 1 token per retry

    @pytest.mark.asyncio
    async def test_paused_blocks_until_resumed(self):
        """While paused, generate waits; after resume, it completes."""
        pause_state = PauseState()
        backend = SGLangBackendWithResubmit(
            backend_addr="http://mock", pause_state=pause_state, resubmit_wait=0.01
        )
        backend._call_sglang = AsyncMock(
            return_value=GenerationResult([100], [-0.5], "stop")
        )

        await pause_state.set_paused(True)
        task = asyncio.create_task(backend.generate([1, 2, 3], {"max_new_tokens": 5}))
        await asyncio.sleep(0.05)  # let it start
        assert not task.done()  # blocked by pause

        await pause_state.set_paused(False)  # unblock
        result = await asyncio.wait_for(task, timeout=2.0)
        assert result.stop_reason == "stop"
        assert result.output_tokens == [100]

    @pytest.mark.asyncio
    async def test_tool_calls_stop_reason_passthrough(self):
        """stop_reason='tool_calls' exits the loop normally."""
        pause_state = PauseState()
        backend = SGLangBackendWithResubmit(
            backend_addr="http://mock", pause_state=pause_state
        )
        backend._call_sglang = AsyncMock(
            return_value=GenerationResult([100, 101], [-0.5, -0.3], "tool_calls")
        )

        result = await backend.generate([1, 2], {"max_new_tokens": 20})

        assert result.stop_reason == "tool_calls"
        assert result.output_tokens == [100, 101]
        backend._call_sglang.assert_called_once()

    @pytest.mark.asyncio
    async def test_length_stop_reason_passthrough(self):
        """stop_reason='length' exits the loop normally."""
        pause_state = PauseState()
        backend = SGLangBackendWithResubmit(
            backend_addr="http://mock", pause_state=pause_state
        )
        backend._call_sglang = AsyncMock(
            return_value=GenerationResult([100], [-0.5], "length")
        )

        result = await backend.generate([1, 2], {"max_new_tokens": 20})

        assert result.stop_reason == "length"
        backend._call_sglang.assert_called_once()


# =============================================================================
# Endpoint tests — /pause_generation and /continue_generation
# =============================================================================


class TestPauseResumeEndpoints:
    """Test POST /pause_generation and POST /continue_generation endpoints."""

    @pytest.mark.asyncio
    async def test_health_includes_paused_false(self, app_client):
        client, app, pause_state = app_client
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "paused" in data
        assert data["paused"] is False

    @pytest.mark.asyncio
    @patch(
        "areal.experimental.gateway.data_proxy.app.pause_backend",
        new_callable=AsyncMock,
    )
    async def test_pause_endpoint(self, mock_pause_backend, app_client):
        client, app, pause_state = app_client

        resp = await client.post("/pause_generation")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["paused"] is True

        # Verify PauseState was set
        assert await pause_state.is_paused()

        # Verify SGLang pause_generation was called
        mock_pause_backend.assert_called_once()

    @pytest.mark.asyncio
    @patch(
        "areal.experimental.gateway.data_proxy.app.resume_backend",
        new_callable=AsyncMock,
    )
    async def test_resume_endpoint(self, mock_resume_backend, app_client):
        client, app, pause_state = app_client

        # First pause
        await pause_state.set_paused(True)

        resp = await client.post("/continue_generation")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["paused"] is False

        # Verify PauseState was cleared
        assert not await pause_state.is_paused()

        # Verify SGLang continue_generation was called
        mock_resume_backend.assert_called_once()

    @pytest.mark.asyncio
    @patch(
        "areal.experimental.gateway.data_proxy.app.pause_backend",
        new_callable=AsyncMock,
    )
    async def test_health_paused_true_after_pause(self, mock_pause_backend, app_client):
        client, app, pause_state = app_client

        await client.post("/pause_generation")

        resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["paused"] is True


# =============================================================================
# Endpoint tests — /generate with resubmit
# =============================================================================


class TestGenerateWithResubmit:
    """Test POST /generate with the resubmit backend."""

    @pytest.mark.asyncio
    async def test_generate_with_single_abort_resubmit(self, app_client):
        """Backend returns abort first, then stop. SSE stream contains all tokens."""
        client, app, pause_state = app_client

        call_count = 0

        async def mock_generate(input_ids, params):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return GenerationResult([100, 101], [-0.5, -0.3], "abort")
            return GenerationResult([200], [-0.4], "stop")

        # Replace the backend's _call_sglang to simulate abort/resubmit
        app.state.backend._call_sglang = mock_generate

        resp = await client.post(
            "/generate",
            json={
                "input_ids": [1, 2, 3],
                "sampling_params": {"max_new_tokens": 20, "temperature": 0.0},
            },
            timeout=10.0,
        )
        assert resp.status_code == 200

        events = _parse_sse_events(resp.content)
        assert len(events) == 3  # tokens: 100, 101, 200

        # All tokens in order
        tokens = [e["token"] for e in events]
        assert tokens == [100, 101, 200]

        # Last event is finished
        assert events[-1]["finished"] is True
        assert events[-1]["stop_reason"] == "stop"

        # Non-last events are not finished
        for evt in events[:-1]:
            assert evt["finished"] is False

    @pytest.mark.asyncio
    async def test_generate_no_abort_normal_flow(self, app_client):
        """No abort — normal generation pass-through."""
        client, app, pause_state = app_client

        resp = await client.post(
            "/generate",
            json={
                "input_ids": [1, 2, 3],
                "sampling_params": {"max_new_tokens": 16, "temperature": 0.0},
            },
            timeout=10.0,
        )
        assert resp.status_code == 200

        events = _parse_sse_events(resp.content)
        assert len(events) == 3  # 3 tokens from mock_backend default
        assert events[-1]["finished"] is True
        assert events[-1]["stop_reason"] == "stop"
