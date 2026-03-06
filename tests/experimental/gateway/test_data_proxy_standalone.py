"""Unit tests for data proxy standalone mode (admin-key without session).

Tests that admin-key callers can use /chat/completions and /generate
without starting a session first. Session-key flows remain unchanged.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import pytest_asyncio

from areal.experimental.gateway.data_proxy.app import create_app
from areal.experimental.gateway.data_proxy.backend import GenerationResult
from areal.experimental.gateway.data_proxy.chat import ChatCompletionHandler
from areal.experimental.gateway.data_proxy.config import DataProxyConfig
from areal.experimental.gateway.data_proxy.session import SessionStore

# =============================================================================
# Fixtures
# =============================================================================

ADMIN_KEY = "areal-admin-key"


@pytest.fixture
def config():
    return DataProxyConfig(
        host="127.0.0.1",
        port=18082,
        backend_addr="http://mock-sglang:30000",
        tokenizer_path="mock-tokenizer",
        request_timeout=10.0,
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

    handler = MagicMock(spec=ChatCompletionHandler)

    completion = ChatCompletion(
        id="chatcmpl-standalone-test",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                logprobs=None,
                message=ChatCompletionMessage(content="Hello!", role="assistant"),
            )
        ],
        created=1234567890,
        model="sglang",
        object="chat.completion",
        usage=CompletionUsage(completion_tokens=3, prompt_tokens=5, total_tokens=8),
    )

    async def _mock_create(*, areal_cache=None, **kwargs):
        return completion

    handler.create = AsyncMock(side_effect=_mock_create)
    return handler


@pytest_asyncio.fixture
async def client(config, mock_tokenizer, mock_backend, mock_chat_handler):
    """Create app with mocked deps and yield an httpx async client (no auth header)."""
    from areal.experimental.gateway.data_proxy.backend import (
        SGLangBackendWithResubmit,
    )
    from areal.experimental.gateway.data_proxy.pause import PauseState

    app = create_app(config)
    pause_state = PauseState()
    resubmit_backend = SGLangBackendWithResubmit(
        base=mock_backend,
        pause_state=pause_state,
        max_resubmit_retries=config.max_resubmit_retries,
        resubmit_wait=0.01,
    )
    app.state.tokenizer = mock_tokenizer
    app.state.backend = mock_backend
    app.state.resubmit_backend = resubmit_backend
    app.state.pause_state = pause_state
    app.state.config = config
    app.state.session_store = SessionStore()
    app.state.chat_handler = mock_chat_handler
    transport = httpx.ASGITransport(app=app)
    # No default auth header — tests supply their own
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def admin_headers():
    return {"Authorization": f"Bearer {ADMIN_KEY}"}


def parse_sse_events(content: bytes) -> list[dict]:
    events = []
    for line in content.decode().strip().split("\n"):
        line = line.strip()
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))
    return events


# =============================================================================
# Admin-key standalone /chat/completions
# =============================================================================


class TestAdminStandaloneChat:
    @pytest.mark.asyncio
    async def test_admin_chat_completions_returns_valid_response(self, client):
        """Admin key can call /chat/completions without a session."""
        resp = await client.post(
            "/chat/completions",
            json={"model": "sglang", "messages": [{"role": "user", "content": "hello"}]},
            headers=admin_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "chatcmpl-standalone-test"
        assert data["choices"][0]["message"]["content"] == "Hello!"

    @pytest.mark.asyncio
    async def test_admin_chat_completions_no_session_created(self, client):
        """Admin-key standalone mode does NOT create a session."""
        resp = await client.post(
            "/chat/completions",
            json={"model": "sglang", "messages": [{"role": "user", "content": "hello"}]},
            headers=admin_headers(),
        )
        assert resp.status_code == 200
        # Verify no sessions were created
        from areal.experimental.gateway.data_proxy.session import SessionStore

        store: SessionStore = client._transport.app.state.session_store  # type: ignore[attr-defined]
        assert store.session_count == 0

    @pytest.mark.asyncio
    async def test_admin_chat_completions_passes_none_cache(
        self, client, mock_chat_handler
    ):
        """Admin-key standalone passes areal_cache=None (no caching)."""
        resp = await client.post(
            "/chat/completions",
            json={"model": "sglang", "messages": [{"role": "user", "content": "hello"}]},
            headers=admin_headers(),
        )
        assert resp.status_code == 200
        # Verify create was called with areal_cache=None
        mock_chat_handler.create.assert_called_once()
        call_kwargs = mock_chat_handler.create.call_args
        assert call_kwargs.kwargs.get("areal_cache") is None


# =============================================================================
# Admin-key standalone /generate
# =============================================================================


class TestAdminStandaloneGenerate:
    @pytest.mark.asyncio
    async def test_admin_generate_returns_sse_stream(self, client):
        """Admin key can call /generate without a session."""
        resp = await client.post(
            "/generate",
            json={"input_ids": [1, 2, 3]},
            headers=admin_headers(),
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        events = parse_sse_events(resp.content)
        assert len(events) == 3
        assert events[-1]["finished"] is True

    @pytest.mark.asyncio
    async def test_admin_generate_with_text(self, client, mock_tokenizer):
        """Admin key can call /generate with text instead of input_ids."""
        resp = await client.post(
            "/generate",
            json={"text": "What is 2+2?"},
            headers=admin_headers(),
        )
        assert resp.status_code == 200
        mock_tokenizer.tokenize.assert_called_once_with("What is 2+2?")


# =============================================================================
# Session-key flows unchanged
# =============================================================================


class TestSessionKeyUnchanged:
    @pytest.mark.asyncio
    async def test_session_chat_completions_still_works(
        self, client, mock_chat_handler
    ):
        """Session key callers still use the session-based flow."""
        # Start a session first
        resp = await client.post(
            "/rl/start_session",
            json={"task_id": "test-task"},
            headers=admin_headers(),
        )
        assert resp.status_code == 201
        session_api_key = resp.json()["api_key"]

        # Now use session key for chat completions
        resp = await client.post(
            "/chat/completions",
            json={"model": "sglang", "messages": [{"role": "user", "content": "hello"}]},
            headers={"Authorization": f"Bearer {session_api_key}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["message"]["content"] == "Hello!"

    @pytest.mark.asyncio
    async def test_session_generate_still_works(self, client):
        """Session key callers can also use /generate."""
        # Start a session
        resp = await client.post(
            "/rl/start_session",
            json={"task_id": "test-task"},
            headers=admin_headers(),
        )
        assert resp.status_code == 201
        session_api_key = resp.json()["api_key"]

        # Use session key for generate
        resp = await client.post(
            "/generate",
            json={"input_ids": [1, 2, 3]},
            headers={"Authorization": f"Bearer {session_api_key}"},
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]


# =============================================================================
# Auth rejection
# =============================================================================


class TestAuthRejection:
    @pytest.mark.asyncio
    async def test_unknown_key_rejected_chat(self, client):
        """Unknown key is rejected with 401 on /chat/completions."""
        resp = await client.post(
            "/chat/completions",
            json={"model": "sglang", "messages": [{"role": "user", "content": "hello"}]},
            headers={"Authorization": "Bearer unknown-key-12345"},
        )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_unknown_key_rejected_generate(self, client):
        """Unknown key is rejected with 401 on /generate."""
        resp = await client.post(
            "/generate",
            json={"input_ids": [1, 2, 3]},
            headers={"Authorization": "Bearer unknown-key-12345"},
        )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_missing_auth_header_chat(self, client):
        """Missing Authorization header is rejected with 401."""
        resp = await client.post(
            "/chat/completions",
            json={"model": "sglang", "messages": [{"role": "user", "content": "hello"}]},
        )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_missing_auth_header_generate(self, client):
        """Missing Authorization header is rejected with 401 on /generate."""
        resp = await client.post(
            "/generate",
            json={"input_ids": [1, 2, 3]},
        )
        assert resp.status_code == 401
