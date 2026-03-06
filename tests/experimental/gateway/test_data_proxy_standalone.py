"""Unit tests for data proxy standalone mode (no-session requests).

Tests that /chat/completions and /generate work without a session.
For /chat/completions: if the bearer token is a known session key, use session
cache; otherwise (no auth, admin key, unknown key) → standalone mode (no caching).
For /generate: no authentication at all — always accepted.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import pytest_asyncio

from areal.experimental.gateway.data_proxy.app import create_app
from areal.experimental.gateway.data_proxy.backend import (
    GenerationResult,
    SGLangBackend,
)
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
async def client(config, mock_tokenizer, mock_chat_handler):
    """Create app with mocked deps and yield an httpx async client (no auth header)."""
    from areal.experimental.gateway.data_proxy.pause import PauseState

    app = create_app(config)
    pause_state = PauseState()
    backend = SGLangBackend(
        backend_addr=config.backend_addr,
        pause_state=pause_state,
        request_timeout=config.request_timeout,
        max_resubmit_retries=config.max_resubmit_retries,
        resubmit_wait=0.01,
    )
    backend._call_sglang = AsyncMock(
        return_value=GenerationResult(
            output_tokens=[1234, 5678, 2],
            output_logprobs=[-0.5, -0.3, -0.1],
            stop_reason="stop",
        )
    )
    app.state.tokenizer = mock_tokenizer
    app.state.backend = backend
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
# Standalone /chat/completions (no session)
# =============================================================================


class TestStandaloneChat:
    @pytest.mark.asyncio
    async def test_no_auth_chat_completions_returns_valid_response(self, client):
        """No auth header → standalone mode, returns valid response."""
        resp = await client.post(
            "/chat/completions",
            json={
                "model": "sglang",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "chatcmpl-standalone-test"
        assert data["choices"][0]["message"]["content"] == "Hello!"

    @pytest.mark.asyncio
    async def test_admin_key_chat_completions_returns_valid_response(self, client):
        """Admin key → standalone mode, returns valid response."""
        resp = await client.post(
            "/chat/completions",
            json={
                "model": "sglang",
                "messages": [{"role": "user", "content": "hello"}],
            },
            headers=admin_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "chatcmpl-standalone-test"
        assert data["choices"][0]["message"]["content"] == "Hello!"

    @pytest.mark.asyncio
    async def test_unknown_key_chat_completions_returns_valid_response(self, client):
        """Unknown key → standalone mode (not rejected)."""
        resp = await client.post(
            "/chat/completions",
            json={
                "model": "sglang",
                "messages": [{"role": "user", "content": "hello"}],
            },
            headers={"Authorization": "Bearer unknown-key-12345"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "chatcmpl-standalone-test"

    @pytest.mark.asyncio
    async def test_standalone_chat_no_session_created(self, client):
        """Standalone mode does NOT create a session."""
        resp = await client.post(
            "/chat/completions",
            json={
                "model": "sglang",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert resp.status_code == 200
        store: SessionStore = client._transport.app.state.session_store  # type: ignore[attr-defined]
        assert store.session_count == 0

    @pytest.mark.asyncio
    async def test_standalone_chat_passes_none_cache(self, client, mock_chat_handler):
        """Standalone mode passes areal_cache=None (no caching)."""
        resp = await client.post(
            "/chat/completions",
            json={
                "model": "sglang",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert resp.status_code == 200
        # Verify create was called with areal_cache=None
        mock_chat_handler.create.assert_called_once()
        call_kwargs = mock_chat_handler.create.call_args
        assert call_kwargs.kwargs.get("areal_cache") is None


# =============================================================================
# Standalone /generate (no auth required)
# =============================================================================


class TestStandaloneGenerate:
    @pytest.mark.asyncio
    async def test_generate_no_auth_returns_sse_stream(self, client):
        """No auth header → /generate works fine."""
        resp = await client.post(
            "/generate",
            json={"input_ids": [1, 2, 3]},
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        events = parse_sse_events(resp.content)
        assert len(events) == 3
        assert events[-1]["finished"] is True

    @pytest.mark.asyncio
    async def test_generate_with_admin_key(self, client):
        """Admin key on /generate → accepted (auth is ignored)."""
        resp = await client.post(
            "/generate",
            json={"input_ids": [1, 2, 3]},
            headers=admin_headers(),
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

    @pytest.mark.asyncio
    async def test_generate_with_unknown_key(self, client):
        """Unknown key on /generate → accepted (auth is ignored)."""
        resp = await client.post(
            "/generate",
            json={"input_ids": [1, 2, 3]},
            headers={"Authorization": "Bearer unknown-key-12345"},
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

    @pytest.mark.asyncio
    async def test_generate_with_text_no_auth(self, client, mock_tokenizer):
        """No auth → /generate with text field works."""
        resp = await client.post(
            "/generate",
            json={"text": "What is 2+2?"},
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
            json={
                "model": "sglang",
                "messages": [{"role": "user", "content": "hello"}],
            },
            headers={"Authorization": f"Bearer {session_api_key}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["message"]["content"] == "Hello!"

    @pytest.mark.asyncio
    async def test_session_generate_still_works(self, client):
        """Session key callers can also use /generate (auth ignored anyway)."""
        # Start a session
        resp = await client.post(
            "/rl/start_session",
            json={"task_id": "test-task"},
            headers=admin_headers(),
        )
        assert resp.status_code == 201
        session_api_key = resp.json()["api_key"]

        # Use session key for generate (accepted regardless)
        resp = await client.post(
            "/generate",
            json={"input_ids": [1, 2, 3]},
            headers={"Authorization": f"Bearer {session_api_key}"},
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
