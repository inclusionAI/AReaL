"""Unit tests for data proxy /generate endpoint (Plan 3a)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import pytest_asyncio

from areal.experimental.gateway.data_proxy.app import create_app
from areal.experimental.gateway.data_proxy.config import DataProxyConfig


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
    tok.eos_token_id = 2
    tok.pad_token_id = 0
    return tok


@pytest_asyncio.fixture
async def client(config, mock_tokenizer):
    """Create app with mocked deps and yield an httpx async client.
    The httpx ASGITransport does not run the ASGI lifespan protocol,
    so we set app.state directly with the mocked deps.
    """
    from areal.experimental.gateway.data_proxy.backend import (
        GenerationResult,
        SGLangBackend,
    )
    from areal.experimental.gateway.data_proxy.pause import PauseState
    from areal.experimental.gateway.data_proxy.session import SessionStore

    app = create_app(config)
    # Bypass lifespan — inject mocks directly into app.state
    pause_state = PauseState()
    backend = SGLangBackend(
        backend_addr=config.backend_addr,
        pause_state=pause_state,
        request_timeout=config.request_timeout,
        max_resubmit_retries=config.max_resubmit_retries,
        resubmit_wait=0.01,
    )
    # Mock _call_sglang to return a canned result
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
    app.state.chat_handler = MagicMock()
    transport = httpx.ASGITransport(app=app)
    # No auth header needed — /generate has no auth
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def parse_sse_events(content: bytes) -> list[dict]:
    """Parse SSE 'data: {...}' lines from response bytes."""
    events = []
    for line in content.decode().strip().split("\n"):
        line = line.strip()
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))
    return events


@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["backend"] == "http://mock-sglang:30000"


@pytest.mark.asyncio
async def test_generate_with_text(client, mock_tokenizer):
    resp = await client.post(
        "/generate",
        json={"text": "What is 2+2?", "sampling_params": {"max_new_tokens": 10}},
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    events = parse_sse_events(resp.content)
    assert len(events) == 3

    # First event
    assert events[0]["token"] == 1234
    assert events[0]["text"] == "tok_1234"
    assert events[0]["logprob"] == -0.5
    assert events[0]["finished"] is False

    # Last event has finished + stop_reason
    assert events[2]["token"] == 2
    assert events[2]["finished"] is True
    assert events[2]["stop_reason"] == "stop"

    # Verify tokenizer was called
    mock_tokenizer.tokenize.assert_called_once_with("What is 2+2?")


@pytest.mark.asyncio
async def test_generate_with_input_ids(client, mock_tokenizer):
    resp = await client.post(
        "/generate",
        json={"input_ids": [1, 2, 3]},
    )
    assert resp.status_code == 200
    events = parse_sse_events(resp.content)
    assert len(events) == 3

    # tokenize should NOT be called when input_ids provided
    mock_tokenizer.tokenize.assert_not_called()

    # _call_sglang should be called with the provided input_ids
    backend = client._transport.app.state.backend
    backend._call_sglang.assert_called_once()
    call_args = backend._call_sglang.call_args
    assert call_args[0][0] == [1, 2, 3]  # input_ids


@pytest.mark.asyncio
async def test_generate_missing_input(client):
    resp = await client.post("/generate", json={})
    assert resp.status_code == 400
    assert (
        "text" in resp.json()["detail"].lower()
        or "input_ids" in resp.json()["detail"].lower()
    )


@pytest.mark.asyncio
async def test_generate_default_sampling_params(client):
    resp = await client.post("/generate", json={"input_ids": [1, 2, 3]})
    assert resp.status_code == 200

    backend = client._transport.app.state.backend
    call_args = backend._call_sglang.call_args
    params = call_args[0][1]  # sampling_params
    assert params["max_new_tokens"] == 512
    assert params["temperature"] == 1.0
    assert params["top_p"] == 1.0
    assert params["skip_special_tokens"] is False


@pytest.mark.asyncio
async def test_generate_custom_sampling_params_override(client):
    resp = await client.post(
        "/generate",
        json={
            "input_ids": [1, 2, 3],
            "sampling_params": {"max_new_tokens": 100, "temperature": 0.7},
        },
    )
    assert resp.status_code == 200

    backend = client._transport.app.state.backend
    call_args = backend._call_sglang.call_args
    params = call_args[0][1]
    assert params["max_new_tokens"] == 100
    assert params["temperature"] == 0.7
    assert params["top_p"] == 1.0  # default preserved


@pytest.mark.asyncio
async def test_generate_abort_stop_reason(client):
    """With SGLangBackend, a single abort followed by stop
    is handled transparently. The abort is never exposed to the caller.
    """
    from areal.experimental.gateway.data_proxy.backend import GenerationResult

    call_count = 0
    original_result = GenerationResult(
        output_tokens=[1234],
        output_logprobs=[-0.5],
        stop_reason="abort",
    )
    stop_result = GenerationResult(
        output_tokens=[5678],
        output_logprobs=[-0.3],
        stop_reason="stop",
    )

    async def mock_call_sglang(input_ids, params):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return original_result
        return stop_result

    backend = client._transport.app.state.backend
    backend._call_sglang = mock_call_sglang

    resp = await client.post("/generate", json={"input_ids": [1, 2, 3]})
    assert resp.status_code == 200
    events = parse_sse_events(resp.content)
    # Resubmit loop: abort(1 token) + stop(1 token) = 2 tokens total
    assert len(events) == 2
    assert events[-1]["finished"] is True
    assert events[-1]["stop_reason"] == "stop"  # abort converted
