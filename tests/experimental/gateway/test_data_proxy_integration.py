"""Integration tests for data proxy /generate with a real SGLang server.

Requires GPU and a model. Marked @pytest.mark.slow to exclude from default CI.
Run manually:
    uv run pytest tests/experimental/gateway/test_data_proxy_integration.py -v -s

The test launches an SGLang server subprocess, starts the data proxy FastAPI app,
and exercises both the SGLangBackend (non-streaming call to SGLang) and the
full /generate endpoint (SSE streaming response from the data proxy).
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from typing import Any

import httpx
import pytest
import torch

from tests.utils import get_model_path

from areal.api.cli_args import SGLangConfig
from areal.utils import network

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LOCAL_MODEL_PATH = "/storage/openpsi/models/Qwen__Qwen3-0.6B/"
HF_MODEL_ID = "Qwen/Qwen3-0.6B"
SERVER_STARTUP_TIMEOUT = 180  # seconds


def _get_test_model_path() -> str:
    return get_model_path(LOCAL_MODEL_PATH, HF_MODEL_ID)


def _has_gpu() -> bool:
    return torch.cuda.is_available() and torch.cuda.device_count() > 0


def _check_server_health(base_url: str) -> bool:
    try:
        resp = httpx.get(f"{base_url}/health", timeout=10)
        return resp.status_code == 200
    except httpx.HTTPError:
        return False


def _parse_sse_events(content: bytes) -> list[dict[str, Any]]:
    """Parse ``data: {...}`` lines from an SSE byte-stream."""
    events: list[dict[str, Any]] = []
    for line in content.decode().strip().split("\n"):
        line = line.strip()
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))
    return events


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sglang_server():
    """Launch an SGLang server and yield its ``(host, port, base_url)``."""
    if not _has_gpu():
        pytest.skip("GPU required for SGLang server")

    from areal.infra.utils.proc import kill_process_tree

    host = network.gethostip()
    port, dist_port = network.find_free_ports(2)

    cmd = SGLangConfig.build_cmd(
        sglang_config=SGLangConfig(
            skip_tokenizer_init=True,
            model_path=_get_test_model_path(),
            mem_fraction_static=0.3,
        ),
        host=host,
        port=port,
        tp_size=1,
        base_gpu_id=0,
        dist_init_addr=f"{host}:{dist_port}",
    )

    process = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stdout)
    base_url = f"http://{host}:{port}"

    # Wait for server readiness
    t0 = time.time()
    while time.time() - t0 < SERVER_STARTUP_TIMEOUT:
        if _check_server_health(base_url):
            break
        time.sleep(1)

    if time.time() - t0 >= SERVER_STARTUP_TIMEOUT:
        kill_process_tree(process.pid, graceful=True)
        pytest.fail("SGLang server did not become healthy within timeout")

    yield {"host": host, "port": port, "base_url": base_url}

    kill_process_tree(process.pid, graceful=True)


@pytest.fixture(scope="module")
def model_path() -> str:
    return _get_test_model_path()


# ---------------------------------------------------------------------------
# Tests — SGLangBackend (non-streaming call to SGLang directly)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(not _has_gpu(), reason="GPU required")
class TestSGLangBackendIntegration:
    """Test ``SGLangBackend.generate`` directly against a real SGLang server."""

    @pytest.mark.asyncio
    async def test_generate_non_streaming(self, sglang_server, model_path):
        """Call SGLang /generate non-streaming and verify output structure."""
        from areal.experimental.gateway.data_proxy.backend import SGLangBackend
        from areal.experimental.gateway.data_proxy.tokenizer_proxy import (
            TokenizerProxy,
        )

        backend = SGLangBackend(sglang_server["base_url"], request_timeout=60.0)
        tok = TokenizerProxy(model_path)

        input_ids = await tok.tokenize("What is 2+2?")
        assert len(input_ids) > 0

        result = await backend.generate(
            input_ids=input_ids,
            sampling_params={
                "max_new_tokens": 32,
                "temperature": 0.0,
                "top_p": 1.0,
                "skip_special_tokens": False,
            },
        )

        # Structure checks
        assert isinstance(result.output_tokens, list)
        assert isinstance(result.output_logprobs, list)
        assert len(result.output_tokens) == len(result.output_logprobs)
        assert len(result.output_tokens) > 0
        assert result.stop_reason in ("stop", "length")

        # Token IDs should be non-negative integers
        for tid in result.output_tokens:
            assert isinstance(tid, int)
            assert tid >= 0

        # Log-probs should be finite floats (typically <= 0)
        for lp in result.output_logprobs:
            assert isinstance(lp, float)

        # Decode to sanity-check
        decoded = tok.decode_tokens(result.output_tokens)
        assert isinstance(decoded, str)
        assert len(decoded) > 0

    @pytest.mark.asyncio
    async def test_generate_with_stop_token_ids(self, sglang_server, model_path):
        """Verify that stop_token_ids causes early stopping."""
        from areal.experimental.gateway.data_proxy.backend import SGLangBackend
        from areal.experimental.gateway.data_proxy.tokenizer_proxy import (
            TokenizerProxy,
        )

        backend = SGLangBackend(sglang_server["base_url"], request_timeout=60.0)
        tok = TokenizerProxy(model_path)

        input_ids = await tok.tokenize("Hello")

        result = await backend.generate(
            input_ids=input_ids,
            sampling_params={
                "max_new_tokens": 64,
                "temperature": 0.0,
                "stop_token_ids": [tok.eos_token_id],
                "skip_special_tokens": False,
            },
        )

        assert result.stop_reason in ("stop", "length")
        assert isinstance(result.output_tokens, list)


# ---------------------------------------------------------------------------
# Tests — /generate endpoint (SSE streaming through data proxy)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(not _has_gpu(), reason="GPU required")
class TestDataProxyGenerateIntegration:
    """Test the full ``/generate`` endpoint with real SGLang backend."""

    @pytest.mark.asyncio
    async def test_generate_streaming_with_text(self, sglang_server, model_path):
        """POST /generate with ``text`` field -> SSE stream of token chunks."""
        from areal.experimental.gateway.data_proxy.app import create_app
        from areal.experimental.gateway.data_proxy.backend import SGLangBackend
        from areal.experimental.gateway.data_proxy.config import DataProxyConfig
        from areal.experimental.gateway.data_proxy.tokenizer_proxy import (
            TokenizerProxy,
        )

        config = DataProxyConfig(
            host="127.0.0.1",
            port=0,  # not binding
            backend_addr=sglang_server["base_url"],
            tokenizer_path=model_path,
            request_timeout=60.0,
        )
        app = create_app(config)

        # httpx.ASGITransport does not trigger ASGI lifespan events,
        # so we must initialize app.state manually.
        app.state.tokenizer = TokenizerProxy(model_path)
        app.state.backend = SGLangBackend(
            sglang_server["base_url"], request_timeout=60.0
        )
        app.state.config = config

        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            # --- health ---
            health_resp = await client.get("/health")
            assert health_resp.status_code == 200
            assert health_resp.json()["status"] == "ok"

            # --- generate with text ---
            resp = await client.post(
                "/generate",
                json={
                    "text": "What is 2+2?",
                    "sampling_params": {
                        "max_new_tokens": 16,
                        "temperature": 0.0,
                    },
                },
                timeout=60.0,
            )
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers.get("content-type", "")

            events = _parse_sse_events(resp.content)
            assert len(events) > 0

            # Every event has the required fields
            for evt in events:
                assert "token" in evt
                assert "text" in evt
                assert "logprob" in evt
                assert "finished" in evt
                assert isinstance(evt["token"], int)
                assert isinstance(evt["text"], str)
                assert isinstance(evt["logprob"], float)

            # Only last event is finished
            for evt in events[:-1]:
                assert evt["finished"] is False
            assert events[-1]["finished"] is True
            assert events[-1]["stop_reason"] in ("stop", "length")

    @pytest.mark.asyncio
    async def test_generate_streaming_with_input_ids(self, sglang_server, model_path):
        """POST /generate with ``input_ids`` -> SSE stream, no tokenization."""
        from areal.experimental.gateway.data_proxy.app import create_app
        from areal.experimental.gateway.data_proxy.backend import SGLangBackend
        from areal.experimental.gateway.data_proxy.config import DataProxyConfig
        from areal.experimental.gateway.data_proxy.tokenizer_proxy import (
            TokenizerProxy,
        )

        tok = TokenizerProxy(model_path)
        input_ids = await tok.tokenize("Hello world")

        config = DataProxyConfig(
            host="127.0.0.1",
            port=0,
            backend_addr=sglang_server["base_url"],
            tokenizer_path=model_path,
            request_timeout=60.0,
        )
        app = create_app(config)

        app.state.tokenizer = tok
        app.state.backend = SGLangBackend(
            sglang_server["base_url"], request_timeout=60.0
        )
        app.state.config = config

        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            resp = await client.post(
                "/generate",
                json={
                    "input_ids": input_ids,
                    "sampling_params": {
                        "max_new_tokens": 8,
                        "temperature": 0.0,
                    },
                },
                timeout=60.0,
            )
            assert resp.status_code == 200

            events = _parse_sse_events(resp.content)
            assert len(events) > 0
            assert events[-1]["finished"] is True

            # Collect all output token IDs and decode
            output_token_ids = [e["token"] for e in events]
            decoded = tok.decode_tokens(output_token_ids)
            assert isinstance(decoded, str)
            assert len(decoded) > 0

    @pytest.mark.asyncio
    async def test_generate_streaming_token_text_consistency(
        self, sglang_server, model_path
    ):
        """Verify that each SSE chunk's ``text`` matches decoding its ``token``."""
        from areal.experimental.gateway.data_proxy.app import create_app
        from areal.experimental.gateway.data_proxy.backend import SGLangBackend
        from areal.experimental.gateway.data_proxy.config import DataProxyConfig
        from areal.experimental.gateway.data_proxy.tokenizer_proxy import (
            TokenizerProxy,
        )

        tok = TokenizerProxy(model_path)

        config = DataProxyConfig(
            host="127.0.0.1",
            port=0,
            backend_addr=sglang_server["base_url"],
            tokenizer_path=model_path,
            request_timeout=60.0,
        )
        app = create_app(config)

        app.state.tokenizer = tok
        app.state.backend = SGLangBackend(
            sglang_server["base_url"], request_timeout=60.0
        )
        app.state.config = config

        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            resp = await client.post(
                "/generate",
                json={
                    "text": "Count from 1 to 5:",
                    "sampling_params": {
                        "max_new_tokens": 32,
                        "temperature": 0.0,
                    },
                },
                timeout=60.0,
            )
            assert resp.status_code == 200
            events = _parse_sse_events(resp.content)

            # Each chunk's text must equal decode_token(chunk's token)
            for evt in events:
                expected_text = tok.decode_token(evt["token"])
                assert evt["text"] == expected_text, (
                    f"Token {evt['token']}: expected {expected_text!r}, "
                    f"got {evt['text']!r}"
                )


# ---------------------------------------------------------------------------
# Tests — /chat/completions endpoint (full session lifecycle)
# ---------------------------------------------------------------------------


def _create_data_proxy_app_with_sessions(sglang_server, model_path):
    """Create a fully-wired data proxy app with session support."""
    from areal.experimental.gateway.data_proxy.app import create_app
    from areal.experimental.gateway.data_proxy.backend import SGLangBackend
    from areal.experimental.gateway.data_proxy.chat import ChatCompletionHandler
    from areal.experimental.gateway.data_proxy.config import DataProxyConfig
    from areal.experimental.gateway.data_proxy.session import SessionStore
    from areal.experimental.gateway.data_proxy.tokenizer_proxy import TokenizerProxy

    config = DataProxyConfig(
        host="127.0.0.1",
        port=0,
        backend_addr=sglang_server["base_url"],
        tokenizer_path=model_path,
        request_timeout=60.0,
    )
    app = create_app(config)

    # httpx.ASGITransport does not trigger ASGI lifespan events,
    # so we must initialize app.state manually.
    tok = TokenizerProxy(model_path)
    backend = SGLangBackend(sglang_server["base_url"], request_timeout=60.0)
    store = SessionStore()

    app.state.tokenizer = tok
    app.state.backend = backend
    app.state.config = config
    app.state.session_store = store
    app.state.chat_handler = ChatCompletionHandler(backend, tok)

    return app, store


ADMIN_KEY = "areal-admin-key"


@pytest.mark.slow
@pytest.mark.skipif(not _has_gpu(), reason="GPU required")
class TestChatCompletionsIntegration:
    """Test the full /chat/completions endpoint with a real SGLang backend."""

    @pytest.mark.asyncio
    async def test_non_streaming_chat_completion(self, sglang_server, model_path):
        """Full lifecycle: start_session → POST /chat/completions → end_session."""
        app, store = _create_data_proxy_app_with_sessions(sglang_server, model_path)

        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            # --- start session ---
            resp = await client.post(
                "/rl/start_session",
                json={"task_id": "integ-chat-ns"},
                headers={"Authorization": f"Bearer {ADMIN_KEY}"},
                timeout=10.0,
            )
            assert resp.status_code == 201, resp.text
            session = resp.json()
            session_api_key = session["api_key"]
            session_id = session["session_id"]

            # --- non-streaming chat completion ---
            resp = await client.post(
                "/chat/completions",
                json={
                    "model": "sglang",
                    "messages": [{"role": "user", "content": "What is 2+2?"}],
                    "max_completion_tokens": 64,
                    "temperature": 0.0,
                    "stream": False,
                },
                headers={"Authorization": f"Bearer {session_api_key}"},
                timeout=60.0,
            )
            assert resp.status_code == 200, resp.text
            completion = resp.json()

            # Validate OpenAI-compatible structure
            assert completion["object"] == "chat.completion"
            assert "id" in completion
            assert "choices" in completion
            assert len(completion["choices"]) == 1

            choice = completion["choices"][0]
            assert "message" in choice
            assert choice["message"]["role"] == "assistant"
            assert isinstance(choice["message"]["content"], str)
            assert len(choice["message"]["content"]) > 0
            assert choice["finish_reason"] in ("stop", "length")

            # Validate usage
            assert "usage" in completion
            usage = completion["usage"]
            assert usage["prompt_tokens"] > 0
            assert usage["completion_tokens"] > 0
            assert usage["total_tokens"] == (
                usage["prompt_tokens"] + usage["completion_tokens"]
            )

            # --- end session ---
            resp = await client.post(
                "/rl/end_session",
                headers={"Authorization": f"Bearer {session_api_key}"},
                timeout=10.0,
            )
            assert resp.status_code == 200, resp.text
            end_data = resp.json()
            assert end_data["interaction_count"] == 1

    @pytest.mark.asyncio
    async def test_streaming_chat_completion(self, sglang_server, model_path):
        """Streaming: start_session → POST /chat/completions stream=True → SSE chunks."""
        app, store = _create_data_proxy_app_with_sessions(sglang_server, model_path)

        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            # --- start session ---
            resp = await client.post(
                "/rl/start_session",
                json={"task_id": "integ-chat-stream"},
                headers={"Authorization": f"Bearer {ADMIN_KEY}"},
                timeout=10.0,
            )
            assert resp.status_code == 201, resp.text
            session_api_key = resp.json()["api_key"]

            # --- streaming chat completion ---
            resp = await client.post(
                "/chat/completions",
                json={
                    "model": "sglang",
                    "messages": [{"role": "user", "content": "Say hello"}],
                    "max_completion_tokens": 32,
                    "temperature": 0.0,
                    "stream": True,
                },
                headers={"Authorization": f"Bearer {session_api_key}"},
                timeout=60.0,
            )
            assert resp.status_code == 200, resp.text
            assert "text/event-stream" in resp.headers.get("content-type", "")

            # Parse SSE events
            chunks = []
            for line in resp.content.decode().strip().split("\n"):
                line = line.strip()
                if line.startswith("data: "):
                    payload = line[6:]
                    if payload == "[DONE]":
                        break
                    chunks.append(json.loads(payload))

            assert len(chunks) >= 2  # At least role chunk + finish chunk

            # First chunk: role
            first = chunks[0]
            assert first["object"] == "chat.completion.chunk"
            assert first["choices"][0]["delta"].get("role") == "assistant"

            # Last chunk: finish_reason
            last = chunks[-1]
            assert last["choices"][0]["finish_reason"] in ("stop", "length")

            # At least one chunk has content
            content_chunks = [
                c for c in chunks
                if c["choices"][0]["delta"].get("content")
            ]
            assert len(content_chunks) >= 1

            # All chunks share the same completion ID
            ids = {c["id"] for c in chunks}
            assert len(ids) == 1

            # --- end session ---
            resp = await client.post(
                "/rl/end_session",
                headers={"Authorization": f"Bearer {session_api_key}"},
                timeout=10.0,
            )
            assert resp.status_code == 200, resp.text

    @pytest.mark.asyncio
    async def test_multi_turn_chat_completion(self, sglang_server, model_path):
        """Multi-turn: two sequential chat completions in the same session."""
        app, store = _create_data_proxy_app_with_sessions(sglang_server, model_path)

        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            # --- start session ---
            resp = await client.post(
                "/rl/start_session",
                json={"task_id": "integ-chat-multi"},
                headers={"Authorization": f"Bearer {ADMIN_KEY}"},
                timeout=10.0,
            )
            assert resp.status_code == 201, resp.text
            session_api_key = resp.json()["api_key"]

            # --- first turn ---
            resp1 = await client.post(
                "/chat/completions",
                json={
                    "model": "sglang",
                    "messages": [{"role": "user", "content": "What is 3+5?"}],
                    "max_completion_tokens": 64,
                    "temperature": 0.0,
                },
                headers={"Authorization": f"Bearer {session_api_key}"},
                timeout=60.0,
            )
            assert resp1.status_code == 200, resp1.text
            turn1 = resp1.json()
            turn1_content = turn1["choices"][0]["message"]["content"]
            assert isinstance(turn1_content, str)
            assert len(turn1_content) > 0

            # --- second turn (builds on first) ---
            resp2 = await client.post(
                "/chat/completions",
                json={
                    "model": "sglang",
                    "messages": [
                        {"role": "user", "content": "What is 3+5?"},
                        {"role": "assistant", "content": turn1_content},
                        {"role": "user", "content": "Now add 2 to that result."},
                    ],
                    "max_completion_tokens": 64,
                    "temperature": 0.0,
                },
                headers={"Authorization": f"Bearer {session_api_key}"},
                timeout=60.0,
            )
            assert resp2.status_code == 200, resp2.text
            turn2 = resp2.json()
            assert turn2["object"] == "chat.completion"
            assert len(turn2["choices"][0]["message"]["content"]) > 0

            # Both completions should have different IDs
            assert turn1["id"] != turn2["id"]

            # --- end session ---
            resp = await client.post(
                "/rl/end_session",
                headers={"Authorization": f"Bearer {session_api_key}"},
                timeout=10.0,
            )
            assert resp.status_code == 200, resp.text
            assert resp.json()["interaction_count"] == 2

    @pytest.mark.asyncio
    async def test_set_reward_and_export(self, sglang_server, model_path):
        """Full lifecycle: start → chat → set_reward → end → export_trajectories."""
        app, store = _create_data_proxy_app_with_sessions(sglang_server, model_path)

        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            # --- start session ---
            resp = await client.post(
                "/rl/start_session",
                json={"task_id": "integ-chat-reward"},
                headers={"Authorization": f"Bearer {ADMIN_KEY}"},
                timeout=10.0,
            )
            assert resp.status_code == 201, resp.text
            session = resp.json()
            session_api_key = session["api_key"]
            session_id = session["session_id"]

            # --- chat completion ---
            resp = await client.post(
                "/chat/completions",
                json={
                    "model": "sglang",
                    "messages": [{"role": "user", "content": "What is 10-3?"}],
                    "max_completion_tokens": 64,
                    "temperature": 0.0,
                },
                headers={"Authorization": f"Bearer {session_api_key}"},
                timeout=60.0,
            )
            assert resp.status_code == 200, resp.text

            # --- set reward ---
            resp = await client.post(
                "/rl/set_reward",
                json={"reward": 1.0},
                headers={"Authorization": f"Bearer {session_api_key}"},
                timeout=10.0,
            )
            assert resp.status_code == 200, resp.text
            assert resp.json()["message"] == "success"

            # --- end session ---
            resp = await client.post(
                "/rl/end_session",
                headers={"Authorization": f"Bearer {session_api_key}"},
                timeout=10.0,
            )
            assert resp.status_code == 200, resp.text
            assert resp.json()["interaction_count"] == 1

            # --- export trajectories ---
            resp = await client.post(
                "/export_trajectories",
                json={"session_id": session_id},
                headers={"Authorization": f"Bearer {ADMIN_KEY}"},
                timeout=10.0,
            )
            assert resp.status_code == 200, resp.text
            export_data = resp.json()
            assert "interactions" in export_data
            interactions = export_data["interactions"]
            assert len(interactions) == 1

            # Each exported interaction should have tensor_dict, reward, interaction_id
            for key, item in interactions.items():
                assert "tensor_dict" in item
                assert "reward" in item
                assert item["reward"] == 1.0
                assert "interaction_id" in item

    @pytest.mark.asyncio
    async def test_auth_rejection(self, sglang_server, model_path):
        """Verify that invalid API keys are rejected."""
        app, store = _create_data_proxy_app_with_sessions(sglang_server, model_path)

        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            # Missing auth header on start_session
            resp = await client.post(
                "/rl/start_session",
                json={"task_id": "no-auth"},
                timeout=10.0,
            )
            assert resp.status_code == 401

            # Wrong admin key on start_session
            resp = await client.post(
                "/rl/start_session",
                json={"task_id": "wrong-key"},
                headers={"Authorization": "Bearer wrong-key"},
                timeout=10.0,
            )
            assert resp.status_code == 403

            # Wrong session key on /chat/completions
            resp = await client.post(
                "/chat/completions",
                json={
                    "model": "sglang",
                    "messages": [{"role": "user", "content": "hi"}],
                },
                headers={"Authorization": "Bearer fake-session-key"},
                timeout=10.0,
            )
            assert resp.status_code == 401
