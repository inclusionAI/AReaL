"""Unit tests for data proxy chat/session endpoints (Plan 3b)."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import pytest_asyncio

from areal.experimental.inference_service.data_proxy.app import create_app
from areal.experimental.inference_service.data_proxy.config import DataProxyConfig
from areal.experimental.inference_service.data_proxy.session import (
    SessionData,
    SessionStore,
)


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
    # Expose underlying _tok for ModelResponse.output_tokens_without_stop
    tok._tok = MagicMock()
    tok._tok.eos_token_id = 2
    tok._tok.pad_token_id = 0
    return tok


@pytest.fixture
def mock_areal_client():
    """Mock ArealOpenAI client that returns a valid ChatCompletion.
    Also stores the interaction in the session's InteractionCache.

    The mock has `.chat.completions.create()` as an AsyncMock to match
    the ArealOpenAI interface used by the data proxy app.
    """
    from openai.types.chat import ChatCompletion, ChatCompletionMessage
    from openai.types.chat.chat_completion import Choice
    from openai.types.completion_usage import CompletionUsage

    from areal.experimental.openai.types import InteractionWithTokenLogpReward

    mock_client = MagicMock()

    completion = ChatCompletion(
        id="chatcmpl-test123",
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
        """Mock create that stores the interaction in session cache via areal_cache."""
        import torch

        messages = kwargs.get("messages", [])

        interaction = InteractionWithTokenLogpReward(
            messages=messages if isinstance(messages, list) else list(messages),
            completion=completion,
            output_message_list=[{"role": "assistant", "content": "Hello!"}],
        )
        # Pre-populate _cache so to_tensor_dict() works without ModelResponse
        interaction._cache = {
            "input_ids": torch.tensor([[100, 200, 300, 1234, 5678, 2]]),
            "loss_mask": torch.tensor([[0, 0, 0, 1, 1, 1]]),
            "logprobs": torch.tensor([[0.0, 0.0, 0.0, -0.5, -0.3, -0.1]]),
            "versions": torch.tensor([[-1, -1, -1, 0, 0, 0]]),
            "attention_mask": torch.ones(6, dtype=torch.bool).unsqueeze(0),
            "rewards": torch.tensor([0.0]),
        }
        if areal_cache is not None:
            areal_cache[completion.id] = interaction
        return completion

    mock_client.chat.completions.create = AsyncMock(side_effect=_mock_create)
    return mock_client


@pytest_asyncio.fixture
async def client(config, mock_tokenizer, mock_areal_client):
    """Create app with mocked deps and yield an httpx async client."""
    from areal.experimental.inference_service.data_proxy.backend import SGLangBridgeBackend
    from areal.experimental.inference_service.data_proxy.inf_bridge import InfBridge
    from areal.experimental.inference_service.data_proxy.pause import PauseState

    app = create_app(config)
    # Bypass lifespan — inject mocks directly into app.state
    pause_state = PauseState()
    inf_bridge = InfBridge(
        backend=SGLangBridgeBackend(),
        backend_addr=config.backend_addr,
        pause_state=pause_state,
        request_timeout=config.request_timeout,
        max_resubmit_retries=5,
        resubmit_wait=0.01,
    )
    app.state.tokenizer = mock_tokenizer
    app.state.inf_bridge = inf_bridge
    app.state.areal_client = mock_areal_client
    app.state.pause_state = pause_state
    app.state.config = config
    store = SessionStore()
    app.state.session_store = store
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def admin_headers():
    return {"Authorization": f"Bearer {ADMIN_KEY}"}


def session_headers(api_key: str):
    return {"Authorization": f"Bearer {api_key}"}


# =============================================================================
# SessionStore unit tests
# =============================================================================


class TestSessionStore:
    def test_start_session_returns_ids(self):
        store = SessionStore()
        session_id, api_key = store.start_session("task-1")
        assert session_id == "task-1-0"
        assert isinstance(api_key, str)
        assert len(api_key) > 0

    def test_get_session_by_api_key(self):
        store = SessionStore()
        session_id, api_key = store.start_session("task-1")
        session = store.get_session_by_api_key(api_key)
        assert session is not None
        assert session.session_id == session_id

    def test_get_session_by_api_key_not_found(self):
        store = SessionStore()
        assert store.get_session_by_api_key("nonexistent") is None

    def test_end_session(self):
        store = SessionStore()
        session_id, _ = store.start_session("task-1")
        count = store.end_session(session_id)
        assert count == 0  # no interactions yet

    def test_end_session_not_found(self):
        store = SessionStore()
        with pytest.raises(KeyError):
            store.end_session("nonexistent")

    def test_session_count(self):
        store = SessionStore()
        assert store.session_count == 0
        store.start_session("task-1")
        assert store.session_count == 1
        store.start_session("task-2")
        assert store.session_count == 2

    def test_remove_session(self):
        store = SessionStore()
        session_id, api_key = store.start_session("task-1")
        store.remove_session(session_id)
        assert store.get_session(session_id) is None
        assert store.get_session_by_api_key(api_key) is None

    def test_duplicate_session_ids_increment(self):
        store = SessionStore()
        sid1, _ = store.start_session("task-1")
        sid2, _ = store.start_session("task-1")
        assert sid1 == "task-1-0"
        assert sid2 == "task-1-1"


# =============================================================================
# Endpoint tests: /rl/start_session
# =============================================================================


@pytest.mark.asyncio
async def test_start_session_with_admin_key(client):
    resp = await client.post(
        "/rl/start_session",
        json={"task_id": "test-task"},
        headers=admin_headers(),
    )
    assert resp.status_code == 201
    data = resp.json()
    assert "session_id" in data
    assert "api_key" in data
    assert data["session_id"].startswith("test-task-")


@pytest.mark.asyncio
async def test_start_session_without_admin_key(client):
    resp = await client.post(
        "/rl/start_session",
        json={"task_id": "test-task"},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_start_session_wrong_admin_key(client):
    resp = await client.post(
        "/rl/start_session",
        json={"task_id": "test-task"},
        headers={"Authorization": "Bearer wrong-key"},
    )
    assert resp.status_code == 403


# =============================================================================
# Endpoint tests: /chat/completions
# =============================================================================


@pytest.mark.asyncio
async def test_chat_completions_with_session_key(client, mock_areal_client):
    # Start session first
    resp = await client.post(
        "/rl/start_session",
        json={"task_id": "chat-test"},
        headers=admin_headers(),
    )
    api_key = resp.json()["api_key"]

    # Call chat/completions (OpenAI-compatible format)
    resp = await client.post(
        "/chat/completions",
        json={
            "model": "sglang",
            "messages": [{"role": "user", "content": "hi"}],
        },
        headers=session_headers(api_key),
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["content"] == "Hello!"
    mock_areal_client.chat.completions.create.assert_called_once()


@pytest.mark.asyncio
async def test_chat_completions_without_session_key(client):
    resp = await client.post(
        "/chat/completions",
        json={
            "model": "sglang",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_chat_completions_with_invalid_key(client):
    resp = await client.post(
        "/chat/completions",
        json={
            "model": "sglang",
            "messages": [{"role": "user", "content": "hi"}],
        },
        headers={"Authorization": "Bearer fake-key"},
    )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_chat_completions_passes_sampling_params(client, mock_areal_client):
    # Start session
    resp = await client.post(
        "/rl/start_session",
        json={"task_id": "sp-test"},
        headers=admin_headers(),
    )
    api_key = resp.json()["api_key"]

    # Call with explicit sampling params (OpenAI-compatible format)
    resp = await client.post(
        "/chat/completions",
        json={
            "model": "sglang",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.5,
            "top_p": 0.9,
            "max_tokens": 100,
        },
        headers=session_headers(api_key),
    )
    assert resp.status_code == 200

    # Check that params were passed to areal_client.chat.completions.create
    call_kwargs = mock_areal_client.chat.completions.create.call_args
    kw = call_kwargs.kwargs if call_kwargs.kwargs else call_kwargs[1]
    assert kw["temperature"] == 0.5
    assert kw["top_p"] == 0.9
    assert kw["max_tokens"] == 100


# =============================================================================
# Endpoint tests: /rl/set_reward
# =============================================================================


@pytest.mark.asyncio
async def test_set_reward_success(client):
    # Start session
    resp = await client.post(
        "/rl/start_session",
        json={"task_id": "reward-test"},
        headers=admin_headers(),
    )
    api_key = resp.json()["api_key"]

    # Do a chat completion to create an interaction
    resp = await client.post(
        "/chat/completions",
        json={
            "model": "sglang",
            "messages": [{"role": "user", "content": "hi"}],
        },
        headers=session_headers(api_key),
    )
    assert resp.status_code == 200

    # Set reward (interaction_id=None → last interaction)
    resp = await client.post(
        "/rl/set_reward",
        json={"reward": 1.0},
        headers=session_headers(api_key),
    )
    assert resp.status_code == 200
    assert resp.json()["message"] == "success"


@pytest.mark.asyncio
async def test_set_reward_no_interactions(client):
    # Start session
    resp = await client.post(
        "/rl/start_session",
        json={"task_id": "reward-empty"},
        headers=admin_headers(),
    )
    api_key = resp.json()["api_key"]

    # Set reward with no interactions
    resp = await client.post(
        "/rl/set_reward",
        json={"reward": 1.0},
        headers=session_headers(api_key),
    )
    assert resp.status_code == 400
    assert "No interactions" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_set_reward_without_session_key(client):
    resp = await client.post(
        "/rl/set_reward",
        json={"reward": 1.0},
    )
    assert resp.status_code == 401


# =============================================================================
# Endpoint tests: /rl/end_session
# =============================================================================


@pytest.mark.asyncio
async def test_end_session_success(client):
    # Start session
    resp = await client.post(
        "/rl/start_session",
        json={"task_id": "end-test"},
        headers=admin_headers(),
    )
    api_key = resp.json()["api_key"]

    # End session
    resp = await client.post(
        "/rl/end_session",
        headers=session_headers(api_key),
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["message"] == "success"
    assert "interaction_count" in data


@pytest.mark.asyncio
async def test_end_session_without_key(client):
    resp = await client.post("/rl/end_session")
    assert resp.status_code == 401


# =============================================================================
# Endpoint tests: /export_trajectories
# =============================================================================


@pytest.mark.asyncio
async def test_export_trajectories_after_end_session(client):
    """Full lifecycle: start → chat → set_reward → end → export."""
    # Start session
    resp = await client.post(
        "/rl/start_session",
        json={"task_id": "export-test"},
        headers=admin_headers(),
    )
    session_id = resp.json()["session_id"]
    api_key = resp.json()["api_key"]

    # Chat completion
    resp = await client.post(
        "/chat/completions",
        json={
            "model": "sglang",
            "messages": [{"role": "user", "content": "hi"}],
        },
        headers=session_headers(api_key),
    )
    assert resp.status_code == 200

    # Set reward
    resp = await client.post(
        "/rl/set_reward",
        json={"reward": 1.0},
        headers=session_headers(api_key),
    )
    assert resp.status_code == 200

    # End session
    resp = await client.post(
        "/rl/end_session",
        headers=session_headers(api_key),
    )
    assert resp.status_code == 200

    # Export trajectories
    resp = await client.post(
        "/export_trajectories",
        json={"session_id": session_id, "discount": 1.0, "style": "individual"},
        headers=admin_headers(),
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "interactions" in data


@pytest.mark.asyncio
async def test_export_trajectories_not_found(client):
    resp = await client.post(
        "/export_trajectories",
        json={"session_id": "nonexistent", "discount": 1.0, "style": "individual"},
        headers=admin_headers(),
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_export_trajectories_without_admin_key(client):
    resp = await client.post(
        "/export_trajectories",
        json={"session_id": "x", "discount": 1.0, "style": "individual"},
    )
    assert resp.status_code == 401


# =============================================================================
# Endpoint tests: /health (updated with sessions count)
# =============================================================================


@pytest.mark.asyncio
async def test_health_includes_sessions(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "sessions" in data
    assert data["sessions"] == 0


@pytest.mark.asyncio
async def test_health_sessions_count_after_start(client):
    # Start a session
    await client.post(
        "/rl/start_session",
        json={"task_id": "health-test"},
        headers=admin_headers(),
    )
    resp = await client.get("/health")
    assert resp.json()["sessions"] == 1


# =============================================================================
# Full lifecycle test
# =============================================================================


@pytest.mark.asyncio
async def test_full_session_lifecycle(client, mock_areal_client):
    """Test the complete flow: start → chat → reward → end → export."""
    # 1. Start session
    resp = await client.post(
        "/rl/start_session",
        json={"task_id": "lifecycle"},
        headers=admin_headers(),
    )
    assert resp.status_code == 201
    session_id = resp.json()["session_id"]
    api_key = resp.json()["api_key"]

    # 2. Chat completion
    resp = await client.post(
        "/chat/completions",
        json={
            "model": "sglang",
            "messages": [{"role": "user", "content": "What is 2+2?"}],
        },
        headers=session_headers(api_key),
    )
    assert resp.status_code == 200
    assert resp.json()["object"] == "chat.completion"

    # 3. Set reward
    resp = await client.post(
        "/rl/set_reward",
        json={"reward": 1.0},
        headers=session_headers(api_key),
    )
    assert resp.status_code == 200

    # 4. End session
    resp = await client.post(
        "/rl/end_session",
        headers=session_headers(api_key),
    )
    assert resp.status_code == 200
    assert resp.json()["interaction_count"] == 1

    # 5. Export trajectories
    resp = await client.post(
        "/export_trajectories",
        json={"session_id": session_id, "discount": 1.0, "style": "individual"},
        headers=admin_headers(),
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "interactions" in data
