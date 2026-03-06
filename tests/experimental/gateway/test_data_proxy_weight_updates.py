"""Tests for data proxy weight update forwarding endpoints."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from areal.experimental.gateway.data_proxy.app import create_app
from areal.experimental.gateway.data_proxy.config import DataProxyConfig


@pytest.fixture
def config():
    return DataProxyConfig(
        backend_addr="http://mock-sglang:30000",
        tokenizer_path="",
        admin_api_key="test-admin-key",
    )


@pytest_asyncio.fixture
async def client(config):
    from areal.experimental.gateway.data_proxy.backend import SGLangBackend
    from areal.experimental.gateway.data_proxy.chat import ChatCompletionHandler
    from areal.experimental.gateway.data_proxy.pause import PauseState
    from areal.experimental.gateway.data_proxy.session import SessionStore

    app = create_app(config)
    pause_state = PauseState()
    backend = SGLangBackend(
        backend_addr=config.backend_addr,
        pause_state=pause_state,
        request_timeout=config.request_timeout,
    )
    tok = MagicMock()
    tok.eos_token_id = 2
    app.state.tokenizer = tok
    app.state.backend = backend
    app.state.pause_state = pause_state
    app.state.config = config
    app.state.session_store = SessionStore()
    app.state.session_store.set_admin_key(config.admin_api_key)
    app.state.chat_handler = ChatCompletionHandler(backend, tok)

    from httpx import ASGITransport, AsyncClient

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def _make_mock_session(resp_status=200, resp_json=None):
    resp_json = resp_json or {"status": "ok"}
    mock_resp = AsyncMock()
    mock_resp.status = resp_status
    mock_resp.json = AsyncMock(return_value=resp_json)
    mock_post_cm = AsyncMock()
    mock_post_cm.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_post_cm.__aexit__ = AsyncMock(return_value=False)
    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=mock_post_cm)
    mock_session_cm = AsyncMock()
    mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session_cm.__aexit__ = AsyncMock(return_value=False)
    return mock_session_cm, mock_session


_PATCH = "areal.experimental.gateway.data_proxy.weight_update.aiohttp.ClientSession"


class TestUpdateWeightsFromDisk:
    @pytest.mark.asyncio
    async def test_forwards_to_sglang(self, client):
        session_cm, mock_session = _make_mock_session(200, {"status": "ok"})
        with patch(_PATCH, return_value=session_cm):
            resp = await client.post(
                "/update_weights_from_disk",
                json={"path": "/tmp/weights", "type": "disk"},
            )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        call_url = mock_session.post.call_args[0][0]
        assert call_url == "http://mock-sglang:30000/update_weights_from_disk"

    @pytest.mark.asyncio
    async def test_propagates_backend_error(self, client):
        session_cm, _ = _make_mock_session(500, {"error": "internal error"})
        with patch(_PATCH, return_value=session_cm):
            resp = await client.post(
                "/update_weights_from_disk",
                json={"path": "/tmp/weights"},
            )
        assert resp.status_code == 500


class TestUpdateWeightsFromDistributed:
    @pytest.mark.asyncio
    async def test_forwards_to_sglang(self, client):
        session_cm, mock_session = _make_mock_session(200, {"status": "ok"})
        with patch(_PATCH, return_value=session_cm):
            resp = await client.post(
                "/update_weights_from_distributed",
                json={"meta": {"type": "xccl"}, "param_specs": []},
            )
        assert resp.status_code == 200
        call_url = mock_session.post.call_args[0][0]
        assert call_url == "http://mock-sglang:30000/update_weights_from_distributed"


class TestInitWeightsUpdateGroup:
    @pytest.mark.asyncio
    async def test_forwards_to_sglang(self, client):
        session_cm, _ = _make_mock_session(200, {"status": "ok"})
        with patch(_PATCH, return_value=session_cm):
            resp = await client.post(
                "/init_weights_update_group",
                json={"meta": {"type": "xccl"}},
            )
        assert resp.status_code == 200


class TestSetVersion:
    @pytest.mark.asyncio
    async def test_forwards_to_sglang(self, client):
        session_cm, _ = _make_mock_session(200, {"status": "ok"})
        with patch(_PATCH, return_value=session_cm):
            resp = await client.post("/set_version", json={"version": 42})
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_forwards_correct_url(self, client):
        session_cm, mock_session = _make_mock_session(200, {"status": "ok"})
        with patch(_PATCH, return_value=session_cm):
            await client.post("/set_version", json={"version": 1})
        call_url = mock_session.post.call_args[0][0]
        assert call_url == "http://mock-sglang:30000/set_version"


class TestBackendUnreachable:
    @pytest.mark.asyncio
    async def test_returns_502_on_connection_error(self, client):
        import aiohttp as _aiohttp

        mock_session = MagicMock()
        mock_session.post = MagicMock(
            side_effect=_aiohttp.ClientError("Connection refused")
        )
        mock_session_cm = AsyncMock()
        mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_cm.__aexit__ = AsyncMock(return_value=False)

        with patch(_PATCH, return_value=mock_session_cm):
            resp = await client.post(
                "/update_weights_from_disk",
                json={"path": "/tmp/test"},
            )
        assert resp.status_code == 502
