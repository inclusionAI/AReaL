from __future__ import annotations

import httpx
import pytest

from areal.api.io_struct import HttpRequest
from areal.experimental.inference_service.data_proxy.app import (
    _post_online_ready_callback,
)
from areal.experimental.inference_service.data_proxy.app import (
    create_app as create_data_proxy_app,
)
from areal.experimental.inference_service.data_proxy.backend import SGLangBridgeBackend
from areal.experimental.inference_service.data_proxy.config import DataProxyConfig
from areal.experimental.inference_service.data_proxy.inf_bridge import InfBridge
from areal.experimental.inference_service.data_proxy.pause import PauseState
from areal.experimental.inference_service.data_proxy.session import ReadyNotification
from areal.experimental.inference_service.gateway.app import (
    create_app as create_gateway_app,
)
from areal.experimental.inference_service.gateway.config import GatewayConfig
from areal.experimental.inference_service.gateway.streaming import (
    grant_capacity_in_router,
    query_router,
)
from areal.experimental.inference_service.router.app import (
    create_app as create_router_app,
)
from areal.experimental.inference_service.router.config import RouterConfig


@pytest.mark.asyncio
async def test_gateway_streaming_helpers_accept_shared_client() -> None:
    calls: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        if request.url.path == "/route":
            return httpx.Response(200, json={"worker_addr": "http://worker-1"})
        if request.url.path == "/grant_capacity":
            return httpx.Response(200, json={"capacity": 1})
        raise AssertionError(f"unexpected path: {request.url.path}")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        worker_addr = await query_router(
            "http://router",
            api_key="session-key",
            path="/chat/completions",
            client=client,
        )
        capacity = await grant_capacity_in_router(
            "http://router",
            "admin-key",
            2.0,
            client=client,
        )

    assert worker_addr == "http://worker-1"
    assert capacity == {"capacity": 1}
    assert calls == [
        ("POST", "/route"),
        ("POST", "/grant_capacity"),
    ]


@pytest.mark.asyncio
async def test_inf_bridge_uses_injected_http_client() -> None:
    calls: list[tuple[str, str, str | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = None
        if request.content:
            payload = request.content.decode("utf-8")
        calls.append((request.method, request.url.path, payload))
        return httpx.Response(200, json={"status": "ok"})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        bridge = InfBridge(
            backend=SGLangBridgeBackend(),
            backend_addr="http://backend",
            pause_state=PauseState(),
            http_client=client,
        )
        data = await bridge._send_request(
            HttpRequest(endpoint="/pause_generation", payload={})
        )

    assert data == {"status": "ok"}
    assert calls == [("POST", "/pause_generation", "{}")]


@pytest.mark.asyncio
async def test_online_ready_callback_uses_injected_http_client() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(200, json={"status": "ok"})

    transport = httpx.MockTransport(handler)
    notification = ReadyNotification(session_id="s1", trajectory_id=3)
    async with httpx.AsyncClient(transport=transport) as client:
        delivered = await _post_online_ready_callback(
            "http://callback",
            "admin-key",
            notification,
            5.0,
            client,
        )

    assert delivered is True
    assert calls == ["/callback/online_ready"]


@pytest.mark.asyncio
async def test_service_apps_expose_shared_http_client_state() -> None:
    gateway_app = create_gateway_app(
        GatewayConfig(
            host="127.0.0.1",
            port=18080,
            admin_api_key="admin-key",
            router_addr="http://router:18081",
        )
    )
    router_app = create_router_app(
        RouterConfig(
            host="127.0.0.1",
            port=18081,
            admin_api_key="admin-key",
        )
    )
    data_proxy_app = create_data_proxy_app(
        DataProxyConfig(
            host="127.0.0.1",
            port=18082,
            backend_addr="http://backend:30000",
            tokenizer_path="mock-tokenizer",
        )
    )

    assert isinstance(gateway_app.state.http_client, httpx.AsyncClient)
    assert isinstance(router_app.state.http_client, httpx.AsyncClient)
    assert isinstance(data_proxy_app.state.http_client, httpx.AsyncClient)

    await gateway_app.state.http_client.aclose()
    await router_app.state.http_client.aclose()
    await data_proxy_app.state.http_client.aclose()
