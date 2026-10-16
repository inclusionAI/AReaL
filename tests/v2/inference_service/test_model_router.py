"""Model routing contracts, without GPUs or a model download."""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import pytest_asyncio
import uvicorn
from omegaconf import OmegaConf

from areal.api.cli_args import GenerationHyperparameters, InferenceEngineConfig
from areal.api.io_struct import LocalInfServerInfo, ModelRequest
from areal.infra.model_router import ModelRouterConfig, create_app
from areal.infra.model_routing import (
    TRAJECTORY_HEADER,
    VERSION_HEADER,
    get_model_routing_key,
    model_routing_context,
)
from areal.v2.inference_service.controller.controller import RolloutControllerV2
from areal.v2.inference_service.data_proxy.pause import PauseState
from areal.v2.inference_service.inf_bridge import InfBridge
from areal.v2.inference_service.sglang.bridge import SGLangBridgeBackend

_KEY = "test-model-router-admin"


def _headers(key: str = "trajectory-0", version: int = 0) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {_KEY}",
        TRAJECTORY_HEADER: key,
        VERSION_HEADER: str(version),
    }


def _payload(rid: str = "call-0") -> dict:
    return {"rid": rid, "input_ids": [1, 2], "stream": False}


@pytest_asyncio.fixture
async def router():
    app = create_app(ModelRouterConfig(("http://a:30000", "http://b:30000"), _KEY))
    calls = []

    async def handler(request):
        calls.append(request)
        return httpx.Response(
            200,
            json={
                "meta_info": {
                    "finish_reason": {"type": "stop"},
                    "output_token_logprobs": [[-0.1, 42]],
                }
            },
        )

    harness = SimpleNamespace(app=app, calls=calls, handler=handler)

    async def dispatch(request):
        return await harness.handler(request)

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(dispatch)
        ) as backend:
            app.state.http_client = backend
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://router"
            ) as client:
                harness.client = client
                yield harness


def test_config_defaults_preserve_existing_controller_and_routing():
    config = InferenceEngineConfig(backend="sglang:d1")
    assert config.model_router == "disabled"
    assert config._version == "v1"
    assert config.routing_strategy == "round_robin"
    assert OmegaConf.structured(config).model_router == "disabled"


@pytest.mark.parametrize("mode", ["node", "global"])
def test_config_model_router_requires_local_v2_sglang(mode):
    assert InferenceEngineConfig(backend="sglang:d2", _version="v2", model_router=mode)
    for overrides in (
        {"_version": "v1"},
        {"backend": "vllm:d1"},
        {"api_url": "http://external"},
    ):
        values = {"backend": "sglang:d1", "_version": "v2", "model_router": mode}
        values.update(overrides)
        with pytest.raises(ValueError, match="requires v2"):
            InferenceEngineConfig(**values)


@pytest.mark.asyncio
async def test_task_context_is_isolated_and_restored():
    async def member(key):
        with model_routing_context(key):
            await asyncio.sleep(0)
            return get_model_routing_key()

    assert await asyncio.gather(member("a"), member("b")) == ["a", "b"]
    with pytest.raises(RuntimeError, match="trajectory context"):
        get_model_routing_key()


@pytest.mark.asyncio
async def test_router_preserves_raw_payload_and_native_response(router):
    payload = {
        **_payload(),
        "image_data": ["data:image/png;base64,abcdef"],
        "return_routed_experts": True,
        "future_extension": {"text": "中文�"},
        "sampling_params": {"top_p": 1.0, "top_k": 100000, "stop": ["<end>"]},
    }
    raw = json.dumps(payload, ensure_ascii=False).encode()
    response = await router.client.post("/generate", headers=_headers(), content=raw)
    assert response.status_code == 200
    assert response.headers[VERSION_HEADER] == "0"
    assert router.calls[0].content == raw
    assert "authorization" not in router.calls[0].headers
    assert response.json()["meta_info"]["output_token_logprobs"] == [[-0.1, 42]]


@pytest.mark.asyncio
async def test_group_members_spread_but_later_turn_keeps_backend(router):
    arrived, release = asyncio.Event(), asyncio.Event()
    original = router.handler

    async def hold(request):
        response = await original(request)
        if len(router.calls) == 12:
            arrived.set()
        await release.wait()
        return response

    router.handler = hold
    tasks = [
        asyncio.create_task(
            router.client.post(
                "/generate",
                headers=_headers(f"group-member-{i}"),
                json=_payload(str(i)),
            )
        )
        for i in range(12)
    ]
    try:
        await asyncio.wait_for(arrived.wait(), timeout=3)
        assert router.app.state.model_router.active == [6, 6]
    finally:
        release.set()
        responses = await asyncio.gather(*tasks)
    assert all(r.status_code == 200 for r in responses)
    first_host = router.calls[0].url.host
    # Make the original backend busy. Existing affinity must win over load.
    state = router.app.state.model_router
    original_index = state.bindings["group-member-0"]
    state.active[original_index] += 10
    await router.client.post(
        "/generate", headers=_headers("group-member-0"), json=_payload("turn-2")
    )
    assert router.calls[-1].url.host == first_host
    await router.client.post(
        "/generate", headers=_headers("new-member"), json=_payload("new")
    )
    assert router.calls[-1].url.host != first_host
    state.active[original_index] -= 10
    assert state.active == [0, 0]


@pytest.mark.asyncio
async def test_close_rejects_late_requests_and_version_changes_keep_binding(router):
    client = router.client
    await client.post("/generate", headers=_headers(), json=_payload())
    binding = dict(router.app.state.model_router.bindings)
    assert (
        await client.post("/set_version", headers=_headers(), json={"version": 1})
    ).status_code == 409
    assert (await client.post("/close", headers=_headers())).status_code == 200
    response = await client.post("/generate", headers=_headers("late"), json=_payload())
    assert response.status_code == 503
    assert response.json()["detail"] == "model_router_paused"
    assert len(router.calls) == 1
    assert (await client.post("/drain", headers=_headers())).status_code == 200
    assert (
        await client.post("/set_version", headers=_headers(), json={"version": 1})
    ).status_code == 200
    assert (await client.post("/open", headers=_headers())).status_code == 200
    stale = await client.post(
        "/generate", headers=_headers(), json=_payload("late-old-version")
    )
    assert stale.status_code == 409
    assert len(router.calls) == 1
    response = await client.post(
        "/generate", headers=_headers(version=1), json=_payload("resumed")
    )
    assert response.status_code == 200
    assert response.headers[VERSION_HEADER] == "1"
    assert router.app.state.model_router.bindings == binding


@pytest.mark.asyncio
async def test_unknown_backend_outcome_closes_pool_without_retry(router):
    async def timeout(request):
        router.calls.append(request)
        raise httpx.ReadTimeout("unknown backend outcome", request=request)

    router.handler = timeout
    assert (
        await router.client.post("/generate", headers=_headers(), json=_payload())
    ).status_code == 502
    state = router.app.state.model_router
    assert state.failed and not state.accepting
    assert state.active == [0, 0]
    assert (
        await router.client.post(
            "/generate", headers=_headers("another"), json=_payload()
        )
    ).status_code == 503
    assert (await router.client.post("/open", headers=_headers())).status_code == 409
    assert (await router.client.post("/drain", headers=_headers())).status_code == 503
    assert len(router.calls) == 1


@pytest.mark.asyncio
async def test_router_rejects_missing_identity_and_releases_exported_binding(router):
    response = await router.client.post("/generate", json=_payload())
    assert response.status_code == 401
    await router.client.post("/generate", headers=_headers(), json=_payload())
    response = await router.client.post(
        "/release", headers=_headers(), json={"keys": ["trajectory-0"]}
    )
    assert response.status_code == 200
    assert not router.app.state.model_router.bindings


@pytest.mark.asyncio
async def test_drain_waits_for_all_backend_responses(router):
    arrived, release = asyncio.Event(), asyncio.Event()
    original = router.handler

    async def hold(request):
        response = await original(request)
        arrived.set()
        await release.wait()
        return response

    router.handler = hold
    task = asyncio.create_task(
        router.client.post("/generate", headers=_headers(), json=_payload())
    )
    try:
        await asyncio.wait_for(arrived.wait(), 3)
        await router.client.post("/close", headers=_headers())
        drain = asyncio.create_task(router.client.post("/drain", headers=_headers()))
        await asyncio.sleep(0.01)
        assert not drain.done()
        assert (
            await router.client.post(
                "/set_version", headers=_headers(), json={"version": 1}
            )
        ).status_code == 409
        release.set()
        assert (await task).status_code == 200
        assert (await drain).status_code == 200
    finally:
        release.set()
        await task


@pytest.mark.asyncio
async def test_inf_bridge_admission_rejection_does_not_consume_abort_budget(router):
    bridge = InfBridge(
        SGLangBridgeBackend(),
        "http://physical:30000",
        PauseState(),
        model_router_addr="http://router",
        model_router_api_key=_KEY,
        resubmit_wait=0.001,
        max_resubmit_retries=1,
    )
    await bridge._client.aclose()
    bridge._client = router.client
    await router.client.post("/close", headers=_headers())
    req = ModelRequest(
        input_ids=[1, 2],
        tokenizer=None,
        gconfig=GenerationHyperparameters(max_new_tokens=2),
    )

    async def generate():
        with model_routing_context("resume-member"):
            return await bridge.agenerate(req)

    task = asyncio.create_task(generate())
    try:
        await asyncio.sleep(0.02)
        assert not task.done() and not router.calls
        await router.client.post(
            "/set_version", headers=_headers(), json={"version": 1}
        )
        bridge.set_version(1)
        await router.client.post("/open", headers=_headers())
        response = await asyncio.wait_for(task, 3)
        assert response.output_versions == [1]
        assert len(router.calls) == 1
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_real_http_router_two_echo_backends_preserve_group_affinity():
    """Exercise actual HTTP forwarding and the same app lifespan as production."""
    servers, threads = [], []
    for index in range(2):

        def handler_type(backend_index):
            class Handler(BaseHTTPRequestHandler):
                def do_POST(self):
                    payload = json.loads(
                        self.rfile.read(int(self.headers["Content-Length"]))
                    )
                    time.sleep(0.03)
                    body = json.dumps(
                        {"backend": backend_index, "payload": payload}
                    ).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

                def log_message(self, *args):
                    pass

            return Handler

        server = ThreadingHTTPServer(("127.0.0.1", 0), handler_type(index))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        servers.append(server)
        threads.append(thread)
    app = create_app(
        ModelRouterConfig(
            tuple(f"http://127.0.0.1:{s.server_port}" for s in servers), _KEY
        )
    )
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    uv_server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="on"))
    uv_thread = threading.Thread(
        target=uv_server.run, kwargs={"sockets": [sock]}, daemon=True
    )
    uv_thread.start()
    try:
        async with httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{port}", trust_env=False
        ) as client:
            async with asyncio.timeout(5):
                while not uv_server.started:
                    await asyncio.sleep(0.01)
            responses = await asyncio.gather(
                *[
                    client.post(
                        "/generate",
                        headers=_headers(f"member-{i}"),
                        json={
                            **_payload(str(i)),
                            "image_data": ["test-image"],
                            "return_routed_experts": True,
                        },
                    )
                    for i in range(12)
                ]
            )
            assert all(r.status_code == 200 for r in responses)
            assert {r.json()["backend"] for r in responses} == {0, 1}
            for i, first in enumerate(responses):
                second = await client.post(
                    "/generate",
                    headers=_headers(f"member-{i}"),
                    json=_payload(f"turn2-{i}"),
                )
                assert second.json()["backend"] == first.json()["backend"]
                assert first.json()["payload"]["return_routed_experts"] is True
                assert first.json()["payload"]["image_data"] == ["test-image"]
            assert (await client.post("/close", headers=_headers())).status_code == 200
            assert (await client.post("/drain", headers=_headers())).status_code == 200
    finally:
        uv_server.should_exit = True
        await asyncio.to_thread(uv_thread.join, 5)
        sock.close()
        for server, thread in zip(servers, threads, strict=True):
            await asyncio.to_thread(server.shutdown)
            server.server_close()
            thread.join(timeout=5)
        assert not uv_thread.is_alive()


@pytest.mark.asyncio
async def test_inf_bridge_routes_generation_but_keeps_physical_control(router):
    bridge = InfBridge(
        SGLangBridgeBackend(),
        "http://physical:30000",
        PauseState(),
        model_router_addr="http://router",
        model_router_api_key=_KEY,
    )
    await bridge._client.aclose()
    bridge._client = router.client
    req = ModelRequest(
        input_ids=[1, 2],
        tokenizer=None,
        gconfig=GenerationHyperparameters(max_new_tokens=2),
    )
    with model_routing_context("member-1"):
        result = await bridge.agenerate(req)
    assert result.output_tokens == [42]
    assert result.output_versions == [0]
    assert bridge.backend_addr == "http://physical:30000"
    assert list(router.app.state.model_router.bindings) == ["member-1"]
    assert json.loads(router.calls[0].content)["rid"]


@pytest.mark.parametrize("mode,expected_pools", [("node", 2), ("global", 1)])
@pytest.mark.asyncio
async def test_controller_builds_pools_from_head_nodes(mode, expected_pools):
    config = InferenceEngineConfig(
        backend="sglang:d3", _version="v2", model_router=mode
    )
    scheduler = MagicMock(n_gpus_per_node=8)
    controller = RolloutControllerV2(config, scheduler)
    controller._inf_addrs = [
        "http://10.0.0.1:30000",
        "http://10.0.0.1:30001",
        "http://10.0.0.2:30000",
    ]
    workers = [
        SimpleNamespace(ip=host, worker_ports=[9000])
        for host in ("10.0.0.1", "10.0.0.1", "10.0.0.2")
    ]
    controller._async_fork_on_guard = AsyncMock(
        side_effect=[("10.0.0.1", 7000), ("10.0.0.2", 7000)]
    )
    try:
        await controller._async_start_model_routers(workers, False, 1)
        assert len(controller._model_router_addrs) == expected_pools
        mappings = controller._model_router_by_backend
        assert mappings[controller._inf_addrs[0]] == mappings[controller._inf_addrs[1]]
        assert (
            mappings[controller._inf_addrs[0]] == mappings[controller._inf_addrs[2]]
        ) == (mode == "global")
    finally:
        controller._sync_client.close()


@pytest.mark.asyncio
async def test_controller_partial_pause_failure_never_drains_or_reopens():
    config = InferenceEngineConfig(
        backend="sglang:d2", _version="v2", model_router="global", admin_api_key=_KEY
    )
    controller = RolloutControllerV2(config, MagicMock(n_gpus_per_node=8))
    controller._model_router_addrs = ["http://router"]
    controller._data_proxy_addrs = ["http://dp0", "http://dp1"]
    controller._async_model_router_post = AsyncMock()
    controller._async_data_proxy_post = AsyncMock(
        side_effect=[{}, RuntimeError("lost ACK")]
    )
    try:
        with pytest.raises(RuntimeError, match="pause_generation failed"):
            await controller._async_pause_generation()
        assert [
            call.args[0] for call in controller._async_model_router_post.await_args_list
        ] == ["/close", "/fail"]
        assert controller._model_router_failed
        with pytest.raises(RuntimeError, match="recover before resuming"):
            await controller._async_continue_generation()
    finally:
        controller._sync_client.close()


@pytest.mark.asyncio
async def test_colocated_train_eval_share_barrier_until_full_onload(router):
    config = InferenceEngineConfig(
        backend="sglang:d2", _version="v2", model_router="global", admin_api_key=_KEY
    )
    owner = RolloutControllerV2(config, MagicMock(n_gpus_per_node=8))
    evaluation = RolloutControllerV2(config, MagicMock(n_gpus_per_node=8))
    for controller in (owner, evaluation):
        controller._model_router_addrs = ["http://router"]
        controller._data_proxy_addrs = ["http://dp0", "http://dp1"]
        controller._get_async_client = AsyncMock(return_value=router.client)
        controller._async_data_proxy_post = AsyncMock(return_value={})
    evaluation._model_router_owner = owner
    try:
        await owner._async_pause_generation()
        await owner._async_offload()
        await owner._async_set_version(1)
        await owner._async_continue_generation()  # weight update while offloaded
        assert not router.app.state.model_router.accepting
        assert owner._model_memory_offloaded
        assert not any(
            call.args[1] == "/continue_generation"
            for call in owner._async_data_proxy_post.await_args_list
        )
        with pytest.raises(ValueError, match="without tags"):
            await evaluation._async_onload(["weights"])
        await evaluation._async_onload()
        assert not owner._model_memory_offloaded
        await evaluation._async_continue_generation()
        assert router.app.state.model_router.accepting
        # RLTrainer republishes the same version to its separate eval DPs.
        await evaluation._async_set_version(1)
        assert router.app.state.model_router.version == 1
        # onload on an already-running pool must close it before draining.
        await owner._async_onload()
        assert not router.app.state.model_router.accepting
        await owner._async_continue_generation()
        assert router.app.state.model_router.accepting
    finally:
        owner._sync_client.close()
        evaluation._sync_client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation,endpoint",
    [
        ("_async_onload", "/resume_memory_occupation"),
        ("_async_offload", "/release_memory_occupation"),
        ("_async_set_version", "/set_version"),
        ("_async_continue_generation", "/continue_generation"),
    ],
)
async def test_partial_control_failure_latches_pool_against_cleanup_reopen(
    router, operation, endpoint
):
    config = InferenceEngineConfig(
        backend="sglang:d2", _version="v2", model_router="global", admin_api_key=_KEY
    )
    controller = RolloutControllerV2(config, MagicMock(n_gpus_per_node=8))
    controller._model_router_addrs = ["http://router"]
    controller._data_proxy_addrs = ["http://dp0", "http://dp1"]
    controller._get_async_client = AsyncMock(return_value=router.client)

    async def physical(addr, path, payload):
        if path == endpoint and addr == "http://dp1":
            raise RuntimeError("lost physical ACK")
        return {}

    controller._async_data_proxy_post = AsyncMock(side_effect=physical)
    try:
        with pytest.raises(RuntimeError):
            method = getattr(controller, operation)
            if operation == "_async_set_version":
                await method(1)
            else:
                await method()
        assert controller._model_router_failed
        assert router.app.state.model_router.failed
        with pytest.raises(RuntimeError, match="recover before resuming"):
            await controller._async_continue_generation()
        assert (
            await router.client.post("/open", headers=_headers())
        ).status_code == 409
    finally:
        controller._sync_client.close()


@pytest.mark.asyncio
async def test_unknown_inflight_request_drain_timeout_permanently_closes_pool():
    app = create_app(ModelRouterConfig(("http://a:30000",), _KEY, request_timeout=0.01))
    state = app.state.model_router
    index, _ = await state.admit("late-request", 0)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://router"
    ) as client:
        await client.post("/close", headers=_headers())
        response = await client.post("/drain", headers=_headers())
        assert response.status_code == 503
        assert response.json()["detail"] == "model_router_drain_timeout"
        await state.finish("late-request", index, failed=False)
        assert (await client.post("/open", headers=_headers())).status_code == 409
        assert (
            await client.post("/set_version", headers=_headers(), json={"version": 1})
        ).status_code == 409


def test_shared_server_initialization_requires_one_matching_router_owner(monkeypatch):
    config = InferenceEngineConfig(
        backend="sglang:d1", _version="v2", model_router="global", admin_api_key=_KEY
    )
    owner = RolloutControllerV2(config, MagicMock(n_gpus_per_node=8))
    child = RolloutControllerV2(config, MagicMock(n_gpus_per_node=8))
    servers = [LocalInfServerInfo("::1", 30000, None)]
    owner._model_router_addrs = ["http://[::1]:7000"]
    owner._model_router_by_backend = {
        "http://[::1]:30000": owner._model_router_addrs[0]
    }
    owner._ensure_initialized = MagicMock()
    try:
        with pytest.raises(ValueError, match="model_router_owner"):
            child.initialize("eval-rollout", server_infos=servers)
        with pytest.raises(ValueError, match="same inference servers"):
            child.initialize(
                "eval-rollout",
                server_infos=[LocalInfServerInfo("other", 30000, None)],
                model_router_owner=owner,
            )
        executor = MagicMock()
        executor.submit.return_value.done.return_value = False
        monkeypatch.setattr(
            "areal.infra.utils.concurrent.get_executor", lambda _: executor
        )
        child._start_online_callback_server = MagicMock()
        child._workers_ready = MagicMock()
        child.initialize("eval-rollout", server_infos=servers, model_router_owner=owner)
        assert child._model_router_owner is owner
        assert child._model_router_addrs == owner._model_router_addrs
        assert child._model_router_addrs is not owner._model_router_addrs
        assert child._model_router_by_backend is not owner._model_router_by_backend
        child._model_router_addrs.clear()
        assert owner._model_router_addrs
    finally:
        owner._sync_client.close()
        child._sync_client.close()
