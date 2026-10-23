# SPDX-License-Identifier: Apache-2.0

import asyncio
from threading import Thread

import pytest
import torch
from flask import Flask
from werkzeug.serving import make_server

from areal.infra.rpc import rtensor
from areal.infra.rpc.guard.data_blueprint import data_bp


@pytest.fixture
def tensor_server():
    app = Flask(__name__)
    app.register_blueprint(data_bp)
    server = make_server("127.0.0.1", 0, app, threaded=True)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield app, f"127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=10)
        server.server_close()


@pytest.mark.parametrize("dtype", [torch.int16, torch.int32, torch.bfloat16])
def test_large_tensor_http_roundtrip_uses_bounded_binary_chunks(
    tensor_server, monkeypatch, dtype
):
    _, address = tensor_server
    values = torch.arange(3 * 1024 * 1024, dtype=torch.int32).remainder(512).to(dtype)
    shard_id = f"r3-chunked-{dtype}"
    rtensor.store(shard_id, values)
    backend = rtensor.HttpRTensorBackend()

    async def forbidden_json(*args, **kwargs):
        raise AssertionError("Large tensor unexpectedly used base64 JSON transfer")

    monkeypatch.setattr(backend, "_fetch_shard_group", forbidden_json)
    try:
        result = backend.fetch(
            [
                rtensor.TensorShardInfo(
                    shard_id,
                    address,
                    shape=list(values.shape),
                    dtype=str(dtype),
                    contiguous=True,
                )
            ]
        )[0]
        torch.testing.assert_close(result, values, rtol=0, atol=0)
    finally:
        rtensor.remove(shard_id)


@pytest.mark.parametrize(
    "query",
    [
        "offset=-1&length=2",
        "offset=0&length=4194305",
        "offset=1&length=100",
        "offset=x&length=1",
    ],
)
def test_binary_chunk_endpoint_rejects_unbounded_or_invalid_ranges(
    tensor_server, query
):
    app, _ = tensor_server
    rtensor.store("r3-range", torch.ones(4, dtype=torch.int16))
    try:
        response = app.test_client().get(f"/data/r3-range/chunk?{query}")
        assert response.status_code == 400
    finally:
        rtensor.remove("r3-range")


def test_routing_fetch_eviction_releases_cache_but_keeps_live_argument():
    values = torch.ones(3, 2, 2, dtype=torch.int16)
    reference = rtensor.RTensor(
        shard=rtensor.TensorShardInfo("r3-evict", "unused"), data=values
    )
    rtensor._fetch_buffer["r3-evict"] = values
    raw = {"routed_experts": reference}
    localized = rtensor.RTensor.localize(raw)
    rtensor.RTensor.evict_routing_fetches(raw)
    assert "r3-evict" not in rtensor._fetch_buffer
    assert reference.data.is_meta
    assert localized["routed_experts"] is values


def test_fetch_bounds_concurrent_storage_nodes(monkeypatch):
    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

    backend = rtensor.HttpRTensorBackend()
    active = peak = 0

    async def fetch_group(session, address, grouped):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return [torch.tensor([index]) for index, _ in grouped]

    monkeypatch.setattr(backend, "_create_session", Session)
    monkeypatch.setattr(backend, "_fetch_shard_group", fetch_group)
    result = backend.fetch(
        [
            rtensor.TensorShardInfo(f"shard-{index}", f"node-{index}")
            for index in range(6)
        ]
    )
    assert peak == 2
    assert [value.item() for value in result] == list(range(6))
