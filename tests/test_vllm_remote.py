from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import patch

from areal.api import WeightUpdateMeta
from areal.api.io_struct import ParamSpec
from areal.engine.vllm_ext.areal_vllm_server import (
    _RUNTIME_LORA_PENDING_PHASE_PREPARED,
    _RUNTIME_LORA_PENDING_PHASE_UPDATING,
    _choose_eviction_candidate,
    _get_runtime_lora_state,
    _register_runtime_lora_name,
    _reserve_runtime_lora_slot,
)
from areal.engine.vllm_remote import VLLMBackend


def _make_fake_app(*, max_loras=4, requests=None):
    if requests is None:
        requests = OrderedDict()
    serving_models = SimpleNamespace(lora_requests=requests)
    state = SimpleNamespace(
        args=SimpleNamespace(max_loras=max_loras),
        openai_serving_models=serving_models,
    )
    return SimpleNamespace(state=state)


def _make_lora_request(name: str, adapter_id: int, path: str | None = None):
    req = SimpleNamespace(
        lora_name=name,
        lora_int_id=adapter_id,
        lora_path=path or f"/tmp/{name}",
        base_model_name="base-model",
    )
    return req


def test_build_distributed_lora_weight_update_requests_uses_versioned_name_only():
    backend = VLLMBackend()
    meta = WeightUpdateMeta(
        type="xccl",
        use_lora=True,
        lora_name="demo-lora",
        lora_int_id=1,
        base_model_name="base-model",
        peft_config={
            "target_modules": ["q_proj", "v_proj"],
            "r": 64,
            "lora_alpha": 16,
            "bias": "none",
        },
        version=3,
        nccl_group_name="weight-update",
    )
    param_specs = [ParamSpec(name="a", shape=(1,), dtype="float16")]

    requests = backend.build_distributed_weight_update_requests(meta, param_specs)

    meta_req, update_req = requests.requests
    assert meta_req.payload["lora_name"] == "demo-lora-v3"
    assert "lora_int_id" not in meta_req.payload
    assert update_req.payload["lora_name"] == "demo-lora-v3"
    assert "lora_int_id" not in update_req.payload
    assert update_req.endpoint == "/areal_update_weights_lora_xccl"


def test_build_disk_lora_weight_update_requests_uses_areal_endpoint():
    backend = VLLMBackend()
    meta = WeightUpdateMeta(
        type="disk",
        path="/tmp/adapter",
        use_lora=True,
        lora_name="demo-lora",
        base_model_name="base-model",
        version=7,
    )

    requests = backend.build_disk_weight_update_requests(meta)

    (req,) = requests.requests
    assert req.endpoint == "/areal_update_weights_lora"
    assert req.payload == {
        "lora_model_path": "/tmp/adapter",
        "lora_name": "demo-lora-v7",
        "base_model_name": "base-model",
    }


def test_reserve_runtime_lora_slot_reuses_existing_version_slot():
    app = _make_fake_app(
        requests=OrderedDict(
            {
                "demo-lora-v1": _make_lora_request("demo-lora-v1", 2),
            }
        )
    )

    slot, replaced = _reserve_runtime_lora_slot(app, "demo-lora-v1")

    assert slot == 2
    assert replaced is None


def test_reserve_runtime_lora_slot_prefers_same_base_eviction():
    app = _make_fake_app(
        max_loras=2,
        requests=OrderedDict(
            {
                "demo-lora-v1": _make_lora_request("demo-lora-v1", 1),
                "other-lora-v4": _make_lora_request("other-lora-v4", 2),
            }
        ),
    )

    slot, replaced = _reserve_runtime_lora_slot(app, "demo-lora-v2")

    assert slot == 1
    assert replaced == "demo-lora-v1"


def test_reserve_runtime_lora_slot_counts_pending_reservations_as_occupied():
    app = _make_fake_app(
        max_loras=3,
        requests=OrderedDict(
            {
                "demo-lora-v1": _make_lora_request("demo-lora-v1", 1),
            }
        ),
    )

    first_slot, first_replaced = _reserve_runtime_lora_slot(app, "demo-lora-v2")
    second_slot, second_replaced = _reserve_runtime_lora_slot(app, "other-lora-v1")

    assert first_slot == 2
    assert first_replaced is None
    assert second_slot == 3
    assert second_replaced is None


def test_choose_eviction_candidate_falls_back_to_oldest_other_base():
    slots = OrderedDict(
        {
            "first-lora-v1": 3,
            "second-lora-v1": 4,
        }
    )

    name, slot = _choose_eviction_candidate(slots, "third-lora-v9")

    assert name == "first-lora-v1"
    assert slot == 3


def test_register_runtime_lora_name_replaces_public_route_and_updates_slots():
    requests = OrderedDict(
        {
            "demo-lora-v1": _make_lora_request(
                "demo-lora-v1", 1, "xccl://demo-lora-v1"
            ),
            "other-lora-v3": _make_lora_request(
                "other-lora-v3", 2, "xccl://other-lora-v3"
            ),
        }
    )
    app = _make_fake_app(max_loras=2, requests=requests)

    # Simulate a reserved replacement of the old version.
    slots, pending = _get_runtime_lora_state(app)
    slots["demo-lora-v1"] = 1
    slots["other-lora-v3"] = 2
    pending["demo-lora-v2"] = (
        1,
        "demo-lora-v1",
        _RUNTIME_LORA_PENDING_PHASE_UPDATING,
        1.0,
    )

    _register_runtime_lora_name(
        app,
        lora_name="demo-lora-v2",
        lora_int_id=1,
        base_model_name="base-model",
        replaced_lora_name="demo-lora-v1",
    )

    assert "demo-lora-v1" not in requests
    assert "demo-lora-v2" in requests
    assert requests["demo-lora-v2"].lora_int_id == 1
    assert requests["demo-lora-v2"].base_model_name == "base-model"
    assert list(slots.items()) == [("other-lora-v3", 2), ("demo-lora-v2", 1)]
    assert "demo-lora-v2" not in pending


def test_register_runtime_lora_name_uses_explicit_disk_lora_path():
    requests = OrderedDict()
    app = _make_fake_app(max_loras=1, requests=requests)

    _, pending = _get_runtime_lora_state(app)
    pending["demo-lora-v2"] = (
        1,
        None,
        _RUNTIME_LORA_PENDING_PHASE_UPDATING,
        1.0,
    )

    _register_runtime_lora_name(
        app,
        lora_name="demo-lora-v2",
        lora_int_id=1,
        lora_path="/tmp/weight_update_v2",
        base_model_name="base-model",
    )

    assert requests["demo-lora-v2"].lora_path == "/tmp/weight_update_v2"


def test_reserve_runtime_lora_slot_prunes_stale_pending_reservations():
    app = _make_fake_app(max_loras=1)
    _, pending = _get_runtime_lora_state(app)
    pending["demo-lora-v1"] = (
        1,
        None,
        _RUNTIME_LORA_PENDING_PHASE_PREPARED,
        0.0,
    )

    with patch(
        "areal.engine.vllm_ext.areal_vllm_server.time.monotonic",
        return_value=301.0,
    ):
        slot, replaced = _reserve_runtime_lora_slot(app, "demo-lora-v2")

    assert slot == 1
    assert replaced is None
    assert "demo-lora-v1" not in pending


def test_reserve_runtime_lora_slot_keeps_inflight_updates_reserved():
    app = _make_fake_app(max_loras=1)
    _, pending = _get_runtime_lora_state(app)
    pending["demo-lora-v1"] = (
        1,
        None,
        _RUNTIME_LORA_PENDING_PHASE_UPDATING,
        0.0,
    )

    with patch(
        "areal.engine.vllm_ext.areal_vllm_server.time.monotonic",
        return_value=301.0,
    ):
        slot, replaced = _reserve_runtime_lora_slot(app, "demo-lora-v1")

    assert slot == 1
    assert replaced is None
    assert pending["demo-lora-v1"][2] == _RUNTIME_LORA_PENDING_PHASE_UPDATING


def test_set_update_weight_meta_lora_clears_reservation_on_collective_failure():
    import asyncio

    from areal.engine.vllm_ext.areal_vllm_server import (
        UpdateWeightsFromXcclRequestLora,
        areal_set_weight_meta_xccl_lora,
    )

    async def _failing_collective_rpc(*args, **kwargs):
        raise RuntimeError("boom")

    app = _make_fake_app(max_loras=1)
    app.state.engine_client = SimpleNamespace(collective_rpc=_failing_collective_rpc)
    raw_request = SimpleNamespace(app=app)
    request = UpdateWeightsFromXcclRequestLora(
        names=["x"],
        dtypes=["float16"],
        shapes=[[1]],
        lora_name="demo-lora-v1",
        lora_target_modules=["q_proj"],
        lora_rank=8,
        lora_alpha=16,
        lora_bias="none",
        base_model_name="base-model",
        group_name="group",
    )

    async def _run():
        try:
            await areal_set_weight_meta_xccl_lora(request, raw_request)
        except RuntimeError:
            pass

    asyncio.run(_run())

    _, pending = _get_runtime_lora_state(app)
    assert pending == {}


def test_static_non_versioned_lora_is_not_an_eviction_candidate():
    app = _make_fake_app(
        max_loras=2,
        requests=OrderedDict(
            {
                "static-ocr": _make_lora_request("static-ocr", 1, "/models/static-ocr"),
                "demo-lora-v1": _make_lora_request(
                    "demo-lora-v1", 2, "xccl://demo-lora-v1"
                ),
            }
        ),
    )

    slot, replaced = _reserve_runtime_lora_slot(app, "demo-lora-v2")

    assert slot == 2
    assert replaced == "demo-lora-v1"
