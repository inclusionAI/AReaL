"""Memory handover decisions for v2 colocated AWEX recovery."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

from areal.v2.weight_update.awex.megatron_adapter import AwexMegatronAdapter
from areal.v2.weight_update.awex.sglang_adapter import AwexSGLangAdapter


def test_colocate_memory_uses_sglang_residency_after_data_proxy_offload():
    """The data proxy can offload SGLang without touching AWEX's local cache."""
    owner = SimpleNamespace(offload_tags={"weights", "kv_cache"})
    scheduler = SimpleNamespace(weight_updater=owner)
    adapter = AwexSGLangAdapter(scheduler)

    assert adapter._memory_transition_tags(["weights", "kv_cache"], release=False) == [
        "weights",
        "kv_cache",
    ]
    assert adapter._memory_transition_tags(
        ["weights", "kv_cache", "cuda_graph"], release=True
    ) == ["cuda_graph"]


def test_colocate_memory_skips_duplicate_resume_from_sglang_state():
    owner = SimpleNamespace(offload_tags=set())
    adapter = AwexSGLangAdapter(SimpleNamespace(weight_updater=owner))
    adapter._released_tags.add("weights")

    assert adapter._memory_transition_tags(["weights"], release=False) == []


def test_megatron_colocate_adapter_shares_engine_residency():
    residency = Mock()
    residency.is_released.return_value = True
    engine = SimpleNamespace(
        _weight_residency=residency,
        init_weight_residency_adapter=Mock(),
    )
    adapter = AwexMegatronAdapter(engine)

    with patch("areal.v2.weight_update.awex.megatron_adapter.httpx.Client"):
        adapter.init_colocate_weight_update(
            pair_name="actor-rollout",
            kv_store_url="http://gateway",
            transfer_rank=1,
            infer_world_size=1,
            train_world_size=1,
            num_engines=1,
            master_port=29500,
        )

    engine.init_weight_residency_adapter.assert_called_once_with()
    assert adapter._is_released("weights") is True
    adapter.release_memory(["optimizer"])
    adapter.resume_memory(["weights"])
    residency.release_memory.assert_called_once_with(["optimizer"])
    residency.resume_memory.assert_called_once_with(["weights"])
