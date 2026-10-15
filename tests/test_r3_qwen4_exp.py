# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tests.test_r3_engine_validation import _fake_engine, _fake_mb_list

from areal.api.cli_args import MegatronEngineConfig, TrainEngineConfig
from areal.engine.core.model import SequencePackingMode
from areal.engine.megatron_engine import MegatronEngine
from areal.engine.r3.asserts import R3Error


def _qwen_engine():
    engine = _fake_engine(torch.ones(1, 4, 2, 2, dtype=torch.int32))
    engine.bridge_cls = "mcore-bridge"
    engine.is_vision_model = True
    engine.hf_config = SimpleNamespace(
        model_type="qwen4_exp", architectures=["Qwen4ExpForConditionalGeneration"]
    )
    engine.tf_config = SimpleNamespace(
        num_layers=2,
        moe_router_topk=2,
        moe_layer_freq=[1, 1],
        moe_enable_routing_replay=True,
        cp_partition_mode="zigzag",
    )
    engine.sequence_packing_mode = SequencePackingMode.WRAPPER_THD
    return engine


def test_qwen4_exp_wrapper_thd_accepts_routing_without_rejecting_vision_model():
    """The token-preserving Qwen bridge may replay expanded image-token rows."""
    engine = _qwen_engine()
    context = engine._prepare_r3_forward_context(_fake_mb_list(batch_size=1))
    assert context["stats"]["routing_valid_samples"] == 1
    torch.testing.assert_close(
        context["routed_mbs"][0], engine._r3_pending_routed_experts, rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "unsupported",
    ["model", "bridge", "padded", "contiguous", "no_replay", "sparse", "tree", "mtp"],
)
def test_qwen4_exp_routing_rejects_unvalidated_layouts(unsupported):
    """Enabling Qwen replay must not admit unrelated bridge/VLM contracts."""
    engine = _qwen_engine()
    if unsupported == "model":
        engine.hf_config.model_type = "qwen3_5"
    elif unsupported == "bridge":
        engine.bridge_cls = "megatron-bridge"
    elif unsupported == "padded":
        engine.sequence_packing_mode = SequencePackingMode.PADDED
    elif unsupported == "contiguous":
        engine.tf_config.cp_partition_mode = "contiguous"
    elif unsupported == "no_replay":
        engine.tf_config.moe_enable_routing_replay = False
    elif unsupported == "sparse":
        engine.tf_config.moe_layer_freq = [1, 0]
    elif unsupported == "mtp":
        engine.tf_config.mtp_num_layers = 1
    else:
        engine.enable_tree_training = True
    with pytest.raises(R3Error, match="supported Qwen4Exp"):
        engine._prepare_r3_forward_context(_fake_mb_list(batch_size=1))


@pytest.mark.parametrize("replay", [False, True])
def test_mcore_bridge_receives_native_replay_flag_before_construction(
    monkeypatch, tmp_path, replay
):
    """Native TopKRouter must construct its replay object with the model."""
    from areal.engine import megatron_engine as module

    engine = object.__new__(MegatronEngine)
    engine.config = TrainEngineConfig(path=str(tmp_path))
    engine.mcore_config = MegatronEngineConfig(
        bridge_type="mcore-bridge", enable_router_replay=replay, moe_router_fusion=False
    )
    engine.bridge_cls = "mcore-bridge"
    engine._r3_enabled = replay
    engine.enable_tree_training = False
    engine.dtype = torch.bfloat16
    engine.logger = Mock()
    engine.parallel_strategy = SimpleNamespace(
        tensor_parallel_size=8,
        pipeline_parallel_size=4,
        context_parallel_size=2,
        expert_parallel_size=16,
        expert_tensor_parallel_size=1,
        virtual_pipeline_parallel_size=1,
    )
    constructor = Mock()
    monkeypatch.setattr(module, "MCoreBridgeAdapter", constructor)
    result = engine._build_hf_mcore_bridge()
    overrides = constructor.call_args.kwargs["transformer_config_overrides"]
    assert overrides.get("moe_enable_routing_replay", False) is replay
    assert result is constructor.return_value
