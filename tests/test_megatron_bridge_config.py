# SPDX-License-Identifier: Apache-2.0

"""CPU coverage of config extraction without importing device-only backends."""

from __future__ import annotations

import ast
import dataclasses
import sys
from functools import wraps
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from areal.api.alloc_mode import ParallelStrategy
from areal.api.cli_args import MegatronEngineConfig, MindSpeedEngineConfig


def _load_function(path, name, namespace):
    source = Path(__file__).resolve().parents[1] / path
    function = next(
        node
        for node in ast.walk(ast.parse(source.read_text()))
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            function,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    return namespace[name]


@dataclasses.dataclass
class _Config:
    expert_model_parallel_size: int = 1
    expert_tensor_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    num_layers: int = 5
    ffn_hidden_size: int | None = None

    def __post_init__(self):
        if self.expert_model_parallel_size <= 1:
            raise ValueError("MoE overlap requires EP > 1")
        if self.num_layers % self.pipeline_model_parallel_size:
            raise ValueError("Pipeline layout has not been configured")


@pytest.mark.parametrize("ep,etp", [(4, 1), (4, 2), (1, 1)])
def test_npu_bridge_config_validates_runtime_expert_dimensions(ep, etp):
    """Validate both configs with real EP while deferring uneven pipeline splits."""
    finalized = []

    class Provider:
        expert_model_parallel_size = 1
        expert_tensor_parallel_size = 1
        pipeline_model_parallel_size = 1
        num_layers = 5
        ffn_hidden_size = None

        def finalize(self):
            _Config(
                expert_model_parallel_size=self.expert_model_parallel_size,
                pipeline_model_parallel_size=self.pipeline_model_parallel_size,
            )
            self.ffn_hidden_size = 128
            finalized.append(self)

    class Bridge:
        hf_pretrained = SimpleNamespace(_name_or_path="")

        def to_megatron_provider(self, *, load_weights):
            assert not load_weights
            return Provider()

        @property
        def transformer_config(self):
            provider = self.to_megatron_provider(load_weights=False)
            provider.finalize()
            return provider

    extract = _load_function(
        "areal/models/mcore/registry.py",
        "make_hf_and_mcore_config",
        {
            "dataclasses": dataclasses,
            "TransformerConfig": _Config,
            "is_npu_available": True,
            "mpu": SimpleNamespace(
                get_expert_model_parallel_world_size=lambda: ep,
                get_expert_tensor_parallel_world_size=lambda: etp,
                get_pipeline_model_parallel_world_size=lambda: 4,
            ),
        },
    )
    if ep == 1:
        with pytest.raises(ValueError, match="MoE overlap requires EP > 1"):
            extract("model", None, Bridge(), "megatron-bridge")
        return

    hf_config, config = extract("model", None, Bridge(), "megatron-bridge")

    assert hf_config._name_or_path == "model"
    assert len(finalized) == 1
    assert config.expert_model_parallel_size == ep
    assert config.expert_tensor_parallel_size == etp
    assert config.pipeline_model_parallel_size == 1
    assert config.ffn_hidden_size == 128


@pytest.mark.parametrize("bridge_type", ["mbridge", "megatron-bridge"])
def test_bridge_config_existing_paths_preserve_config(bridge_type):
    """The GPU and mbridge paths keep using their original config objects."""
    config = object()
    hf_config = SimpleNamespace(_name_or_path="")
    bridge = SimpleNamespace(
        hf_pretrained=hf_config,
        hf_config=hf_config,
        config=config,
        transformer_config=config,
    )
    extract = _load_function(
        "areal/models/mcore/registry.py",
        "make_hf_and_mcore_config",
        {"is_npu_available": bridge_type == "mbridge"},
    )

    result = extract("model", None, bridge, bridge_type)

    assert result == (hf_config, config)


@pytest.mark.parametrize("pp,etp", [(1, 1), (4, 2)])
def test_mindspeed_repatch_uses_expert_group_size(monkeypatch, pp, etp):
    """MindSpeed EP excludes pipeline and expert tensor parallel dimensions."""
    captured = {}
    adaptor = ModuleType("mindspeed.megatron_adaptor")
    adaptor.repatch = captured.update
    monkeypatch.setitem(sys.modules, adaptor.__name__, adaptor)
    layout = ModuleType("areal.engine.megatron_utils.mindspeed_pipeline_layout_patch")
    layout.ensure_mindspeed_pipeline_layout_stage_count = lambda: None
    monkeypatch.setitem(sys.modules, layout.__name__, layout)
    patch = _load_function("areal/engine/megatron_engine.py", "_patch_mindspeed", {})
    strategy = ParallelStrategy(
        data_parallel_size=4,
        tensor_parallel_size=2,
        pipeline_parallel_size=pp,
        expert_parallel_size=4,
        expert_tensor_parallel_size=etp,
    )
    engine = SimpleNamespace(
        mindspeed_config=SimpleNamespace(as_dict=lambda: {}),
        mcore_config=SimpleNamespace(
            recompute_method="uniform",
            recompute_granularity="full",
            recompute_num_layers=1,
            ddp=SimpleNamespace(use_distributed_optimizer=True),
        ),
        bridge_cls="mbridge",
    )

    patch(engine, strategy)

    assert captured["expert_model_parallel_size"] == 4


@pytest.mark.parametrize("experts", [None, 8])
def test_qwen_vision_config_isolates_dense_moe_options(monkeypatch, experts):
    """Dense vision construction and replacement leave actor MoE flags intact."""
    calls = []

    @dataclasses.dataclass
    class VisionConfig:
        num_moe_experts: int | None = None
        moe_grouped_gemm: bool = False
        gemm_gradient_accumulation_fusion: bool = True
        moe_alltoall_overlap_comm: bool = True
        moe_permute_fusion: bool = True
        use_fused_moe_token_permute_and_unpermute: bool = True
        moe_zero_memory: str = "level0"
        moe_zero_memory_num_layers: int | None = 1

        def __post_init__(self):
            if not self.num_moe_experts:
                assert not self.gemm_gradient_accumulation_fusion
                assert not self.moe_alltoall_overlap_comm
                assert not self.moe_permute_fusion
                assert not self.use_fused_moe_token_permute_and_unpermute
                assert self.moe_zero_memory == "disable"
                assert self.moe_zero_memory_num_layers is None
            calls.append(self)

    module = ModuleType(
        "megatron.bridge.models.qwen_vl.modelling_qwen3_vl.transformer_config"
    )
    module.Qwen3VLTransformerConfig = VisionConfig
    monkeypatch.setitem(sys.modules, module.__name__, module)
    patch = _load_function(
        "areal/engine/megatron_utils/megatron_bridge_patches.py",
        "patch_qwen3_vl_vision_config",
        {"wraps": wraps},
    )
    patch()
    installed = VisionConfig.__post_init__
    patch()
    assert VisionConfig.__post_init__ is installed

    config = VisionConfig(num_moe_experts=experts)
    clone = dataclasses.replace(config)

    assert len(calls) == 2
    assert config == clone
    assert config.gemm_gradient_accumulation_fusion == bool(experts)
    assert config.moe_alltoall_overlap_comm == bool(experts)
    assert VisionConfig.gemm_gradient_accumulation_fusion
    assert VisionConfig.moe_alltoall_overlap_comm


@pytest.mark.parametrize("is_npu", [False, True])
@pytest.mark.parametrize("aux_coeff", [0.0, 0.02, None])
def test_bridge_model_honors_mindspeed_aux_settings_before_finalize(
    monkeypatch, is_npu, aux_coeff
):
    """NPU Bridge construction honors explicit zero and nonzero auxiliary losses."""
    provider_module = ModuleType("megatron.bridge.models.gpt_provider")
    provider_module.default_layer_spec = object()
    provider_module.local_layer_spec = object()
    monkeypatch.setitem(sys.modules, provider_module.__name__, provider_module)
    provider = SimpleNamespace(
        moe_aux_loss_coeff=0.001,
        moe_router_load_balancing_type="global_aux_loss",
    )
    finalized = []

    def finalize():
        finalized.append(
            (provider.moe_aux_loss_coeff, provider.moe_router_load_balancing_type)
        )

    def provide_distributed_model(**kwargs):
        assert len(finalized) == 1
        return []

    provider.finalize = finalize
    provider.provide_distributed_model = provide_distributed_model
    make_model = _load_function(
        "areal/models/mcore/registry.py",
        "make_mcore_model",
        {
            "dataclasses": dataclasses,
            "is_npu_available": is_npu,
            "mpu": SimpleNamespace(
                get_tensor_model_parallel_world_size=lambda: 2,
                get_pipeline_model_parallel_world_size=lambda: 1,
                get_context_parallel_world_size=lambda: 8,
                get_expert_model_parallel_world_size=lambda: 8,
                get_expert_tensor_parallel_world_size=lambda: 1,
            ),
            "MCoreDDPConfig": SimpleNamespace,
            "logger": SimpleNamespace(warning=lambda *args: None),
        },
    )
    tf_config = SimpleNamespace(params_dtype="bf16", fp16=False, bf16=True)
    mindspeed_config = (
        MindSpeedEngineConfig(
            moe_aux_loss_coeff=aux_coeff, moe_router_load_balancing_type="aux_loss"
        )
        if aux_coeff is not None
        else None
    )

    make_model(
        hf_config=SimpleNamespace(model_type="test_moe"),
        tf_config=tf_config,
        mcore_config=MegatronEngineConfig(),
        bridge=SimpleNamespace(to_megatron_provider=lambda **kw: provider),
        bridge_type="megatron-bridge",
        mindspeed_config=mindspeed_config,
    )

    if is_npu and mindspeed_config is not None:
        assert finalized == [(aux_coeff, "aux_loss")]
        assert tf_config.moe_aux_loss_coeff == aux_coeff
        assert tf_config.moe_router_load_balancing_type == "aux_loss"
    else:
        assert finalized == [(0.001, "global_aux_loss")]
