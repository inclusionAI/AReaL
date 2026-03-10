from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import torch.nn as nn

from areal.api.io_struct import WeightUpdateMeta
from areal.experimental.engine import archon_weight_sync
from areal.experimental.engine.archon_engine import ArchonEngine
from areal.experimental.models.archon.lora import LoRALinear, get_adapter_params
from areal.experimental.models.archon.qwen2.infra import parallelize as qwen2_parallelize


class _ToyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.wq = nn.Linear(8, 8)
        self.other = nn.Linear(8, 8)
        self.inner = nn.Module()
        self.inner.wv = nn.Linear(8, 8)


def _make_engine(model: nn.Module, target_modules: list[str]) -> ArchonEngine:
    engine = ArchonEngine.__new__(ArchonEngine)
    engine.model = model
    engine.model_parts = [model]
    engine.logger = Mock()
    engine.lora_config = SimpleNamespace(
        rank=4,
        alpha=8.0,
        target_modules=target_modules,
    )
    engine.state_dict_adapter = SimpleNamespace(
        to_peft_module_map={
            "wq": "q_proj",
            "wv": "v_proj",
        }
    )
    return engine


def test_apply_lora_replaces_target_linear_modules():
    model = _ToyBlock()
    engine = _make_engine(model, ["wq", "v_proj"])

    engine._apply_lora()

    assert isinstance(model.wq, LoRALinear)
    assert isinstance(model.inner.wv, LoRALinear)
    assert isinstance(model.other, nn.Linear)
    assert get_adapter_params(model)


def test_from_linear_preserves_weight_identity():
    """from_linear must transfer the original parameter, not copy it.

    After TP the weight is a DTensor; copying would fail with
    ``got mixed torch.Tensor and DTensor``.
    """
    import torch

    linear = nn.Linear(8, 8)
    original_weight = linear.weight
    lora = LoRALinear.from_linear(linear, rank=4, alpha=8.0)
    assert lora.weight is original_weight


def test_freeze_non_lora_params_keeps_only_adapter_trainable():
    model = _ToyBlock()
    engine = _make_engine(model, ["wq"])

    engine._apply_lora()
    engine._freeze_non_lora_params()

    assert model.wq.weight.requires_grad is False
    assert model.wq.lora_a.weight.requires_grad is True
    assert model.wq.lora_b.weight.requires_grad is True
    assert model.other.weight.requires_grad is False
    assert model.inner.wv.weight.requires_grad is False


class _ImmediateFuture:
    def result(self):
        return None


def test_update_weights_from_disk_uses_lora_adapter(monkeypatch, tmp_path):
    meta = WeightUpdateMeta.from_disk(
        experiment_name="exp",
        trial_name="trial",
        file_root=str(tmp_path),
        use_lora=True,
        lora_name="lora",
        base_model_name="base-model",
    )

    calls: list[tuple[str, str, str]] = []
    rollout_engine = SimpleNamespace(
        update_weights_from_disk=lambda _: _ImmediateFuture()
    )
    engine = SimpleNamespace(
        rollout_engine=rollout_engine,
        lora_config=SimpleNamespace(rank=4),
        config=SimpleNamespace(
            experiment_name="exp",
            trial_name="trial",
            path="engine-model",
        ),
        cpu_group=None,
        get_version=lambda: 0,
    )

    monkeypatch.setattr(archon_weight_sync.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(archon_weight_sync.dist, "barrier", lambda group=None: None)
    monkeypatch.setattr(archon_weight_sync.current_platform, "synchronize", lambda: None)
    monkeypatch.setattr(archon_weight_sync.name_resolve, "add", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        archon_weight_sync.names,
        "update_weights_from_disk",
        lambda *args: "update-name",
    )
    monkeypatch.setattr(
        "areal.experimental.engine.archon_lora_checkpoint.save_lora_adapter",
        lambda engine_arg, path_arg, base_model_path: calls.append(
            ("lora", path_arg, base_model_path)
        ),
    )
    monkeypatch.setattr(
        archon_weight_sync,
        "save_model_to_hf",
        lambda *args, **kwargs: calls.append(("full", "", "")),
    )

    archon_weight_sync.update_weights_from_disk(meta, engine)

    assert calls == [("lora", meta.path, "base-model")]


def test_update_weights_from_disk_falls_back_to_full_model(monkeypatch, tmp_path):
    meta = WeightUpdateMeta.from_disk(
        experiment_name="exp",
        trial_name="trial",
        file_root=str(tmp_path),
    )

    calls: list[str] = []
    rollout_engine = SimpleNamespace(
        update_weights_from_disk=lambda _: _ImmediateFuture()
    )
    engine = SimpleNamespace(
        rollout_engine=rollout_engine,
        lora_config=None,
        tokenizer=None,
        config=SimpleNamespace(
            experiment_name="exp",
            trial_name="trial",
            path="engine-model",
        ),
        cpu_group=None,
        get_version=lambda: 0,
    )

    monkeypatch.setattr(archon_weight_sync.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(archon_weight_sync.dist, "barrier", lambda group=None: None)
    monkeypatch.setattr(archon_weight_sync.current_platform, "synchronize", lambda: None)
    monkeypatch.setattr(archon_weight_sync.name_resolve, "add", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        archon_weight_sync.names,
        "update_weights_from_disk",
        lambda *args: "update-name",
    )
    monkeypatch.setattr(
        archon_weight_sync,
        "save_model_to_hf",
        lambda *args, **kwargs: calls.append("full"),
    )

    archon_weight_sync.update_weights_from_disk(meta, engine)

    assert calls == ["full"]


def test_qwen2_parallelize_applies_lora_after_tp_and_cp(monkeypatch):
    order: list[str] = []
    model = SimpleNamespace(
        model_args=SimpleNamespace(enable_weight_tying=False),
    )
    parallel_dims = SimpleNamespace(
        tp_enabled=True,
        cp_enabled=True,
        pp_enabled=False,
        tp=2,
        get_mesh=lambda name: object(),
        get_group=lambda name: object(),
    )

    monkeypatch.setattr(
        qwen2_parallelize,
        "apply_tp",
        lambda *args, **kwargs: order.append("tp"),
    )
    monkeypatch.setattr(
        qwen2_parallelize,
        "apply_cp",
        lambda *args, **kwargs: order.append("cp"),
    )
    monkeypatch.setattr(
        qwen2_parallelize,
        "apply_compile",
        lambda *args, **kwargs: order.append("compile"),
    )
    monkeypatch.setattr(
        qwen2_parallelize,
        "apply_fsdp",
        lambda *args, **kwargs: order.append("fsdp"),
    )

    qwen2_parallelize.parallelize_qwen2(
        model,
        parallel_dims,
        enable_compile=True,
        apply_lora_fn=lambda module: order.append("lora"),
    )

    assert order == ["tp", "cp", "lora", "compile", "fsdp"]
