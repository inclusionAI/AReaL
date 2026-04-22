# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import areal.engine.fsdp_engine as fsdp_engine_mod


def _make_engine():
    engine = object.__new__(fsdp_engine_mod.FSDPEngine)
    engine.model = SimpleNamespace(
        active_adapter="default",
        peft_config={"default": SimpleNamespace(peft_type="mock_lora")},
    )
    engine.config = SimpleNamespace(use_lora=True)
    engine.cpu_offload = None
    engine.model_config = SimpleNamespace(tie_word_embeddings=False)
    engine.logger = Mock()
    return engine


def test_load_lora_adapter_collects_empty_partial_state_on_nonzero_rank(monkeypatch):
    engine = _make_engine()
    events = []

    monkeypatch.setattr(fsdp_engine_mod.dist, "get_rank", lambda: 1)
    monkeypatch.setattr(
        fsdp_engine_mod,
        "load_safetensors_file",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("nonzero ranks must not read adapter safetensors")
        ),
    )
    monkeypatch.setattr(
        fsdp_engine_mod,
        "fsdp2_load_full_state_dict",
        lambda model, partial_state, cpu_offload, strict=None: events.append(
            ("fsdp2_load_full_state_dict", dict(partial_state), strict)
        ),
    )

    engine._load_lora_adapter_from_hf("/tmp/adapter")

    assert events == [
        ("fsdp2_load_full_state_dict", {}, False),
    ]


def test_load_lora_adapter_rank0_remaps_adapter_keys(monkeypatch, tmp_path):
    engine = _make_engine()
    events = []
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    adapter_path = adapter_dir / "adapter_model.safetensors"
    adapter_path.write_bytes(b"stub")

    monkeypatch.setattr(fsdp_engine_mod.dist, "get_rank", lambda: 0)
    monkeypatch.setitem(
        fsdp_engine_mod.PEFT_TYPE_TO_PREFIX_MAPPING,
        "mock_lora",
        "lora_",
    )
    monkeypatch.setattr(
        fsdp_engine_mod,
        "load_safetensors_file",
        lambda path, device="cpu": events.append(
            ("load_safetensors_file", path, device)
        )
        or {
            "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": 2,
            "base_model.model.layers.0.self_attn.q_proj.lora_B.weight": 3,
        },
    )
    monkeypatch.setattr(
        fsdp_engine_mod,
        "fsdp2_load_full_state_dict",
        lambda model, partial_state, cpu_offload, strict=None: events.append(
            ("fsdp2_load_full_state_dict", dict(partial_state), strict)
        ),
    )

    engine._load_lora_adapter_from_hf(str(adapter_dir))

    assert events == [
        ("load_safetensors_file", str(adapter_path), "cpu"),
        (
            "fsdp2_load_full_state_dict",
            {
                "base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight": 2,
                "base_model.model.layers.0.self_attn.q_proj.lora_B.default.weight": 3,
            },
            False,
        ),
    ]
