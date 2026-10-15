# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from areal.engine.r3.asserts import R3Error
from areal.engine.r3.config import resolve_moe_layer_indices, resolve_r3_moe_config
from areal.engine.r3.discovery import discover_native_router_replay


def test_resolve_r3_moe_config_handles_dense_prefix_and_freq_list():
    cfg = SimpleNamespace(
        num_layers=6,
        moe_layer_freq=[1, 1, 0, 0, 1, 1],
        first_k_dense_replace=2,
        moe_router_topk=4,
    )

    resolved = resolve_r3_moe_config(cfg)

    assert resolved.num_moe_layers == 2
    assert resolved.moe_layer_indices == (4, 5)
    assert resolved.global_to_moe_index == {4: 0, 5: 1}
    assert resolved.topk == 4


def test_resolve_r3_moe_config_accepts_omegaconf_list_config():
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {
            "num_layers": 4,
            "moe_layer_freq": [0, 0, 1, 1],
            "moe_router_topk": 2,
        }
    )

    resolved = resolve_r3_moe_config(cfg)

    assert resolved.moe_layer_indices == (2, 3)
    assert resolved.num_moe_layers == 2


def test_resolve_r3_moe_config_accepts_hf_num_hidden_layers():
    cfg = SimpleNamespace(
        num_hidden_layers=4,
        first_k_dense_replace=1,
        num_experts_per_tok=2,
    )

    resolved = resolve_r3_moe_config(cfg)

    assert resolved.num_layers == 4
    assert resolved.moe_layer_indices == (1, 2, 3)
    assert resolved.num_moe_layers == 3
    assert resolved.topk == 2


def test_resolve_r3_moe_config_rejects_flat_dim_guessing():
    cfg = SimpleNamespace(num_layers=2, moe_layer_freq=1)

    with pytest.raises(R3Error, match="Unable to resolve MoE router topk"):
        resolve_r3_moe_config(cfg)


@pytest.mark.parametrize(
    ("freq", "dense_prefix", "expected"),
    [
        (1, 0, (0, 1, 2, 3, 4, 5)),
        (2, 0, (0, 2, 4)),
        (3, 0, (0, 3)),
        (2, 2, (2, 4)),
        (9, 0, (0,)),
    ],
)
def test_integer_moe_frequency_matches_native_mcore(freq, dense_prefix, expected):
    """MCore counts periodic MoE layers from zero, including dense prefixes."""
    config = SimpleNamespace(
        num_layers=6,
        moe_router_topk=2,
        moe_layer_freq=freq,
        first_k_dense_replace=dense_prefix,
    )
    assert resolve_moe_layer_indices(config) == expected


@pytest.mark.parametrize("freq", [0, -2])
def test_integer_moe_frequency_rejects_nonpositive_values(freq):
    """Do not invent a decoder layout for unsupported periodic frequencies."""
    with pytest.raises(R3Error, match="moe_layer_freq must be positive"):
        resolve_r3_moe_config(
            SimpleNamespace(num_layers=4, moe_router_topk=2, moe_layer_freq=freq)
        )


@pytest.mark.parametrize("use_getter", [False, True])
def test_composite_config_uses_text_moe_fields(use_getter):
    """Composite model metadata must not override its decoder's MoE layout."""
    text = SimpleNamespace(
        num_hidden_layers=6,
        num_experts_per_tok=3,
        moe_layer_freq=1,
        first_k_dense_replace=2,
    )
    outer = SimpleNamespace(
        text_config=text, num_hidden_layers=99, num_experts_per_tok=8, moe_layer_freq=1
    )
    if use_getter:
        outer.get_text_config = lambda: text
    resolved = resolve_r3_moe_config(outer)
    assert resolved.num_layers == 6
    assert resolved.topk == 3
    assert resolved.moe_layer_indices == (2, 3, 4, 5)


@pytest.mark.parametrize("freq", [2, [1, 0, 1, 0, 1, 0], [0, 0, 1, 1, 0, 0]])
@pytest.mark.parametrize("composite", [False, True])
def test_r3_config_sparse_or_dense_tail_layout_raises(freq, composite):
    """Reject layouts whose full-layer capture cannot be trimmed by a prefix."""
    config = SimpleNamespace(num_layers=6, moe_router_topk=2, moe_layer_freq=freq)
    if composite:
        config = SimpleNamespace(text_config=config)

    with pytest.raises(R3Error, match="MoE layers must form a contiguous suffix"):
        resolve_r3_moe_config(config)


def test_r3_config_sparse_omegaconf_layout_raises():
    """OmegaConf layer masks obey the same rollout layout restriction."""
    from omegaconf import OmegaConf

    config = OmegaConf.create(
        {"num_layers": 4, "moe_router_topk": 2, "moe_layer_freq": [1, 0, 1, 0]}
    )
    with pytest.raises(R3Error, match="MoE layers must form a contiguous suffix"):
        resolve_r3_moe_config(config)


def test_discover_native_router_replay_collects_instance_refs(monkeypatch):
    class FakeRouter(torch.nn.Module):
        def __init__(self, replay):
            super().__init__()
            self.router_replay = replay

    class FakeLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.router = FakeRouter(replay=object())

    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([FakeLayer(), FakeLayer()])
    monkeypatch.setattr(
        "areal.engine.r3.discovery._resolve_topk_router_type",
        lambda: FakeRouter,
    )

    grouped = discover_native_router_replay(model)

    refs = grouped[0]
    assert len(refs) == 2
    assert [ref.layer_number for ref in refs] == [0, 1]
    assert [ref.layer_number_is_global for ref in refs] == [False, False]
    assert refs[0].router_replay is model.layers[0].router.router_replay
    assert refs[1].router_replay is model.layers[1].router.router_replay


def test_discover_native_router_replay_normalizes_router_layer_number(monkeypatch):
    class FakeRouter(torch.nn.Module):
        def __init__(self, layer_number: int):
            super().__init__()
            self.router_replay = object()
            self.layer_number = layer_number

    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([FakeRouter(3)])
    monkeypatch.setattr(
        "areal.engine.r3.discovery._resolve_topk_router_type",
        lambda: FakeRouter,
    )

    refs = discover_native_router_replay(model)[0]

    assert refs[0].layer_number == 2
    assert refs[0].layer_number_is_global


def test_discover_native_router_replay_missing_native_replay_raises(monkeypatch):
    class FakeRouter(torch.nn.Module):
        router_replay = None

    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([FakeRouter()])
    monkeypatch.setattr(
        "areal.engine.r3.discovery._resolve_topk_router_type",
        lambda: FakeRouter,
    )

    with pytest.raises(R3Error, match="missing native router_replay"):
        discover_native_router_replay(model)
