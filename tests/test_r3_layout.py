# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from areal.engine.r3.asserts import R3Error
from areal.engine.r3.layout import prepare_native_replay_slabs


def _mb_input(
    *,
    orig_cu: list[int],
    padded_cu: list[int],
) -> SimpleNamespace:
    old_cu = torch.tensor(orig_cu, dtype=torch.int32)
    padded = torch.tensor(padded_cu, dtype=torch.int32)
    return SimpleNamespace(
        orig_mb={"cu_seqlens": old_cu},
        padded_mb={"cu_seqlens": padded},
        old_cu_seqlens=old_cu,
    )


def _patch_parallel(monkeypatch, *, tp_size: int = 1) -> None:
    monkeypatch.setattr(
        "areal.engine.megatron_utils.packed_context_parallel.mpu.get_context_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(
        "areal.engine.r3.layout._scatter_to_sequence_parallel_region",
        lambda tensor: tensor[: tensor.shape[0] // tp_size] if tp_size > 1 else tensor,
    )


def test_prepare_native_replay_slabs_packs_alignment_and_padding(monkeypatch):
    _patch_parallel(monkeypatch)
    routed = torch.arange(1, 1 + 2 * 4 * 3 * 2, dtype=torch.int32).reshape(2, 4, 3, 2)
    valid = torch.tensor([True, True])
    mb_input = _mb_input(orig_cu=[0, 3, 5], padded_cu=[0, 4, 8, 12])

    slabs = prepare_native_replay_slabs(
        routed,
        valid,
        mb_input,
        local_moe_indices=[0, 2],
    )

    assert not slabs.skip_replay
    assert len(slabs.slabs) == 2
    assert slabs.slabs[0].shape == (12, 2)
    torch.testing.assert_close(slabs.slabs[0][0], routed[0, 0, 0].long())
    torch.testing.assert_close(slabs.slabs[0][3], routed[0, 2, 0].long())
    torch.testing.assert_close(slabs.slabs[0][6], routed[1, 1, 0].long())
    torch.testing.assert_close(slabs.slabs[0][11], routed[1, 1, 0].long())
    torch.testing.assert_close(slabs.slabs[1][0], routed[0, 0, 2].long())


def test_prepare_native_replay_slabs_invalid_sample_skips(monkeypatch):
    _patch_parallel(monkeypatch)
    routed = torch.ones(2, 4, 1, 2, dtype=torch.int32)
    valid = torch.tensor([True, False])

    slabs = prepare_native_replay_slabs(
        routed,
        valid,
        _mb_input(orig_cu=[0, 3, 5], padded_cu=[0, 4, 8]),
        local_moe_indices=[0],
    )

    assert slabs.skip_replay
    assert slabs.reason == "invalid_sample"


def test_prepare_native_replay_slabs_trusts_preprocessed_routing_valid(
    monkeypatch,
):
    _patch_parallel(monkeypatch)
    routed = torch.ones(1, 4, 1, 2, dtype=torch.int32)
    routed[0, 1] = 0
    valid = torch.tensor([True])

    slabs = prepare_native_replay_slabs(
        routed,
        valid,
        _mb_input(orig_cu=[0, 4], padded_cu=[0, 4]),
        local_moe_indices=[0],
    )

    assert not slabs.skip_replay
    torch.testing.assert_close(slabs.slabs[0], routed[0, :, 0].long())


def test_prepare_native_replay_slabs_tp_scatter_preserves_layer_slices(monkeypatch):
    _patch_parallel(monkeypatch, tp_size=2)
    routed = torch.arange(1, 1 + 1 * 4 * 2 * 2, dtype=torch.int32).reshape(1, 4, 2, 2)

    slabs = prepare_native_replay_slabs(
        routed,
        torch.tensor([True]),
        _mb_input(orig_cu=[0, 4], padded_cu=[0, 4]),
        local_moe_indices=[1],
    )

    assert not slabs.skip_replay
    assert slabs.slabs[0].shape == (2, 2)
    torch.testing.assert_close(slabs.slabs[0], routed[0, :2, 1].long())


def test_prepare_native_replay_slabs_skips_tp_scatter_without_sequence_parallel(
    monkeypatch,
):
    _patch_parallel(monkeypatch, tp_size=2)
    routed = torch.arange(1, 1 + 1 * 4 * 2 * 2, dtype=torch.int32).reshape(1, 4, 2, 2)

    slabs = prepare_native_replay_slabs(
        routed,
        torch.tensor([True]),
        _mb_input(orig_cu=[0, 4], padded_cu=[0, 4]),
        local_moe_indices=[1],
        sequence_parallel=False,
    )

    assert not slabs.skip_replay
    assert slabs.slabs[0].shape == (4, 2)
    torch.testing.assert_close(slabs.slabs[0], routed[0, :, 1].long())


def test_prepare_native_replay_slabs_rejects_bad_local_moe_index(monkeypatch):
    _patch_parallel(monkeypatch)
    routed = torch.ones(1, 4, 2, 2, dtype=torch.int32)

    with pytest.raises(R3Error, match="out of routed_experts range"):
        prepare_native_replay_slabs(
            routed,
            torch.tensor([True]),
            _mb_input(orig_cu=[0, 4], padded_cu=[0, 4]),
            local_moe_indices=[2],
        )


@pytest.mark.parametrize("cp_rank", [0, 1])
@pytest.mark.parametrize("tp_rank", [0, 1])
@pytest.mark.parametrize("sequence_parallel", [False, True])
def test_cpu_replay_layout_preserves_cp_zigzag_then_native_sp(
    monkeypatch, cp_rank, tp_rank, sequence_parallel
):
    """Local CPU slabs preserve padding, reordered layers and native SP order."""
    from megatron.core import parallel_state as mpu

    monkeypatch.setattr(mpu, "get_context_parallel_world_size", lambda: 2)
    monkeypatch.setattr(mpu, "get_context_parallel_rank", lambda: cp_rank)
    monkeypatch.setattr(mpu, "get_tensor_model_parallel_world_size", lambda: 2)
    group = SimpleNamespace(size=lambda: 2, rank=lambda: tp_rank)
    monkeypatch.setattr(
        "megatron.core.tensor_parallel.mappings.get_tensor_model_parallel_group_if_none",
        lambda _: group,
    )
    routed = torch.arange(2 * 6 * 4 * 2, dtype=torch.int32).reshape(2, 6, 4, 2)
    mb = _mb_input(orig_cu=[0, 5, 8], padded_cu=[0, 8, 16, 24])
    # Explicitly build the three padded segments, including batch padding.
    packed = torch.cat(
        [
            routed[0, :5],
            routed[0, 4:5].expand(3, -1, -1),
            routed[1, :3],
            routed[1, 2:3].expand(5, -1, -1),
            routed[1, 2:3].expand(8, -1, -1),
        ]
    )
    token_indices = [
        offset + index
        for offset in (0, 8, 16)
        for index in ([0, 1, 6, 7] if cp_rank == 0 else [2, 3, 4, 5])
    ]
    expected = packed[token_indices]
    if sequence_parallel:
        expected = expected.chunk(2, dim=0)[tp_rank]
    result = prepare_native_replay_slabs(
        routed,
        torch.tensor([True, True]),
        mb,
        [3, 1],
        sequence_parallel=sequence_parallel,
    )
    assert not result.skip_replay
    for slab, layer in zip(result.slabs, [3, 1], strict=True):
        assert slab.device.type == "cpu"
        assert slab.dtype == torch.long
        assert slab.is_contiguous()
        torch.testing.assert_close(slab, expected[:, layer].long(), rtol=0, atol=0)


@pytest.mark.parametrize("local_moe_indices", [[], ()])
def test_prepare_native_replay_slabs_no_local_routers_noops(local_moe_indices):
    slabs = prepare_native_replay_slabs(
        torch.ones(1, 2, 1, 2, dtype=torch.int32),
        torch.tensor([True]),
        _mb_input(orig_cu=[0, 2], padded_cu=[0, 2]),
        local_moe_indices=local_moe_indices,
    )

    assert not slabs.skip_replay
    assert slabs.slabs == []
