# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from examples.swe.qwen38_flash_next.batch_snapshot import (
    capture_training_batches,
    load_batch_snapshot,
    save_batch_snapshot,
)

from areal.infra.rpc import rtensor
from areal.infra.rpc.rtensor import RTensor, TensorShardInfo
from areal.infra.rpc.serialization import serialize_value


def test_capture_limit_keeps_training_calls_and_stops_snapshot_writes(tmp_path):
    batch = {"input_ids": torch.tensor([[10, 11]])}
    actor = SimpleNamespace(
        prepare_batch=Mock(return_value=batch),
        compute_advantages=Mock(side_effect=lambda data: data),
    )
    original = actor.prepare_batch
    with capture_training_batches(actor, tmp_path, {}, max_batches=1):
        for _ in range(3):
            assert actor.prepare_batch() is batch
            assert actor.compute_advantages(batch) is batch
    assert actor.prepare_batch is original
    assert actor.prepare_batch.call_count == 3
    assert actor.compute_advantages.call_count == 3
    assert len(list(tmp_path.glob("*.pt"))) == 3
    assert not list(tmp_path.glob("*-0001.*"))


def test_zero_limit_does_not_wrap_actor_or_create_directory(tmp_path):
    actor = SimpleNamespace(prepare_batch=Mock(), compute_advantages=Mock())
    original = actor.prepare_batch
    directory = tmp_path / "disabled"
    with capture_training_batches(actor, directory, {}, max_batches=0):
        assert actor.prepare_batch is original
        actor.prepare_batch()
    assert not directory.exists()


@pytest.mark.parametrize("limit", [-1, 1.5, True])
def test_invalid_limit_fails_before_mutating_actor(tmp_path, limit):
    actor = SimpleNamespace(prepare_batch=Mock(), compute_advantages=Mock())
    original = actor.prepare_batch
    with pytest.raises(ValueError, match="nonnegative"):
        with capture_training_batches(actor, tmp_path, {}, max_batches=limit):
            pytest.fail("Invalid limit should not enter the capture context")
    assert actor.prepare_batch is original


def test_snapshot_preserves_raw_tokens_masks_logps_pixels_and_remote_references(
    tmp_path, monkeypatch
):
    tensors = {
        "input_ids": torch.tensor([[10, 70011, 70012, 13]]),
        "loss_mask": torch.tensor([[0, 1, 0, 1]]),
        "attention_mask": torch.ones((1, 4), dtype=torch.bool),
        "versions": torch.tensor([[0, 2, 0, 2]]),
        "logprobs": torch.tensor([[0.0, -2.0, 0.0, -3.0]]),
        "pixel_values": torch.arange(12, dtype=torch.bfloat16).reshape(3, 4),
        "image_grid_thw": torch.tensor([[1, 2, 2]]),
    }
    backend = Mock()
    backend.fetch.side_effect = lambda shards: [tensors[s.shard_id] for s in shards]
    monkeypatch.setattr(rtensor, "get_backend", lambda: backend)
    monkeypatch.setattr(rtensor, "_fetch_buffer", {})
    refs = {
        key: RTensor(
            shard=TensorShardInfo(shard_id=key, node_addr="test.invalid:1234"),
            data=tensor.to("meta"),
        )
        for key, tensor in tensors.items()
    }
    batch = {
        k: v for k, v in refs.items() if k not in ("pixel_values", "image_grid_thw")
    }
    image = {k: refs[k] for k in ("pixel_values", "image_grid_thw")}
    batch["multi_modal_input"] = [image, image]
    before = serialize_value(batch)
    path = tmp_path / "raw.pt"
    save_batch_snapshot(path, batch, {"checkpoint": "owned-diagnostic-v2"})
    loaded = load_batch_snapshot(path)
    restored = loaded["batch"]
    assert loaded["metadata"]["checkpoint"] == "owned-diagnostic-v2"
    assert serialize_value(batch) == before
    assert all(ref.data.is_meta for ref in refs.values())
    for key, expected in tensors.items():
        actual = (
            restored["multi_modal_input"][0][key] if key in image else restored[key]
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert actual.device.type == "cpu"
    assert (
        restored["multi_modal_input"][0]["pixel_values"]
        is restored["multi_modal_input"][1]["pixel_values"]
    )
    with pytest.raises(FileExistsError):
        save_batch_snapshot(path, batch, {})
