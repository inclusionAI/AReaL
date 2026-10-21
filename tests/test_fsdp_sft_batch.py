# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from areal.api.cli_args import MicroBatchSpec
from areal.engine.fsdp_engine import FSDPPPOActor
from areal.utils.data import (
    amend_position_ids,
    concat_batch,
    pack_tensor_dict,
    split_padded_tensor_dict_into_mb_list,
    unsqueeze_packed_tensor_dict,
)


@pytest.mark.parametrize("lengths", [(64, 64), (32, 64)])
@pytest.mark.parametrize("with_offsets", [True, False])
def test_sft_batch_packing_keeps_sequences_and_batch_dimension(lengths, with_offsets):
    """Per-example sequence offsets must not overwrite packed batch offsets."""
    captured = {}

    def train_batch(data, **kwargs):
        batched, _ = concat_batch(data)
        batched = amend_position_ids(batched)
        micro_batches = split_padded_tensor_dict_into_mb_list(
            batched, MicroBatchSpec(max_tokens_per_mb=256, granularity=2)
        )
        assert len(micro_batches.mbs) == 1
        packed = pack_tensor_dict(micro_batches.mbs[0])
        captured.update(unsqueeze_packed_tensor_dict(packed))
        return {"update_successful": 1}

    engine = SimpleNamespace(train_batch=train_batch)
    data = [
        {
            "input_ids": list(range(start, start + length)),
            "attention_mask": [1] * length,
            "loss_mask": [0] + [1] * (length - 1),
            **({"cu_seqlens": [0, length]} if with_offsets else {}),
        }
        for start, length in zip((100, 200), lengths)
    ]

    result = FSDPPPOActor.sft_train_batch(engine, data)

    assert result["update_successful"] == 1
    assert captured["input_ids"].shape == (1, sum(lengths))
    assert captured["position_ids"].shape == (1, sum(lengths))
    assert captured["loss_mask"].shape == (1, sum(lengths))
    torch.testing.assert_close(
        captured["cu_seqlens"],
        torch.tensor([0, lengths[0], sum(lengths)], dtype=torch.int32),
        rtol=0,
        atol=0,
    )
    assert captured["input_ids"][0, lengths[0]].item() == 200
