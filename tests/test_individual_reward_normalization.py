# SPDX-License-Identifier: Apache-2.0

"""Individual turn scores retain one explicit normalization reference per episode."""

from unittest.mock import Mock

import pytest
import torch

from areal.api import RolloutWorkflow
from areal.infra import workflow_context
from areal.infra.remote_inf_engine import GroupedRolloutWorkflow
from areal.utils.data import (
    concat_batch,
    split_batch,
)


def _reward_batch():
    return {
        "input_ids": torch.ones(3, 3, dtype=torch.long),
        "attention_mask": torch.ones(3, 3, dtype=torch.bool),
        "loss_mask": torch.tensor([[0, 1, 1]] * 3, dtype=torch.bool),
        "logprobs": torch.zeros(3, 3),
        "rewards": torch.tensor([0.9, 1.0, 3.0]),
        "begin_of_trajectory": torch.tensor([1, 0, 1], dtype=torch.int32),
        "rollout_rewards": [1.0, 1.0, None],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("reference_source", ["first", "second", "all-none", "absent"])
async def test_grouped_tensor_references_are_collected_outside_concat(reference_source):
    """Mixed reference columns do not change markers or mutate episode inputs."""
    episodes = [
        {
            "input_ids": torch.ones(rows, 3, dtype=torch.long),
            "attention_mask": torch.ones(rows, 3, dtype=torch.bool),
            "rewards": torch.ones(rows),
        }
        for rows in (2, 1)
    ]
    expected = None
    if reference_source == "first":
        episodes[0]["rollout_rewards"] = [1.0, 1.0]
        expected = [1.0, 1.0, None]
    elif reference_source == "second":
        episodes[1]["rollout_rewards"] = [3.0]
        expected = [None, None, 3.0]
    elif reference_source == "all-none":
        episodes[0]["rollout_rewards"] = [None, None]
        expected = [None, None, None]
    keys = [set(episode) for episode in episodes]

    class Episodes(RolloutWorkflow):
        async def arun_episode(self, engine, data):
            return episodes[workflow_context.get().sample_idx]

    result = await GroupedRolloutWorkflow(Episodes(), 2, Mock()).arun_episode(None, {})

    torch.testing.assert_close(
        result["begin_of_trajectory"],
        torch.tensor([1, 0, 1], dtype=torch.int32),
        rtol=0,
        atol=0,
    )
    if expected is None:
        assert "rollout_rewards" not in result
    else:
        assert result["rollout_rewards"] == expected
    assert [set(episode) for episode in episodes] == keys


@pytest.mark.parametrize("second_has_metadata", [False, True])
def test_concat_split_restores_each_prompt_reference_column(second_has_metadata):
    first = _reward_batch()
    second = {
        "input_ids": torch.ones(1, 4, dtype=torch.long),
        "attention_mask": torch.ones(1, 4, dtype=torch.bool),
        "loss_mask": torch.ones(1, 4, dtype=torch.bool),
        "logprobs": torch.zeros(1, 4),
        "rewards": torch.tensor([5.0]),
        "begin_of_trajectory": torch.ones(1, dtype=torch.int32),
    }
    if second_has_metadata:
        second["rollout_rewards"] = [5.0]
    originals = [first, second]

    batched, meta = concat_batch(originals)
    assert "rollout_rewards" not in batched
    restored = split_batch(batched, meta)

    for original, result in zip(originals, restored, strict=True):
        assert set(result) == set(original)
        for key, value in original.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(result[key], value, rtol=0, atol=0)
            else:
                assert result[key] == value
