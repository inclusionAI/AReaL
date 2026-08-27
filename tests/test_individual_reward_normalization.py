# SPDX-License-Identifier: Apache-2.0

"""Individual turn scores retain one explicit normalization reference per episode."""

from unittest.mock import Mock

import pytest
import torch

from areal.api import RolloutWorkflow
from areal.api.cli_args import NormConfig
from areal.infra import workflow_context
from areal.infra.remote_inf_engine import GroupedRolloutWorkflow
from areal.utils.data import (
    Normalization,
    TrajBatchMeta,
    concat_batch,
    normalize_rollout_rewards,
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


@pytest.mark.parametrize("bias,scale,clip", [(0.0, 1.0, float("inf")), (1.0, 2.0, 9.0)])
def test_shaped_normalization_rewards_use_upstream_logical_unit_contract(
    bias, scale, clip
):
    """The override contains raw shaped references per trajectory, not per row."""
    data = _reward_batch()
    batch, meta = concat_batch([data])
    norm = Normalization(
        NormConfig(mean_level="group", std_level="group", std_unbiased=False)
    )
    shaped = torch.tensor([2.0, 4.0])

    actual = normalize_rollout_rewards(
        batch["rewards"],
        norm,
        meta,
        reward_bias=bias,
        reward_scaling=scale,
        reward_clip=clip,
        normalization_rewards=shaped,
        traj_starts=batch["begin_of_trajectory"],
    )

    scores = ((batch["rewards"] + bias) * scale).clamp(-clip, clip)
    references = ((shaped + bias) * scale).clamp(-clip, clip)
    expected = (scores - references.mean()) / (
        references.std(unbiased=False) + norm.eps
    )
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    "bias,scale,clip", [(0.0, 1.0, float("inf")), (2.0, 2.0, 11.0)]
)
def test_mixed_references_keep_existing_split_and_singleton_penalty_policy(
    bias, scale, clip
):
    """Explicit outcomes coexist with the existing split/singleton penalty references."""
    scores = torch.tensor([0.9, 1.0, 1.0, 2.0, 3.0])
    unpenalized = torch.tensor([0.9, 1.0, 2.0, 2.0, 4.0])
    starts = torch.tensor([1, 0, 1, 0, 1])
    meta = TrajBatchMeta(1, [5], [1], [[1.0, 1.0, None, None, None]])
    norm = Normalization(
        NormConfig(mean_level="group", std_level="group", std_unbiased=False)
    )

    actual = normalize_rollout_rewards(
        scores,
        norm,
        meta,
        reward_bias=bias,
        reward_scaling=scale,
        reward_clip=clip,
        unpenalized_rewards=unpenalized,
        traj_starts=starts,
    )

    # Explicit reference 1; split reference 2 before penalties; singleton score 3.
    references = ((torch.tensor([1.0, 2.0, 3.0]) + bias) * scale).clamp(-clip, clip)
    transformed = ((scores + bias) * scale).clamp(-clip, clip)
    expected = (transformed - references.mean()) / (
        references.std(unbiased=False) + norm.eps
    )
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_shaped_override_ignores_unused_raw_reference_transform():
    """A shaped override replaces raw references before their affine transform."""
    data = _reward_batch()
    data["rollout_rewards"] = [1e300, 1e300, None]
    batch, meta = concat_batch([data])
    norm = Normalization(
        NormConfig(mean_level="group", std_level="group", std_unbiased=False)
    )

    actual = normalize_rollout_rewards(
        batch["rewards"],
        norm,
        meta,
        normalization_rewards=torch.tensor([2.0, 4.0]),
        traj_starts=batch["begin_of_trajectory"],
    )

    expected = (batch["rewards"] - 3.0) / (1.0 + norm.eps)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
