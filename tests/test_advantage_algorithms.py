# SPDX-License-Identifier: Apache-2.0

"""Regression tests for independently composed temporal reward channels."""

from dataclasses import replace

import pytest
import torch

from areal.trainer.ppo.advantage_algorithms import TemporalInputs, token_gae, turn_gae


def _inputs() -> TemporalInputs:
    return TemporalInputs(
        rewards=torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
        values=torch.tensor([[0.2, 0.3, 2.0, 99.0]]),
        loss_mask=torch.tensor([[1.0, 1.0, 0.0, 0.0]]),
        seq_no_eos_mask=torch.tensor([True]),
        bootstrap_values=torch.tensor([2.0]),
        discount=1.0,
        gae_lambda=1.0,
        turn_ids=torch.tensor([[0, 0, -1, -1]]),
        kl_rewards=torch.tensor([[-0.1, -0.2, 0.0, 0.0]]),
        process_rewards=torch.tensor([[0.0, 0.5, 0.0, 0.0]]),
    )


@pytest.mark.parametrize("estimator", [token_gae, turn_gae])
def test_temporal_estimator_padding_does_not_change_valid_targets(estimator):
    """The explicit endpoint bootstrap must win over padded value slots."""
    inputs = _inputs()
    result = estimator(inputs)
    padded_values = inputs.values.clone()
    padded_values[:, -1] = -12345.0
    padded = estimator(replace(inputs, values=padded_values))
    torch.testing.assert_close(
        result.advantages[:, :2], padded.advantages[:, :2], rtol=0, atol=0
    )
    torch.testing.assert_close(
        result.returns[:, :2], padded.returns[:, :2], rtol=0, atol=0
    )
    expected = [3.2, 3.3] if estimator is token_gae else [3.5, 3.5]
    torch.testing.assert_close(
        result.returns[:, :2], torch.tensor([expected]), rtol=0, atol=1e-6
    )


def test_turn_gae_kl_changes_only_actor_targets():
    """Token KL must not be broadcast through the turn or enter returns."""
    inputs = _inputs()
    result = turn_gae(inputs)
    without_kl = turn_gae(replace(inputs, kl_rewards=None))
    torch.testing.assert_close(result.returns, without_kl.returns, rtol=0, atol=0)
    torch.testing.assert_close(
        result.advantages,
        without_kl.advantages + inputs.kl_rewards,
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("estimator", [token_gae, turn_gae])
def test_temporal_estimator_preserves_reward_addition_order(estimator):
    """Cancellation exposes accidental reassociation of the reward channels."""
    inputs = _inputs()
    inputs = replace(
        inputs,
        rewards=torch.full_like(inputs.rewards, 1e8),
        kl_rewards=torch.full_like(inputs.rewards, -1e8),
        process_rewards=torch.ones_like(inputs.rewards),
    )
    result = estimator(inputs)
    expected = (
        torch.ones_like(inputs.rewards)
        if estimator is token_gae
        else torch.zeros_like(inputs.rewards)
    )
    torch.testing.assert_close(result.total_rewards, expected, rtol=0, atol=0)


def test_turn_gae_missing_turn_ids_rejected():
    """Turn estimation requires explicit rollout structure."""
    with pytest.raises(ValueError, match="requires rollout data.*turn_ids"):
        turn_gae(replace(_inputs(), turn_ids=None))
