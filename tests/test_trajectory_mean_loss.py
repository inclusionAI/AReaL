# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest
import torch
from omegaconf import OmegaConf

from tests.test_prompt_mean_actor import _actor

from areal.api.cli_args import MOPDLossConfig, PPOActorConfig
from areal.infra.dist_rollout import _pack_gathered_trajectories
from areal.infra.rpc.serialization import deserialize_value, serialize_value
from areal.trainer.ppo.actor import PPOActor, _group_training_metrics
from areal.trainer.ppo.loss_reduction import (
    PG_TOKEN_WEIGHTS,
    prepare_policy_gradient_batch,
)
from areal.utils.data import (
    RolloutGroup,
    concat_batch,
    pack_tensor_dict,
    split_batch,
    split_training_batch_into_microbatches,
)
from areal.utils.functional.loss_aggregation import SequenceMean


def _groups():
    """Three active rollouts over two prompts, plus an empty logical rollout."""
    masks = (
        [[0, 1, 1, 0], [0, 1, 0, 0], [0, 1, 1, 1]],
        [[0, 1, 0, 0], [0, 0, 0, 0]],
    )
    groups = []
    offset = 0
    for counts, mask in zip(((2, 1), (1, 1)), masks, strict=True):
        mask = torch.tensor(mask, dtype=torch.bool)
        values = (
            torch.arange(offset * 4 + 1, (offset + len(mask)) * 4 + 1)
            .reshape(-1, 4)
            .float()
        )
        groups.append(
            dict(
                input_ids=torch.arange(offset, offset + len(mask))[:, None]
                .expand(-1, 4)
                .clone(),
                attention_mask=torch.ones_like(mask),
                loss_mask=mask,
                advantages=values,
                logprobs=torch.zeros_like(values),
                prox_logp=torch.zeros_like(values),
                kl_rewards=torch.zeros_like(values),
                tot_rewards=torch.zeros_like(values),
                rewards=torch.ones(len(mask)),
                rollout_group=RolloutGroup(counts),
            )
        )
        offset += len(mask)
    return groups


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("requested_steps", [1, 2, 8])
@pytest.mark.parametrize("surrogate", ["ppo", "sapo", "cispo"])
def test_actor_trajectory_mean_preserves_frozen_objective_and_response_schedule(
    packed, requested_steps, surrogate
):
    """Full-rollout coefficients survive cross-step, packed and filtered fragments."""
    groups = _groups()
    full, meta = concat_batch(groups)
    schedule = split_training_batch_into_microbatches(full, n_mbs=requested_steps)
    # Independent oracle: numerator filtering drops row 1 but not its denominator.
    parameter = torch.tensor(0.0, requires_grad=True)
    numerator_mask = full["loss_mask"].clone()
    numerator_mask[1] = False
    oracle = parameter * 0
    offset, active = 0, 0
    for rows in meta.rollout_row_counts:
        mask = full["loss_mask"][offset : offset + rows]
        count = mask.count_nonzero()
        if count:
            oracle = (
                oracle
                - (
                    full["advantages"][offset : offset + rows]
                    * numerator_mask[offset : offset + rows]
                ).sum()
                / count
            )
            active += 1
        offset += rows
    oracle = oracle / active
    step_losses = []

    def train_batch(batch, *, loss_fn, loss_weight_fn):
        index = len(step_losses)
        torch.testing.assert_close(
            batch["input_ids"], schedule[index]["input_ids"], rtol=0, atol=0
        )
        losses, weights = [], []
        # Reverse engine microbatches to exercise row reordering with coefficients.
        for row in reversed(range(len(batch["loss_mask"]))):
            mb = {key: value[row : row + 1] for key, value in batch.items()}
            retained = numerator_mask[mb["input_ids"][:, 0]]
            if packed:
                mb = pack_tensor_dict(mb)
                retained = retained.flatten()
            with patch(
                "areal.trainer.ppo.actor._apply_m2po_masking", return_value=retained
            ):
                loss = loss_fn(
                    logprobs=parameter.expand_as(mb["logprobs"]),
                    entropy=torch.zeros_like(mb["logprobs"]),
                    input_data=mb,
                )
            losses.append(loss)
            weights.append(loss_weight_fn(mb))
        total_weight = sum(weights)
        torch.testing.assert_close(
            total_weight, torch.tensor(1.0), rtol=1e-6, atol=1e-6
        )
        step_losses.append(
            sum(loss * weight for loss, weight in zip(losses, weights, strict=True))
            / total_weight
        )
        return {}

    actor = _actor("traj_mean", requested_steps, train_batch)
    actor.m2_threshold = 0.1
    if surrogate == "sapo":
        actor.config.use_sapo_loss = True
    if surrogate == "cispo":
        actor.config.use_cispo_loss = True
        actor.config.eps_clip = 1.0
        actor.config.eps_clip_higher = 4.0
    with patch("areal.trainer.ppo.actor.stats_tracker", MagicMock()):
        actor.ppo_update(groups)
    actual = torch.stack(step_losses).mean()
    # At ratio=1 SAPO has value -2*A/tau_pos and CISPO has value zero.
    expected_loss = oracle * {"ppo": 1.0, "sapo": 2.0, "cispo": 0.0}[surrogate]
    torch.testing.assert_close(
        actual,
        expected_loss,
        rtol=1e-6,
        atol=1e-6,
    )
    # At logprob=0 each surrogate has d(loss)/d(logprob) = -advantage.
    gradient = torch.autograd.grad(actual, parameter)[0]
    torch.testing.assert_close(gradient, oracle, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("split", [False, True])
def test_same_rollout_as_one_or_multiple_rows_has_same_loss_and_gradient(split):
    """Export representation must not multiply a logical rollout's weight."""
    parameter = torch.tensor(2.0, requires_grad=True)
    values = torch.tensor([[1.0, 3.0, 5.0, 7.0], [9.0, 11.0, 0.0, 0.0]])
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)
    sizes = [1, 1]
    if split:
        values = torch.cat((values[:1].reshape(2, 2), values[1:, :2]))
        mask = torch.ones_like(values, dtype=torch.bool)
        sizes = [2, 1]
    data = {"loss_mask": mask}
    step = prepare_policy_gradient_batch(
        data, mode="traj_mean", trajectory_sizes=sizes
    ).for_steps([data])[0]
    loss = step.bind(data).aggregate(parameter * values, mask)
    torch.testing.assert_close(loss, torch.tensor(14.0), rtol=0, atol=0)
    torch.testing.assert_close(
        torch.autograd.grad(loss, parameter)[0], torch.tensor(7.0), rtol=0, atol=0
    )


def test_unannotated_rows_match_sequence_mean_in_one_step():
    """Fallback defines one logical rollout per row without guessing membership."""
    mask = torch.tensor([[1, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=torch.bool)
    loss = torch.arange(9).reshape(3, 3).float()
    data = {"loss_mask": mask}
    step = prepare_policy_gradient_batch(data, mode="traj_mean").for_steps([data])[0]
    torch.testing.assert_close(
        step.bind(data).aggregate(loss, mask),
        SequenceMean(mask).aggregate(loss, mask),
        rtol=0,
        atol=0,
    )


def test_rollout_membership_survives_rpc_redistribution_and_concat():
    """Use the collector's validated logical metadata rather than row count."""
    groups = _groups()
    wire = deserialize_value(serialize_value(groups))
    gathered = _pack_gathered_trajectories([wire, []], world_size=2, rank=0)
    data, meta = concat_batch(gathered.all_data)
    assert meta.rollout_row_counts == [2, 1, 1, 1]
    assert [group["rollout_group"].row_counts for group in split_batch(data, meta)] == [
        (2, 1),
        (1, 1),
    ]
    # A mixed batch leaves unannotated prompt rows as singleton rollouts.
    groups[1].pop("rollout_group")
    assert concat_batch(groups)[1].rollout_row_counts == [2, 1, 1, 1]


def test_trajectory_group_metrics_count_active_rollouts_not_export_rows():
    """The first prompt has three rows but only two active logical rollouts."""
    data, meta = concat_batch(_groups())
    prepare_policy_gradient_batch(
        data, mode="traj_mean", trajectory_sizes=meta.rollout_row_counts
    )
    starts, sizes, weights = _group_training_metrics(
        data["loss_mask"],
        meta.traj_group_sizes,
        meta.logical_group_sizes,
        "traj_mean",
        data[PG_TOKEN_WEIGHTS],
    )
    torch.testing.assert_close(
        weights[starts], torch.tensor([2.0, 1.0]), rtol=1e-6, atol=1e-6
    )
    torch.testing.assert_close(sizes[starts], torch.tensor([2.0, 2.0]), rtol=0, atol=0)


@pytest.mark.parametrize(
    "sizes", [[0, 2], [-1, 4], [1.5, 1.5], [True, 2], [1], torch.tensor([1, 2])]
)
def test_invalid_trajectory_membership_fails_before_scheduling(sizes):
    """Malformed membership cannot silently regroup loss denominators."""
    with pytest.raises((ValueError, TypeError)):
        prepare_policy_gradient_batch(
            {"loss_mask": torch.ones(3, 2, dtype=torch.bool)},
            mode="traj_mean",
            trajectory_sizes=sizes,
        )


def test_trajectory_mean_rejects_globally_empty_update():
    data = {"loss_mask": torch.zeros(2, 3, dtype=torch.bool)}
    prepared = prepare_policy_gradient_batch(
        data, mode="traj_mean", trajectory_sizes=[2]
    )
    with pytest.raises(ValueError, match="active trajectories"):
        prepared.for_steps([data])


def test_trajectory_mean_config_and_unsupported_objectives():
    config = OmegaConf.structured(PPOActorConfig(loss_aggregation="traj_mean"))
    assert config.loss_aggregation == "traj_mean"
    with pytest.raises(ValueError, match="only used"):
        PPOActorConfig(loss_aggregation="traj_mean", loss_aggregation_divisor=4)
    with pytest.raises(ValueError, match="tree"):
        PPOActorConfig(loss_aggregation="traj_mean", enable_tree_training=True)
    actor = _actor("traj_mean", 1, MagicMock())
    with pytest.raises(ValueError, match="Distillation"):
        actor._ppo_update({"teacher_logp": torch.zeros(1)})
    with pytest.raises(ValueError, match="MOPD distillation"):
        PPOActor.configure_mopd_loss(
            actor, MOPDLossConfig(distillation_coefficient=1.0)
        )
