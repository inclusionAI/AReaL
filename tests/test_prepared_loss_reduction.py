# SPDX-License-Identifier: Apache-2.0

"""Full-update unit coefficients survive row-level optimizer schedules."""

from unittest.mock import patch

import pytest
import torch

from areal.trainer.ppo.actor import grpo_loss_fn
from areal.utils.data import split_training_batch_into_microbatches
from areal.utils.functional.loss_aggregation import PolicyGradientReduction


def test_optimizer_steps_are_capped_by_available_rows():
    data = {"attention_mask": torch.ones(2, 3, dtype=torch.bool)}
    batches = split_training_batch_into_microbatches(data, 8)
    assert len(batches) == 2
    assert all(batch["attention_mask"].shape[0] == 1 for batch in batches)


@pytest.mark.parametrize("use_sapo", [False, True])
def test_actor_loss_consumes_prepared_mode_free_reduction(use_sapo):
    mask = torch.ones(2, 2, dtype=torch.bool)
    logprobs = torch.zeros(2, 2, requires_grad=True)
    data = {
        "logprobs": torch.zeros_like(logprobs),
        "advantages": torch.ones_like(logprobs),
        "loss_mask": mask,
        "prox_logp": torch.zeros_like(logprobs),
    }

    class DoubleTokenMean:
        def __bool__(self):
            return False

        def normalizer_fn(self, data):
            return PolicyGradientReduction().normalizer_fn(data)

        def aggregate(self, loss, numerator_mask, **kwargs):
            return 2 * PolicyGradientReduction().aggregate(
                loss, numerator_mask, **kwargs
            )

    with patch("areal.trainer.ppo.actor.stats_tracker"):
        loss = grpo_loss_fn(
            logprobs,
            torch.zeros_like(logprobs),
            data,
            0.2,
            None,
            None,
            use_sapo_loss=use_sapo,
            pg_reduction=DoubleTokenMean(),
        )
    torch.testing.assert_close(
        loss, torch.tensor(-4.0 if use_sapo else -2.0), rtol=1e-6, atol=1e-6
    )
    torch.testing.assert_close(
        torch.autograd.grad(loss, logprobs)[0],
        -2 * mask.float() / 4,
        rtol=1e-6,
        atol=1e-6,
    )
