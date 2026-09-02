# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from areal.api.cli_args import MicroBatchSpec
from areal.utils.data import (
    split_padded_tensor_dict_into_mb_list,
)
from areal.utils.functional.loss_aggregation import PolicyGradientReduction

# Three packed sequences of length 2, 3 and 5. Per-token losses are constant
# within a sequence so every expected value below can be read off by hand.
CU_SEQLENS = torch.tensor([0, 2, 5, 10], dtype=torch.int32)
LOSS = torch.tensor([1.0, 1.0, 2.0, 2.0, 2.0, 4.0, 4.0, 4.0, 4.0, 4.0])
MASK = torch.ones(10, dtype=torch.bool)


def legacy_token_mean(loss, loss_mask):
    """The reduction ppo_actor_loss_fn used before loss_aggregation existed."""
    return torch.where(loss_mask, loss, 0).sum() / (loss_mask.count_nonzero() or 1)


class TestTokenMeanParity:
    """token-mean must stay bit-identical to the pre-feature reduction."""

    def test_matches_legacy_reduction(self):
        reduction = PolicyGradientReduction("token-mean")
        assert torch.equal(
            reduction.aggregate(LOSS, MASK, cu_seqlens=CU_SEQLENS),
            legacy_token_mean(LOSS, MASK),
        )

    def test_matches_legacy_reduction_with_holes(self):
        mask = MASK.clone()
        mask[0] = False
        mask[7] = False
        reduction = PolicyGradientReduction("token-mean")
        assert torch.equal(
            reduction.aggregate(LOSS, mask, cu_seqlens=CU_SEQLENS),
            legacy_token_mean(LOSS, mask),
        )

    def test_normalizer_matches_legacy_loss_weight_fn(self):
        reduction = PolicyGradientReduction("token-mean")
        data = {"loss_mask": MASK, "cu_seqlens": CU_SEQLENS}
        assert torch.equal(reduction.normalizer_fn(data), MASK.count_nonzero())

    def test_token_mean_needs_no_sequence_boundaries(self):
        reduction = PolicyGradientReduction("token-mean")
        assert reduction.aggregate(LOSS, MASK) == pytest.approx(2.8)


class TestSeqMean:
    def test_packed(self):
        # Per-sequence token means are 1, 2 and 4.
        reduction = PolicyGradientReduction("seq-mean")
        result = reduction.aggregate(LOSS, MASK, cu_seqlens=CU_SEQLENS)
        assert result.item() == pytest.approx(7 / 3)

    def test_padded_matches_packed(self):
        loss = torch.tensor(
            [
                [1.0, 1.0, 0.0, 0.0, 0.0],
                [2.0, 2.0, 2.0, 0.0, 0.0],
                [4.0, 4.0, 4.0, 4.0, 4.0],
            ]
        )
        mask = torch.tensor(
            [
                [1, 1, 0, 0, 0],
                [1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1],
            ],
            dtype=torch.bool,
        )
        reduction = PolicyGradientReduction("seq-mean")
        assert reduction.aggregate(loss, mask).item() == pytest.approx(7 / 3)

    def test_ignores_sequence_length(self):
        """A long and a short sequence with the same per-token loss tie."""
        cu_seqlens = torch.tensor([0, 1, 9], dtype=torch.int32)
        loss = torch.tensor([1.0] + [3.0] * 8)
        mask = torch.ones(9, dtype=torch.bool)
        reduction = PolicyGradientReduction("seq-mean")
        assert reduction.aggregate(loss, mask, cu_seqlens=cu_seqlens).item() == (
            pytest.approx(2.0)
        )

    def test_normalizer_counts_sequences(self):
        reduction = PolicyGradientReduction("seq-mean")
        data = {"loss_mask": MASK, "cu_seqlens": CU_SEQLENS}
        assert reduction.normalizer_fn(data).item() == pytest.approx(3.0)

    def test_packed_input_requires_cu_seqlens(self):
        reduction = PolicyGradientReduction("seq-mean")
        with pytest.raises(ValueError, match="requires cu_seqlens"):
            reduction.aggregate(LOSS, MASK)


class TestConstant:
    def test_divides_by_fixed_denominator(self):
        # Masked sum is 28 over 3 active sequences, divisor 2 -> 28 / 6.
        reduction = PolicyGradientReduction("constant", divisor=2.0)
        result = reduction.aggregate(LOSS, MASK, cu_seqlens=CU_SEQLENS)
        assert result.item() == pytest.approx(28 / 6)

    def test_divisor_is_required(self):
        with pytest.raises(ValueError, match="divisor must be a positive"):
            PolicyGradientReduction("constant")

    def test_divisor_rejected_by_other_modes(self):
        with pytest.raises(ValueError, match="only valid for"):
            PolicyGradientReduction("seq-mean", divisor=2.0)

    def test_unknown_mode(self):
        with pytest.raises(ValueError, match="loss_aggregation must be one of"):
            PolicyGradientReduction("mean")


class TestNarrowedNumerator:
    """Rejection sampling narrows the numerator but keeps the denominator."""

    @pytest.mark.parametrize("mode", ["token-mean", "seq-mean"])
    def test_denominator_mask_is_honoured(self, mode):
        narrowed = MASK.clone()
        narrowed[0] = False
        reduction = PolicyGradientReduction(mode)
        kept = reduction.aggregate(
            LOSS, narrowed, denominator_mask=MASK, cu_seqlens=CU_SEQLENS
        )
        rescaled = reduction.aggregate(LOSS, narrowed, cu_seqlens=CU_SEQLENS)
        # Dropping a token must shrink the loss, not redistribute its weight.
        assert kept.item() < rescaled.item()

    def test_token_mean_denominator_is_the_original_count(self):
        narrowed = MASK.clone()
        narrowed[0] = False
        reduction = PolicyGradientReduction("token-mean")
        result = reduction.aggregate(
            LOSS, narrowed, denominator_mask=MASK, cu_seqlens=CU_SEQLENS
        )
        assert result.item() == pytest.approx((LOSS.sum().item() - 1.0) / 10)


class TestMicrobatchGranularity:
    """The ordinary splitter preserves configured consecutive-row granularity."""

    @pytest.mark.parametrize("bs,granularity", [(6, 1), (8, 2), (12, 3), (130, 1)])
    def test_matches_legacy_granularity_split(self, bs, granularity):
        data = {
            "input_ids": torch.arange(bs).reshape(-1, 1).expand(-1, 2),
            "attention_mask": torch.ones(bs, 2, dtype=torch.bool),
        }
        batches = split_padded_tensor_dict_into_mb_list(
            data, MicroBatchSpec(n_mbs=bs // granularity, granularity=granularity)
        ).mbs
        assert [batch["input_ids"][:, 0].tolist() for batch in batches] == [
            list(range(i * granularity, (i + 1) * granularity))
            for i in range(bs // granularity)
        ]

    def test_indivisible_batch_still_raises(self):
        with pytest.raises(RuntimeError, match="cannot divide granularity"):
            split_padded_tensor_dict_into_mb_list(
                {"attention_mask": torch.ones(7, 2, dtype=torch.bool)},
                MicroBatchSpec(granularity=2),
            )
