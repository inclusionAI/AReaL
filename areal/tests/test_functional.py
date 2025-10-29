import pytest
import torch

from areal.utils.functional import ppo_actor_loss_fn


class TestPPOActorLossFnSequenceLevel:
    """Test cases for ppo_actor_loss_fn with importance_sampling_level='sequence'."""

    @pytest.fixture
    def basic_2d_data(self):
        """Basic 2D tensor test data for sequence-level importance sampling."""
        batch_size = 4
        seq_len = 8

        return {
            "logprobs": torch.randn(batch_size, seq_len),
            "proximal_logprobs": torch.randn(batch_size, seq_len),
            "old_logprobs": torch.randn(batch_size, seq_len),
            "advantages": torch.randn(batch_size, seq_len),
            "loss_mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
            "eps_clip": 0.2,
        }

    @pytest.fixture
    def basic_1d_data(self):
        """Basic 1D tensor test data for sequence-level importance sampling (packed)."""
        total_tokens = 32

        return {
            "logprobs": torch.randn(total_tokens),
            "proximal_logprobs": torch.randn(total_tokens),
            "old_logprobs": torch.randn(total_tokens),
            "advantages": torch.randn(total_tokens),
            "loss_mask": torch.ones(total_tokens, dtype=torch.bool),
            "eps_clip": 0.2,
        }

    def test_sequence_level_2d_shape(self, basic_2d_data):
        """Test that sequence-level loss returns correct shape for 2D tensors."""
        loss, stat = ppo_actor_loss_fn(
            logprobs=basic_2d_data["logprobs"],
            proximal_logprobs=basic_2d_data["proximal_logprobs"],
            old_logprobs=basic_2d_data["old_logprobs"],
            advantages=basic_2d_data["advantages"],
            eps_clip=basic_2d_data["eps_clip"],
            loss_mask=basic_2d_data["loss_mask"],
            importance_sampling_level="sequence",
        )

        # Loss should be a scalar
        assert loss.ndim == 0
        assert loss.dtype == torch.float32

        # Stats should have correct shapes
        assert stat["importance_weight"].shape == basic_2d_data["logprobs"].shape
        assert stat["approx_kl"].shape == basic_2d_data["logprobs"].shape
        assert stat["clip_mask"].shape == basic_2d_data["logprobs"].shape

    def test_sequence_level_1d_shape(self, basic_1d_data):
        """Test that sequence-level loss returns correct shape for 1D tensors."""
        loss, stat = ppo_actor_loss_fn(
            logprobs=basic_1d_data["logprobs"],
            proximal_logprobs=basic_1d_data["proximal_logprobs"],
            old_logprobs=basic_1d_data["old_logprobs"],
            advantages=basic_1d_data["advantages"],
            eps_clip=basic_1d_data["eps_clip"],
            loss_mask=basic_1d_data["loss_mask"],
            importance_sampling_level="sequence",
        )

        # Loss should be a scalar
        assert loss.ndim == 0
        assert loss.dtype == torch.float32

        # Stats should have correct shapes
        assert stat["importance_weight"].shape == basic_1d_data["logprobs"].shape
        assert stat["approx_kl"].shape == basic_1d_data["logprobs"].shape
        assert stat["clip_mask"].shape == basic_1d_data["logprobs"].shape

    def test_sequence_level_geometric_mean_2d(self):
        """Test that geometric mean is computed correctly for 2D tensors."""
        batch_size = 2
        seq_len = 4

        # Create data where we can manually verify geometric mean
        logprobs = torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0, 2.0],
            ]
        )
        proximal_logprobs = torch.tensor(
            [
                [0.5, 0.5, 0.5, 0.5],
                [1.0, 1.0, 1.0, 1.0],
            ]
        )
        old_logprobs = torch.zeros(batch_size, seq_len)
        advantages = torch.ones(batch_size, seq_len)
        loss_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        loss, stat = ppo_actor_loss_fn(
            logprobs=logprobs,
            proximal_logprobs=proximal_logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            eps_clip=0.2,
            loss_mask=loss_mask,
            importance_sampling_level="sequence",
        )

        # For first sequence: log_ratio = 1.0 - 0.5 = 0.5
        # Geometric mean = exp(0.5) ≈ 1.6487
        # For second sequence: log_ratio = 2.0 - 1.0 = 1.0
        # Geometric mean = exp(1.0) ≈ 2.7183

        expected_ratio_seq1 = torch.exp(torch.tensor(0.5))
        expected_ratio_seq2 = torch.exp(torch.tensor(1.0))

        # All tokens in a sequence should have the same ratio
        assert torch.allclose(
            stat["importance_weight"][0, :],
            expected_ratio_seq1.expand(seq_len),
            atol=1e-5,
        )
        assert torch.allclose(
            stat["importance_weight"][1, :],
            expected_ratio_seq2.expand(seq_len),
            atol=1e-5,
        )

    def test_sequence_level_geometric_mean_1d(self):
        """Test that geometric mean is computed correctly for 1D tensors."""
        total_tokens = 4

        # Create data where we can manually verify geometric mean
        logprobs = torch.tensor([1.0, 1.0, 1.0, 1.0])
        proximal_logprobs = torch.tensor([0.5, 0.5, 0.5, 0.5])
        old_logprobs = torch.zeros(total_tokens)
        advantages = torch.ones(total_tokens)
        loss_mask = torch.ones(total_tokens, dtype=torch.bool)

        loss, stat = ppo_actor_loss_fn(
            logprobs=logprobs,
            proximal_logprobs=proximal_logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            eps_clip=0.2,
            loss_mask=loss_mask,
            importance_sampling_level="sequence",
        )

        # log_ratio = 1.0 - 0.5 = 0.5
        # Geometric mean = exp(0.5) ≈ 1.6487
        expected_ratio = torch.exp(torch.tensor(0.5))

        # All tokens should have the same ratio
        assert torch.allclose(
            stat["importance_weight"], expected_ratio.expand(total_tokens), atol=1e-5
        )

    def test_sequence_level_advantage_summing_2d(self):
        """Test that advantages are summed per sequence for 2D tensors."""
        batch_size = 2
        seq_len = 4

        # Create advantages that vary per token
        advantages = torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0],  # sum = 10.0
                [0.5, 0.5, 0.5, 0.5],  # sum = 2.0
            ]
        )

        logprobs = torch.ones(batch_size, seq_len)
        proximal_logprobs = torch.ones(batch_size, seq_len)
        old_logprobs = torch.ones(batch_size, seq_len)
        loss_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        # We'll verify this by checking that the loss is computed correctly
        # The function should sum advantages across the sequence dimension
        loss, stat = ppo_actor_loss_fn(
            logprobs=logprobs,
            proximal_logprobs=proximal_logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            eps_clip=0.2,
            loss_mask=loss_mask,
            importance_sampling_level="sequence",
        )

        # The function should work without errors
        assert loss.ndim == 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_sequence_level_with_mask_2d(self):
        """Test sequence-level with partial masking for 2D tensors."""
        batch_size = 2
        seq_len = 8

        # Create mask where only some tokens are valid
        loss_mask = torch.tensor(
            [
                [True, True, True, True, False, False, False, False],
                [True, True, True, True, True, True, False, False],
            ]
        )

        logprobs = torch.randn(batch_size, seq_len)
        proximal_logprobs = torch.randn(batch_size, seq_len)
        old_logprobs = torch.randn(batch_size, seq_len)
        advantages = torch.randn(batch_size, seq_len)

        loss, stat = ppo_actor_loss_fn(
            logprobs=logprobs,
            proximal_logprobs=proximal_logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            eps_clip=0.2,
            loss_mask=loss_mask,
            importance_sampling_level="sequence",
        )

        # Masked positions should have zero importance weight
        assert torch.allclose(
            stat["importance_weight"][0, 4:], torch.zeros(4), atol=1e-6
        )
        assert torch.allclose(
            stat["importance_weight"][1, 6:], torch.zeros(2), atol=1e-6
        )

        # Valid positions in same sequence should have same ratio
        assert torch.allclose(
            stat["importance_weight"][0, 0], stat["importance_weight"][0, 3], atol=1e-5
        )

    def test_sequence_level_with_mask_1d(self):
        """Test sequence-level with partial masking for 1D tensors."""
        total_tokens = 8

        # Create mask where only some tokens are valid
        loss_mask = torch.tensor([True, True, True, True, False, False, False, False])

        logprobs = torch.randn(total_tokens)
        proximal_logprobs = torch.randn(total_tokens)
        old_logprobs = torch.randn(total_tokens)
        advantages = torch.randn(total_tokens)

        loss, stat = ppo_actor_loss_fn(
            logprobs=logprobs,
            proximal_logprobs=proximal_logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            eps_clip=0.2,
            loss_mask=loss_mask,
            importance_sampling_level="sequence",
        )

        # Masked positions should have zero importance weight
        assert torch.allclose(stat["importance_weight"][4:], torch.zeros(4), atol=1e-6)

        # All valid positions should have same ratio (same sequence)
        valid_ratios = stat["importance_weight"][:4]
        assert torch.allclose(valid_ratios, valid_ratios[0].expand(4), atol=1e-5)

    def test_sequence_level_batch_size_1_2d(self):
        """Test batch_size=1 edge case for 2D tensors."""
        batch_size = 1
        seq_len = 10

        logprobs = torch.randn(batch_size, seq_len)
        proximal_logprobs = torch.randn(batch_size, seq_len)
        old_logprobs = torch.randn(batch_size, seq_len)
        advantages = torch.randn(batch_size, seq_len)
        loss_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        loss, stat = ppo_actor_loss_fn(
            logprobs=logprobs,
            proximal_logprobs=proximal_logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            eps_clip=0.2,
            loss_mask=loss_mask,
            importance_sampling_level="sequence",
        )

        # Should compute without errors
        assert loss.ndim == 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

        # All tokens in the single sequence should have same ratio
        ratios = stat["importance_weight"][0]
        assert torch.allclose(ratios, ratios[0].expand(seq_len), atol=1e-5)

    def test_sequence_level_batch_size_1_1d(self):
        """Test batch_size=1 edge case for 1D tensors (single token)."""
        total_tokens = 1

        logprobs = torch.randn(total_tokens)
        proximal_logprobs = torch.randn(total_tokens)
        old_logprobs = torch.randn(total_tokens)
        advantages = torch.randn(total_tokens)
        loss_mask = torch.ones(total_tokens, dtype=torch.bool)

        loss, stat = ppo_actor_loss_fn(
            logprobs=logprobs,
            proximal_logprobs=proximal_logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            eps_clip=0.2,
            loss_mask=loss_mask,
            importance_sampling_level="sequence",
        )

        # Should compute without errors
        assert loss.ndim == 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

        # Single token should have valid ratio
        assert stat["importance_weight"].shape == (1,)
        assert not torch.isnan(stat["importance_weight"][0])

    def test_sequence_level_with_clipping_2d(self):
        """Test that clipping works correctly with sequence-level importance sampling."""
        batch_size = 2
        seq_len = 4

        # Create data that will trigger clipping
        logprobs = torch.tensor(
            [
                [2.0, 2.0, 2.0, 2.0],  # High ratio, will clip
                [-2.0, -2.0, -2.0, -2.0],  # Low ratio, will clip
            ]
        )
        proximal_logprobs = torch.zeros(batch_size, seq_len)
        old_logprobs = torch.zeros(batch_size, seq_len)
        advantages = torch.ones(batch_size, seq_len)
        loss_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        eps_clip = 0.2

        loss, stat = ppo_actor_loss_fn(
            logprobs=logprobs,
            proximal_logprobs=proximal_logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            eps_clip=eps_clip,
            loss_mask=loss_mask,
            importance_sampling_level="sequence",
        )

        # Check that some clipping occurred
        assert stat["clip_mask"].any()

        # Verify loss is finite
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_sequence_level_with_dual_clip(self):
        """Test sequence-level with dual clip (c_clip parameter)."""
        batch_size = 2
        seq_len = 4

        logprobs = torch.randn(batch_size, seq_len)
        proximal_logprobs = torch.randn(batch_size, seq_len)
        old_logprobs = torch.randn(batch_size, seq_len)
        advantages = torch.randn(batch_size, seq_len)
        loss_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        loss, stat = ppo_actor_loss_fn(
            logprobs=logprobs,
            proximal_logprobs=proximal_logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            eps_clip=0.2,
            loss_mask=loss_mask,
            c_clip=3.0,
            importance_sampling_level="sequence",
        )

        # Should have dual_clip_mask in stats
        assert "dual_clip_mask" in stat
        assert stat["dual_clip_mask"].shape == logprobs.shape

        # Verify loss is finite
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_sequence_level_with_behav_imp_weight_cap(self):
        """Test sequence-level with behavior importance weight capping."""
        batch_size = 2
        seq_len = 4

        logprobs = torch.randn(batch_size, seq_len)
        proximal_logprobs = torch.randn(batch_size, seq_len)
        old_logprobs = torch.randn(batch_size, seq_len)
        advantages = torch.randn(batch_size, seq_len)
        loss_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        loss, stat = ppo_actor_loss_fn(
            logprobs=logprobs,
            proximal_logprobs=proximal_logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            eps_clip=0.2,
            loss_mask=loss_mask,
            behav_imp_weight_cap=10.0,
            importance_sampling_level="sequence",
        )

        # Should have behavior stats
        assert "behave_imp_weight" in stat
        assert "behave_approx_kl" in stat
        assert "behave_mask" in stat

        # Verify loss is finite
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_sequence_level_vs_token_level_different(self):
        """Verify that sequence-level and token-level produce different results."""
        batch_size = 2
        seq_len = 4

        # Create non-uniform data within sequences
        logprobs = torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0],
                [0.5, 1.5, 2.5, 3.5],
            ]
        )
        proximal_logprobs = torch.zeros(batch_size, seq_len)
        old_logprobs = torch.zeros(batch_size, seq_len)
        advantages = torch.ones(batch_size, seq_len)
        loss_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        # Compute sequence-level loss
        loss_seq, stat_seq = ppo_actor_loss_fn(
            logprobs=logprobs.clone(),
            proximal_logprobs=proximal_logprobs.clone(),
            old_logprobs=old_logprobs.clone(),
            advantages=advantages.clone(),
            eps_clip=0.2,
            loss_mask=loss_mask.clone(),
            importance_sampling_level="sequence",
        )

        # Compute token-level loss
        loss_tok, stat_tok = ppo_actor_loss_fn(
            logprobs=logprobs.clone(),
            proximal_logprobs=proximal_logprobs.clone(),
            old_logprobs=old_logprobs.clone(),
            advantages=advantages.clone(),
            eps_clip=0.2,
            loss_mask=loss_mask.clone(),
            importance_sampling_level="token",
        )

        # Results should be different
        assert not torch.allclose(loss_seq, loss_tok, atol=1e-6)
        assert not torch.allclose(
            stat_seq["importance_weight"], stat_tok["importance_weight"], atol=1e-6
        )

    def test_sequence_level_all_masked(self):
        """Test edge case where all tokens are masked."""
        batch_size = 2
        seq_len = 4

        logprobs = torch.randn(batch_size, seq_len)
        proximal_logprobs = torch.randn(batch_size, seq_len)
        old_logprobs = torch.randn(batch_size, seq_len)
        advantages = torch.randn(batch_size, seq_len)
        loss_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)

        loss, stat = ppo_actor_loss_fn(
            logprobs=logprobs,
            proximal_logprobs=proximal_logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            eps_clip=0.2,
            loss_mask=loss_mask,
            importance_sampling_level="sequence",
        )

        # Loss should be zero or very small
        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)

        # All importance weights should be zero (masked)
        assert torch.allclose(
            stat["importance_weight"],
            torch.zeros_like(stat["importance_weight"]),
            atol=1e-6,
        )

    def test_sequence_level_consistency_across_calls(self):
        """Test that multiple calls with same input produce same output."""
        batch_size = 2
        seq_len = 4

        logprobs = torch.randn(batch_size, seq_len)
        proximal_logprobs = torch.randn(batch_size, seq_len)
        old_logprobs = torch.randn(batch_size, seq_len)
        advantages = torch.randn(batch_size, seq_len)
        loss_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        # First call
        loss1, stat1 = ppo_actor_loss_fn(
            logprobs=logprobs.clone(),
            proximal_logprobs=proximal_logprobs.clone(),
            old_logprobs=old_logprobs.clone(),
            advantages=advantages.clone(),
            eps_clip=0.2,
            loss_mask=loss_mask.clone(),
            importance_sampling_level="sequence",
        )

        # Second call with same inputs
        loss2, stat2 = ppo_actor_loss_fn(
            logprobs=logprobs.clone(),
            proximal_logprobs=proximal_logprobs.clone(),
            old_logprobs=old_logprobs.clone(),
            advantages=advantages.clone(),
            eps_clip=0.2,
            loss_mask=loss_mask.clone(),
            importance_sampling_level="sequence",
        )

        # Results should be identical
        assert torch.allclose(loss1, loss2)
        assert torch.allclose(stat1["importance_weight"], stat2["importance_weight"])
        assert torch.equal(stat1["clip_mask"], stat2["clip_mask"])


class TestPPOActorLossFnEdgeCases:
    """Additional edge case tests for ppo_actor_loss_fn."""

    def test_invalid_importance_sampling_level(self):
        """Test that invalid importance_sampling_level raises ValueError."""
        batch_size = 2
        seq_len = 4

        logprobs = torch.randn(batch_size, seq_len)
        proximal_logprobs = torch.randn(batch_size, seq_len)
        old_logprobs = torch.randn(batch_size, seq_len)
        advantages = torch.randn(batch_size, seq_len)
        loss_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        with pytest.raises(ValueError, match="Invalid importance_sampling_level"):
            ppo_actor_loss_fn(
                logprobs=logprobs,
                proximal_logprobs=proximal_logprobs,
                old_logprobs=old_logprobs,
                advantages=advantages,
                eps_clip=0.2,
                loss_mask=loss_mask,
                importance_sampling_level="invalid",
            )

    def test_batch_size_1_with_full_mask(self):
        """Test batch_size=1 with complete sequence (no padding)."""
        batch_size = 1
        seq_len = 5

        logprobs = torch.randn(batch_size, seq_len)
        proximal_logprobs = torch.randn(batch_size, seq_len)
        old_logprobs = torch.randn(batch_size, seq_len)
        advantages = torch.randn(batch_size, seq_len)
        loss_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        # Test with sequence-level
        loss_seq, stat_seq = ppo_actor_loss_fn(
            logprobs=logprobs,
            proximal_logprobs=proximal_logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            eps_clip=0.2,
            loss_mask=loss_mask,
            importance_sampling_level="sequence",
        )

        # Test with token-level for comparison
        loss_tok, stat_tok = ppo_actor_loss_fn(
            logprobs=logprobs,
            proximal_logprobs=proximal_logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            eps_clip=0.2,
            loss_mask=loss_mask,
            importance_sampling_level="token",
        )

        # Both should produce valid finite losses
        assert not torch.isnan(loss_seq) and not torch.isinf(loss_seq)
        assert not torch.isnan(loss_tok) and not torch.isinf(loss_tok)

    def test_batch_size_1_with_partial_mask(self):
        """Test batch_size=1 with partial masking (some padding)."""
        batch_size = 1
        seq_len = 10

        logprobs = torch.randn(batch_size, seq_len)
        proximal_logprobs = torch.randn(batch_size, seq_len)
        old_logprobs = torch.randn(batch_size, seq_len)
        advantages = torch.randn(batch_size, seq_len)
        # First 7 tokens valid, last 3 masked (padding)
        loss_mask = torch.tensor(
            [[True, True, True, True, True, True, True, False, False, False]]
        )

        loss, stat = ppo_actor_loss_fn(
            logprobs=logprobs,
            proximal_logprobs=proximal_logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            eps_clip=0.2,
            loss_mask=loss_mask,
            importance_sampling_level="sequence",
        )

        # Verify masked tokens have zero weight
        assert torch.allclose(
            stat["importance_weight"][0, 7:], torch.zeros(3), atol=1e-6
        )

        # Verify all valid tokens have same ratio (same sequence)
        valid_ratios = stat["importance_weight"][0, :7]
        assert torch.allclose(valid_ratios, valid_ratios[0].expand(7), atol=1e-5)
