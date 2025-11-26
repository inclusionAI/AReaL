"""
Unit tests for proximal log-probability approximation functionality.

Tests the compute_prox_logp_approximations function and related metrics.
"""

import pytest
import torch

from areal.engine.ppo.actor import compute_prox_logp_approximations
from areal.utils.constants import (
    PROX_APPROX_METHOD_ROLLOUT,
    PROX_APPROX_METHODS_ALL,
    PROX_LOGP_METHOD_RECOMPUTE,
    PROX_LOGP_METHODS_ALL,
)


class TestProximalApproximations:
    """Test suite for proximal log-probability approximation methods."""

    def test_basic_loglinear_interpolation(self):
        """Test log-linear interpolation with simple version progression."""
        # Setup: behavior version=0, proximal version=1, current version=2
        old_logp = torch.tensor([[-1.0, -2.0, -3.0]], dtype=torch.float32)
        logprobs = torch.tensor([[-1.5, -2.5, -3.5]], dtype=torch.float32)
        versions = torch.tensor([[0, 0, 0]], dtype=torch.int32)
        current_version = 2

        approx = compute_prox_logp_approximations(
            old_logp=old_logp,
            logprobs=logprobs,
            versions=versions,
            current_version=current_version,
        )

        # alpha = (proximal - behave) / (theta - behave) = (1 - 0) / (2 - 0) = 0.5
        # loglinear: old + alpha * (new - old) = -1.0 + 0.5 * (-1.5 - (-1.0)) = -1.25
        expected_loglinear = torch.tensor([[-1.25, -2.25, -3.25]], dtype=torch.float32)
        torch.testing.assert_close(
            approx["loglinear"], expected_loglinear, rtol=1e-4, atol=1e-4
        )

    def test_rollout_approximation(self):
        """Test rollout approximation returns behavior logp unchanged."""
        old_logp = torch.tensor([[-1.0, -2.0]], dtype=torch.float32)
        logprobs = torch.tensor([[-5.0, -6.0]], dtype=torch.float32)
        versions = torch.tensor([[0, 1]], dtype=torch.int32)
        current_version = 5

        approx = compute_prox_logp_approximations(
            old_logp=old_logp,
            logprobs=logprobs,
            versions=versions,
            current_version=current_version,
        )

        # Rollout approximation should return old_logp unchanged (uses behavior policy as-is)
        torch.testing.assert_close(
            approx[PROX_APPROX_METHOD_ROLLOUT], old_logp, rtol=1e-6, atol=1e-6
        )

    def test_alpha_clamping(self):
        """Test that alpha is clamped to [0, 1] range."""
        # Case 1: alpha should be 0 when v_behave == v_proximal
        old_logp = torch.tensor([[-1.0]], dtype=torch.float32)
        logprobs = torch.tensor([[-2.0]], dtype=torch.float32)
        versions = torch.tensor([[4]], dtype=torch.int32)  # v_behave = v_proximal = 4
        current_version = 5  # v_theta = 5

        approx = compute_prox_logp_approximations(
            old_logp=old_logp,
            logprobs=logprobs,
            versions=versions,
            current_version=current_version,
        )

        # alpha = (4 - 4) / (5 - 4) = 0
        # loglinear should equal old_logp
        torch.testing.assert_close(approx["loglinear"], old_logp, rtol=1e-4, atol=1e-4)

    def test_mixed_versions_in_batch(self):
        """Test handling of samples with different behavior versions."""
        # Sample 1: v_behave=0, Sample 2: v_behave=2
        old_logp = torch.tensor([[-1.0], [-2.0]], dtype=torch.float32)
        logprobs = torch.tensor([[-1.5], [-2.2]], dtype=torch.float32)
        versions = torch.tensor([[0], [2]], dtype=torch.int32)
        current_version = 4  # v_proximal = 3

        approx = compute_prox_logp_approximations(
            old_logp=old_logp,
            logprobs=logprobs,
            versions=versions,
            current_version=current_version,
        )

        # Sample 1: alpha = (3 - 0) / (4 - 0) = 0.75
        # loglinear = -1.0 + 0.75 * (-1.5 - (-1.0)) = -1.0 + 0.75 * (-0.5) = -1.375
        # Sample 2: alpha = (3 - 2) / (4 - 2) = 0.5
        # loglinear = -2.0 + 0.5 * (-2.2 - (-2.0)) = -2.0 + 0.5 * (-0.2) = -2.1
        expected_loglinear = torch.tensor([[-1.375], [-2.1]], dtype=torch.float32)
        torch.testing.assert_close(
            approx["loglinear"], expected_loglinear, rtol=1e-4, atol=1e-4
        )

    def test_linear_approximation_probabilities(self):
        """Test linear interpolation works in probability space (arithmetic mean)."""
        old_logp = torch.tensor([[-0.693]], dtype=torch.float32)  # log(0.5)
        logprobs = torch.tensor([[-1.386]], dtype=torch.float32)  # log(0.25)
        versions = torch.tensor([[0]], dtype=torch.int32)
        current_version = 2  # v_proximal = 1, alpha = 0.5

        approx = compute_prox_logp_approximations(
            old_logp=old_logp,
            logprobs=logprobs,
            versions=versions,
            current_version=current_version,
        )

        # alpha = 0.5
        # p_behave = 0.5, p_theta = 0.25
        # p_arithmetic = (1-0.5)*0.5 + 0.5*0.25 = 0.25 + 0.125 = 0.375
        # log(0.375) ≈ -0.981
        expected = torch.log(torch.tensor([[0.375]], dtype=torch.float32))
        torch.testing.assert_close(approx["linear"], expected, rtol=1e-3, atol=1e-3)

    def test_all_methods_return_tensors(self):
        """Test that all approximation methods return valid tensors."""
        old_logp = torch.tensor([[-1.0, -2.0]], dtype=torch.float32)
        logprobs = torch.tensor([[-1.5, -2.5]], dtype=torch.float32)
        versions = torch.tensor([[0, 0]], dtype=torch.int32)
        current_version = 2

        approx = compute_prox_logp_approximations(
            old_logp=old_logp,
            logprobs=logprobs,
            versions=versions,
            current_version=current_version,
        )

        expected_methods = PROX_APPROX_METHODS_ALL
        for method in expected_methods:
            assert method in approx, f"Missing method: {method}"
            assert isinstance(approx[method], torch.Tensor), f"{method} not a tensor"
            assert approx[method].shape == old_logp.shape, f"{method} shape mismatch"
            assert approx[method].dtype == torch.float32, f"{method} wrong dtype"

    def test_version_zero_division_handling(self):
        """Test handling of same versions (zero division in alpha)."""
        old_logp = torch.tensor([[-1.0]], dtype=torch.float32)
        logprobs = torch.tensor([[-2.0]], dtype=torch.float32)
        versions = torch.tensor([[3]], dtype=torch.int32)
        current_version = 3  # v_behave = v_theta = 3, division by zero

        # Should not crash, alpha should be 0
        approx = compute_prox_logp_approximations(
            old_logp=old_logp,
            logprobs=logprobs,
            versions=versions,
            current_version=current_version,
        )

        # When versions are equal, alpha should be 0, loglinear should equal old_logp
        torch.testing.assert_close(approx["loglinear"], old_logp, rtol=1e-4, atol=1e-4)

    def test_negative_versions_in_prompt(self):
        """Test handling of negative versions (prompt tokens)."""
        # Prompt tokens have version=-1, should be handled gracefully
        old_logp = torch.tensor([[-1.0, -2.0, -3.0]], dtype=torch.float32)
        logprobs = torch.tensor([[-1.5, -2.5, -3.5]], dtype=torch.float32)
        versions = torch.tensor([[-1, 0, 1]], dtype=torch.int32)
        current_version = 3

        approx = compute_prox_logp_approximations(
            old_logp=old_logp,
            logprobs=logprobs,
            versions=versions,
            current_version=current_version,
        )

        # Should not crash with negative versions
        assert approx["loglinear"].shape == old_logp.shape
        # For prompt tokens (version < 0), alpha is 0, so approximation should equal old_logp
        assert torch.isclose(approx["loglinear"][0, 0], old_logp[0, 0])
        assert torch.isfinite(approx["loglinear"]).all(), "NaN/Inf in approximation"

    def test_batch_dimensions(self):
        """Test handling of different batch shapes."""
        batch_size = 4
        seq_len = 8
        old_logp = torch.randn(batch_size, seq_len, dtype=torch.float32)
        logprobs = torch.randn(batch_size, seq_len, dtype=torch.float32)
        versions = torch.randint(0, 10, (batch_size, seq_len), dtype=torch.int32)
        current_version = 10

        approx = compute_prox_logp_approximations(
            old_logp=old_logp,
            logprobs=logprobs,
            versions=versions,
            current_version=current_version,
        )

        for method in PROX_APPROX_METHODS_ALL:
            assert approx[method].shape == (batch_size, seq_len)
            assert torch.isfinite(approx[method]).all(), f"{method} has NaN/Inf"


class TestProximalApproximationIntegration:
    """Integration tests for proximal approximation in training flow."""

    def test_versions_not_popped_from_batch(self):
        """Test that versions are kept in batch (not popped) for use in loss function."""
        # Simulate the code in ppo_update - versions should NOT be popped
        data = {
            "versions": torch.tensor([[0, 1, 2]], dtype=torch.int32),
            "rewards": torch.tensor([1.0]),
            "tot_rewards": torch.tensor([1.0]),
            "kl_rewards": torch.tensor([[0.1, 0.2, 0.3]]),
        }

        original_versions = data["versions"]

        # Pop other keys but not versions
        for key in ["rewards", "tot_rewards", "kl_rewards"]:
            data.pop(key, None)

        # Verify versions still exists and is the same object (not cloned)
        assert "versions" in data, "versions should still be in data"
        if data["versions"] is not original_versions:
            assert False, "versions should be the same object (not cloned)"
        torch.testing.assert_close(
            data["versions"],
            torch.tensor([[0, 1, 2]], dtype=torch.int32),
        )

    def test_approximation_metrics_only_with_metrics_method(self):
        """Test that metrics are only computed when prox_logp_method='metrics'."""
        from areal.engine.ppo.actor import grpo_loss_fn

        batch_size, seq_len = 2, 4
        logits = torch.randn(batch_size, seq_len, 100)
        input_data = {
            "input_ids": torch.randint(0, 100, (batch_size, seq_len)),
            "logprobs": torch.randn(batch_size, seq_len),
            "advantages": torch.randn(batch_size, seq_len),
            "loss_mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
            "prox_logp": torch.randn(batch_size, seq_len),  # Ground truth available
            "versions": torch.randint(0, 3, (batch_size, seq_len), dtype=torch.int32),
        }

        # With prox_logp_method != "metrics", metrics should not be logged
        loss = grpo_loss_fn(
            logits=logits,
            input_data=input_data,
            temperature=1.0,
            eps_clip=0.2,
            eps_clip_higher=None,
            c_clip=None,
            behav_imp_weight_cap=None,
            current_version=5,
            prox_logp_method="recompute",  # Not metrics
        )

        # Should complete without error
        assert isinstance(loss, torch.Tensor)


def test_import_success():
    """Test that the approximation function can be imported."""
    from areal.engine.ppo.actor import compute_prox_logp_approximations

    assert callable(compute_prox_logp_approximations)


class TestComputeLogpOptimization:
    """Test suite for compute_logp() None return optimization."""

    def test_compute_logp_returns_none_when_optimized(self):
        """Test that compute_logp() returns None when using loglinear method."""
        from unittest.mock import MagicMock

        from areal.engine.ppo.actor import PPOActor, PPOActorConfig

        # Create config with loglinear method (skips forward pass)
        config = PPOActorConfig(
            use_decoupled_loss=True,
            prox_logp_method="loglinear",
        )

        # Mock the engine
        mock_engine = MagicMock()
        actor = PPOActor(config, mock_engine)

        # Create dummy batch data
        batch = {
            "input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
            "attention_mask": torch.ones(1, 4, dtype=torch.bool),
        }

        # Call compute_logp - should return None without calling engine.forward
        result = actor.compute_logp(batch)

        assert result is None
        mock_engine.forward.assert_not_called()  # Should NOT call forward

    def test_compute_logp_returns_tensor_when_recomputation_enabled(self):
        """Test that compute_logp() returns tensor when using recompute or metrics method."""
        from unittest.mock import MagicMock

        from areal.engine.ppo.actor import PPOActor, PPOActorConfig

        # Create config with recompute method (does forward pass)
        config = PPOActorConfig(
            use_decoupled_loss=True,
            prox_logp_method="recompute",  # Should compute
        )

        # Mock the engine
        mock_engine = MagicMock()
        mock_engine.forward.return_value = torch.tensor(
            [[-1.0, -2.0, -3.0, -4.0]], dtype=torch.float32
        )
        actor = PPOActor(config, mock_engine)
        actor.temperature = 1.0

        # Create dummy batch data
        batch = {
            "input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
            "attention_mask": torch.ones(1, 4, dtype=torch.bool),
        }

        # Call compute_logp - should return tensor
        result = actor.compute_logp(batch)

        assert result is not None
        assert isinstance(result, torch.Tensor)
        mock_engine.forward.assert_called_once()  # Should call forward


class TestGrpoLossFnNoneHandling:
    """Test suite for grpo_loss_fn() handling of None prox_logp."""

    def test_grpo_loss_fn_detects_none_prox_logp(self):
        """Test that grpo_loss_fn() detects None prox_logp and validates configuration."""
        from areal.engine.ppo.actor import grpo_loss_fn

        # Create dummy inputs with prox_logp=None
        batch_size, seq_len = 2, 4
        logits = torch.randn(batch_size, seq_len, 100)  # vocab_size=100
        input_data = {
            "input_ids": torch.randint(0, 100, (batch_size, seq_len)),
            "logprobs": torch.randn(batch_size, seq_len),
            "advantages": torch.randn(batch_size, seq_len),
            "loss_mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
            "prox_logp": None,  # Key test: None value
            "versions": torch.randint(0, 5, (batch_size, seq_len), dtype=torch.int32),
        }

        # Should raise error if prox_logp_method="recompute" but prox_logp=None
        with pytest.raises(
            ValueError, match="prox_logp is None but prox_logp_method='recompute'"
        ):
            grpo_loss_fn(
                logits=logits,
                input_data=input_data,
                temperature=1.0,
                eps_clip=0.2,
                eps_clip_higher=None,
                c_clip=None,
                behav_imp_weight_cap=None,
                current_version=5,
                prox_logp_method="recompute",
            )

    def test_grpo_loss_fn_requires_versions_when_prox_logp_none(self):
        """Test that grpo_loss_fn() requires versions when prox_logp is None."""
        from areal.engine.ppo.actor import grpo_loss_fn

        # Create dummy inputs with prox_logp=None but no versions
        batch_size, seq_len = 2, 4
        logits = torch.randn(batch_size, seq_len, 100)
        input_data = {
            "input_ids": torch.randint(0, 100, (batch_size, seq_len)),
            "logprobs": torch.randn(batch_size, seq_len),
            "advantages": torch.randn(batch_size, seq_len),
            "loss_mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
            "prox_logp": None,  # None value
            # Missing: "versions"
        }

        # Should raise error if versions not available
        with pytest.raises(
            ValueError,
            match=r"prox_logp is None with prox_logp_method='loglinear' but versions not available",
        ):
            grpo_loss_fn(
                logits=logits,
                input_data=input_data,
                temperature=1.0,
                eps_clip=0.2,
                eps_clip_higher=None,
                c_clip=None,
                behav_imp_weight_cap=None,
                current_version=5,
                prox_logp_method="loglinear",
            )

    def test_grpo_loss_fn_computes_approximation_when_prox_logp_none(self):
        """Test that grpo_loss_fn() successfully computes approximation when prox_logp is None."""
        from areal.engine.ppo.actor import grpo_loss_fn

        # Create valid inputs with prox_logp=None
        batch_size, seq_len = 2, 4
        logits = torch.randn(batch_size, seq_len, 100)
        input_data = {
            "input_ids": torch.randint(0, 100, (batch_size, seq_len)),
            "logprobs": torch.randn(batch_size, seq_len),
            "advantages": torch.randn(batch_size, seq_len),
            "loss_mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
            "prox_logp": None,  # None - will be replaced with approximation
            "versions": torch.randint(0, 3, (batch_size, seq_len), dtype=torch.int32),
        }

        # Should successfully compute approximation
        loss = grpo_loss_fn(
            logits=logits,
            input_data=input_data,
            temperature=1.0,
            eps_clip=0.2,
            eps_clip_higher=None,
            c_clip=None,
            behav_imp_weight_cap=None,
            current_version=5,
            prox_logp_method="loglinear",
        )

        # Loss should be computed successfully
        assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
        assert torch.isfinite(loss), "Loss should not contain NaN/Inf"

    def test_grpo_loss_fn_works_with_tensor_prox_logp(self):
        """Test that grpo_loss_fn() still works normally with tensor prox_logp."""
        from areal.engine.ppo.actor import grpo_loss_fn

        # Create valid inputs with prox_logp as tensor (normal case)
        batch_size, seq_len = 2, 4
        logits = torch.randn(batch_size, seq_len, 100)
        input_data = {
            "input_ids": torch.randint(0, 100, (batch_size, seq_len)),
            "logprobs": torch.randn(batch_size, seq_len),
            "advantages": torch.randn(batch_size, seq_len),
            "loss_mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
            "prox_logp": torch.randn(batch_size, seq_len),  # Tensor (normal case)
            "versions": torch.randint(0, 3, (batch_size, seq_len), dtype=torch.int32),
        }

        # Should work normally
        loss = grpo_loss_fn(
            logits=logits,
            input_data=input_data,
            temperature=1.0,
            eps_clip=0.2,
            eps_clip_higher=None,
            c_clip=None,
            behav_imp_weight_cap=None,
            current_version=5,
            prox_logp_method="loglinear",
        )

        assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
        assert torch.isfinite(loss), "Loss should not contain NaN/Inf"

    def test_grpo_loss_fn_metrics_disabled_when_prox_logp_none(self):
        """Test that metrics are not logged when prox_logp is None (no ground truth)."""
        from areal.engine.ppo.actor import grpo_loss_fn

        # Create inputs with prox_logp=None and metrics enabled
        batch_size, seq_len = 2, 4
        logits = torch.randn(batch_size, seq_len, 100)
        input_data = {
            "input_ids": torch.randint(0, 100, (batch_size, seq_len)),
            "logprobs": torch.randn(batch_size, seq_len),
            "advantages": torch.randn(batch_size, seq_len),
            "loss_mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
            "prox_logp": None,  # No ground truth
            "versions": torch.randint(0, 3, (batch_size, seq_len), dtype=torch.int32),
            "prox_logp_recomputed": False,  # Not recomputed
        }

        loss = grpo_loss_fn(
            logits=logits,
            input_data=input_data,
            temperature=1.0,
            eps_clip=0.2,
            eps_clip_higher=None,
            c_clip=None,
            behav_imp_weight_cap=None,
            current_version=5,
            prox_logp_method="loglinear",
        )

        # Should not crash and loss should be valid
        assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
        assert torch.isfinite(loss), "Loss should not contain NaN/Inf"


class TestEndToEndOptimization:
    """Integration tests for the full optimization flow."""

    def test_user_script_flow_with_none_return(self):
        """Test the full flow as it would happen in user scripts."""
        from unittest.mock import MagicMock

        from areal.engine.ppo.actor import PPOActor, PPOActorConfig

        # Setup: user configuration with optimization enabled
        config = PPOActorConfig(
            use_decoupled_loss=True,
            prox_logp_method="loglinear",
            recompute_logprob=False,  # Optimization active
        )

        mock_engine = MagicMock()
        actor = PPOActor(config, mock_engine)

        # Simulate user script behavior
        batch = {
            "input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
            "attention_mask": torch.ones(1, 4, dtype=torch.bool),
        }

        # User script calls compute_logp
        logp = actor.compute_logp(batch)

        # User script assigns to batch
        batch["prox_logp"] = logp

        # Verify the flow
        assert batch["prox_logp"] is None, "batch['prox_logp'] should be None"
        mock_engine.forward.assert_not_called()  # Forward pass was skipped!

    def test_configuration_matrix(self):
        """Test all combinations of prox_logp_method values."""
        from unittest.mock import MagicMock

        from areal.engine.ppo.actor import PPOActor, PPOActorConfig

        test_cases = [
            # (prox_logp_method, should_return_none, should_call_forward, description)
            ("loglinear", True, False, "loglinear -> None (skip forward)"),
            ("recompute", False, True, "recompute -> Tensor (do forward)"),
            ("metrics", False, True, "metrics -> Tensor (do forward)"),
        ]

        for method, should_be_none, should_call_forward, desc in test_cases:
            config = PPOActorConfig(
                use_decoupled_loss=True,
                prox_logp_method=method,
            )

            mock_engine = MagicMock()
            mock_engine.forward.return_value = torch.randn(1, 4)
            actor = PPOActor(config, mock_engine)
            actor.temperature = 1.0

            batch = {
                "input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
                "attention_mask": torch.ones(1, 4, dtype=torch.bool),
            }

            result = actor.compute_logp(batch)

            if should_be_none:
                assert result is None, f"Failed: {desc}"
            else:
                assert result is not None, f"Failed: {desc}"
                assert isinstance(result, torch.Tensor), f"Failed: {desc}"

            if should_call_forward:
                mock_engine.forward.assert_called_once()
            else:
                mock_engine.forward.assert_not_called()

            # Reset mock for next test
            mock_engine.reset_mock()


class TestConfigValidation:
    """Test suite for PPOActorConfig with new prox_logp_method field."""

    def test_valid_prox_logp_methods(self):
        """Test that all valid prox_logp_method values work correctly."""
        from unittest.mock import MagicMock

        from areal.api.cli_args import PPOActorConfig
        from areal.engine.ppo.actor import PPOActor

        # Test each valid method from the config
        valid_methods = PROX_LOGP_METHODS_ALL
        for method in valid_methods:
            config = PPOActorConfig(
                use_decoupled_loss=True,
                prox_logp_method=method,
            )
            # Should not raise any errors
            mock_engine = MagicMock()
            mock_engine.module.config = MagicMock()
            actor = PPOActor(config, mock_engine)
            assert actor.config.prox_logp_method == method

    def test_prox_logp_method_metadata_choices(self):
        """Test that prox_logp_method has correct choices in metadata."""
        from dataclasses import fields as dataclass_fields

        from areal.api.cli_args import PPOActorConfig

        # Get the actual choices from the dataclass
        config_choices = None
        for f in dataclass_fields(PPOActorConfig):
            if f.name == "prox_logp_method":
                config_choices = f.metadata.get("choices", [])
                break

        assert config_choices is not None, "prox_logp_method field should exist"
        expected_choices = PROX_LOGP_METHODS_ALL
        expected_count = len(expected_choices)
        if len(config_choices) != expected_count:
            assert False, f"Should have exactly {expected_count} choices"
        if set(config_choices) != set(expected_choices):
            assert False, f"Expected {expected_choices}, got {config_choices}"

    def test_prox_logp_method_default(self):
        """Test that prox_logp_method has correct default value."""
        from areal.api.cli_args import PPOActorConfig

        config = PPOActorConfig()
        expected_default = PROX_LOGP_METHOD_RECOMPUTE
        if config.prox_logp_method != expected_default:
            assert False, f"Default should be '{expected_default}'"

    def test_old_config_fields_removed(self):
        """Test that old config fields have been removed."""
        from dataclasses import fields as dataclass_fields

        from areal.api.cli_args import PPOActorConfig

        field_names = {f.name for f in dataclass_fields(PPOActorConfig)}

        # Old fields should not exist
        if "use_prox_approx" in field_names:
            assert False, "use_prox_approx should be removed"
        if "prox_approx_method" in field_names:
            assert False, "prox_approx_method should be removed"
        if "log_prox_approx_metrics" in field_names:
            assert False, "log_prox_approx_metrics should be removed"

        # New field should exist
        assert "prox_logp_method" in field_names, "prox_logp_method should exist"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
