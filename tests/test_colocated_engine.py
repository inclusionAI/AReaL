"""Unit tests for colocated orchestration and trainer validation."""

from __future__ import annotations

import os
import shutil
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from areal.infra.colocated import ColocatedConfig, ColocatedOrchestrator
from areal.trainer.rl_trainer import PPOTrainer


@pytest.fixture
def temp_weight_dir():
    """Create a temporary directory for weight storage."""
    tmpdir = tempfile.mkdtemp(prefix="areal_colocated_test_")
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def mock_train_engine():
    """Create a mock training engine."""
    engine = MagicMock()
    engine.offload = MagicMock()
    engine.onload = MagicMock()
    engine.save = MagicMock()
    return engine


@pytest.fixture
def mock_inf_engine():
    """Create a mock inference engine."""
    engine = MagicMock()
    engine.offload = MagicMock()
    engine.onload = MagicMock()
    engine.backend = MagicMock()
    engine.addresses = ["127.0.0.1:30000"]
    engine.config = MagicMock()
    engine.config.request_retries = 1
    engine.config.request_timeout = 30.0
    return engine


@pytest.fixture
def orchestrator(mock_train_engine, mock_inf_engine, temp_weight_dir):
    """Create a ColocatedOrchestrator with mock engines."""
    config = ColocatedConfig(
        weight_path=temp_weight_dir,
        cleanup_weights_after_load=True,
    )
    return ColocatedOrchestrator(
        train_engine=mock_train_engine,
        inf_engine=mock_inf_engine,
        config=config,
    )


class TestColocatedConfig:
    """Tests for ColocatedConfig dataclass."""

    def test_default_config(self):
        config = ColocatedConfig()
        assert config.weight_path == "/dev/shm/areal_colocated_weights"
        assert config.cleanup_weights_after_load is True

    def test_custom_config(self):
        config = ColocatedConfig(
            weight_path="/tmp/custom_weights",
            cleanup_weights_after_load=False,
        )
        assert config.weight_path == "/tmp/custom_weights"
        assert config.cleanup_weights_after_load is False


class TestColocatedOrchestrator:
    """Tests for ColocatedOrchestrator."""

    def test_initial_state(self, orchestrator):
        """Test that orchestrator starts with both engines on GPU.

        The caller must call ``initial_offload_training()`` before the
        first rollout to move the training engine off GPU.
        """
        assert orchestrator._inf_on_gpu is True
        assert orchestrator._train_on_gpu is True

    def test_initial_offload_training(
        self, orchestrator, mock_train_engine
    ):
        """Test that initial_offload_training offloads the training engine."""
        assert orchestrator._train_on_gpu is True
        orchestrator.initial_offload_training()
        mock_train_engine.offload.assert_called_once()
        assert orchestrator._train_on_gpu is False

    def test_initial_offload_training_idempotent(
        self, orchestrator, mock_train_engine
    ):
        """Test that calling initial_offload_training twice is safe."""
        orchestrator.initial_offload_training()
        orchestrator.initial_offload_training()  # Should skip
        mock_train_engine.offload.assert_called_once()

    def test_prepare_for_training_offloads_inf_and_onloads_train(
        self, orchestrator, mock_train_engine, mock_inf_engine
    ):
        """Test that prepare_for_training switches GPU ownership correctly."""
        # Must offload training first (mimics real init sequence)
        orchestrator.initial_offload_training()
        mock_train_engine.offload.reset_mock()

        orchestrator.prepare_for_training()

        mock_inf_engine.offload.assert_called_once()
        mock_train_engine.onload.assert_called_once()
        assert orchestrator._inf_on_gpu is False
        assert orchestrator._train_on_gpu is True

    def test_prepare_for_training_idempotent(
        self, orchestrator, mock_train_engine, mock_inf_engine
    ):
        """Test that calling prepare_for_training twice doesn't double offload/onload."""
        orchestrator.initial_offload_training()
        mock_train_engine.offload.reset_mock()

        orchestrator.prepare_for_training()
        orchestrator.prepare_for_training()

        # Should only be called once (second call is a no-op)
        mock_inf_engine.offload.assert_called_once()
        mock_train_engine.onload.assert_called_once()

    def test_prepare_for_inference_offloads_train_and_onloads_inf(
        self, orchestrator, mock_train_engine, mock_inf_engine
    ):
        """Test that prepare_for_inference switches GPU ownership correctly."""
        # Must offload training first, then switch to training mode
        orchestrator.initial_offload_training()
        orchestrator.prepare_for_training()
        mock_inf_engine.offload.reset_mock()
        mock_train_engine.onload.reset_mock()
        mock_train_engine.offload.reset_mock()

        meta = MagicMock()
        meta.path = "/tmp/test_weights"

        with patch.object(orchestrator, "_direct_disk_weight_update"):
            orchestrator.prepare_for_inference(meta)

        mock_train_engine.offload.assert_called_once()
        mock_inf_engine.onload.assert_called_once()
        assert orchestrator._train_on_gpu is False
        assert orchestrator._inf_on_gpu is True

    def test_prepare_for_inference_calls_weight_update(
        self, orchestrator, mock_train_engine, mock_inf_engine
    ):
        """Test that prepare_for_inference triggers disk weight update."""
        # Must offload training first, then switch to training mode
        orchestrator.initial_offload_training()
        orchestrator.prepare_for_training()

        meta = MagicMock()
        meta.path = "/tmp/test_weights"

        with patch.object(
            orchestrator, "_direct_disk_weight_update"
        ) as mock_update:
            orchestrator.prepare_for_inference(meta)
            mock_update.assert_called_once_with(meta)

    def test_cleanup_removes_weight_directory(self, orchestrator, temp_weight_dir):
        """Test that cleanup removes the weight directory."""
        # Create the directory
        os.makedirs(temp_weight_dir, exist_ok=True)
        assert os.path.exists(temp_weight_dir)

        orchestrator.cleanup()
        assert not os.path.exists(temp_weight_dir)

    def test_cleanup_ignores_missing_directory(self, orchestrator):
        """Test that cleanup handles missing directory gracefully."""
        orchestrator.config.weight_path = "/nonexistent/path"
        # Should not raise
        orchestrator.cleanup()

    def test_full_cycle_train_infer(
        self, orchestrator, mock_train_engine, mock_inf_engine
    ):
        """Test a full cycle: init offload → training → inference."""
        # Initial state: both engines on GPU
        assert orchestrator._inf_on_gpu is True
        assert orchestrator._train_on_gpu is True

        # Offload training engine (real init sequence)
        orchestrator.initial_offload_training()
        assert orchestrator._train_on_gpu is False
        assert orchestrator._inf_on_gpu is True

        # Reset mocks after initial offload
        mock_train_engine.offload.reset_mock()

        # Switch to training
        orchestrator.prepare_for_training()
        assert orchestrator._train_on_gpu is True
        assert orchestrator._inf_on_gpu is False

        # Switch back to inference
        meta = MagicMock()
        meta.path = "/tmp/test_weights"
        with patch.object(orchestrator, "_direct_disk_weight_update"):
            orchestrator.prepare_for_inference(meta)
        assert orchestrator._train_on_gpu is False
        assert orchestrator._inf_on_gpu is True

        # Verify call order
        assert mock_inf_engine.offload.call_count == 1
        assert mock_train_engine.onload.call_count == 1
        assert mock_train_engine.offload.call_count == 1
        assert mock_inf_engine.onload.call_count == 1


class TestWeightUpdateMetaColocated:
    """Tests for WeightUpdateMeta.from_colocated_disk factory method."""

    def test_from_colocated_disk_default(self):
        from areal.api.io_struct import WeightUpdateMeta

        meta = WeightUpdateMeta.from_colocated_disk()
        assert meta.type == "disk"
        assert meta.path == "/dev/shm/areal_colocated_weights/weight_update"
        assert meta.use_lora is False
        assert meta.clear_checkpoint_after_load is False

    def test_from_colocated_disk_custom_path(self):
        from areal.api.io_struct import WeightUpdateMeta

        meta = WeightUpdateMeta.from_colocated_disk(
            weight_path="/tmp/custom_weights"
        )
        assert meta.path == "/tmp/custom_weights/weight_update"

    def test_from_colocated_disk_with_lora(self):
        from areal.api.io_struct import WeightUpdateMeta

        meta = WeightUpdateMeta.from_colocated_disk(
            use_lora=True,
            lora_name="test_lora",
            base_model_name="Qwen/Qwen2.5-1.5B",
        )
        assert meta.use_lora is True
        assert meta.lora_name == "test_lora"
        assert meta.base_model_name == "Qwen/Qwen2.5-1.5B"

    def test_from_colocated_disk_with_version(self):
        from areal.api.io_struct import WeightUpdateMeta

        meta = WeightUpdateMeta.from_colocated_disk(
            weight_path="/dev/shm/test_weights"
        )
        versioned = meta.with_version(5)
        assert versioned.path == "/dev/shm/test_weights/weight_update_v5"
        assert versioned.version == 5


def _make_validation_trainer() -> PPOTrainer:
    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer.allocation_mode = SimpleNamespace(gen_backend="sglang")
    trainer.config = SimpleNamespace(
        actor=SimpleNamespace(colocated=True, kl_ctl=0),
        rollout=SimpleNamespace(return_routed_experts=False, openai=None),
        critic=None,
        ref=None,
        teacher=None,
        cluster=SimpleNamespace(n_nodes=1),
    )
    return trainer


class TestPPOTrainerColocatedValidation:
    def test_validate_cfg_rejects_single_controller(self):
        trainer = _make_validation_trainer()

        with patch(
            "areal.trainer.rl_trainer.is_single_controller", return_value=True
        ), pytest.raises(ValueError, match="only supports SPMD mode"):
            trainer._validate_cfg(train_dataset=object())

    def test_validate_cfg_rejects_multi_node(self):
        trainer = _make_validation_trainer()
        trainer.config.cluster.n_nodes = 2

        with patch(
            "areal.trainer.rl_trainer.is_single_controller", return_value=False
        ), pytest.raises(ValueError, match="single-node runs"):
            trainer._validate_cfg(train_dataset=object())

    def test_validate_cfg_rejects_missing_train_dataset(self):
        trainer = _make_validation_trainer()

        with patch(
            "areal.trainer.rl_trainer.is_single_controller", return_value=False
        ), pytest.raises(ValueError, match="requires a train_dataset"):
            trainer._validate_cfg(train_dataset=None)

    def test_validate_cfg_rejects_online_mode(self):
        trainer = _make_validation_trainer()
        trainer.config.rollout.openai = SimpleNamespace(mode="online")

        with patch(
            "areal.trainer.rl_trainer.is_single_controller", return_value=False
        ), pytest.raises(ValueError, match="rollout.openai.mode='online'"):
            trainer._validate_cfg(train_dataset=object())

    @pytest.mark.parametrize(
        ("mutate", "expected_error"),
        [
            (
                lambda trainer: setattr(trainer.config, "critic", object()),
                "critic is not supported",
            ),
            (
                lambda trainer: setattr(trainer.config, "ref", object()),
                "ref/kl_ctl is not supported",
            ),
            (
                lambda trainer: setattr(trainer.config.actor, "kl_ctl", 0.1),
                "ref/kl_ctl is not supported",
            ),
            (
                lambda trainer: setattr(trainer.config, "teacher", object()),
                "teacher is not supported",
            ),
        ],
    )
    def test_validate_cfg_rejects_non_actor_only_components(
        self, mutate, expected_error
    ):
        trainer = _make_validation_trainer()
        mutate(trainer)

        with patch(
            "areal.trainer.rl_trainer.is_single_controller", return_value=False
        ), pytest.raises(ValueError, match=expected_error):
            trainer._validate_cfg(train_dataset=object())

    def test_validate_cfg_does_not_apply_colocated_restrictions_to_standard_mode(self):
        trainer = _make_validation_trainer()
        trainer.config.actor.colocated = False
        trainer.config.cluster.n_nodes = 4
        trainer.config.rollout.openai = SimpleNamespace(mode="online")
        trainer.config.critic = object()
        trainer.config.ref = object()
        trainer.config.teacher = object()

        with patch(
            "areal.trainer.rl_trainer.is_single_controller", return_value=True
        ):
            trainer._validate_cfg(train_dataset=None)
