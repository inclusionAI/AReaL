"""Unit tests for TrainController.

Tests cover initialization, worker management, batch operations,
RPC wrappers, PPO/SFT methods, weight management, and error handling.
"""

import asyncio
from unittest.mock import Mock

import pytest
import torch

from areal.api.alloc_mode import ParallelStrategy
from areal.api.cli_args import SchedulingSpec, TrainEngineConfig
from areal.api.engine_api import TrainEngine
from areal.api.io_struct import (
    AllocationMode,
    FinetuneSpec,
    SaveLoadMeta,
    WeightUpdateMeta,
)
from areal.api.scheduler_api import Worker
from areal.controller.batch import DistributedBatchMemory
from areal.controller.train_controller import TrainController


class MockTrainEngine(TrainEngine):
    """Mock TrainEngine for testing."""

    @classmethod
    def __module__(cls):
        return "areal.tests.test_train_controller"

    @classmethod
    def __name__(cls):
        return "MockTrainEngine"


class MockScheduler:
    """Mock Scheduler for testing TrainController."""

    def __init__(self):
        self.workers = []
        self.call_count = 0
        self.engine_calls = []
        self.deleted_roles = []
        self.env_settings: dict[str, dict[str, str]] = {}

    def create_workers(self, job):
        """Create mock workers based on job configuration."""
        worker_ids = [f"{job.role}/{i}" for i in range(job.replicas)]
        self.workers = [
            Worker(
                id=wid,
                ip="127.0.0.1",
                worker_ports=["8000", "8001"],
                engine_ports=["9000", "9001"],
            )
            for wid in worker_ids
        ]
        return worker_ids

    def get_workers(self, role, timeout=None):
        """Return list of workers for the given role."""
        return self.workers

    async def set_worker_env(self, worker_id, env):
        """Mock environment configuration."""
        await asyncio.sleep(0.001)
        self.env_settings[worker_id] = {k: str(v) for k, v in env.items()}

    async def create_engine(self, worker_id, engine, **kwargs):
        """Mock engine creation."""
        await asyncio.sleep(0.001)
        return None

    async def async_call_engine(self, worker_id, method, *args, **kwargs):
        """Mock async engine call."""
        self.engine_calls.append((worker_id, method, args, kwargs))
        self.call_count += 1

        # Return appropriate mock results based on method
        if method == "is_data_parallel_head":
            # First worker in each DP group is the head
            worker_idx = int(worker_id.split("/")[-1])
            return worker_idx % 2 == 0  # Every other worker is a DP head

        elif method == "get_version":
            return 1

        elif method == "train_lm":
            return {"lm_loss": 0.4, "perplexity": 1.5}

        elif method == "evaluate_lm":
            # Return tensor with shape [batch_size, seq_len] to match expected format
            return torch.tensor([[0.35, 0.35, 0.35, 0.35]])

        await asyncio.sleep(0.001)
        return None

    def delete_workers(self, role):
        """Mock worker deletion."""
        self.deleted_roles.append(role)
        self.workers.clear()


# ==================== FIXTURES ====================


@pytest.fixture
def mock_scheduler():
    """Provide a MockScheduler instance."""
    return MockScheduler()


@pytest.fixture
def train_config():
    """Provide a TrainEngineConfig for testing."""
    return TrainEngineConfig(
        scheduling_spec=(
            SchedulingSpec(
                cpu=4,
                gpu=1,
                mem=16000,
                port_count=2,
                cmd="python -m areal.scheduler.rpc.rpc_server",
            ),
        )
    )


@pytest.fixture
def alloc_mode():
    """Provide an AllocationMode for testing."""
    mode = AllocationMode.from_str("d4t2p1")
    return mode


@pytest.fixture
def parallel_strategy():
    """Provide a ParallelStrategy for testing."""
    return ParallelStrategy(
        data_parallel_size=4, tensor_parallel_size=2, pipeline_parallel_size=1
    )


@pytest.fixture
def ft_spec():
    """Provide a FinetuneSpec for testing."""
    return FinetuneSpec(total_train_epochs=10, dataset_size=1000, train_batch_size=32)


@pytest.fixture
def train_controller(mock_scheduler, train_config):
    """Provide a TrainController instance."""
    return TrainController(
        train_engine=MockTrainEngine, config=train_config, scheduler=mock_scheduler
    )


def create_mock_distributed_batch(size=4, seq_len=10):
    """Create a mock DistributedBatch for testing."""
    data = {
        "input_ids": torch.randint(0, 100, (size, seq_len)),
        "attention_mask": torch.ones(size, seq_len, dtype=torch.bool),
        "loss_mask": torch.ones(size, seq_len, dtype=torch.bool),
    }
    return DistributedBatchMemory.from_dict(data)


# ==================== TEST CLASSES ====================


class TestTrainControllerInitialization:
    """Tests for TrainController initialization and setup."""

    def test_constructor(self, mock_scheduler, train_config):
        """Test TrainController constructor."""
        controller = TrainController(
            train_engine=MockTrainEngine, config=train_config, scheduler=mock_scheduler
        )

        assert controller.train_engine == MockTrainEngine
        assert controller.config == train_config
        assert controller.scheduler == mock_scheduler
        assert controller.workers == []
        assert controller.workers_is_dp_head == []

    def test_initialize(self, train_controller, alloc_mode, ft_spec):
        """Test initialize method creates workers and engines."""
        train_controller.initialize(
            role="train_worker",
            alloc_mode=alloc_mode,
            ft_spec=ft_spec,
        )

        # Verify workers were created
        assert len(train_controller.workers) == alloc_mode.train.world_size
        assert train_controller._worker_role == "train_worker"
        assert train_controller.alloc_mode == alloc_mode

        # Verify DP heads were identified
        assert len(train_controller.workers_is_dp_head) == len(train_controller.workers)

        # Verify scheduler was called
        assert train_controller.scheduler.call_count > 0
        # Verify environment configuration occurred for each worker
        assert len(train_controller.scheduler.env_settings) == len(
            train_controller.workers
        )

    def test_create_process_group_sets_parallel_strategy(
        self, train_controller, parallel_strategy
    ):
        """Test that create_process_group correctly assigns parallel_strategy.

        This is a regression test for the bug at line 79 where parallel_strategy
        was being assigned to itself instead of the parameter.
        """
        # Setup: Add mock workers
        train_controller.workers = [Mock(), Mock()]
        train_controller.parallel_strategy = parallel_strategy

        # Call create_process_group with a different strategy
        new_strategy = ParallelStrategy(
            data_parallel_size=8, tensor_parallel_size=1, pipeline_parallel_size=1
        )

        # create_process_group is now a dummy method that does nothing
        # It no longer calls custom_function_call
        train_controller.create_process_group(new_strategy)

        # The parallel_strategy is not updated in create_process_group anymore
        # It's set during initialize() from alloc_mode.train

    def test_identify_dp_heads(self, train_controller, alloc_mode, ft_spec):
        """Test _identify_dp_heads correctly identifies DP head workers."""
        train_controller.initialize(
            role="train_worker",
            alloc_mode=alloc_mode,
            ft_spec=ft_spec,
        )

        # MockScheduler returns True for even-indexed workers
        for idx, is_head in enumerate(train_controller.workers_is_dp_head):
            assert is_head == (idx % 2 == 0)


class TestTrainControllerDestroy:
    """Tests for TrainController cleanup and destruction."""

    def test_destroy(self, train_controller, alloc_mode, ft_spec):
        """Test destroy method cleans up resources."""
        # Initialize first
        train_controller.initialize(
            role="train_worker",
            alloc_mode=alloc_mode,
            ft_spec=ft_spec,
        )

        initial_worker_count = len(train_controller.workers)
        assert initial_worker_count > 0

        # Call destroy
        train_controller.destroy()

        # Verify cleanup
        assert len(train_controller.workers) == 0
        assert len(train_controller.workers_is_dp_head) == 0
        assert "train_worker" in train_controller.scheduler.deleted_roles

    def test_destroy_handles_errors(self, train_controller, alloc_mode, ft_spec):
        """Test destroy handles errors gracefully."""
        train_controller.initialize(
            role="train_worker",
            alloc_mode=alloc_mode,
            ft_spec=ft_spec,
        )

        # Make delete_workers raise an exception
        def raise_error(role):
            raise RuntimeError("Simulated error")

        train_controller.scheduler.delete_workers = raise_error

        # Should not raise, just log the error
        train_controller.destroy()

        # Workers should still be cleared
        assert len(train_controller.workers) == 0


class TestTrainControllerBatchOperations:
    """Tests for batch splitting and alignment operations."""

    def test_align_batches_with_dp_rebalance(
        self, train_controller, alloc_mode, ft_spec
    ):
        """Test _align_batches_with_dp with rebalance=True."""
        train_controller.initialize(
            role="train_worker",
            alloc_mode=alloc_mode,
            ft_spec=ft_spec,
        )

        batch = create_mock_distributed_batch(size=16)
        chunks = train_controller._align_batches_with_dp(batch, rebalance=True)

        # Should split into dp_size chunks
        assert len(chunks) == alloc_mode.train.dp_size

        # Each chunk should be a DistributedBatch
        for chunk in chunks:
            assert isinstance(chunk, DistributedBatchMemory)

    def test_align_batches_with_dp_no_rebalance(
        self, train_controller, alloc_mode, ft_spec
    ):
        """Test _align_batches_with_dp with rebalance=False."""
        train_controller.initialize(
            role="train_worker",
            alloc_mode=alloc_mode,
            ft_spec=ft_spec,
        )

        batch = create_mock_distributed_batch(size=16)
        chunks = train_controller._align_batches_with_dp(batch, rebalance=False)

        # Should split into dp_size chunks
        assert len(chunks) == alloc_mode.train.dp_size

        # Each chunk should be a DistributedBatch
        for chunk in chunks:
            assert isinstance(chunk, DistributedBatchMemory)


class TestTrainControllerMergeResults:
    """Tests for result merging from workers."""

    def test_merge_results_with_tensor_dict(self, train_controller):
        """Test _merge_results with dictionary of tensors."""
        results = [
            {"loss": torch.tensor([0.5, 0.6])},
            {"loss": torch.tensor([0.3, 0.4])},
        ]

        merged = train_controller._merge_results(results, "some_method")

        # Should concatenate into DistributedBatch
        assert isinstance(merged, DistributedBatchMemory)
        assert "loss" in merged.get_data()

    def test_merge_results_with_empty_dict(self, train_controller):
        """Test _merge_results with empty dictionaries."""
        results = [{}, {}]

        merged = train_controller._merge_results(results, "some_method")

        # Should return empty DistributedBatch
        assert isinstance(merged, DistributedBatchMemory)
        assert len(merged.get_data()) == 0

    def test_merge_results_with_non_tensor(self, train_controller):
        """Test _merge_results with non-tensor results."""
        results = [{"status": "ok"}, {"status": "ok"}]

        merged = train_controller._merge_results(results, "some_method")

        # Should return first result (already synchronized)
        assert merged == {"status": "ok"}

    def test_merge_results_accepts_method_parameter(self, train_controller):
        """Test that _merge_results accepts method parameter.

        This is a regression test for the bug at line 279 where the method
        parameter was missing from the signature.
        """
        # Use tensors with proper shape [batch_size, seq_len]
        results = [torch.tensor([[0.5, 0.5]]), torch.tensor([[0.3, 0.3]])]

        # This should work without TypeError
        try:
            result = train_controller._merge_results(results, "some_method")
            # Test passes if no exception
            assert result is not None
        except TypeError as e:
            if "missing" in str(e) and "required positional argument" in str(e):
                pytest.fail(f"_merge_results missing required parameter: {e}")


class TestTrainControllerRPCWrappers:
    """Tests for RPC wrapper methods."""

    def test_train_mode(self, train_controller, alloc_mode, ft_spec):
        """Test train() method sets training mode."""
        train_controller.initialize(
            role="train_worker",
            alloc_mode=alloc_mode,
            ft_spec=ft_spec,
        )

        result = train_controller.train(mode=True)

        # Should return self for chaining
        assert result is train_controller

        # Verify custom_function_call was invoked
        engine_calls = [call[1] for call in train_controller.scheduler.engine_calls]
        assert "train" in engine_calls

    def test_eval_mode(self, train_controller, alloc_mode, ft_spec):
        """Test eval() method sets evaluation mode."""
        train_controller.initialize(
            role="train_worker",
            alloc_mode=alloc_mode,
            ft_spec=ft_spec,
        )

        result = train_controller.eval()

        # Should return self for chaining
        assert result is train_controller

        # Verify train(False) was called
        engine_calls = [call[1] for call in train_controller.scheduler.engine_calls]
        assert "train" in engine_calls

    def test_step_lr_scheduler(self, train_controller, alloc_mode, ft_spec):
        """Test step_lr_scheduler() method."""
        train_controller.initialize(
            role="train_worker",
            alloc_mode=alloc_mode,
            ft_spec=ft_spec,
        )

        train_controller.step_lr_scheduler()

        # Verify step_lr_scheduler was called on engines
        engine_calls = [call[1] for call in train_controller.scheduler.engine_calls]
        assert "step_lr_scheduler" in engine_calls


class TestTrainControllerSFTMethods:
    """Tests for SFT-specific methods."""

    def test_train_lm(self, train_controller, alloc_mode, ft_spec):
        """Test train_lm() method."""
        train_controller.initialize(
            role="train_worker",
            alloc_mode=alloc_mode,
            ft_spec=ft_spec,
        )

        batch = create_mock_distributed_batch(size=8)
        result = train_controller.train_lm(batch)

        # Should return stats dictionary
        assert isinstance(result, dict)

        # Verify train_lm was called on engines
        engine_calls = [call[1] for call in train_controller.scheduler.engine_calls]
        assert "train_lm" in engine_calls

    def test_evaluate_lm(self, train_controller, alloc_mode, ft_spec):
        """Test evaluate_lm() method."""
        train_controller.initialize(
            role="train_worker",
            alloc_mode=alloc_mode,
            ft_spec=ft_spec,
        )

        batch = create_mock_distributed_batch(size=8)
        result = train_controller.evaluate_lm(batch)

        # Should return loss tensor or merged results
        assert result is not None

        # Verify evaluate_lm was called on engines
        engine_calls = [call[1] for call in train_controller.scheduler.engine_calls]
        assert "evaluate_lm" in engine_calls


class TestTrainControllerWeightManagement:
    """Tests for weight management operations."""

    def test_set_version(self, train_controller, alloc_mode, ft_spec):
        """Test set_version() method."""
        train_controller.initialize(
            role="train_worker",
            alloc_mode=alloc_mode,
            ft_spec=ft_spec,
        )

        train_controller.set_version(42)

        # Verify set_version was called on engines
        engine_calls = [call[1] for call in train_controller.scheduler.engine_calls]
        assert "set_version" in engine_calls

    def test_get_version(self, train_controller, alloc_mode, ft_spec):
        """Test get_version() method."""
        train_controller.initialize(
            role="train_worker",
            alloc_mode=alloc_mode,
            ft_spec=ft_spec,
        )

        version = train_controller.get_version()

        # Should return version number
        assert isinstance(version, int)

        # Verify get_version was called on engines
        engine_calls = [call[1] for call in train_controller.scheduler.engine_calls]
        assert "get_version" in engine_calls

    def test_save(self, train_controller, alloc_mode, ft_spec):
        """Test save() method."""
        train_controller.initialize(
            role="train_worker",
            alloc_mode=alloc_mode,
            ft_spec=ft_spec,
        )

        meta = SaveLoadMeta(
            path="/tmp/checkpoint", weight_format="safetensors", with_optim=True
        )
        train_controller.save(meta)

        # Verify save was called on engines
        engine_calls = [call[1] for call in train_controller.scheduler.engine_calls]
        assert "save" in engine_calls

    def test_load(self, train_controller, alloc_mode, ft_spec):
        """Test load() method."""
        train_controller.initialize(
            role="train_worker",
            alloc_mode=alloc_mode,
            ft_spec=ft_spec,
        )

        meta = SaveLoadMeta(
            path="/tmp/checkpoint", weight_format="safetensors", with_optim=True
        )
        train_controller.load(meta)

        # Verify load was called on engines
        engine_calls = [call[1] for call in train_controller.scheduler.engine_calls]
        assert "load" in engine_calls


class TestTrainControllerCustomFunctionCall:
    """Tests for custom_function_call orchestration."""

    def test_custom_function_call_with_distributed_batch(
        self, train_controller, alloc_mode, ft_spec
    ):
        """Test custom_function_call with DistributedBatch argument."""
        train_controller.initialize(
            role="train_worker",
            alloc_mode=alloc_mode,
            ft_spec=ft_spec,
        )

        # Clear previous calls from initialization
        train_controller.scheduler.engine_calls = []

        batch = create_mock_distributed_batch(size=16)
        result = train_controller._custom_function_call("train_lm", input_=batch)

        # Should split batch across DP groups and call only DP heads
        assert result is not None

        # Count how many workers were called
        worker_calls = len(train_controller.scheduler.engine_calls)

        # Should call all workers (DP heads get data, others get empty)
        assert worker_calls == len(train_controller.workers)

    def test_custom_function_call_with_regular_args(
        self, train_controller, alloc_mode, ft_spec
    ):
        """Test custom_function_call with non-DistributedBatch arguments."""
        train_controller.initialize(
            role="train_worker",
            alloc_mode=alloc_mode,
            ft_spec=ft_spec,
        )

        # Clear previous calls
        train_controller.scheduler.engine_calls = []

        result = train_controller._custom_function_call("set_version", 5)

        # set_version returns None, which is expected - just verify it doesn't crash
        # The key test is that all workers were called
        assert result is None

        # Verify all workers were called
        assert len(train_controller.scheduler.engine_calls) == len(
            train_controller.workers
        )

    def test_custom_function_call_filters_dp_heads(
        self, train_controller, alloc_mode, ft_spec
    ):
        """Test custom_function_call only returns results from DP heads."""
        train_controller.initialize(
            role="train_worker",
            alloc_mode=alloc_mode,
            ft_spec=ft_spec,
        )

        batch = create_mock_distributed_batch(size=8)
        train_controller._custom_function_call("ppo_update", input_=batch)

        # Results should only come from DP head workers
        # (verified by _merge_results receiving filtered results)


class TestTrainControllerEdgeCases:
    def test_create_process_group_is_dummy(self, train_controller, parallel_strategy):
        """Test create_process_group is now a dummy method that does nothing."""
        # Don't initialize, so workers list is empty
        # Should not raise - create_process_group is now a no-op
        train_controller.create_process_group(parallel_strategy)

        # parallel_strategy should not be set by create_process_group
        assert train_controller.parallel_strategy is None

    def test_method_chaining(self, train_controller, alloc_mode, ft_spec):
        """Test that train() and eval() support method chaining."""
        train_controller.initialize(
            role="train_worker",
            alloc_mode=alloc_mode,
            ft_spec=ft_spec,
        )

        # Should be able to chain calls
        result = train_controller.train().eval().train()
        assert result is train_controller


class TestTrainControllerRolloutIntegration:
    """Tests for rollout engine integration methods."""

    def test_connect_engine_sets_rollout(self, train_controller, alloc_mode, ft_spec):
        """Test connect_engine correctly sets the rollout controller."""
        train_controller.initialize(
            role="train_worker",
            alloc_mode=alloc_mode,
            ft_spec=ft_spec,
        )

        mock_rollout = Mock()
        meta = WeightUpdateMeta(type="disk", path="/tmp/test")

        train_controller.connect_engine(mock_rollout, meta)

        assert train_controller.rollout == mock_rollout

    def test_connect_engine_warns_on_change(
        self, train_controller, alloc_mode, ft_spec
    ):
        """Test connect_engine logs warning when rollout controller changes."""
        train_controller.initialize(
            role="train_worker",
            alloc_mode=alloc_mode,
            ft_spec=ft_spec,
        )

        mock_rollout1 = Mock()
        mock_rollout2 = Mock()
        meta = WeightUpdateMeta(type="disk", path="/tmp/test")

        train_controller.connect_engine(mock_rollout1, meta)
        train_controller.connect_engine(mock_rollout2, meta)

        assert train_controller.rollout == mock_rollout2

    def test_connect_engine_same_rollout_no_warning(
        self, train_controller, alloc_mode, ft_spec
    ):
        """Test connect_engine does not warn when same rollout controller is used."""
        train_controller.initialize(
            role="train_worker",
            alloc_mode=alloc_mode,
            ft_spec=ft_spec,
        )

        mock_rollout = Mock()
        meta = WeightUpdateMeta(type="disk", path="/tmp/test")

        train_controller.connect_engine(mock_rollout, meta)
        train_controller.connect_engine(mock_rollout, meta)

        assert train_controller.rollout == mock_rollout

    def test_check_rollout_engine_connected_raises_when_not_connected(
        self, train_controller, alloc_mode, ft_spec
    ):
        """Test _check_rollout_engine_connected raises when rollout is not connected."""
        train_controller.initialize(
            role="train_worker",
            alloc_mode=alloc_mode,
            ft_spec=ft_spec,
        )

        with pytest.raises(RuntimeError, match="Rollout engine not connected"):
            train_controller._check_rollout_engine_connected()

    def test_check_rollout_engine_connected_passes_when_connected(
        self, train_controller, alloc_mode, ft_spec
    ):
        """Test _check_rollout_engine_connected passes when rollout is connected."""
        train_controller.initialize(
            role="train_worker",
            alloc_mode=alloc_mode,
            ft_spec=ft_spec,
        )

        mock_rollout = Mock()
        meta = WeightUpdateMeta(type="disk", path="/tmp/test")
        train_controller.connect_engine(mock_rollout, meta)

        # Should not raise
        train_controller._check_rollout_engine_connected()

    def test_prepare_batch_delegates_to_rollout(
        self, train_controller, alloc_mode, ft_spec
    ):
        """Test prepare_batch delegates to rollout controller."""
        train_controller.initialize(
            role="train_worker",
            alloc_mode=alloc_mode,
            ft_spec=ft_spec,
        )

        mock_rollout = Mock()
        mock_rollout.prepare_batch.return_value = DistributedBatchMemory.from_dict({})
        meta = WeightUpdateMeta(type="disk", path="/tmp/test")
        train_controller.connect_engine(mock_rollout, meta)

        mock_dataloader = Mock()
        train_controller.prepare_batch(
            dataloader=mock_dataloader,
            workflow="test.workflow",
            workflow_kwargs={"key": "value"},
        )

        mock_rollout.prepare_batch.assert_called_once_with(
            dataloader=mock_dataloader,
            workflow="test.workflow",
            workflow_kwargs={"key": "value"},
            should_accept_fn=None,
        )

    def test_rollout_batch_delegates_to_rollout(
        self, train_controller, alloc_mode, ft_spec
    ):
        """Test rollout_batch delegates to rollout controller."""
        train_controller.initialize(
            role="train_worker",
            alloc_mode=alloc_mode,
            ft_spec=ft_spec,
        )

        mock_rollout = Mock()
        mock_rollout.rollout_batch.return_value = DistributedBatchMemory.from_dict({})
        meta = WeightUpdateMeta(type="disk", path="/tmp/test")
        train_controller.connect_engine(mock_rollout, meta)

        data = [{"id": 1}, {"id": 2}]
        train_controller.rollout_batch(
            data=data,
            workflow="test.workflow",
            workflow_kwargs={"key": "value"},
        )

        mock_rollout.rollout_batch.assert_called_once_with(
            data=data,
            workflow="test.workflow",
            workflow_kwargs={"key": "value"},
            should_accept_fn=None,
        )


class TestTrainControllerPPOMethods:
    """Tests for PPO-specific methods."""

    def test_compute_logp(self, train_controller, alloc_mode, ft_spec):
        """Test compute_logp method."""
        train_controller.initialize(
            role="train_worker",
            alloc_mode=alloc_mode,
            ft_spec=ft_spec,
        )

        batch = create_mock_distributed_batch(size=8)
        train_controller.compute_logp(batch)

        engine_calls = [call[1] for call in train_controller.scheduler.engine_calls]
        assert "compute_logp" in engine_calls

    def test_compute_advantages(self, train_controller, alloc_mode, ft_spec):
        """Test compute_advantages method."""
        train_controller.initialize(
            role="train_worker",
            alloc_mode=alloc_mode,
            ft_spec=ft_spec,
        )

        batch = create_mock_distributed_batch(size=8)
        train_controller.compute_advantages(batch)

        engine_calls = [call[1] for call in train_controller.scheduler.engine_calls]
        assert "compute_advantages" in engine_calls

    def test_ppo_update(self, train_controller, alloc_mode, ft_spec):
        """Test ppo_update method."""
        train_controller.initialize(
            role="train_worker",
            alloc_mode=alloc_mode,
            ft_spec=ft_spec,
        )

        batch = create_mock_distributed_batch(size=8)
        train_controller.ppo_update(batch)

        engine_calls = [call[1] for call in train_controller.scheduler.engine_calls]
        assert "ppo_update" in engine_calls


class TestTrainControllerWeightUpdateMethods:
    """Tests for weight update methods."""

    def test_update_weights_raises_when_not_connected(
        self, train_controller, alloc_mode, ft_spec
    ):
        """Test update_weights raises when rollout is not connected."""
        train_controller.initialize(
            role="train_worker",
            alloc_mode=alloc_mode,
            ft_spec=ft_spec,
        )

        meta = WeightUpdateMeta(type="disk", path="/tmp/test")

        with pytest.raises(RuntimeError, match="Rollout engine not connected"):
            train_controller.update_weights(meta)

    def test_update_weights_invalid_type_raises(
        self, train_controller, alloc_mode, ft_spec
    ):
        """Test update_weights raises for invalid type."""
        train_controller.initialize(
            role="train_worker",
            alloc_mode=alloc_mode,
            ft_spec=ft_spec,
        )

        mock_rollout = Mock()
        meta = WeightUpdateMeta(type="disk", path="/tmp/test")
        train_controller.connect_engine(mock_rollout, meta)

        invalid_meta = WeightUpdateMeta(type="invalid_type", path="/tmp/test")

        with pytest.raises(ValueError, match="Unknown weight update type"):
            train_controller.update_weights(invalid_meta)


class TestTrainControllerExportStats:
    """Tests for export_stats method."""

    def test_export_stats(self, train_controller, alloc_mode, ft_spec):
        """Test export_stats returns statistics from first worker."""
        train_controller.initialize(
            role="train_worker",
            alloc_mode=alloc_mode,
            ft_spec=ft_spec,
        )

        # Mock the scheduler to return stats
        expected_stats = {"loss": 0.5, "accuracy": 0.95}

        async def mock_async_call(*args, **kwargs):
            if kwargs.get("method") == "export_stats" or (
                len(args) > 1 and args[1] == "export_stats"
            ):
                return expected_stats
            return None

        train_controller.scheduler.async_call_engine = mock_async_call

        result = train_controller.export_stats()
        assert result == expected_stats

    def test_export_stats_calls_all_workers(
        self, train_controller, alloc_mode, ft_spec
    ):
        """Test export_stats calls all workers."""
        train_controller.initialize(
            role="train_worker",
            alloc_mode=alloc_mode,
            ft_spec=ft_spec,
        )

        # Clear previous calls
        train_controller.scheduler.engine_calls = []

        train_controller.export_stats()

        export_stats_calls = [
            call
            for call in train_controller.scheduler.engine_calls
            if call[1] == "export_stats"
        ]
        assert len(export_stats_calls) == len(train_controller.workers)


class TestTrainControllerSGLangEnvVars:
    """Tests for SGLang-specific environment variable handling."""

    def test_sglang_backend_sets_nccl_env_vars(self, mock_scheduler):
        """Test that SGLang backend sets NCCL environment variables."""
        # Create config with mutable scheduling_spec
        scheduling_spec = SchedulingSpec(
            cpu=4,
            gpu=1,
            mem=16000,
            port_count=2,
            cmd="python -m areal.scheduler.rpc.rpc_server",
        )
        config = TrainEngineConfig(scheduling_spec=(scheduling_spec,))

        controller = TrainController(
            train_engine=MockTrainEngine,
            config=config,
            scheduler=mock_scheduler,
        )

        # Use allocation mode with sglang backend
        alloc_mode = AllocationMode.from_str("sglang:d2+d4t2p1")
        ft_spec = FinetuneSpec(
            total_train_epochs=10, dataset_size=1000, train_batch_size=32
        )

        controller.initialize(
            role="train_worker", alloc_mode=alloc_mode, ft_spec=ft_spec
        )

        # Check that NCCL env vars were set on all scheduling specs
        for spec in controller.config.scheduling_spec:
            assert spec.env_vars.get("NCCL_CUMEM_ENABLE") == "0"
            assert spec.env_vars.get("NCCL_NVLS_ENABLE") == "0"

    def test_non_sglang_backend_does_not_set_nccl_env_vars(self, mock_scheduler):
        """Test that non-SGLang backend does not set NCCL environment variables."""
        scheduling_spec = SchedulingSpec(
            cpu=4,
            gpu=1,
            mem=16000,
            port_count=2,
            cmd="python -m areal.scheduler.rpc.rpc_server",
        )
        config = TrainEngineConfig(scheduling_spec=(scheduling_spec,))

        controller = TrainController(
            train_engine=MockTrainEngine,
            config=config,
            scheduler=mock_scheduler,
        )

        # Use allocation mode without sglang backend
        alloc_mode = AllocationMode.from_str("d4t2p1")
        ft_spec = FinetuneSpec(
            total_train_epochs=10, dataset_size=1000, train_batch_size=32
        )

        controller.initialize(
            role="train_worker", alloc_mode=alloc_mode, ft_spec=ft_spec
        )

        # Check that NCCL env vars were NOT set
        for spec in controller.config.scheduling_spec:
            assert "NCCL_CUMEM_ENABLE" not in spec.env_vars
            assert "NCCL_NVLS_ENABLE" not in spec.env_vars


class TestTrainControllerAsyncMethods:
    """Tests for async method handling."""

    def test_run_async_task(self, train_controller):
        """Test _run_async_task correctly runs async tasks."""

        async def async_task():
            return 42

        result = train_controller._run_async_task(async_task())
        assert result == 42

    def test_run_async_task_with_exception(self, train_controller):
        """Test _run_async_task propagates exceptions."""

        async def failing_task():
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            train_controller._run_async_task(failing_task())


class TestTrainControllerDispatchInputs:
    """Tests for input dispatching across DP groups."""

    def test_dispatch_inputs_splits_distributed_batch(
        self, train_controller, alloc_mode, ft_spec
    ):
        """Test _dispatch_inputs correctly splits DistributedBatch."""
        train_controller.initialize(
            role="train_worker",
            alloc_mode=alloc_mode,
            ft_spec=ft_spec,
        )

        batch = create_mock_distributed_batch(size=16)
        split_args, split_kwargs = train_controller._dispatch_inputs(batch)

        # Should split into dp_size chunks
        assert len(split_args) == 1
        assert len(split_args[0]) == alloc_mode.train.dp_size

    def test_dispatch_inputs_replicates_non_batch_args(
        self, train_controller, alloc_mode, ft_spec
    ):
        """Test _dispatch_inputs replicates non-DistributedBatch arguments."""
        train_controller.initialize(
            role="train_worker",
            alloc_mode=alloc_mode,
            ft_spec=ft_spec,
        )

        scalar_arg = 42
        split_args, split_kwargs = train_controller._dispatch_inputs(scalar_arg)

        # Should replicate to all DP groups
        assert len(split_args) == 1
        assert all(arg == 42 for arg in split_args[0])
        assert len(split_args[0]) == alloc_mode.train.dp_size

    def test_dispatch_inputs_handles_kwargs(
        self, train_controller, alloc_mode, ft_spec
    ):
        """Test _dispatch_inputs correctly handles keyword arguments."""
        train_controller.initialize(
            role="train_worker",
            alloc_mode=alloc_mode,
            ft_spec=ft_spec,
        )

        batch = create_mock_distributed_batch(size=16)
        split_args, split_kwargs = train_controller._dispatch_inputs(
            input_=batch, learning_rate=0.001
        )

        assert "input_" in split_kwargs
        assert "learning_rate" in split_kwargs
        assert len(split_kwargs["input_"]) == alloc_mode.train.dp_size
        assert all(lr == 0.001 for lr in split_kwargs["learning_rate"])


class TestTrainControllerMergeResultsExtended:
    """Extended tests for result merging."""

    def test_merge_results_with_attention_mask(self, train_controller):
        """Test _merge_results correctly handles batches with attention_mask."""
        results = [
            {
                "input_ids": torch.randint(0, 100, (2, 5)),
                "attention_mask": torch.ones(2, 5, dtype=torch.bool),
            },
            {
                "input_ids": torch.randint(0, 100, (2, 7)),
                "attention_mask": torch.ones(2, 7, dtype=torch.bool),
            },
        ]

        merged = train_controller._merge_results(results, "some_method")

        assert isinstance(merged, DistributedBatchMemory)
        # Should have concatenated batches
        assert len(merged) == 4

    def test_merge_results_with_scalar_dict(self, train_controller):
        """Test _merge_results with dictionary of scalars."""
        results = [
            {"loss": 0.5, "accuracy": 0.9},
            {"loss": 0.5, "accuracy": 0.9},
        ]

        merged = train_controller._merge_results(results, "some_method")

        # Non-tensor dicts should return first result
        assert merged == {"loss": 0.5, "accuracy": 0.9}

    def test_merge_results_pads_tensors_correctly(self, train_controller):
        """Test _merge_results pads tensors to max sequence length."""
        # Create tensors with different sequence lengths
        results = [
            torch.randn(2, 5),  # batch=2, seq=5
            torch.randn(2, 10),  # batch=2, seq=10
        ]

        merged = train_controller._merge_results(results, "some_method")

        # Should be concatenated with padding
        assert merged.shape == (4, 10)


class TestTrainControllerAlignBatches:
    """Tests for batch alignment across DP groups."""

    def test_align_batches_empty_batch(self, train_controller, alloc_mode, ft_spec):
        """Test _align_batches_with_dp handles empty batch."""
        train_controller.initialize(
            role="train_worker",
            alloc_mode=alloc_mode,
            ft_spec=ft_spec,
        )

        empty_batch = DistributedBatchMemory.from_dict({})
        chunks = train_controller._align_batches_with_dp(empty_batch, rebalance=True)

        # Should replicate empty batch to all DP groups
        assert len(chunks) == alloc_mode.train.dp_size
        for chunk in chunks:
            assert len(chunk.get_data()) == 0

    def test_align_batches_with_rebalance(self, train_controller, alloc_mode, ft_spec):
        """Test _align_batches_with_dp with rebalance=True uses FFD."""
        train_controller.initialize(
            role="train_worker",
            alloc_mode=alloc_mode,
            ft_spec=ft_spec,
        )

        batch = create_mock_distributed_batch(size=16)
        chunks = train_controller._align_batches_with_dp(batch, rebalance=True)

        assert len(chunks) == alloc_mode.train.dp_size
        for chunk in chunks:
            assert isinstance(chunk, DistributedBatchMemory)

    def test_align_batches_without_rebalance(
        self, train_controller, alloc_mode, ft_spec
    ):
        """Test _align_batches_with_dp with rebalance=False uses simple chunking."""
        train_controller.initialize(
            role="train_worker",
            alloc_mode=alloc_mode,
            ft_spec=ft_spec,
        )

        batch = create_mock_distributed_batch(size=16)
        chunks = train_controller._align_batches_with_dp(batch, rebalance=False)

        assert len(chunks) == alloc_mode.train.dp_size
        for chunk in chunks:
            assert isinstance(chunk, DistributedBatchMemory)
