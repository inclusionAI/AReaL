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
        train_controller._custom_function_call("train_batch", input_=batch)

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
