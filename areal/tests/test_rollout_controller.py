import asyncio
import queue
from concurrent.futures import Future
from unittest.mock import Mock, patch

import pytest
import torch

from areal.api.alloc_mode import AllocationMode
from areal.api.cli_args import InferenceEngineConfig
from areal.api.io_struct import ModelRequest, ParamSpec, WeightUpdateMeta
from areal.api.scheduler_api import Worker
from areal.controller.batch import DistributedBatchMemory
from areal.controller.rollout_controller import RolloutController
from areal.core.async_task_runner import TaskQueueFullError


class MockScheduler:
    def __init__(self):
        self.workers = []
        self.call_count = 0
        self.engine_calls = []

    def create_workers(self, role, scheduler_config, *args, **kwargs):
        worker_ids = [f"{role}/{i}" for i in range(scheduler_config.replicas)]
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
        return self.workers

    async def create_engine(self, worker_id, engine, config):
        pass

    async def async_call_engine(self, worker_id, method, *args, **kwargs):
        self.engine_calls.append((worker_id, method, args, kwargs))
        self.call_count += 1

        if method == "run_workflow":
            await asyncio.sleep(0.01)
            return {
                "input_ids": torch.randint(0, 100, (1, 10)),
                "attention_mask": torch.ones(1, 10, dtype=torch.bool),
                "loss_mask": torch.tensor(
                    [0] * 5 + [1] * 5, dtype=torch.bool
                ).unsqueeze(0),
                "rewards": torch.randn(1),
            }
        elif method == "agenerate":
            return Mock()
        return None

    def call_engine(self, worker_id, method, *args, **kwargs):
        self.engine_calls.append((worker_id, method, args, kwargs))

        # For weight update methods that await call_engine, return a coroutine
        if method in [
            "update_weights_from_distributed",
            "update_weights_from_disk",
            "init_weights_update_group",
        ]:
            return self._async_call_engine_internal(worker_id, method, *args, **kwargs)

        return None

    async def _async_call_engine_internal(self, worker_id, method, *args, **kwargs):
        await asyncio.sleep(0.001)
        return None

    def delete_workers(self, role):
        self.workers.clear()


class MockInferenceEngine:
    @classmethod
    def __module__(cls):
        return "areal.tests.test_rollout_controller"

    @classmethod
    def __name__(cls):
        return "MockInferenceEngine"


class TestRolloutControllerInitialization:
    def test_constructor(self):
        config = InferenceEngineConfig(consumer_batch_size=16)
        scheduler = MockScheduler()

        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        assert controller.config == config
        assert controller.scheduler == scheduler
        assert controller.workers == []
        assert controller._current_worker_idx == 0
        assert controller._version == 0
        assert controller.runner is None
        assert controller.executor is None
        assert controller.staleness_manager is None

    def test_initialize_creates_workers(self):
        config = InferenceEngineConfig(
            consumer_batch_size=16,
            max_head_offpolicyness=2,
            enable_rollout_tracing=False,
        )
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        alloc_mode = AllocationMode.from_str("sglang.d2")
        controller.initialize(alloc_mode=alloc_mode)

        assert len(controller.workers) == 2
        assert controller.runner is not None
        assert controller.executor is not None
        assert controller.staleness_manager is not None

        controller.destroy()

    def test_initialize_creates_staleness_manager(self):
        config = InferenceEngineConfig(
            consumer_batch_size=32,
            max_head_offpolicyness=5,
            max_concurrent_rollouts=100,
        )
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        alloc_mode = AllocationMode.from_str("sglang.d1")
        controller.initialize(alloc_mode=alloc_mode)

        assert controller.staleness_manager.max_concurrent_rollouts == 100
        assert controller.staleness_manager.consumer_batch_size == 32
        assert controller.staleness_manager.max_staleness == 5

        controller.destroy()

    def test_initialize_uses_consumer_batch_size_as_fallback(self):
        config = InferenceEngineConfig(
            consumer_batch_size=64,
            max_head_offpolicyness=3,
            max_concurrent_rollouts=None,
        )
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        alloc_mode = AllocationMode.from_str("sglang.d1")
        controller.initialize(alloc_mode=alloc_mode)

        assert controller.staleness_manager.max_concurrent_rollouts == 64

        controller.destroy()

    def test_initialize_with_tracing_enabled(self):
        config = InferenceEngineConfig(
            consumer_batch_size=16,
            max_head_offpolicyness=2,
            enable_rollout_tracing=True,
        )
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        alloc_mode = AllocationMode.from_str("sglang.d1")
        controller.initialize(alloc_mode=alloc_mode)

        assert controller.runner.enable_tracing is True

        controller.destroy()


class TestRolloutControllerDestroy:
    def test_destroy_cleans_up_resources(self):
        config = InferenceEngineConfig(consumer_batch_size=16)
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        alloc_mode = AllocationMode.from_str("sglang.d1")
        controller.initialize(alloc_mode=alloc_mode)

        assert controller.runner is not None
        assert controller.executor is not None
        assert len(controller.workers) > 0

        controller.destroy()

        assert controller.runner is None
        assert controller.executor is None
        assert len(controller.workers) == 0

    def test_destroy_deletes_workers_via_scheduler(self):
        config = InferenceEngineConfig(consumer_batch_size=16)
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        alloc_mode = AllocationMode.from_str("sglang.d2")
        controller.initialize(alloc_mode=alloc_mode)

        assert len(scheduler.workers) == 2

        controller.destroy()

        assert len(scheduler.workers) == 0

    def test_destroy_handles_scheduler_error(self):
        config = InferenceEngineConfig(consumer_batch_size=16)
        scheduler = MockScheduler()
        scheduler.delete_workers = Mock(side_effect=Exception("Test error"))

        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        alloc_mode = AllocationMode.from_str("sglang.d1")
        controller.initialize(alloc_mode=alloc_mode)

        controller.destroy()


class TestRolloutControllerCapacity:
    def test_get_capacity_initial_state(self):
        config = InferenceEngineConfig(
            consumer_batch_size=16,
            max_concurrent_rollouts=32,
            max_head_offpolicyness=2,
        )
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        alloc_mode = AllocationMode.from_str("sglang.d1")
        controller.initialize(alloc_mode=alloc_mode)

        capacity = controller.get_capacity()
        assert capacity == 32

        controller.destroy()

    def test_get_capacity_uses_version(self):
        config = InferenceEngineConfig(
            consumer_batch_size=8,
            max_concurrent_rollouts=1000,
            max_head_offpolicyness=2,
        )
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        alloc_mode = AllocationMode.from_str("sglang.d1")
        controller.initialize(alloc_mode=alloc_mode)

        capacity_v0 = controller.get_capacity()

        controller.set_version(5)
        capacity_v5 = controller.get_capacity()

        assert capacity_v5 > capacity_v0

        controller.destroy()


class TestRolloutControllerWorkerSelection:
    def test_choose_worker_round_robin(self):
        config = InferenceEngineConfig(consumer_batch_size=16)
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        alloc_mode = AllocationMode.from_str("sglang.d3")
        controller.initialize(alloc_mode=alloc_mode)

        worker_ids = []
        for _ in range(6):
            worker = controller._choose_worker()
            worker_ids.append(worker.id)

        assert worker_ids[0] == "rollout/0"
        assert worker_ids[1] == "rollout/1"
        assert worker_ids[2] == "rollout/2"
        assert worker_ids[3] == "rollout/0"
        assert worker_ids[4] == "rollout/1"
        assert worker_ids[5] == "rollout/2"

        controller.destroy()


class TestRolloutControllerSubmitAndWait:
    def test_submit_adds_to_pending_queue(self):
        config = InferenceEngineConfig(consumer_batch_size=16)
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        alloc_mode = AllocationMode.from_str("sglang.d1")
        controller.initialize(alloc_mode=alloc_mode)

        data = {"test": "data"}
        controller.submit(
            data, workflow_path="areal.tests.utils.TestWorkflow", workflow_kwargs={}
        )

        assert len(controller._pending_inputs) == 1
        assert controller._pending_inputs[0].data == data

        controller.destroy()

    def test_submit_multiple_requests(self):
        config = InferenceEngineConfig(consumer_batch_size=16)
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        alloc_mode = AllocationMode.from_str("sglang.d1")
        controller.initialize(alloc_mode=alloc_mode)

        for i in range(5):
            controller.submit(
                {"id": i},
                workflow_path="areal.tests.utils.TestWorkflow",
                workflow_kwargs={},
            )

        assert len(controller._pending_inputs) == 5

        controller.destroy()

    def test_wait_returns_distributed_batch(self):
        config = InferenceEngineConfig(
            consumer_batch_size=16, max_concurrent_rollouts=50
        )
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        alloc_mode = AllocationMode.from_str("sglang.d1")
        controller.initialize(alloc_mode=alloc_mode)

        for i in range(3):
            controller.submit(
                {"id": i},
                workflow_path="areal.tests.utils.TestWorkflow",
                workflow_kwargs={},
            )

        batch = controller.wait(count=3, timeout=5.0)

        assert isinstance(batch, DistributedBatchMemory)
        assert len(batch) == 3

        controller.destroy()

    def test_wait_timeout_when_insufficient_results(self):
        config = InferenceEngineConfig(
            consumer_batch_size=16, max_concurrent_rollouts=10
        )
        scheduler = MockScheduler()

        async def slow_workflow(*args, **kwargs):
            await asyncio.sleep(10.0)
            return None

        scheduler.async_call_engine = slow_workflow

        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        alloc_mode = AllocationMode.from_str("sglang.d1")
        controller.initialize(alloc_mode=alloc_mode)

        controller.submit(
            {"id": 0},
            workflow_path="areal.tests.utils.TestWorkflow",
            workflow_kwargs={},
        )

        with pytest.raises(TimeoutError, match="Timed out waiting for"):
            controller.wait(count=1, timeout=0.2)

        controller.destroy()

    def test_wait_handles_rejected_rollouts(self):
        config = InferenceEngineConfig(
            consumer_batch_size=16, max_concurrent_rollouts=20
        )
        scheduler = MockScheduler()

        call_count = 0

        async def mixed_results(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)
            if call_count % 2 == 0:
                return None
            return {
                "input_ids": torch.randint(0, 100, (1, 10)),
                "attention_mask": torch.ones(1, 10, dtype=torch.bool),
                "loss_mask": torch.ones(1, 10, dtype=torch.bool),
                "rewards": torch.randn(1),
            }

        scheduler.async_call_engine = mixed_results

        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        alloc_mode = AllocationMode.from_str("sglang.d1")
        controller.initialize(alloc_mode=alloc_mode)

        for i in range(6):
            controller.submit(
                {"id": i},
                workflow_path="areal.tests.utils.TestWorkflow",
                workflow_kwargs={},
            )

        batch = controller.wait(count=3, timeout=2.0)
        assert len(batch) == 3

        controller.destroy()


class TestRolloutControllerBatchOperations:
    def test_rollout_batch_submits_all_data(self):
        config = InferenceEngineConfig(
            consumer_batch_size=16, max_concurrent_rollouts=50
        )
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        alloc_mode = AllocationMode.from_str("sglang.d1")
        controller.initialize(alloc_mode=alloc_mode)

        batch_data = [{"id": i, "value": f"item_{i}"} for i in range(4)]
        batch = controller.rollout_batch(
            batch_data,
            workflow_path="areal.tests.utils.TestWorkflow",
            workflow_kwargs={},
        )

        assert isinstance(batch, DistributedBatchMemory)
        assert len(batch) == 4

        controller.destroy()

    def test_rollout_batch_waits_for_all_results(self):
        config = InferenceEngineConfig(
            consumer_batch_size=16, max_concurrent_rollouts=100
        )
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        alloc_mode = AllocationMode.from_str("sglang.d2")
        controller.initialize(alloc_mode=alloc_mode)

        batch_data = [{"id": i} for i in range(10)]
        batch = controller.rollout_batch(
            batch_data,
            workflow_path="areal.tests.utils.TestWorkflow",
            workflow_kwargs={},
        )

        assert len(batch) == 10

        controller.destroy()


class TestRolloutControllerVersionManagement:
    def test_get_version_initial(self):
        config = InferenceEngineConfig(consumer_batch_size=16)
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        assert controller.get_version() == 0

    def test_set_version_updates_controller_version(self):
        config = InferenceEngineConfig(consumer_batch_size=16)
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        alloc_mode = AllocationMode.from_str("sglang.d2")
        controller.initialize(alloc_mode=alloc_mode)

        controller.set_version(42)
        assert controller.get_version() == 42

        controller.destroy()

    def test_set_version_calls_workers(self):
        config = InferenceEngineConfig(consumer_batch_size=16)
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        alloc_mode = AllocationMode.from_str("sglang.d2")
        controller.initialize(alloc_mode=alloc_mode)

        controller.set_version(10)

        version_calls = [
            call for call in scheduler.engine_calls if call[1] == "set_version"
        ]
        assert len(version_calls) == 2

        controller.destroy()

    def test_set_version_handles_worker_error(self):
        config = InferenceEngineConfig(consumer_batch_size=16)
        scheduler = MockScheduler()

        def failing_call(*args, **kwargs):
            raise Exception("Worker error")

        scheduler.call_engine = failing_call

        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        alloc_mode = AllocationMode.from_str("sglang.d1")
        controller.initialize(alloc_mode=alloc_mode)

        controller.set_version(5)


class TestRolloutControllerWeightUpdates:
    def test_init_weights_update_group_returns_future(self):
        config = InferenceEngineConfig(consumer_batch_size=16)
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        alloc_mode = AllocationMode.from_str("sglang.d1")
        controller.initialize(alloc_mode=alloc_mode)

        meta = WeightUpdateMeta(type="disk", path="/tmp/test")
        future = controller.init_weights_update_group(meta)

        assert isinstance(future, Future)
        future.result(timeout=5.0)

        controller.destroy()

    def test_update_weights_from_distributed_returns_future(self):
        config = InferenceEngineConfig(consumer_batch_size=16)
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        alloc_mode = AllocationMode.from_str("sglang.d1")
        controller.initialize(alloc_mode=alloc_mode)

        meta = WeightUpdateMeta(type="disk", path="/tmp/test")
        param_specs = [ParamSpec(name="test", shape=(10, 10), dtype="float32")]
        future = controller.update_weights_from_distributed(meta, param_specs)

        assert isinstance(future, Future)
        future.result(timeout=5.0)

        controller.destroy()

    def test_update_weights_from_disk_returns_future(self):
        config = InferenceEngineConfig(consumer_batch_size=16)
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        alloc_mode = AllocationMode.from_str("sglang.d1")
        controller.initialize(alloc_mode=alloc_mode)

        meta = WeightUpdateMeta(type="disk", path="/tmp/test")
        future = controller.update_weights_from_disk(meta)

        assert isinstance(future, Future)

        controller.destroy()


class TestRolloutControllerLifecycle:
    def test_pause_calls_all_workers(self):
        config = InferenceEngineConfig(consumer_batch_size=16)
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        alloc_mode = AllocationMode.from_str("sglang.d3")
        controller.initialize(alloc_mode=alloc_mode)

        controller.pause()

        pause_calls = [call for call in scheduler.engine_calls if call[1] == "pause"]
        assert len(pause_calls) == 3

        controller.destroy()

    def test_resume_calls_all_workers(self):
        config = InferenceEngineConfig(consumer_batch_size=16)
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        alloc_mode = AllocationMode.from_str("sglang.d3")
        controller.initialize(alloc_mode=alloc_mode)

        controller.resume()

        resume_calls = [call for call in scheduler.engine_calls if call[1] == "resume"]
        assert len(resume_calls) == 3

        controller.destroy()

    def test_pause_handles_worker_error(self):
        config = InferenceEngineConfig(consumer_batch_size=16)
        scheduler = MockScheduler()

        def failing_call(*args, **kwargs):
            raise Exception("Worker error")

        scheduler.call_engine = failing_call

        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        alloc_mode = AllocationMode.from_str("sglang.d1")
        controller.initialize(alloc_mode=alloc_mode)

        controller.pause()

    def test_resume_handles_worker_error(self):
        config = InferenceEngineConfig(consumer_batch_size=16)
        scheduler = MockScheduler()

        def failing_call(*args, **kwargs):
            raise Exception("Worker error")

        scheduler.call_engine = failing_call

        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        alloc_mode = AllocationMode.from_str("sglang.d1")
        controller.initialize(alloc_mode=alloc_mode)

        controller.resume()


class TestRolloutControllerAgenerate:
    def test_agenerate_chooses_worker(self):
        config = InferenceEngineConfig(consumer_batch_size=16)
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        alloc_mode = AllocationMode.from_str("sglang.d2")
        controller.initialize(alloc_mode=alloc_mode)

        req = ModelRequest(input_ids=[1, 2, 3, 4, 5])

        async def test_agenerate():
            result = await controller.agenerate(req)
            return result

        asyncio.run(test_agenerate())

        agenerate_calls = [
            call for call in scheduler.engine_calls if call[1] == "agenerate"
        ]
        assert len(agenerate_calls) == 1
        assert agenerate_calls[0][3]["req"] == req

        controller.destroy()

    def test_agenerate_round_robin(self):
        config = InferenceEngineConfig(consumer_batch_size=16)
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        alloc_mode = AllocationMode.from_str("sglang.d3")
        controller.initialize(alloc_mode=alloc_mode)

        async def test_multiple_agenerate():
            for _ in range(6):
                req = ModelRequest(input_ids=[1, 2, 3])
                await controller.agenerate(req)

        asyncio.run(test_multiple_agenerate())

        agenerate_calls = [
            call for call in scheduler.engine_calls if call[1] == "agenerate"
        ]
        worker_ids = [call[0] for call in agenerate_calls]

        assert worker_ids[0] == "rollout/0"
        assert worker_ids[1] == "rollout/1"
        assert worker_ids[2] == "rollout/2"
        assert worker_ids[3] == "rollout/0"

        controller.destroy()


class TestRolloutControllerErrorHandling:
    def test_commit_raises_on_queue_full(self):
        config = InferenceEngineConfig(consumer_batch_size=16)
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        alloc_mode = AllocationMode.from_str("sglang.d1")
        controller.initialize(alloc_mode=alloc_mode)

        with patch.object(
            controller.runner, "submit", side_effect=TaskQueueFullError("Queue full")
        ):
            controller.submit(
                {"id": 0},
                workflow_path="areal.tests.utils.TestWorkflow",
                workflow_kwargs={},
            )

            with pytest.raises(queue.Full):
                controller._commit_one_to_runner()

        controller.destroy()

    def test_wait_returns_empty_batch_on_no_results(self):
        config = InferenceEngineConfig(
            consumer_batch_size=16, max_concurrent_rollouts=50
        )
        scheduler = MockScheduler()

        async def reject_all(*args, **kwargs):
            await asyncio.sleep(0.01)
            return None

        scheduler.async_call_engine = reject_all

        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        alloc_mode = AllocationMode.from_str("sglang.d1")
        controller.initialize(alloc_mode=alloc_mode)

        with pytest.raises(TimeoutError):
            controller.wait(count=1, timeout=0.5)

        controller.destroy()


class TestRolloutControllerIntegration:
    def test_end_to_end_workflow(self):
        config = InferenceEngineConfig(
            consumer_batch_size=8,
            max_concurrent_rollouts=20,
            max_head_offpolicyness=2,
        )
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        alloc_mode = AllocationMode.from_str("sglang.d2")
        controller.initialize(alloc_mode=alloc_mode)

        capacity = controller.get_capacity()
        assert capacity == 20

        for i in range(5):
            controller.submit(
                {"id": i},
                workflow_path="areal.tests.utils.TestWorkflow",
                workflow_kwargs={},
            )

        batch = controller.wait(count=5, timeout=5.0)
        assert len(batch) == 5

        controller.set_version(1)
        assert controller.get_version() == 1

        controller.destroy()

    def test_multiple_batch_cycles(self):
        config = InferenceEngineConfig(
            consumer_batch_size=4,
            max_concurrent_rollouts=50,
            max_head_offpolicyness=5,
        )
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        alloc_mode = AllocationMode.from_str("sglang.d1")
        controller.initialize(alloc_mode=alloc_mode)

        for cycle in range(3):
            batch_data = [{"id": i, "cycle": cycle} for i in range(4)]
            batch = controller.rollout_batch(
                batch_data,
                workflow_path="areal.tests.utils.TestWorkflow",
                workflow_kwargs={},
            )
            assert len(batch) == 4

        controller.destroy()


class TestRolloutControllerNotImplemented:
    def test_register_callback_not_implemented(self):
        config = InferenceEngineConfig(consumer_batch_size=16)
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        with pytest.raises(NotImplementedError):
            controller.register_callback_to_all_worker("test", lambda: None)

    def test_abort_all_requests_not_implemented(self):
        config = InferenceEngineConfig(consumer_batch_size=16)
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        with pytest.raises(NotImplementedError):
            controller.abort_all_requests()


@pytest.mark.parametrize("num_workers", [1, 2, 4])
def test_parametrized_worker_count(num_workers):
    config = InferenceEngineConfig(consumer_batch_size=16)
    scheduler = MockScheduler()
    controller = RolloutController(
        inf_engine=MockInferenceEngine,
        config=config,
        scheduler=scheduler,
    )

    alloc_mode = AllocationMode.from_str(f"sglang.d{num_workers}")
    controller.initialize(alloc_mode=alloc_mode)

    assert len(controller.workers) == num_workers

    controller.destroy()


@pytest.mark.parametrize(
    "consumer_batch_size,max_concurrent_rollouts,expected_capacity",
    [(16, 32, 32), (32, 64, 64), (8, 100, 24)],
)
def test_parametrized_capacity_settings(
    consumer_batch_size, max_concurrent_rollouts, expected_capacity
):
    config = InferenceEngineConfig(
        consumer_batch_size=consumer_batch_size,
        max_concurrent_rollouts=max_concurrent_rollouts,
        max_head_offpolicyness=2,
    )
    scheduler = MockScheduler()
    controller = RolloutController(
        inf_engine=MockInferenceEngine,
        config=config,
        scheduler=scheduler,
    )

    alloc_mode = AllocationMode.from_str("sglang.d1")
    controller.initialize(alloc_mode=alloc_mode)

    capacity = controller.get_capacity()
    assert capacity == expected_capacity

    controller.destroy()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
