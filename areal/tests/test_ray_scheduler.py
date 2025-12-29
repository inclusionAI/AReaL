import asyncio
from unittest.mock import Mock, patch

import pytest
import ray
from ray.util.state import summarize_actors

from areal.api.cli_args import BaseExperimentConfig
from areal.api.scheduler_api import Job, SchedulingSpec, Worker
from areal.scheduler.ray import RayScheduler, RayWorkerInfo, ray_resource_type

pytestmark = pytest.mark.skipif(
    not ray.is_initialized(),
    reason="Ray scheduler tests will only run if ray is explicitly initialized.",
)


class TestRaySchedulerInitialization:
    def test_init(self):
        scheduler = RayScheduler(
            startup_timeout=60.0, exp_config=BaseExperimentConfig()
        )
        assert scheduler.startup_timeout == 60.0


class TestWorkerCreationAndDeletion:
    def test_create_delete_workers(self):
        ray.init()

        config = BaseExperimentConfig()

        scheduler = RayScheduler(startup_timeout=60.0, exp_config=config)

        job = Job(
            replicas=2,
            role="train",
            tasks=[
                SchedulingSpec(
                    cpu=1,
                    mem=1024,
                    gpu=1,
                ),
                SchedulingSpec(
                    cpu=1,
                    mem=1024,
                    gpu=1,
                ),
            ],
        )

        # create workers
        worker_ids = scheduler.create_workers(job)
        assert len(worker_ids) == 2
        assert len(scheduler._workers["train"]) == 2

        actor_summary = summarize_actors()

        assert (
            actor_summary["cluster"]["summary"]["RayRPCServer"]["state_counts"]["ALIVE"]
            == 2
        )
        assert len(scheduler.get_workers("train")) == 2

        scheduler._ping_workers("train")

        # delete workers
        scheduler.delete_workers()
        assert len(scheduler._workers["train"]) == 0

        actor_summary = summarize_actors()
        assert (
            actor_summary["cluster"]["summary"]["RayRPCServer"]["state_counts"]["DEAD"]
            == 2
        )


class TestWorkerCallEngine:
    def test_create_call_engine(self):
        # to simulate an awaitable None
        async def async_none(*args, **kwargs):
            return None

        config = BaseExperimentConfig()

        scheduler = RayScheduler(startup_timeout=60.0, exp_config=config)
        ray_actor_handle = Mock()
        ray_actor_handle.create_engine.remote = async_none

        worker = RayWorkerInfo(
            worker=Worker(id="test/0", ip="0.0.0.0"),
            actor=ray_actor_handle,
            role="test",
            placement_group=None,
            bundle_index=0,
            created_at=0,
        )

        scheduler._workers["test"] = [worker]
        scheduler._worker_info_by_id[worker.worker.id] = worker

        # create engine
        result = asyncio.run(
            scheduler.create_engine(
                worker.worker.id, "test_engines.DummyEngine", name="TestEngine"
            )
        )
        assert result is None

        # sync
        ray_actor_handle.call.remote = lambda x: None
        with patch("areal.scheduler.ray.ray.get", return_value=None):
            result = scheduler.call_engine(worker.worker.id, "test_fn")
        assert result is None

        # async
        ray_actor_handle.call.remote = async_none
        result = asyncio.run(scheduler.async_call_engine(worker.worker.id, "test_fn"))
        assert result is None


class TestUtilityFunctions:
    def test_utilities(self):
        _num_gpu_per_node = 16
        config = BaseExperimentConfig()

        config.cluster.n_gpus_per_node = _num_gpu_per_node

        scheduler = RayScheduler(startup_timeout=60.0, exp_config=config)

        schedulings = [
            SchedulingSpec(
                cpu=1,
                mem=1024,
                gpu=1,
            ),
            SchedulingSpec(
                cpu=1,
                mem=1024,
                gpu=1,
            ),
        ]

        new_schedulings = scheduler._prepare_worker_specs("train", 2, schedulings)
        assert len(new_schedulings) == 2
        for spec in new_schedulings:
            assert spec.cpu == 1
            assert spec.mem == 1024
            assert spec.gpu == 1

        # case where only 1 spec is passed but multiple workers
        new_schedulings = scheduler._prepare_worker_specs("train", 2, schedulings[0:])
        assert len(new_schedulings) == 2
        for spec in new_schedulings:
            assert spec.cpu == 1
            assert spec.mem == 1024
            assert spec.gpu == 1

        bundle_list = scheduler._create_bundle_list_gpu(1, 24, 1024)
        assert len(bundle_list) == 2
        for bundle in bundle_list:
            assert bundle[ray_resource_type()] <= _num_gpu_per_node
