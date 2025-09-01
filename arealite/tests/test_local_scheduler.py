import logging

import pytest

from arealite.scheduler.local import LocalScheduler
from arealite.scheduler.test.my_engine import MyEngine


@pytest.fixture(scope="module")
def scheduler_and_worker():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    sched = LocalScheduler({"type": "local"})
    sched.create_workers("train", {"num_workers": 1})
    workers = sched.get_workers("train")
    sched.create_workers("infer", {"num_workers": 2})
    workers = sched.get_workers("infer")
    return sched, workers


def test_create_engine_and_infer(scheduler_and_worker):
    sched, workers = scheduler_and_worker
    engine_obj = MyEngine({"value": 24})
    # 测试 create_engine
    assert sched.create_engine(workers[0].id, engine_obj, {"init": 1})
    # 测试 infer 方法
    result = sched.call_engine(workers[0].id, "infer", 100, 10)
    assert result == 100 * 10 + 24


def test_multiple_workers(scheduler_and_worker):
    sched, workers = scheduler_and_worker
    workers = sched.get_workers("infer")
    assert len(workers) == 2

    for idx, worker in enumerate(workers):
        engine_obj = MyEngine({"value": idx})
        assert sched.create_engine(worker.id, engine_obj, {"init": 1})
        result = sched.call_engine(worker.id, "infer", 2, 3)
        assert result == 2 * 3 + idx
