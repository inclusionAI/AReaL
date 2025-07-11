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
    workers = sched.create_workers({"num_workers": 2})
    sched.wait_workers()
    return sched, workers


def test_create_engine_and_infer(scheduler_and_worker):
    sched, workers = scheduler_and_worker
    worker_id, ip, port = workers[0]
    engine_obj = MyEngine({"value": 24})
    # 测试 create_engine
    assert sched.create_engine(worker_id, engine_obj, {"init": 1})
    # 测试 infer 方法
    result = sched.call(worker_id, "infer", 100, 10)
    assert result == 100 * 10 + 24


def test_multiple_workers(scheduler_and_worker):
    sched, workers = scheduler_and_worker
    for idx, (worker_id, ip, port) in enumerate(workers):
        engine_obj = MyEngine({"value": idx})
        assert sched.create_engine(worker_id, engine_obj, {"init": 1})
        result = sched.call(worker_id, "infer", 2, 3)
        assert result == 2 * 3 + idx
