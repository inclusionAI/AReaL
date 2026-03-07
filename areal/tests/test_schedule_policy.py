import asyncio
import random

import pytest

from areal.infra.schedule_policy import (
    LeastRequestPrioritySchedulePolicy,
    RoundRobinSchedulePolicy,
)


class MockWorker:
    def __init__(self, worker_id):
        self.id = worker_id

    def __repr__(self):
        return f"MockWorker(id={self.id})"


@pytest.fixture
def mock_workers():
    return [MockWorker(1), MockWorker(2), MockWorker(3)]


@pytest.fixture
def max_num_seqs():
    return 2


# -------------------------- LeastRequestPrioritySchedulePolicy 测试 --------------------------
@pytest.mark.asyncio
async def test_least_request_init(mock_workers, max_num_seqs):
    """测试最小请求策略初始化"""
    policy = LeastRequestPrioritySchedulePolicy(mock_workers, max_num_seqs)
    assert policy.max_concurrent_per_worker == 1
    assert len(policy.current_process_requests_state) == len(mock_workers)
    for item in policy.current_process_requests_state:
        assert item[0] == 0
        assert item[1][0] in [1, 2, 3]


@pytest.mark.asyncio
async def test_least_request_choose_worker_block(mock_workers):
    """测试worker达到最大并发时的阻塞等待"""
    max_num_seqs = 1
    policy = LeastRequestPrioritySchedulePolicy(mock_workers, max_num_seqs)

    await policy.choose_worker()
    await policy.choose_worker()
    await policy.choose_worker()

    async def wait_for_worker():
        return await policy.choose_worker()

    task = asyncio.create_task(wait_for_worker())

    await asyncio.sleep(1)
    assert task.done() is False

    await policy.release_worker(mock_workers[1])

    await asyncio.sleep(1)
    assert task.done() is True
    assert task.result().id == 2


@pytest.mark.asyncio
async def test_least_request_release_worker_not_found(mock_workers, max_num_seqs):
    """测试释放不存在的worker触发异常"""
    policy = LeastRequestPrioritySchedulePolicy(mock_workers, max_num_seqs)
    fake_worker = MockWorker(999)

    with pytest.raises(RuntimeError) as excinfo:
        await policy.release_worker(fake_worker)
    assert "No workers available to release" in str(excinfo.value)


# -------------------------- RoundRobinSchedulePolicy 测试 --------------------------
@pytest.mark.asyncio
async def test_round_robin_init(mock_workers, max_num_seqs):
    """测试轮询策略初始化"""
    policy = RoundRobinSchedulePolicy(mock_workers, max_num_seqs, no_block=False)
    assert policy.max_concurrent_per_worker == 1
    assert policy.index == 0
    assert policy.no_block is False
    assert len(policy.current_process_requests_state) == 3


@pytest.mark.asyncio
async def test_round_robin_choose_worker_no_block(mock_workers, max_num_seqs):
    """测试非阻塞模式的轮询选择"""
    policy = RoundRobinSchedulePolicy(mock_workers, max_num_seqs, no_block=True)

    worker1 = await policy.choose_worker()
    assert worker1.id == 1
    assert policy.index == 1

    worker2 = await policy.choose_worker()
    assert worker2.id == 2
    assert policy.index == 2

    worker3 = await policy.choose_worker()
    assert worker3.id == 3
    assert policy.index == 0

    worker1_again = await policy.choose_worker()
    assert worker1_again.id == 1
    assert policy.index == 1


@pytest.mark.asyncio
async def test_round_robin_choose_worker_block(mock_workers):
    """测试阻塞模式的轮询选择（达到最大并发时等待）"""
    max_num_seqs = 1
    policy = RoundRobinSchedulePolicy(mock_workers, max_num_seqs, no_block=False)

    await policy.choose_worker()
    assert policy.current_process_requests_state[0][0] == 1
    assert policy.index == 1

    await policy.choose_worker()
    assert policy.current_process_requests_state[1][0] == 1
    assert policy.index == 2

    await policy.choose_worker()
    assert policy.current_process_requests_state[2][0] == 1
    assert policy.index == 0

    async def wait_for_robin_worker():
        return await policy.choose_worker()

    task = asyncio.create_task(wait_for_robin_worker())
    await asyncio.sleep(1)
    assert task.done() is False

    await policy.release_worker(mock_workers[1])
    await asyncio.sleep(1)
    assert task.done() is False

    await policy.release_worker(mock_workers[0])
    await asyncio.sleep(1)
    assert task.done() is True


@pytest.mark.asyncio
async def test_concurrent_for_least_request_choose_worker(mock_workers):
    max_num_seqs = 4
    policy = LeastRequestPrioritySchedulePolicy(mock_workers, max_num_seqs)

    async def do_worker():
        async with policy as worker:
            await asyncio.sleep(0.1 * random.randint(5, 20))
            return worker.id

    task = [asyncio.create_task(do_worker()) for _ in range(20)]
    results = await asyncio.gather(*task)
    assert len(results) == 20
    assert all(value is None for value in policy._chosen_worker_dict.values())
