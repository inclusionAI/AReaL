"""Unit tests for WorkerPoolManager."""

from __future__ import annotations

import pytest

from areal.experimental.agent_service.worker_pool import WorkerPoolManager


class TestWorkerRegistration:
    """Tests for worker registration and unregistration."""

    @pytest.mark.asyncio
    async def test_register_worker_adds_healthy_worker(self):
        """register_worker() should add a worker with status 'healthy'."""
        pool = WorkerPoolManager()
        await pool.register_worker("w1", "http://localhost:9001")

        workers = await pool.get_all_workers()
        assert len(workers) == 1
        assert workers[0].worker_id == "w1"
        assert workers[0].address == "http://localhost:9001"
        assert workers[0].status == "healthy"

    @pytest.mark.asyncio
    async def test_register_multiple_workers(self):
        """register_worker() should support multiple workers."""
        pool = WorkerPoolManager()
        for i in range(3):
            await pool.register_worker(f"w{i}", f"http://localhost:900{i}")

        workers = await pool.get_all_workers()
        assert len(workers) == 3

    @pytest.mark.asyncio
    async def test_register_same_id_replaces_worker(self):
        """Registering with same ID should replace the existing worker."""
        pool = WorkerPoolManager()
        await pool.register_worker("w1", "http://localhost:9001")
        await pool.register_worker("w1", "http://localhost:9999")  # new address

        workers = await pool.get_all_workers()
        assert len(workers) == 1
        assert workers[0].address == "http://localhost:9999"

    @pytest.mark.asyncio
    async def test_unregister_worker_removes_it(self):
        """unregister_worker() should remove the worker from the pool."""
        pool = WorkerPoolManager()
        await pool.register_worker("w1", "http://localhost:9001")
        await pool.unregister_worker("w1")

        workers = await pool.get_all_workers()
        assert len(workers) == 0

    @pytest.mark.asyncio
    async def test_unregister_unknown_worker_is_noop(self):
        """unregister_worker() with unknown ID should not raise."""
        pool = WorkerPoolManager()
        # Should not raise
        await pool.unregister_worker("nonexistent")

        workers = await pool.get_all_workers()
        assert len(workers) == 0


class TestWorkerSelection:
    """Tests for round-robin selection and availability."""

    @pytest.mark.asyncio
    async def test_next_worker_returns_healthy_worker(self):
        """next_worker() should return a healthy worker."""
        pool = WorkerPoolManager()
        await pool.register_worker("w1", "http://localhost:9001")

        worker = await pool.next_worker()
        assert worker is not None
        assert worker.worker_id == "w1"
        assert worker.status == "healthy"

    @pytest.mark.asyncio
    async def test_next_worker_returns_none_when_empty(self):
        """next_worker() should return None when no workers registered."""
        pool = WorkerPoolManager()
        worker = await pool.next_worker()
        assert worker is None

    @pytest.mark.asyncio
    async def test_next_worker_returns_none_when_all_unhealthy(self):
        """next_worker() should return None when all workers are unhealthy."""
        pool = WorkerPoolManager()
        await pool.register_worker("w1", "http://localhost:9001")
        await pool.mark_unhealthy("w1")

        worker = await pool.next_worker()
        assert worker is None

    @pytest.mark.asyncio
    async def test_round_robin_cycles_through_workers(self):
        """next_worker() should cycle through workers in round-robin order."""
        pool = WorkerPoolManager()
        await pool.register_worker("w1", "http://localhost:9001")
        await pool.register_worker("w2", "http://localhost:9002")
        await pool.register_worker("w3", "http://localhost:9003")

        # Collect workers in round-robin order
        seen = []
        for _ in range(3):
            worker = await pool.next_worker()
            assert worker is not None
            seen.append(worker.worker_id)

        # All 3 workers should have been returned (round-robin)
        assert set(seen) == {"w1", "w2", "w3"}

    @pytest.mark.asyncio
    async def test_next_worker_skips_unhealthy(self):
        """next_worker() should skip unhealthy workers."""
        pool = WorkerPoolManager()
        await pool.register_worker("w1", "http://localhost:9001")
        await pool.register_worker("w2", "http://localhost:9002")
        await pool.mark_unhealthy("w1")

        worker = await pool.next_worker()
        assert worker is not None
        assert worker.worker_id == "w2"


class TestWorkerHealthTransitions:
    """Tests for mark_unhealthy and mark_healthy state transitions."""

    @pytest.mark.asyncio
    async def test_mark_unhealthy_changes_status(self):
        """mark_unhealthy() should change worker status to 'unhealthy'."""
        pool = WorkerPoolManager()
        await pool.register_worker("w1", "http://localhost:9001")
        await pool.mark_unhealthy("w1")

        workers = await pool.get_all_workers()
        assert workers[0].status == "unhealthy"

    @pytest.mark.asyncio
    async def test_mark_healthy_changes_status(self):
        """mark_healthy() should change worker status back to 'healthy'."""
        pool = WorkerPoolManager()
        await pool.register_worker("w1", "http://localhost:9001")
        await pool.mark_unhealthy("w1")
        await pool.mark_healthy("w1")

        workers = await pool.get_all_workers()
        assert workers[0].status == "healthy"

    @pytest.mark.asyncio
    async def test_state_transitions_healthy_unhealthy_healthy(self):
        """Full healthy -> unhealthy -> healthy cycle should work correctly."""
        pool = WorkerPoolManager()
        await pool.register_worker("w1", "http://localhost:9001")

        # healthy -> unhealthy
        await pool.mark_unhealthy("w1")
        stats = await pool.get_stats()
        assert stats["unhealthy"] == 1
        assert stats["healthy"] == 0

        # unhealthy -> healthy
        await pool.mark_healthy("w1")
        stats = await pool.get_stats()
        assert stats["unhealthy"] == 0
        assert stats["healthy"] == 1

    @pytest.mark.asyncio
    async def test_mark_unhealthy_unknown_worker_is_noop(self):
        """mark_unhealthy() with unknown ID should not raise."""
        pool = WorkerPoolManager()
        await pool.mark_unhealthy("nonexistent")  # Should not raise

    @pytest.mark.asyncio
    async def test_mark_healthy_unknown_worker_is_noop(self):
        """mark_healthy() with unknown ID should not raise."""
        pool = WorkerPoolManager()
        await pool.mark_healthy("nonexistent")  # Should not raise


class TestWorkerStats:
    """Tests for get_stats() accuracy."""

    @pytest.mark.asyncio
    async def test_get_stats_empty_pool(self):
        """get_stats() should return zeros for empty pool."""
        pool = WorkerPoolManager()
        stats = await pool.get_stats()
        assert stats == {"total": 0, "healthy": 0, "unhealthy": 0}

    @pytest.mark.asyncio
    async def test_get_stats_mixed_states(self):
        """get_stats() should accurately count workers in each state."""
        pool = WorkerPoolManager()
        await pool.register_worker("w1", "http://localhost:9001")
        await pool.register_worker("w2", "http://localhost:9002")
        await pool.register_worker("w3", "http://localhost:9003")
        await pool.register_worker("w4", "http://localhost:9004")

        await pool.mark_unhealthy("w3")
        # w1, w2, w4 remain healthy

        stats = await pool.get_stats()
        assert stats["total"] == 4
        assert stats["healthy"] == 3
        assert stats["unhealthy"] == 1

    @pytest.mark.asyncio
    async def test_get_stats_updates_after_unregister(self):
        """get_stats() should reflect changes after unregistering a worker."""
        pool = WorkerPoolManager()
        await pool.register_worker("w1", "http://localhost:9001")
        await pool.register_worker("w2", "http://localhost:9002")
        await pool.unregister_worker("w1")

        stats = await pool.get_stats()
        assert stats["total"] == 1
        assert stats["healthy"] == 1


class TestWorkerPoolLifecycle:
    """Tests for start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_start_creates_health_check_task(self):
        """start() should create the background health check task."""
        pool = WorkerPoolManager()
        await pool.start()
        try:
            assert pool._health_check_task is not None
            assert not pool._health_check_task.done()
        finally:
            await pool.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_health_check_task(self):
        """stop() should cancel the health check task."""
        pool = WorkerPoolManager()
        await pool.start()
        await pool.stop()

        assert pool._health_check_task is None

    @pytest.mark.asyncio
    async def test_stop_without_start_is_noop(self):
        """stop() without start() should not raise."""
        pool = WorkerPoolManager()
        await pool.stop()  # Should not raise
