"""Unit tests for WorkerPoolManager."""

from __future__ import annotations

import pytest

from areal.experimental.agent_service.worker_pool import WorkerPoolManager


class TestWorkerRegistration:
    """Tests for worker registration and unregistration."""

    @pytest.mark.asyncio
    async def test_register_worker_adds_idle_worker(self):
        """register_worker() should add a worker with status 'idle'."""
        pool = WorkerPoolManager()
        await pool.register_worker("w1", "http://localhost:9001")

        workers = await pool.get_all_workers()
        assert len(workers) == 1
        assert workers[0].worker_id == "w1"
        assert workers[0].address == "http://localhost:9001"
        assert workers[0].status == "idle"

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


class TestWorkerDispatch:
    """Tests for round-robin dispatch and availability."""

    @pytest.mark.asyncio
    async def test_get_available_worker_returns_idle_worker(self):
        """get_available_worker() should return an idle worker."""
        pool = WorkerPoolManager()
        await pool.register_worker("w1", "http://localhost:9001")

        worker = await pool.get_available_worker()
        assert worker is not None
        assert worker.worker_id == "w1"
        assert worker.status == "idle"

    @pytest.mark.asyncio
    async def test_get_available_worker_returns_none_when_empty(self):
        """get_available_worker() should return None when no workers registered."""
        pool = WorkerPoolManager()
        worker = await pool.get_available_worker()
        assert worker is None

    @pytest.mark.asyncio
    async def test_get_available_worker_returns_none_when_all_busy(self):
        """get_available_worker() should return None when all workers are busy."""
        pool = WorkerPoolManager()
        await pool.register_worker("w1", "http://localhost:9001")
        await pool.mark_busy("w1")

        worker = await pool.get_available_worker()
        assert worker is None

    @pytest.mark.asyncio
    async def test_round_robin_cycles_through_workers(self):
        """get_available_worker() should cycle through workers in round-robin order."""
        pool = WorkerPoolManager()
        await pool.register_worker("w1", "http://localhost:9001")
        await pool.register_worker("w2", "http://localhost:9002")
        await pool.register_worker("w3", "http://localhost:9003")

        # Collect workers in round-robin order
        seen = []
        for _ in range(3):
            worker = await pool.get_available_worker()
            assert worker is not None
            seen.append(worker.worker_id)

        # All 3 workers should have been returned (round-robin)
        assert set(seen) == {"w1", "w2", "w3"}

    @pytest.mark.asyncio
    async def test_get_available_worker_skips_unhealthy(self):
        """get_available_worker() should skip unhealthy workers."""
        pool = WorkerPoolManager()
        await pool.register_worker("w1", "http://localhost:9001")
        await pool.register_worker("w2", "http://localhost:9002")
        await pool.mark_unhealthy("w1")

        worker = await pool.get_available_worker()
        assert worker is not None
        assert worker.worker_id == "w2"


class TestWorkerStateTransitions:
    """Tests for mark_busy, mark_idle, mark_unhealthy state transitions."""

    @pytest.mark.asyncio
    async def test_mark_busy_changes_status(self):
        """mark_busy() should change worker status to 'busy'."""
        pool = WorkerPoolManager()
        await pool.register_worker("w1", "http://localhost:9001")
        await pool.mark_busy("w1")

        workers = await pool.get_all_workers()
        assert workers[0].status == "busy"

    @pytest.mark.asyncio
    async def test_mark_idle_changes_status(self):
        """mark_idle() should change worker status back to 'idle'."""
        pool = WorkerPoolManager()
        await pool.register_worker("w1", "http://localhost:9001")
        await pool.mark_busy("w1")
        await pool.mark_idle("w1")

        workers = await pool.get_all_workers()
        assert workers[0].status == "idle"

    @pytest.mark.asyncio
    async def test_mark_unhealthy_changes_status(self):
        """mark_unhealthy() should change worker status to 'unhealthy'."""
        pool = WorkerPoolManager()
        await pool.register_worker("w1", "http://localhost:9001")
        await pool.mark_unhealthy("w1")

        workers = await pool.get_all_workers()
        assert workers[0].status == "unhealthy"

    @pytest.mark.asyncio
    async def test_state_transitions_idle_busy_idle(self):
        """Full idle -> busy -> idle cycle should work correctly."""
        pool = WorkerPoolManager()
        await pool.register_worker("w1", "http://localhost:9001")

        # idle -> busy
        await pool.mark_busy("w1")
        stats = await pool.get_stats()
        assert stats["busy"] == 1
        assert stats["idle"] == 0

        # busy -> idle
        await pool.mark_idle("w1")
        stats = await pool.get_stats()
        assert stats["busy"] == 0
        assert stats["idle"] == 1

    @pytest.mark.asyncio
    async def test_mark_busy_unknown_worker_is_noop(self):
        """mark_busy() with unknown ID should not raise."""
        pool = WorkerPoolManager()
        await pool.mark_busy("nonexistent")  # Should not raise


class TestWorkerStats:
    """Tests for get_stats() accuracy."""

    @pytest.mark.asyncio
    async def test_get_stats_empty_pool(self):
        """get_stats() should return zeros for empty pool."""
        pool = WorkerPoolManager()
        stats = await pool.get_stats()
        assert stats == {"total": 0, "idle": 0, "busy": 0, "unhealthy": 0}

    @pytest.mark.asyncio
    async def test_get_stats_mixed_states(self):
        """get_stats() should accurately count workers in each state."""
        pool = WorkerPoolManager()
        await pool.register_worker("w1", "http://localhost:9001")
        await pool.register_worker("w2", "http://localhost:9002")
        await pool.register_worker("w3", "http://localhost:9003")
        await pool.register_worker("w4", "http://localhost:9004")

        await pool.mark_busy("w2")
        await pool.mark_unhealthy("w3")
        # w1 and w4 remain idle

        stats = await pool.get_stats()
        assert stats["total"] == 4
        assert stats["idle"] == 2
        assert stats["busy"] == 1
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
        assert stats["idle"] == 1
