"""Worker pool manager for tracking and dispatching to Agent Workers.

This module provides the WorkerPoolManager class, which maintains a registry
of Agent Worker processes, tracks their availability, and dispatches requests
using round-robin scheduling among idle workers.

Thread Safety
-------------
All state mutations are protected by an asyncio.Lock, making concurrent
access from multiple coroutines safe.

Health Detection
----------------
Only passive health detection is supported. Workers are marked unhealthy
explicitly by the caller (e.g., on HTTP timeout). No active polling is
performed.
"""

from __future__ import annotations

import asyncio

from areal.utils import logging

from .schemas import WorkerInfo

logger = logging.getLogger("WorkerPoolManager")


class WorkerPoolManager:
    """Manages a pool of Agent Workers with round-robin dispatch.

    Attributes:
        _workers: Mapping from worker_id to WorkerInfo.
        _round_robin_idx: Current index for round-robin scheduling.
        _worker_timeout: Timeout threshold in seconds (informational).
        _lock: asyncio.Lock guarding all state mutations.
    """

    def __init__(self, worker_timeout: float = 300.0) -> None:
        """Initialize pool state.

        Args:
            worker_timeout: Timeout in seconds for worker operations.
        """
        self._workers: dict[str, WorkerInfo] = {}
        self._round_robin_idx: int = 0
        self._worker_timeout: float = worker_timeout
        self._lock = asyncio.Lock()

    async def register_worker(self, worker_id: str, address: str) -> None:
        """Add a worker to the pool with status 'idle'.

        If a worker with the same ID already exists, it is replaced.

        Args:
            worker_id: Unique identifier for the worker.
            address: HTTP address of the worker (e.g., 'http://host:port').
        """
        async with self._lock:
            self._workers[worker_id] = WorkerInfo.create(
                worker_id=worker_id, address=address
            )
            logger.info("Registered worker %s at %s", worker_id, address)

    async def unregister_worker(self, worker_id: str) -> None:
        """Remove a worker from the pool.

        No-op if the worker is not registered.

        Args:
            worker_id: Unique identifier for the worker to remove.
        """
        async with self._lock:
            if worker_id in self._workers:
                del self._workers[worker_id]
                logger.info("Unregistered worker %s", worker_id)
            else:
                logger.warning("Attempted to unregister unknown worker %s", worker_id)

    async def get_available_worker(self) -> WorkerInfo | None:
        """Get an idle worker using round-robin scheduling.

        Cycles through all registered workers starting from the current
        round-robin index and returns the first idle worker found.

        Returns:
            A WorkerInfo for an idle worker, or None if no idle workers exist.
        """
        async with self._lock:
            worker_ids = list(self._workers.keys())
            n = len(worker_ids)
            if n == 0:
                return None

            # Scan all workers starting from round-robin index
            for i in range(n):
                idx = (self._round_robin_idx + i) % n
                worker = self._workers[worker_ids[idx]]
                if worker.status == "idle":
                    self._round_robin_idx = (idx + 1) % n
                    return worker

            return None

    async def mark_busy(self, worker_id: str) -> None:
        """Set a worker's status to 'busy'.

        Args:
            worker_id: Unique identifier for the worker.
        """
        async with self._lock:
            if worker_id in self._workers:
                self._workers[worker_id].status = "busy"
            else:
                logger.warning("Cannot mark unknown worker %s as busy", worker_id)

    async def mark_idle(self, worker_id: str) -> None:
        """Set a worker's status to 'idle'.

        Args:
            worker_id: Unique identifier for the worker.
        """
        async with self._lock:
            if worker_id in self._workers:
                self._workers[worker_id].status = "idle"
            else:
                logger.warning("Cannot mark unknown worker %s as idle", worker_id)

    async def mark_unhealthy(self, worker_id: str) -> None:
        """Set a worker's status to 'unhealthy'.

        Args:
            worker_id: Unique identifier for the worker.
        """
        async with self._lock:
            if worker_id in self._workers:
                self._workers[worker_id].status = "unhealthy"
                logger.warning("Worker %s marked unhealthy", worker_id)
            else:
                logger.warning("Cannot mark unknown worker %s as unhealthy", worker_id)

    async def get_all_workers(self) -> list[WorkerInfo]:
        """Return a list of all registered workers.

        Returns:
            List of WorkerInfo for every registered worker.
        """
        async with self._lock:
            return list(self._workers.values())

    async def get_stats(self) -> dict:
        """Return worker pool statistics.

        Returns:
            Dict with keys: total, idle, busy, unhealthy.
        """
        async with self._lock:
            workers = list(self._workers.values())
            return {
                "total": len(workers),
                "idle": sum(1 for w in workers if w.status == "idle"),
                "busy": sum(1 for w in workers if w.status == "busy"),
                "unhealthy": sum(1 for w in workers if w.status == "unhealthy"),
            }
