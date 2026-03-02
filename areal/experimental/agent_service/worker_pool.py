"""Worker pool manager for tracking Agent Workers with round-robin dispatch.

This module provides the WorkerPoolManager class, which maintains a registry
of Agent Worker processes, tracks their health, and selects workers using
round-robin scheduling.

Workers only have two states: **healthy** and **unhealthy**.  Workers are
async FastAPI servers that handle concurrency natively, so idle/busy tracking
is not needed — the Gateway simply forwards requests to healthy workers.

Health Detection
----------------
Both passive and active health detection are supported:
- Passive: Workers are marked unhealthy by the Gateway on HTTP errors.
- Active: A background ``health_check_loop`` periodically polls ``/health``
  on unhealthy workers and restores them to healthy when they respond.
"""

from __future__ import annotations

import asyncio

import aiohttp

from areal.utils import logging

from .schemas import WorkerInfo

logger = logging.getLogger("WorkerPoolManager")


class WorkerPoolManager:
    """Manages a pool of Agent Workers with round-robin dispatch.

    Attributes:
        _workers: Mapping from worker_id to WorkerInfo.
        _round_robin_idx: Current index for round-robin scheduling.
        _worker_timeout: Timeout threshold in seconds for HTTP calls.
        _health_check_interval: Seconds between health check polls.
        _lock: asyncio.Lock guarding all state mutations.
        _health_check_task: Background task for active health checking.
    """

    def __init__(
        self,
        worker_timeout: float = 300.0,
        health_check_interval: float = 30.0,
    ) -> None:
        """Initialize pool state.

        Args:
            worker_timeout: Timeout in seconds for worker operations.
            health_check_interval: Seconds between health check polls.
        """
        self._workers: dict[str, WorkerInfo] = {}
        self._round_robin_idx: int = 0
        self._worker_timeout: float = worker_timeout
        self._health_check_interval: float = health_check_interval
        self._lock = asyncio.Lock()
        self._health_check_task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the background health check loop."""
        if self._health_check_task is None:
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            logger.info(
                "WorkerPoolManager started (health_check_interval=%.1fs)",
                self._health_check_interval,
            )

    async def stop(self) -> None:
        """Stop the background health check loop."""
        if self._health_check_task is not None:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            self._health_check_task = None
            logger.info("WorkerPoolManager stopped")

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    async def register_worker(self, worker_id: str, address: str) -> None:
        """Add a worker to the pool with status 'healthy'.

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

    # ------------------------------------------------------------------
    # Worker selection
    # ------------------------------------------------------------------

    async def next_worker(self) -> WorkerInfo | None:
        """Get the next healthy worker using round-robin scheduling.

        Cycles through all registered workers starting from the current
        round-robin index and returns the first healthy worker found.

        Returns:
            A WorkerInfo for a healthy worker, or None if none exist.
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
                if worker.status == "healthy":
                    self._round_robin_idx = (idx + 1) % n
                    return worker

            return None

    # ------------------------------------------------------------------
    # Health management
    # ------------------------------------------------------------------

    async def mark_unhealthy(self, worker_id: str) -> None:
        """Set a worker's status to 'unhealthy'."""
        async with self._lock:
            if worker_id in self._workers:
                old = self._workers[worker_id].status
                self._workers[worker_id].status = "unhealthy"
                logger.warning("Worker %s: %s -> unhealthy", worker_id, old)
            else:
                logger.warning(
                    "Cannot mark unknown worker %s as unhealthy",
                    worker_id,
                )

    async def mark_healthy(self, worker_id: str) -> None:
        """Set a worker's status to 'healthy'."""
        async with self._lock:
            if worker_id in self._workers:
                old = self._workers[worker_id].status
                if old != "healthy":
                    self._workers[worker_id].status = "healthy"
                    logger.info("Worker %s: %s -> healthy", worker_id, old)
            else:
                logger.warning(
                    "Cannot mark unknown worker %s as healthy",
                    worker_id,
                )

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    async def get_all_workers(self) -> list[WorkerInfo]:
        """Return a list of all registered workers."""
        async with self._lock:
            return list(self._workers.values())

    async def get_stats(self) -> dict:
        """Return worker pool statistics.

        Returns:
            Dict with keys: total, healthy, unhealthy.
        """
        async with self._lock:
            workers = list(self._workers.values())
            return {
                "total": len(workers),
                "healthy": sum(1 for w in workers if w.status == "healthy"),
                "unhealthy": sum(1 for w in workers if w.status == "unhealthy"),
            }

    # ------------------------------------------------------------------
    # Background health check
    # ------------------------------------------------------------------

    async def _health_check_loop(self) -> None:
        """Periodically poll /health on unhealthy workers."""
        try:
            while True:
                await asyncio.sleep(self._health_check_interval)
                await self._check_unhealthy_workers()
        except asyncio.CancelledError:
            logger.info("Health check loop cancelled")

    async def _check_unhealthy_workers(self) -> None:
        """Poll /health on all unhealthy workers, restore those OK."""
        async with self._lock:
            unhealthy = [w for w in self._workers.values() if w.status == "unhealthy"]

        if not unhealthy:
            return

        timeout = aiohttp.ClientTimeout(total=5.0)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for worker in unhealthy:
                try:
                    async with session.get(f"{worker.address}/health") as resp:
                        if resp.status == 200:
                            await self.mark_healthy(worker.worker_id)
                            logger.info(
                                "Health check: worker %s recovered",
                                worker.worker_id,
                            )
                except Exception:
                    # Still unhealthy — will retry next interval
                    pass
