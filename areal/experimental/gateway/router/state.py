"""Worker and session registries for the Router service.

All state is in-memory (lost on restart). Thread-safe via asyncio.Lock.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field


@dataclass
class WorkerInfo:
    """A registered data proxy worker."""

    worker_addr: str
    is_healthy: bool = True
    active_requests: int = 0
    registered_at: float = field(default_factory=time.time)


class WorkerRegistry:
    """Thread-safe worker registry with health tracking."""

    def __init__(self) -> None:
        self._workers: dict[str, WorkerInfo] = {}
        self._lock = asyncio.Lock()

    async def register(self, worker_addr: str) -> None:
        """Add a worker. No-op if already registered."""
        async with self._lock:
            if worker_addr not in self._workers:
                self._workers[worker_addr] = WorkerInfo(worker_addr=worker_addr)

    async def deregister(self, worker_addr: str) -> None:
        """Remove a worker. No-op if not found."""
        async with self._lock:
            self._workers.pop(worker_addr, None)

    async def update_health(self, worker_addr: str, healthy: bool) -> None:
        """Set the health flag for a worker."""
        async with self._lock:
            w = self._workers.get(worker_addr)
            if w:
                w.is_healthy = healthy

    async def get_healthy_workers(self) -> list[WorkerInfo]:
        """Return only workers with ``is_healthy == True``."""
        async with self._lock:
            return [w for w in self._workers.values() if w.is_healthy]

    async def get_all_workers(self) -> list[WorkerInfo]:
        """Return all workers regardless of health."""
        async with self._lock:
            return list(self._workers.values())

    async def list_worker_addrs(self) -> list[str]:
        """Return all registered worker addresses."""
        async with self._lock:
            return list(self._workers.keys())


class CapacityManager:
    """Tracks available capacity for new RL sessions (staleness control).

    The rollout controller calls ``grant()`` once per episode to add one
    permit.  ``try_acquire()`` is called when ``/rl/start_session`` arrives
    — if no permits remain, it returns *False* so the gateway can respond
    with HTTP 429.  This prevents users from starting sessions outside
    the allowed weight-staleness window.
    """

    def __init__(self) -> None:
        self._capacity: int = 0
        self._lock = asyncio.Lock()

    async def grant(self) -> int:
        """Increment capacity by 1. Returns the new capacity value."""
        async with self._lock:
            self._capacity += 1
            return self._capacity

    async def try_acquire(self) -> bool:
        """Try to decrement capacity by 1. Returns True on success."""
        async with self._lock:
            if self._capacity <= 0:
                return False
            self._capacity -= 1
            return True

    async def get_capacity(self) -> int:
        """Return current capacity (for health / debug endpoints)."""
        async with self._lock:
            return self._capacity


class SessionRegistry:
    """Maps session API keys and session IDs to worker addresses.

    Pinning persists even after ``/rl/end_session`` (needed for
    ``/export_trajectories``). Revoked only when worker is deleted.
    """

    def __init__(self) -> None:
        self._key_to_worker: dict[str, str] = {}  # session_api_key -> worker_addr
        self._id_to_worker: dict[str, str] = {}  # session_id -> worker_addr
        self._lock = asyncio.Lock()

    async def register_session(
        self, session_key: str, session_id: str, worker_addr: str
    ) -> None:
        """Store both session_key→worker and session_id→worker. Upsert semantics."""
        async with self._lock:
            self._key_to_worker[session_key] = worker_addr
            self._id_to_worker[session_id] = worker_addr

    async def lookup_by_key(self, session_key: str) -> str | None:
        """Return the worker address pinned to a session API key, or None."""
        async with self._lock:
            return self._key_to_worker.get(session_key)

    async def lookup_by_id(self, session_id: str) -> str | None:
        """Return the worker address pinned to a session ID, or None."""
        async with self._lock:
            return self._id_to_worker.get(session_id)

    async def revoke_by_worker(self, worker_addr: str) -> int:
        """Remove all sessions pinned to a worker (cascade on deletion).

        Returns the number of session keys removed.
        """
        async with self._lock:
            keys_to_remove = [
                k for k, v in self._key_to_worker.items() if v == worker_addr
            ]
            ids_to_remove = [
                k for k, v in self._id_to_worker.items() if v == worker_addr
            ]
            for k in keys_to_remove:
                del self._key_to_worker[k]
            for k in ids_to_remove:
                del self._id_to_worker[k]
            return len(keys_to_remove)

    async def count(self) -> int:
        """Return the number of registered session keys."""
        async with self._lock:
            return len(self._key_to_worker)
