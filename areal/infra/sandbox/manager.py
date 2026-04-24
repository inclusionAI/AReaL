# SPDX-License-Identifier: Apache-2.0

"""Sandbox pool and lifecycle management.

Provides :class:`SandboxManager` — a per-thread sandbox pool manager
analogous to :class:`~areal.infra.workflow_context.HttpClientManager`.
Workflows access sandbox instances through this manager, which handles
pooling, lifecycle, and cleanup.
"""

from __future__ import annotations

import asyncio
import threading
from collections import deque

from areal.api.sandbox_api import SandboxConfig, SandboxExecutor
from areal.infra.sandbox.factory import create_sandbox
from areal.infra.utils.concurrent import register_loop_cleanup
from areal.utils import logging

logger = logging.getLogger("SandboxManager")


class SandboxManager:
    """Per-thread sandbox pool manager.

    Similar to :class:`~areal.infra.workflow_context.HttpClientManager`,
    workflows access sandbox instances through this manager which handles
    pooling, lifecycle, and cleanup.

    Parameters
    ----------
    config : SandboxConfig
        Sandbox configuration for creating new instances.
    """

    def __init__(self, config: SandboxConfig) -> None:
        self._config = config
        self._pool: deque[SandboxExecutor] = deque()
        self._lock = asyncio.Lock()
        self._active_count = 0
        self._total_created = 0
        self._closed = False

    async def checkout(self) -> SandboxExecutor:
        """Get a sandbox from the pool or create a new one.

        Returns
        -------
        SandboxExecutor
            A ready-to-use sandbox executor.
        """
        if self._closed:
            raise RuntimeError("SandboxManager is closed.")

        async with self._lock:
            if self._pool:
                sandbox = self._pool.popleft()
                self._active_count += 1
                logger.debug(
                    "Checked out sandbox from pool (pool=%d, active=%d)",
                    len(self._pool),
                    self._active_count,
                )
                return sandbox

        # Create new sandbox outside the lock to avoid blocking
        sandbox = await create_sandbox(self._config)
        self._total_created += 1
        self._active_count += 1
        logger.debug(
            "Created new sandbox (total_created=%d, active=%d)",
            self._total_created,
            self._active_count,
        )
        return sandbox

    async def checkin(self, sandbox: SandboxExecutor) -> None:
        """Return a sandbox to the pool or destroy it.

        If the pool is full (at ``pool_size``), the sandbox is destroyed.

        Parameters
        ----------
        sandbox : SandboxExecutor
            The sandbox to return.
        """
        self._active_count = max(0, self._active_count - 1)

        if self._closed:
            await sandbox.close()
            return

        # Check if sandbox is still usable
        is_closed = getattr(sandbox, "is_closed", False)
        if is_closed:
            logger.debug("Discarding closed sandbox.")
            return

        async with self._lock:
            if self._config.pool_size > 0 and len(self._pool) < self._config.pool_size:
                self._pool.append(sandbox)
                logger.debug(
                    "Returned sandbox to pool (pool=%d, active=%d)",
                    len(self._pool),
                    self._active_count,
                )
                return

        # Pool is full or pool_size is 0 (on-demand mode), destroy
        await sandbox.close()
        logger.debug(
            "Destroyed sandbox (pool full or on-demand mode, active=%d)",
            self._active_count,
        )

    async def warmup(self) -> None:
        """Pre-warm the sandbox pool to ``pool_size``.

        Creates sandbox instances concurrently up to the configured pool size.
        """
        if self._config.pool_size <= 0:
            return

        to_create = self._config.pool_size - len(self._pool)
        if to_create <= 0:
            return

        logger.info("Warming up sandbox pool with %d instances...", to_create)
        tasks = [create_sandbox(self._config) for _ in range(to_create)]
        sandboxes = await asyncio.gather(*tasks, return_exceptions=True)

        created = 0
        for result in sandboxes:
            if isinstance(result, Exception):
                logger.warning("Failed to create sandbox during warmup: %s", result)
            else:
                self._pool.append(result)
                self._total_created += 1
                created += 1

        logger.info("Sandbox pool warmup complete: %d/%d created.", created, to_create)

    async def cleanup(self) -> None:
        """Destroy all sandboxes in the pool.

        Should be called during shutdown. After cleanup, new checkouts
        will raise RuntimeError.
        """
        self._closed = True
        logger.info(
            "Cleaning up sandbox manager (pool=%d, active=%d)...",
            len(self._pool),
            self._active_count,
        )
        while self._pool:
            sandbox = self._pool.popleft()
            try:
                await sandbox.close()
            except Exception as exc:
                logger.warning("Error closing pooled sandbox: %s", exc)

        logger.info("Sandbox manager cleanup complete.")

    @property
    def pool_size(self) -> int:
        """Current number of sandboxes in the pool."""
        return len(self._pool)

    @property
    def active_count(self) -> int:
        """Number of sandboxes currently checked out."""
        return self._active_count


# ---------------------------------------------------------------------------
# Global per-thread SandboxManager registry
# (mirrors workflow_context._managers pattern)
# ---------------------------------------------------------------------------
_managers: dict[int, SandboxManager] = {}
_managers_lock = threading.Lock()
_configs: dict[int, SandboxConfig] = {}


def configure(config: SandboxConfig) -> None:
    """Configure the sandbox manager for the current thread.

    Must be called before :func:`get_manager`. Typically called during
    workflow executor initialization.

    Parameters
    ----------
    config : SandboxConfig
        Sandbox configuration.
    """
    thread_id = threading.get_ident()
    with _managers_lock:
        _configs[thread_id] = config


def get_manager() -> SandboxManager:
    """Get or create the SandboxManager for the current thread.

    Returns
    -------
    SandboxManager
        The manager for the current thread.

    Raises
    ------
    RuntimeError
        If :func:`configure` has not been called for this thread.
    """
    thread_id = threading.get_ident()
    with _managers_lock:
        if thread_id in _managers:
            return _managers[thread_id]
        if thread_id not in _configs:
            raise RuntimeError(
                "SandboxManager not configured for this thread. "
                "Call areal.infra.sandbox.manager.configure(config) first."
            )
        manager = SandboxManager(_configs[thread_id])
        _managers[thread_id] = manager

        # Register cleanup with the event loop
        async def _cleanup_sandbox_manager():
            await manager.cleanup()

        try:
            loop = asyncio.get_running_loop()
            register_loop_cleanup(_cleanup_sandbox_manager, loop=loop)
        except RuntimeError:
            pass  # No running loop; cleanup will happen manually

        return manager


async def checkout_sandbox() -> SandboxExecutor:
    """Convenience: get a sandbox from the current thread's pool.

    Returns
    -------
    SandboxExecutor
        A ready-to-use sandbox.
    """
    return await get_manager().checkout()


async def checkin_sandbox(sandbox: SandboxExecutor) -> None:
    """Convenience: return a sandbox to the current thread's pool.

    Parameters
    ----------
    sandbox : SandboxExecutor
        The sandbox to return.
    """
    await get_manager().checkin(sandbox)
