from __future__ import annotations

import asyncio
import threading
import weakref
from contextvars import ContextVar
from dataclasses import dataclass
from functools import partial

import aiohttp
import httpx

from areal.utils import logging
from areal.utils.http import DEFAULT_REQUEST_TIMEOUT, get_default_connector

logger = logging.getLogger("WorkflowContext")


@dataclass(frozen=True)
class WorkflowContext:
    """Execution context available to workflows via contextvars.

    Attributes
    ----------
    is_eval : bool
        Whether the workflow is running in evaluation mode.
    task_id : int | None
        The task ID assigned by the workflow executor.
    """

    is_eval: bool = False
    task_id: int | None = None


_current_context: ContextVar[WorkflowContext] = ContextVar(
    "workflow_context", default=WorkflowContext()
)


def set(ctx: WorkflowContext) -> None:
    """Set the current workflow context."""
    _current_context.set(ctx)


def get() -> WorkflowContext:
    """Get the current workflow context."""
    return _current_context.get()


def stat_scope() -> str:
    """Get the appropriate stats_tracker scope based on current context.

    Returns
    -------
    str
        "eval-rollout" if in eval mode, "rollout" otherwise.
    """
    return "eval-rollout" if get().is_eval else "rollout"


class _LoopCleanupEntry:
    """Registry entry for HTTP client cleanup on event loop close.

    Stores reference to HttpClientManager and original loop.close method.
    Handles the complexity of weakrefs for extension loops (uvloop).
    """

    def __init__(self, loop: asyncio.AbstractEventLoop, manager: HttpClientManager):
        # Store manager reference
        self.manager = manager

        # Save original close method (avoid double-patching)
        if not hasattr(loop, "_http_cleanup_orig_close"):
            loop._http_cleanup_orig_close = loop.close

        # Try to create weakref to original close
        # Some loop implementations (uvloop) can't be weakref'd
        try:
            self._close_ref = weakref.WeakMethod(loop._http_cleanup_orig_close)
        except TypeError:
            # Fallback: store regular reference on the loop object itself
            self._close_ref = lambda: loop._http_cleanup_orig_close

    def get_original_close(self):
        """Get the original close method."""
        return self._close_ref()


# Global registry: event_loop -> _LoopCleanupEntry
# WeakKeyDictionary ensures loops can be garbage collected
_loop_registry: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, _LoopCleanupEntry
] = weakref.WeakKeyDictionary()


async def _run_http_cleanup(manager: HttpClientManager) -> None:
    """Run async cleanup for HTTP clients.

    Called by patched loop.close() via run_until_complete.
    """
    try:
        if manager._aiohttp_session is not None:
            await manager._aiohttp_session.close()
            manager._aiohttp_session = None
        if manager._httpx_client is not None:
            await manager._httpx_client.aclose()
            manager._httpx_client = None
        manager._event_loop = None
    except Exception as e:
        # Log but don't fail - cleanup is best-effort during shutdown
        logger.warning(f"Error during HTTP client cleanup: {e}")


def _patched_loop_close(loop: asyncio.AbstractEventLoop) -> None:
    """Patched EventLoop.close method to run HTTP cleanup before closing.

    This is the core of the asyncio-atexit pattern.
    """
    entry = _loop_registry.get(loop)
    if entry is not None and entry.manager is not None:
        # Run async cleanup if loop is not already closed
        if not loop.is_closed():
            try:
                loop.run_until_complete(_run_http_cleanup(entry.manager))
            except Exception as e:
                logger.warning(f"Failed to run HTTP cleanup on loop close: {e}")

    # Call original close method
    original_close = (
        entry.get_original_close() if entry else loop._http_cleanup_orig_close
    )
    return original_close()


def _register_loop_cleanup(
    loop: asyncio.AbstractEventLoop, manager: HttpClientManager
) -> None:
    """Register HTTP client cleanup for an event loop.

    Patches the loop's close() method if not already patched.
    """
    # Check if already registered
    if loop in _loop_registry:
        # Update manager reference (loop might be reused with new manager)
        _loop_registry[loop].manager = manager
        return

    # Create new registry entry
    entry = _LoopCleanupEntry(loop, manager)
    _loop_registry[loop] = entry

    # Patch loop.close with our cleanup wrapper
    loop.close = partial(_patched_loop_close, loop)


class HttpClientManager:
    """Per-thread manager for shared HTTP clients.

    This class manages shared aiohttp and httpx clients for workflow execution,
    providing connection pooling and DNS caching for improved performance.
    Each thread gets its own instance via a global dict indexed by thread ID.
    """

    def __init__(self) -> None:
        """Initialize the HttpClientManager instance."""
        self._aiohttp_session: aiohttp.ClientSession | None = None
        self._httpx_client: httpx.AsyncClient | None = None
        self._event_loop: asyncio.AbstractEventLoop | None = None

    async def _check_event_loop_change(self) -> None:
        """Check if event loop changed and close stale clients if so."""
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop - nothing to check
            return

        if self._event_loop is not None and self._event_loop is not current_loop:
            # Event loop changed - need to close stale clients
            logger.warning(
                "Event loop changed. Closing stale HTTP clients and creating new ones."
            )

            old_aiohttp = self._aiohttp_session
            old_httpx = self._httpx_client

            # Reset state first
            self._aiohttp_session = None
            self._httpx_client = None
            self._event_loop = None

            # Close old clients asynchronously (in current loop - they may warn but won't fail)
            if old_aiohttp is not None:
                await old_aiohttp.close()
            if old_httpx is not None:
                await old_httpx.aclose()

    async def get_aiohttp_session(self) -> aiohttp.ClientSession:
        """Get the shared aiohttp.ClientSession for the current workflow execution.

        The session is lazily created on first access and reused for all subsequent
        calls within the same AsyncTaskRunner background thread. The session is
        automatically closed during AsyncTaskRunner shutdown.

        If the event loop has changed since the session was created (e.g., in tests),
        the old session is closed and a new one is created.

        Returns
        -------
        aiohttp.ClientSession
            The shared session for HTTP requests.
        """
        await self._check_event_loop_change()

        if self._aiohttp_session is None:
            timeout = aiohttp.ClientTimeout(total=DEFAULT_REQUEST_TIMEOUT)
            self._aiohttp_session = aiohttp.ClientSession(
                timeout=timeout,
                read_bufsize=1024 * 1024 * 10,
                connector=get_default_connector(),
            )
            # Track which event loop this session belongs to
            self._event_loop = asyncio.get_running_loop()

            # Register cleanup with the event loop
            _register_loop_cleanup(self._event_loop, self)

        return self._aiohttp_session

    async def get_httpx_client(self) -> httpx.AsyncClient:
        """Get the shared httpx.AsyncClient for the current workflow execution.

        The client is lazily created on first access and reused for all subsequent
        calls within the same AsyncTaskRunner background thread. The client is
        automatically closed during AsyncTaskRunner shutdown.

        If the event loop has changed since the client was created (e.g., in tests),
        the old client is closed and a new one is created.

        Returns
        -------
        httpx.AsyncClient
            The shared client for HTTP requests.
        """
        await self._check_event_loop_change()

        if self._httpx_client is None:
            self._httpx_client = httpx.AsyncClient(
                timeout=httpx.Timeout(DEFAULT_REQUEST_TIMEOUT)
            )
            # Track which event loop this client belongs to
            self._event_loop = asyncio.get_running_loop()

            # Register cleanup with the event loop
            _register_loop_cleanup(self._event_loop, self)

        return self._httpx_client


# Global dict mapping thread ID -> HttpClientManager
# Each thread gets its own HttpClientManager instance
_managers: dict[int, HttpClientManager] = {}
_managers_lock = threading.Lock()


def _get_manager() -> HttpClientManager:
    """Get or create the HttpClientManager for the current thread.

    Returns
    -------
    HttpClientManager
        The HttpClientManager instance for the current thread.
    """
    thread_id = threading.get_ident()
    with _managers_lock:
        if thread_id not in _managers:
            _managers[thread_id] = HttpClientManager()
        return _managers[thread_id]


async def get_aiohttp_session() -> aiohttp.ClientSession:
    """Get the shared aiohttp.ClientSession for the current workflow execution.

    The session is lazily created on first access and reused for all subsequent
    calls within the same AsyncTaskRunner background thread. The session is
    automatically closed during AsyncTaskRunner shutdown.

    Returns
    -------
    aiohttp.ClientSession
        The shared session for HTTP requests.
    """
    return await _get_manager().get_aiohttp_session()


async def get_httpx_client() -> httpx.AsyncClient:
    """Get the shared httpx.AsyncClient for the current workflow execution.

    The client is lazily created on first access and reused for all subsequent
    calls within the same AsyncTaskRunner background thread. The client is
    automatically closed during AsyncTaskRunner shutdown.

    Returns
    -------
    httpx.AsyncClient
        The shared client for HTTP requests.
    """
    return await _get_manager().get_httpx_client()
