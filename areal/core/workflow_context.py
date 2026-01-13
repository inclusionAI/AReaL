from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING

import aiohttp
import httpx

from areal.utils import logging
from areal.utils.http import DEFAULT_REQUEST_TIMEOUT, get_default_connector

if TYPE_CHECKING:
    from areal.core.async_task_runner import AsyncTaskRunner

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


class HttpClientManager:
    """Per-thread and per-event-loop manager for shared HTTP clients.

    This class manages shared aiohttp and httpx clients for workflow execution,
    providing connection pooling and DNS caching for improved performance.
    Each thread/event loop combination gets its own instance via a global dict
    indexed by (thread_id, loop_id) tuple.
    """

    def __init__(self) -> None:
        """Initialize the HttpClientManager instance."""
        self._aiohttp_session: aiohttp.ClientSession | None = None
        self._httpx_client: httpx.AsyncClient | None = None
        self._shutdown_registrar: (
            Callable[[Callable[[AsyncTaskRunner], Awaitable[None]]], None] | None
        ) = None
        self._cleanup_registered: bool = False
        self._client_lock = threading.Lock()

    def configure(
        self,
        shutdown_hook_registrar: Callable[
            [Callable[[AsyncTaskRunner], Awaitable[None]]], None
        ],
    ) -> None:
        """Configure the shutdown hook registrar for HTTP client cleanup.

        Called by WorkflowExecutor.initialize() to set up the cleanup mechanism.

        Parameters
        ----------
        shutdown_hook_registrar : Callable
            A function to register the HTTP client cleanup hook (typically
            AsyncTaskRunner.register_shutdown_hook).
        """
        with self._client_lock:
            self._shutdown_registrar = shutdown_hook_registrar
            self._cleanup_registered = False

    def _ensure_cleanup_registered(self) -> None:
        """Register the cleanup hook if not already registered."""
        if not self._cleanup_registered and self._shutdown_registrar is not None:
            self._shutdown_registrar(self._async_cleanup)
            self._cleanup_registered = True

    def get_aiohttp_session(self) -> aiohttp.ClientSession:
        """Get the shared aiohttp.ClientSession for the current workflow execution.

        The session is lazily created on first access and reused for all subsequent
        calls within the same AsyncTaskRunner background thread. The session is
        automatically closed during AsyncTaskRunner shutdown.

        Returns
        -------
        aiohttp.ClientSession
            The shared session for HTTP requests.
        """
        with self._client_lock:
            if self._aiohttp_session is None:
                timeout = aiohttp.ClientTimeout(total=DEFAULT_REQUEST_TIMEOUT)
                self._aiohttp_session = aiohttp.ClientSession(
                    timeout=timeout,
                    read_bufsize=1024 * 1024 * 10,
                    connector=get_default_connector(),
                )
                if self._shutdown_registrar is None:
                    logger.warning(
                        "HTTP session created before configure() was called. "
                        "Session cleanup may not be automatic."
                    )
                self._ensure_cleanup_registered()
            return self._aiohttp_session

    def get_httpx_client(self) -> httpx.AsyncClient:
        """Get the shared httpx.AsyncClient for the current workflow execution.

        The client is lazily created on first access and reused for all subsequent
        calls within the same AsyncTaskRunner background thread. The client is
        automatically closed during AsyncTaskRunner shutdown.

        Returns
        -------
        httpx.AsyncClient
            The shared client for HTTP requests.
        """
        with self._client_lock:
            if self._httpx_client is None:
                self._httpx_client = httpx.AsyncClient(
                    timeout=httpx.Timeout(DEFAULT_REQUEST_TIMEOUT)
                )
                if self._shutdown_registrar is None:
                    logger.warning(
                        "HTTP client created before configure() was called. "
                        "Client cleanup may not be automatic."
                    )
                self._ensure_cleanup_registered()
            return self._httpx_client

    async def _async_cleanup(self, runner: AsyncTaskRunner) -> None:
        """Async cleanup hook called during shutdown to close all HTTP clients.

        Parameters
        ----------
        runner : AsyncTaskRunner
            The runner instance that invoked this hook (unused but required by signature).
        """
        thread_id = threading.get_ident()
        try:
            loop = asyncio.get_running_loop()
            loop_id = id(loop)
        except RuntimeError:
            raise RuntimeError(
                "No running event loop; HttpClientManager requires an active event loop"
            )

        with self._client_lock:
            if self._aiohttp_session is not None:
                await self._aiohttp_session.close()
                self._aiohttp_session = None
            if self._httpx_client is not None:
                await self._httpx_client.aclose()
                self._httpx_client = None
            self._cleanup_registered = False
        # Remove from global dict so next initialization creates fresh instance
        _remove_manager((thread_id, loop_id))


# Global dict mapping (thread_id, loop_id) -> HttpClientManager
# Each thread/event loop combination gets its own HttpClientManager instance
_managers: dict[tuple[int, int], HttpClientManager] = {}
_managers_lock = threading.Lock()


def _get_manager() -> HttpClientManager:
    """Get or create the HttpClientManager for the current thread and event loop.

    Returns
    -------
    HttpClientManager
        The HttpClientManager instance for the current thread and event loop.
    """
    thread_id = threading.get_ident()
    try:
        loop = asyncio.get_running_loop()
        loop_id = id(loop)
    except RuntimeError:
        raise RuntimeError(
            "No running event loop; HttpClientManager requires an active event loop"
        )

    key = (thread_id, loop_id)
    with _managers_lock:
        if key not in _managers:
            _managers[key] = HttpClientManager()
        return _managers[key]


def _remove_manager(key: tuple[int, int]) -> None:
    """Remove the HttpClientManager for a specific thread/loop from the global dict.

    Parameters
    ----------
    key : tuple[int, int]
        The (thread_id, loop_id) key whose manager should be removed.
    """
    with _managers_lock:
        _managers.pop(key, None)


def configure_http_clients(
    shutdown_hook_registrar: Callable[
        [Callable[[AsyncTaskRunner], Awaitable[None]]], None
    ],
) -> None:
    """Configure the shutdown hook registrar for HTTP client cleanup.

    Called by WorkflowExecutor.initialize() to set up the cleanup mechanism.

    Parameters
    ----------
    shutdown_hook_registrar : Callable
        A function to register the HTTP client cleanup hook (typically
        AsyncTaskRunner.register_shutdown_hook).
    """
    _get_manager().configure(shutdown_hook_registrar)


def get_aiohttp_session() -> aiohttp.ClientSession:
    """Get the shared aiohttp.ClientSession for the current workflow execution.

    The session is lazily created on first access and reused for all subsequent
    calls within the same AsyncTaskRunner background thread. The session is
    automatically closed during AsyncTaskRunner shutdown.

    Returns
    -------
    aiohttp.ClientSession
        The shared session for HTTP requests.
    """
    return _get_manager().get_aiohttp_session()


def get_httpx_client() -> httpx.AsyncClient:
    """Get the shared httpx.AsyncClient for the current workflow execution.

    The client is lazily created on first access and reused for all subsequent
    calls within the same AsyncTaskRunner background thread. The client is
    automatically closed during AsyncTaskRunner shutdown.

    Returns
    -------
    httpx.AsyncClient
        The shared client for HTTP requests.
    """
    return _get_manager().get_httpx_client()
