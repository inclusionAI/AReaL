from __future__ import annotations

import threading
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass

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


class _ThreadState:
    """Container for per-thread HTTP client state.

    Each thread that uses HTTP clients gets its own instance of this class
    via thread-local storage. This ensures aiohttp.ClientSession and
    httpx.AsyncClient instances are bound to the correct event loop.
    """

    __slots__ = (
        "aiohttp_session",
        "httpx_client",
        "cleanup_registered",
    )

    def __init__(self):
        self.aiohttp_session: aiohttp.ClientSession | None = None
        self.httpx_client: httpx.AsyncClient | None = None
        self.cleanup_registered: bool = False


class HttpClientManager:
    """Thread-local manager for HTTP clients.

    This class manages aiohttp and httpx clients for workflow execution,
    providing connection pooling and DNS caching for improved performance.

    Each thread gets its own HTTP client instances that are bound to that
    thread's event loop. This is necessary because aiohttp.ClientSession
    and httpx.AsyncClient are not safe to share across different event loops.

    The shutdown_registrar is configured globally (typically from the main thread)
    and shared across all threads.
    """

    _thread_local = threading.local()
    _shutdown_registrar: Callable[[Callable[[], Awaitable[None]]], None] | None = None

    def _get_thread_state(self) -> _ThreadState:
        """Get or create thread-local state for the current thread."""
        if not hasattr(self._thread_local, "state"):
            self._thread_local.state = _ThreadState()
        return self._thread_local.state

    def configure(
        self,
        shutdown_hook_registrar: Callable[[Callable[[], Awaitable[None]]], None],
    ) -> None:
        """Configure the shutdown hook registrar for HTTP clients.

        Called once from the main thread to set up the cleanup mechanism.
        The registrar is shared globally across all threads.

        Parameters
        ----------
        shutdown_hook_registrar : Callable
            A function to register the HTTP client cleanup hook (typically
            AsyncTaskRunner.register_shutdown_hook).
        """
        HttpClientManager._shutdown_registrar = shutdown_hook_registrar

    def _ensure_cleanup_registered(self, state: _ThreadState) -> None:
        """Register the cleanup hook for the given thread state if not already done."""
        registrar = HttpClientManager._shutdown_registrar
        if not state.cleanup_registered and registrar is not None:
            registrar(self._make_cleanup_for_state(state))
            state.cleanup_registered = True

    def _make_cleanup_for_state(
        self, state: _ThreadState
    ) -> Callable[[], Awaitable[None]]:
        """Create a cleanup coroutine bound to specific thread state."""

        async def _async_cleanup() -> None:
            if state.aiohttp_session is not None:
                await state.aiohttp_session.close()
                state.aiohttp_session = None
            if state.httpx_client is not None:
                await state.httpx_client.aclose()
                state.httpx_client = None

        return _async_cleanup

    def get_aiohttp_session(self) -> aiohttp.ClientSession:
        """Get the aiohttp.ClientSession for the current thread.

        The session is lazily created on first access and bound to the
        current thread's event loop. Each thread gets its own session
        instance to avoid cross-event-loop issues.

        Returns
        -------
        aiohttp.ClientSession
            The session for HTTP requests in the current thread.
        """
        state = self._get_thread_state()
        if state.aiohttp_session is None:
            timeout = aiohttp.ClientTimeout(total=DEFAULT_REQUEST_TIMEOUT)
            state.aiohttp_session = aiohttp.ClientSession(
                timeout=timeout,
                read_bufsize=1024 * 1024 * 10,
                connector=get_default_connector(),
            )
            if HttpClientManager._shutdown_registrar is None:
                logger.warning(
                    "HTTP session created before configure() was called. "
                    "Session cleanup may not be automatic."
                )
            self._ensure_cleanup_registered(state)
        return state.aiohttp_session

    def get_httpx_client(self) -> httpx.AsyncClient:
        """Get the httpx.AsyncClient for the current thread.

        The client is lazily created on first access and bound to the
        current thread's event loop. Each thread gets its own client
        instance to avoid cross-event-loop issues.

        Returns
        -------
        httpx.AsyncClient
            The client for HTTP requests in the current thread.
        """
        state = self._get_thread_state()
        if state.httpx_client is None:
            state.httpx_client = httpx.AsyncClient(
                timeout=httpx.Timeout(DEFAULT_REQUEST_TIMEOUT)
            )
            if HttpClientManager._shutdown_registrar is None:
                logger.warning(
                    "HTTP client created before configure() was called. "
                    "Client cleanup may not be automatic."
                )
            self._ensure_cleanup_registered(state)
        return state.httpx_client

    def reset(self) -> None:
        """Reset the current thread's HTTP client state.

        This resets only the current thread's client instances. The clients
        themselves should be closed via the registered shutdown hook before
        calling this. The global shutdown_registrar is not affected.
        """
        state = self._get_thread_state()
        state.aiohttp_session = None
        state.httpx_client = None
        state.cleanup_registered = False


# Module-level singleton instance
_http_client_manager = HttpClientManager()


def configure_http_clients(
    shutdown_hook_registrar: Callable[[Callable[[], Awaitable[None]]], None],
) -> None:
    """Configure the shutdown hook registrar for HTTP clients.

    Called once from the main thread to set up the cleanup mechanism.
    The registrar is shared globally across all threads.

    Parameters
    ----------
    shutdown_hook_registrar : Callable
        A function to register the HTTP client cleanup hook (typically
        AsyncTaskRunner.register_shutdown_hook).
    """
    _http_client_manager.configure(shutdown_hook_registrar)


def get_aiohttp_session() -> aiohttp.ClientSession:
    """Get the aiohttp.ClientSession for the current thread.

    The session is lazily created on first access and bound to the current
    thread's event loop. Each thread gets its own session instance to avoid
    cross-event-loop issues. The session is automatically closed during
    AsyncTaskRunner shutdown.

    Returns
    -------
    aiohttp.ClientSession
        The session for HTTP requests in the current thread.
    """
    return _http_client_manager.get_aiohttp_session()


def get_httpx_client() -> httpx.AsyncClient:
    """Get the httpx.AsyncClient for the current thread.

    The client is lazily created on first access and bound to the current
    thread's event loop. Each thread gets its own client instance to avoid
    cross-event-loop issues. The client is automatically closed during
    AsyncTaskRunner shutdown.

    Returns
    -------
    httpx.AsyncClient
        The client for HTTP requests in the current thread.
    """
    return _http_client_manager.get_httpx_client()


def reset_http_clients() -> None:
    """Reset the current thread's HTTP client state.

    This resets only the current thread's client instances. The clients
    themselves should be closed via the registered shutdown hook before
    calling this. The global shutdown_registrar is not affected.
    """
    _http_client_manager.reset()
