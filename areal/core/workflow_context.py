from __future__ import annotations

import threading
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Self

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


class HttpClientManager:
    """Thread-safe singleton manager for shared HTTP clients.

    This class manages shared aiohttp and httpx clients for workflow execution,
    providing connection pooling and DNS caching for improved performance.
    """

    _instance: HttpClientManager | None = None
    _instance_lock = threading.Lock()

    def __new__(cls) -> Self:
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._aiohttp_session = None
                    instance._httpx_client = None
                    instance._shutdown_registrar = None
                    instance._cleanup_registered = False
                    instance._client_lock = threading.Lock()
                    cls._instance = instance
        return cls._instance

    def configure(
        self,
        shutdown_hook_registrar: Callable[[Callable[[], Awaitable[None]]], None],
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

    async def _async_cleanup(self) -> None:
        """Async cleanup hook called during shutdown to close all HTTP clients."""
        with self._client_lock:
            if self._aiohttp_session is not None:
                await self._aiohttp_session.close()
                self._aiohttp_session = None
            if self._httpx_client is not None:
                await self._httpx_client.aclose()
                self._httpx_client = None

    def reset(self) -> None:
        """Reset all HTTP client state for reinitialization.

        This resets the instance state. The clients themselves should be
        closed via the registered shutdown hook before calling this.
        """
        with self._client_lock:
            self._aiohttp_session = None
            self._httpx_client = None
            self._shutdown_registrar = None
            self._cleanup_registered = False


# Module-level singleton instance
_http_client_manager = HttpClientManager()


def configure_http_clients(
    shutdown_hook_registrar: Callable[[Callable[[], Awaitable[None]]], None],
) -> None:
    """Configure the shutdown hook registrar for HTTP client cleanup.

    Called by WorkflowExecutor.initialize() to set up the cleanup mechanism.

    Parameters
    ----------
    shutdown_hook_registrar : Callable
        A function to register the HTTP client cleanup hook (typically
        AsyncTaskRunner.register_shutdown_hook).
    """
    _http_client_manager.configure(shutdown_hook_registrar)


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
    return _http_client_manager.get_aiohttp_session()


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
    return _http_client_manager.get_httpx_client()


def reset_http_clients() -> None:
    """Reset all HTTP client state for reinitialization.

    This resets the module-level state. The clients themselves should be
    closed via the registered shutdown hook before calling this.
    """
    _http_client_manager.reset()
