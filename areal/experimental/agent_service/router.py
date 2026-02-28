"""Async producer-consumer dispatcher for the Agent Service Gateway.

The Router queues incoming DispatchRequests and dispatches them to Agent
Workers via HTTP using a background ``_dispatcher`` coroutine.  Back-pressure
is provided by a bounded :class:`asyncio.Queue`.

Failure Handling
----------------
When an HTTP request to a worker times out or errors, the worker is marked
unhealthy and the request is requeued **once**.  If the second attempt also
fails the future is resolved with an error :class:`DispatchResponse`.
"""

from __future__ import annotations

import asyncio
from typing import Any

import aiohttp

from areal.utils import logging

from .schemas import DispatchRequest, DispatchResponse, WorkerInfo
from .worker_pool import WorkerPoolManager

logger = logging.getLogger("Router")

# Type alias for items sitting in the internal queue.
_QueueItem = tuple[DispatchRequest, asyncio.Future[DispatchResponse]]


class Router:
    """Async dispatcher that bridges the Gateway API to Agent Workers.

    Incoming requests are placed into a bounded :pyclass:`asyncio.Queue`.  A
    background ``_dispatcher`` task continuously dequeues items, selects an
    idle worker via :class:`WorkerPoolManager`, and POSTs the request payload
    over HTTP.

    Attributes:
        _pool: Worker pool used for round-robin worker selection.
        _queue: Bounded async queue providing back-pressure.
        _session: Shared ``aiohttp.ClientSession`` for connection pooling.
        _dispatcher_task: Handle to the background dispatcher coroutine.
        _worker_timeout: Per-request HTTP timeout in seconds.
    """

    def __init__(
        self,
        pool: WorkerPoolManager,
        queue_size: int = 1000,
        worker_timeout: float = 300.0,
    ) -> None:
        """Initialize queue and state.

        Args:
            pool: Worker pool manager for worker selection.
            queue_size: Maximum number of pending requests before
                :meth:`enqueue` raises :class:`asyncio.QueueFull`.
            worker_timeout: HTTP timeout in seconds for each worker request.
        """
        self._pool: WorkerPoolManager = pool
        self._queue: asyncio.Queue[_QueueItem] = asyncio.Queue(maxsize=queue_size)
        self._session: aiohttp.ClientSession | None = None
        self._dispatcher_task: asyncio.Task[None] | None = None
        self._worker_timeout: float = worker_timeout

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Create the shared HTTP session and start the dispatcher loop."""
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self._worker_timeout),
        )
        self._dispatcher_task = asyncio.create_task(self._dispatcher())
        logger.info("Router started (queue_size=%d)", self._queue.maxsize)

    async def stop(self) -> None:
        """Cancel the dispatcher task and close the HTTP session."""
        if self._dispatcher_task is not None:
            self._dispatcher_task.cancel()
            try:
                await self._dispatcher_task
            except asyncio.CancelledError:
                pass
            self._dispatcher_task = None

        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None

        logger.info("Router stopped")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def enqueue(self, request: DispatchRequest) -> DispatchResponse:
        """Submit a request and wait for the dispatch result.

        The request is placed into the internal queue using ``put_nowait``.
        If the queue is full, :class:`asyncio.QueueFull` is raised
        immediately so the caller can apply its own back-pressure strategy.

        Args:
            request: The dispatch request to send to a worker.

        Returns:
            The :class:`DispatchResponse` produced once a worker handles the
            request (or an error response on failure).

        Raises:
            asyncio.QueueFull: If the internal queue has reached capacity.
        """
        loop = asyncio.get_running_loop()
        future: asyncio.Future[DispatchResponse] = loop.create_future()
        # put_nowait raises asyncio.QueueFull when the queue is at capacity.
        self._queue.put_nowait((request, future))
        return await future

    # ------------------------------------------------------------------
    # Background dispatcher
    # ------------------------------------------------------------------

    async def _dispatcher(self) -> None:
        """Background loop: dequeue requests and dispatch to workers."""
        try:
            while True:
                request, future = await self._queue.get()
                await self._dispatch_one(request, future, is_retry=False)
        except asyncio.CancelledError:
            logger.info("Dispatcher cancelled, draining remaining futures")
            # Resolve any futures still sitting in the queue so callers
            # awaiting them are not stuck forever.
            while not self._queue.empty():
                try:
                    request, future = self._queue.get_nowait()
                    if not future.done():
                        future.set_result(
                            DispatchResponse(
                                status="error",
                                error="Router is shutting down",
                            )
                        )
                except asyncio.QueueEmpty:
                    break

    async def _dispatch_one(
        self,
        request: DispatchRequest,
        future: asyncio.Future[DispatchResponse],
        *,
        is_retry: bool,
    ) -> None:
        """Find an idle worker, POST the request, and resolve *future*.

        On failure the request is requeued once (``is_retry=False`` ->
        ``is_retry=True``).  A second failure resolves *future* with an error
        response.
        """
        # Wait until an idle worker is available.
        worker = await self._wait_for_worker()

        await self._pool.mark_busy(worker.worker_id)
        try:
            response = await self._post_to_worker(worker, request)
            await self._pool.mark_idle(worker.worker_id)
            if not future.done():
                future.set_result(response)
        except Exception as exc:
            logger.warning(
                "Dispatch to worker %s failed: %s",
                worker.worker_id,
                exc,
            )
            await self._pool.mark_unhealthy(worker.worker_id)

            if not is_retry:
                # Requeue once for a different worker.
                logger.info("Requeuing request (first failure)")
                await self._dispatch_one(request, future, is_retry=True)
            else:
                # Second failure – give up.
                if not future.done():
                    future.set_result(
                        DispatchResponse(
                            status="error",
                            error=f"Worker dispatch failed after retry: {exc}",
                        )
                    )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _wait_for_worker(self) -> WorkerInfo:
        """Spin until an idle worker becomes available."""
        while True:
            worker = await self._pool.get_available_worker()
            if worker is not None:
                return worker
            await asyncio.sleep(0.01)

    async def _post_to_worker(
        self,
        worker: WorkerInfo,
        request: DispatchRequest,
    ) -> DispatchResponse:
        """POST *request* to ``{worker.address}/run_episode``.

        Returns:
            Parsed :class:`DispatchResponse` from the worker.

        Raises:
            aiohttp.ClientError: On network / HTTP errors.
            asyncio.TimeoutError: If the worker does not respond in time.
        """
        assert self._session is not None, "Router.start() must be called first"

        url = f"{worker.address}/run_episode"
        timeout = aiohttp.ClientTimeout(total=self._worker_timeout)
        async with self._session.post(
            url,
            json=request.model_dump(),
            timeout=timeout,
        ) as resp:
            data: dict[str, Any] = await resp.json()
            return DispatchResponse(**data)
