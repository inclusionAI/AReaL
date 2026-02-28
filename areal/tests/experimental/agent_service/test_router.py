"""Unit tests for the Agent Service Router."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from areal.experimental.agent_service.router import Router
from areal.experimental.agent_service.schemas import (
    DispatchRequest,
    DispatchResponse,
    WorkerInfo,
)
from areal.experimental.agent_service.worker_pool import WorkerPoolManager


def _make_request(session_url: str = "http://proxy/session/1") -> DispatchRequest:
    """Create a minimal DispatchRequest for testing."""
    return DispatchRequest(data={"prompt": "test"}, session_url=session_url)


async def _make_router_with_worker(
    worker_id: str = "w1",
    worker_addr: str = "http://localhost:9001",
    queue_size: int = 100,
    worker_timeout: float = 5.0,
) -> tuple[Router, WorkerPoolManager]:
    """Create a started Router with one registered worker."""
    pool = WorkerPoolManager(worker_timeout=worker_timeout)
    await pool.register_worker(worker_id, worker_addr)
    router = Router(pool=pool, queue_size=queue_size, worker_timeout=worker_timeout)
    await router.start()
    return router, pool


class TestRouterLifecycle:
    """Tests for Router start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_start_creates_session_and_dispatcher(self):
        """Router.start() should create HTTP session and dispatcher task."""
        pool = WorkerPoolManager()
        router = Router(pool=pool, queue_size=10)

        await router.start()
        try:
            assert router._session is not None
            assert not router._session.closed
            assert router._dispatcher_task is not None
            assert not router._dispatcher_task.done()
        finally:
            await router.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_dispatcher_and_closes_session(self):
        """Router.stop() should cancel dispatcher and close HTTP session."""
        pool = WorkerPoolManager()
        router = Router(pool=pool, queue_size=10)

        await router.start()
        await router.stop()

        assert router._dispatcher_task is None
        assert router._session is None

    @pytest.mark.asyncio
    async def test_stop_resolves_queued_futures(self):
        """Router.stop() should resolve futures still in the queue with error response."""
        pool = WorkerPoolManager()
        router = Router(pool=pool, queue_size=10, worker_timeout=5.0)
        await router.start()

        # Register a worker that blocks indefinitely so the dispatcher stays busy
        # on the first item, leaving subsequent items in the queue.
        await pool.register_worker("w1", "http://localhost:9001")

        blocked = asyncio.Event()

        async def blocking_post(
            worker: WorkerInfo, request: DispatchRequest
        ) -> DispatchResponse:
            blocked.set()
            await asyncio.sleep(100)  # block until cancelled
            return DispatchResponse(status="success", result=0.0)

        with patch.object(router, "_post_to_worker", new=blocking_post):
            loop = asyncio.get_running_loop()
            # First item: dispatcher will pick it up and block in blocking_post
            future1: asyncio.Future[DispatchResponse] = loop.create_future()
            router._queue.put_nowait((_make_request(), future1))

            # Wait until dispatcher is inside blocking_post
            await asyncio.wait_for(blocked.wait(), timeout=2.0)

            # Second item: stays in queue since dispatcher is busy
            future2: asyncio.Future[DispatchResponse] = loop.create_future()
            router._queue.put_nowait((_make_request("http://proxy/session/2"), future2))

            # stop() cancels the dispatcher; drain loop resolves future2
            await router.stop()

        # future2 was in the queue when stop() ran — should be resolved with error
        assert future2.done()
        assert future2.result().status == "error"
        assert "shutting down" in future2.result().error.lower()


class TestRouterEnqueue:
    """Tests for Router.enqueue() behavior."""

    @pytest.mark.asyncio
    async def test_enqueue_raises_queue_full_when_at_capacity(self):
        """enqueue() should raise asyncio.QueueFull when queue is full."""
        pool = WorkerPoolManager()
        # queue_size=0 means maxsize=0 which is unlimited in asyncio.Queue,
        # so use maxsize=1 and fill it manually
        router = Router(pool=pool, queue_size=1, worker_timeout=5.0)
        await router.start()

        try:
            # Stop dispatcher so items stay in queue
            router._dispatcher_task.cancel()
            try:
                await router._dispatcher_task
            except asyncio.CancelledError:
                pass
            router._dispatcher_task = None

            # Fill the queue
            loop = asyncio.get_running_loop()
            future1: asyncio.Future[DispatchResponse] = loop.create_future()
            router._queue.put_nowait((_make_request(), future1))

            # Second enqueue should raise QueueFull
            with pytest.raises(asyncio.QueueFull):
                router._queue.put_nowait(
                    (_make_request("http://proxy/session/2"), loop.create_future())
                )
        finally:
            # Clean up pending futures
            while not router._queue.empty():
                _, f = router._queue.get_nowait()
                if not f.done():
                    f.set_result(DispatchResponse(status="error", error="cleanup"))
            if router._session and not router._session.closed:
                await router._session.close()
                router._session = None

    @pytest.mark.asyncio
    async def test_enqueue_returns_dispatch_response_on_success(self):
        """enqueue() should return DispatchResponse when worker succeeds."""
        router, pool = await _make_router_with_worker()

        success_resp = DispatchResponse(status="success", result=1.0)
        with patch.object(
            router, "_post_to_worker", new=AsyncMock(return_value=success_resp)
        ):
            try:
                response = await asyncio.wait_for(
                    router.enqueue(_make_request()), timeout=3.0
                )
            finally:
                await router.stop()

        assert response.status == "success"
        assert response.result == 1.0

    @pytest.mark.asyncio
    async def test_enqueue_marks_worker_busy_then_idle(self):
        """Worker should be marked busy during dispatch and idle after."""
        router, pool = await _make_router_with_worker()

        busy_during_call = False

        async def mock_post(
            worker: WorkerInfo, request: DispatchRequest
        ) -> DispatchResponse:
            nonlocal busy_during_call
            stats = await pool.get_stats()
            busy_during_call = stats["busy"] == 1
            return DispatchResponse(status="success", result=0.5)

        with patch.object(router, "_post_to_worker", new=mock_post):
            try:
                await asyncio.wait_for(router.enqueue(_make_request()), timeout=3.0)
            finally:
                await router.stop()

        assert busy_during_call
        # After dispatch, worker should be idle again
        stats = await pool.get_stats()
        assert stats["idle"] == 1
        assert stats["busy"] == 0


class TestRouterFailureHandling:
    """Tests for Router failure and retry behavior."""

    @pytest.mark.asyncio
    async def test_worker_failure_marks_unhealthy_and_retries(self):
        """On worker failure, worker should be marked unhealthy and request requeued."""
        pool = WorkerPoolManager()
        await pool.register_worker("w1", "http://localhost:9001")
        await pool.register_worker("w2", "http://localhost:9002")
        router = Router(pool=pool, queue_size=10, worker_timeout=5.0)
        await router.start()

        call_count = 0

        async def mock_post(
            worker: WorkerInfo, request: DispatchRequest
        ) -> DispatchResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("Worker w1 unreachable")
            return DispatchResponse(status="success", result=0.9)

        with patch.object(router, "_post_to_worker", new=mock_post):
            try:
                response = await asyncio.wait_for(
                    router.enqueue(_make_request()), timeout=5.0
                )
            finally:
                await router.stop()

        # First call failed, second succeeded (retry)
        assert call_count == 2
        assert response.status == "success"
        assert response.result == 0.9

    @pytest.mark.asyncio
    async def test_double_failure_resolves_with_error(self):
        """If both attempts fail, future should be resolved with error response."""
        # Register 2 workers so the retry can find a second one (which also fails)
        pool = WorkerPoolManager()
        await pool.register_worker("w1", "http://localhost:9001")
        await pool.register_worker("w2", "http://localhost:9002")
        router = Router(pool=pool, queue_size=10, worker_timeout=5.0)
        await router.start()

        async def always_fail(
            worker: WorkerInfo, request: DispatchRequest
        ) -> DispatchResponse:
            raise ConnectionError("Always fails")

        with patch.object(router, "_post_to_worker", new=always_fail):
            try:
                response = await asyncio.wait_for(
                    router.enqueue(_make_request()), timeout=5.0
                )
            finally:
                await router.stop()

        assert response.status == "error"
        assert response.error is not None
        assert "failed after retry" in response.error

    @pytest.mark.asyncio
    async def test_worker_failure_marks_worker_unhealthy(self):
        """Failed worker should be marked unhealthy in the pool."""
        router, pool = await _make_router_with_worker()

        async def always_fail(
            worker: WorkerInfo, request: DispatchRequest
        ) -> DispatchResponse:
            raise ConnectionError("Unreachable")

        with patch.object(router, "_post_to_worker", new=always_fail):
            try:
                await asyncio.wait_for(router.enqueue(_make_request()), timeout=5.0)
            except Exception:
                pass
            finally:
                await router.stop()

        stats = await pool.get_stats()
        assert stats["unhealthy"] >= 1
