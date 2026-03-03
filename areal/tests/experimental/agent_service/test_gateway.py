"""Tests for the stateless HTTP↔ZMQ Gateway.

Verifies: submit, result polling, 404 for unknown tasks, health, metrics, configure.
Uses real ZMQ sockets (PULL absorber + PUSH injector) to avoid mocking transport.
"""

from __future__ import annotations

import asyncio

import pytest
import pytest_asyncio
import zmq
from httpx import ASGITransport, AsyncClient

from areal.experimental.agent_service.gateway import create_app
from areal.utils.network import find_free_ports

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Valid body for POST /submit
_EPISODE_BODY = {
    "data": {"question": "What is 2+2?"},
    "session_url": "http://localhost:8000",
}


@pytest.fixture
def zmq_addrs():
    """Allocate free ports and bind absorber/injector ZMQ sockets.

    * absorber (PULL) — binds to req_frontend; absorbs Gateway PUSH messages.
    * injector (PUSH) — binds to res_backend; injects results to Gateway PULL.
    """
    ports = find_free_ports(2)
    req_frontend = f"tcp://127.0.0.1:{ports[0]}"
    res_backend = f"tcp://127.0.0.1:{ports[1]}"

    ctx = zmq.Context()

    absorber = ctx.socket(zmq.PULL)
    absorber.setsockopt(zmq.LINGER, 0)
    absorber.setsockopt(zmq.RCVTIMEO, 1000)
    absorber.bind(req_frontend)

    injector = ctx.socket(zmq.PUSH)
    injector.setsockopt(zmq.LINGER, 0)
    injector.bind(res_backend)

    yield req_frontend, res_backend, absorber, injector

    absorber.close()
    injector.close()
    ctx.term()


@pytest_asyncio.fixture
async def gateway_client(zmq_addrs):
    """Create a Gateway FastAPI app and yield (client, absorber, injector)."""
    req_frontend, res_backend, absorber, injector = zmq_addrs
    app = create_app(
        req_frontend_addr=req_frontend,
        res_backend_addr=res_backend,
        task_timeout=5.0,
        result_ttl=60.0,
    )
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client, absorber, injector


# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------


class TestGatewaySubmit:
    """Tests for POST /submit endpoint."""

    @pytest.mark.asyncio
    async def test_submit_returns_task_id(self, gateway_client):
        """Test that POST /submit returns 200 with task_id and status=submitted."""
        client, _absorber, _injector = gateway_client

        resp = await client.post("/submit", json=_EPISODE_BODY)

        assert resp.status_code == 200
        body = resp.json()
        assert body["task_id"]  # non-empty string
        assert body["status"] == "submitted"

    @pytest.mark.asyncio
    async def test_submit_increments_metrics(self, gateway_client):
        """Test that submitting 3 tasks increments total_submitted to 3."""
        client, _absorber, _injector = gateway_client

        for _ in range(3):
            resp = await client.post("/submit", json=_EPISODE_BODY)
            assert resp.status_code == 200

        resp = await client.get("/metrics")
        assert resp.status_code == 200
        assert resp.json()["total_submitted"] == 3


# ---------------------------------------------------------------------------
# Result polling
# ---------------------------------------------------------------------------


class TestGatewayResult:
    """Tests for GET /result/{task_id} endpoint."""

    @pytest.mark.asyncio
    async def test_result_pending_before_completion(self, gateway_client):
        """Test that GET /result returns pending before any result arrives."""
        client, _absorber, _injector = gateway_client

        resp = await client.post("/submit", json=_EPISODE_BODY)
        task_id = resp.json()["task_id"]

        resp = await client.get(f"/result/{task_id}")

        assert resp.status_code == 200
        body = resp.json()
        assert body["task_id"] == task_id
        assert body["status"] == "pending"

    @pytest.mark.asyncio
    async def test_result_completed_after_result_arrives(self, gateway_client):
        """Test that GET /result returns completed after ZMQ result injection."""
        client, _absorber, injector = gateway_client

        resp = await client.post("/submit", json=_EPISODE_BODY)
        task_id = resp.json()["task_id"]

        # Inject result via ZMQ PUSH → Gateway's receiver thread (PULL)
        injector.send_json({"task_id": task_id, "result": 1.0})
        await asyncio.sleep(0.2)  # let receiver thread process

        resp = await client.get(f"/result/{task_id}")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "completed"
        assert body["result"] == 1.0

    @pytest.mark.asyncio
    async def test_result_unknown_task_id_returns_404(self, gateway_client):
        """Test that GET /result with unknown task_id returns 404."""
        client, _absorber, _injector = gateway_client

        resp = await client.get("/result/nonexistent-id")

        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_result_not_evicted_before_retrieval(self):
        """Test that a result completed near the TTL boundary is not prematurely evicted.

        Uses a short result_ttl (2.0s) and a short task_timeout (10.0s).
        Submits a task, injects a result at ~0.5s, waits 1.5s more (total ~2.0s from
        submission but only ~1.5s from completion), then asserts the result is still
        accessible (not 404).
        """
        ports = find_free_ports(2)
        req_frontend = f"tcp://127.0.0.1:{ports[0]}"
        res_backend = f"tcp://127.0.0.1:{ports[1]}"

        ctx = zmq.Context()
        absorber = ctx.socket(zmq.PULL)
        absorber.setsockopt(zmq.LINGER, 0)
        absorber.setsockopt(zmq.RCVTIMEO, 1000)
        absorber.bind(req_frontend)

        injector = ctx.socket(zmq.PUSH)
        injector.setsockopt(zmq.LINGER, 0)
        injector.bind(res_backend)

        app = create_app(
            req_frontend_addr=req_frontend,
            res_backend_addr=res_backend,
            task_timeout=10.0,
            result_ttl=2.0,
        )
        try:
            async with app.router.lifespan_context(app):
                transport = ASGITransport(app=app)
                async with AsyncClient(
                    transport=transport, base_url="http://test"
                ) as client:
                    # Submit task
                    resp = await client.post("/submit", json=_EPISODE_BODY)
                    assert resp.status_code == 200
                    task_id = resp.json()["task_id"]

                    # Wait 0.5s, then inject result (simulates slow task)
                    await asyncio.sleep(0.5)
                    injector.send_json({"task_id": task_id, "result": 42.0})
                    await asyncio.sleep(0.2)  # let receiver thread process

                    # Wait 1.5s more — total 2.2s from submission, but only 1.7s from completion
                    # With TTL measured from completed_at, result should still be alive
                    await asyncio.sleep(1.5)

                    resp = await client.get(f"/result/{task_id}")
                    assert resp.status_code == 200, (
                        f"Result was prematurely evicted (status={resp.status_code}). "
                        "TTL should be measured from completion time, not submission time."
                    )
                    assert resp.json()["status"] == "completed"
        finally:
            absorber.close()
            injector.close()
            ctx.term()


# ---------------------------------------------------------------------------
# Health / Metrics / Configure
# ---------------------------------------------------------------------------


class TestGatewayEndpoints:
    """Tests for /health, /metrics, and /configure endpoints."""

    @pytest.mark.asyncio
    async def test_health_endpoint(self, gateway_client):
        """Test that GET /health returns ok with pending and completed counts."""
        client, _absorber, _injector = gateway_client

        resp = await client.get("/health")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "pending" in body
        assert "completed" in body

    @pytest.mark.asyncio
    async def test_metrics_endpoint(self, gateway_client):
        """Test that GET /metrics returns the correct structure."""
        client, _absorber, _injector = gateway_client

        resp = await client.get("/metrics")

        assert resp.status_code == 200
        body = resp.json()
        assert "total_submitted" in body
        assert "total_completed" in body
        assert "total_errors" in body
        assert "avg_latency_s" in body

    @pytest.mark.asyncio
    async def test_configure_endpoint(self, gateway_client):
        """Test that POST /configure returns ok."""
        client, _absorber, _injector = gateway_client

        resp = await client.post("/configure")

        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# Metrics double-counting check
# ---------------------------------------------------------------------------


class TestMetricsNotDoubleCounted:
    """Tests to verify metrics are not double-counted on repeated polling."""

    @pytest.mark.asyncio
    async def test_metrics_not_double_counted(self, gateway_client):
        """Test that polling /result/{task_id} 5 times doesn't increment metrics 5x."""
        client, _absorber, injector = gateway_client

        # Submit a task
        resp = await client.post("/submit", json=_EPISODE_BODY)
        assert resp.status_code == 200
        task_id = resp.json()["task_id"]

        # Inject completed result
        injector.send_json({"task_id": task_id, "result": 1.5})
        await asyncio.sleep(0.2)  # let receiver thread process

        # Poll /result/{task_id} 5 times
        for i in range(5):
            resp = await client.get(f"/result/{task_id}")
            assert resp.status_code == 200
            body = resp.json()
            assert body["status"] == "completed"
            await asyncio.sleep(0.05)  # small delay between polls

        # Verify metrics: total_completed should be 1, not 5
        resp = await client.get("/metrics")
        assert resp.status_code == 200
        metrics = resp.json()
        assert metrics["total_completed"] == 1, (
            f"Expected total_completed=1 after 5 polls, got {metrics['total_completed']}"
        )
        # Verify latency is reasonable (not 5x inflated)
        avg_latency = metrics["avg_latency_s"]
        assert avg_latency > 0, "Average latency should be positive"
        assert avg_latency < 10.0, "Average latency seems inflated"
