"""Integration tests for the full Gateway → Router → Worker flow.

Tests the complete ZMQ request path:
  Client → Gateway (HTTP) → [ZMQ PUSH] → Router (PULL)
         → [ZMQ ROUTER/DEALER] → Simulated Worker (DEALER)
  Client ← Gateway (HTTP) ← [ZMQ PULL] ← Router (PUSH)
         ← [ZMQ PUSH] ← Simulated Worker (PUSH)

Workers are simulated with raw ZMQ DEALER+PUSH sockets — no real
AgentService instances are needed.  This keeps tests self-contained.

ZMQ port mapping (indices into ``zmq_ports`` fixture):
  [0] req_frontend  — Gateway PUSH  → Router PULL
  [1] req_backend   — Router ROUTER → Worker DEALER
  [2] res_frontend  — Worker PUSH   → Router PULL
  [3] res_backend   — Router PUSH   → Gateway PULL
  [4] Router HTTP health port
"""

from __future__ import annotations

import asyncio
import json
import threading
import time

import pytest
import pytest_asyncio
import zmq
from httpx import ASGITransport, AsyncClient

from areal.experimental.agent_service.gateway import create_app
from areal.experimental.agent_service.router import RoundRobinStrategy, Router
from areal.utils.network import find_free_ports

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EPISODE_BODY = {
    "data": {"question": "What is 2+2?"},
    "session_url": "http://localhost:8000",
}


# ---------------------------------------------------------------------------
# Simulated worker (raw ZMQ sockets, no AgentService)
# ---------------------------------------------------------------------------


def _run_simulated_worker(
    zmq_ports: list[int],
    worker_id: str,
    max_tasks: int = 1,
    result_value: float = 1.0,
    error_msg: str | None = None,
    received_tasks: list | None = None,
) -> None:
    """Simulated worker thread: DEALER + PUSH, processes up to *max_tasks*.

    Parameters
    ----------
    zmq_ports : list[int]
        Port list from the ``zmq_ports`` fixture.
    worker_id : str
        ZMQ DEALER identity string.
    max_tasks : int
        Number of tasks to process before exiting.
    result_value : float
        Value to return in successful results.
    error_msg : str | None
        If set, send an error result instead of a success result.
    received_tasks : list | None
        If provided, ``(worker_id, task_id)`` tuples are appended here.
    """
    ctx = zmq.Context()

    # DEALER — receives tasks from Router's ROUTER socket
    dealer = ctx.socket(zmq.DEALER)
    dealer.setsockopt(zmq.IDENTITY, worker_id.encode())
    dealer.setsockopt(zmq.LINGER, 0)
    dealer.setsockopt(zmq.RCVTIMEO, 5000)
    dealer.connect(f"tcp://127.0.0.1:{zmq_ports[1]}")  # req_backend

    # PUSH — sends results to Router's res_frontend (PULL)
    push = ctx.socket(zmq.PUSH)
    push.setsockopt(zmq.LINGER, 0)
    push.connect(f"tcp://127.0.0.1:{zmq_ports[2]}")  # res_frontend

    # Register with Router
    dealer.send_multipart([b"", json.dumps({"type": "READY"}).encode()])

    processed = 0
    try:
        while processed < max_tasks:
            try:
                frames = dealer.recv_multipart()
            except zmq.Again:
                break  # timed out waiting for a task

            if len(frames) < 2:
                continue

            task = json.loads(frames[1])
            task_id = task.get("task_id")

            if received_tasks is not None:
                received_tasks.append((worker_id, task_id))

            if error_msg:
                push.send_json(
                    {"task_id": task_id, "status": "error", "error": error_msg}
                )
            else:
                push.send_json({"task_id": task_id, "result": result_value})

            processed += 1
    finally:
        dealer.close()
        push.close()
        ctx.term()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def zmq_ports():
    """Allocate 5 free ports: 4 ZMQ channels + 1 Router HTTP."""
    return find_free_ports(5)


@pytest.fixture
def router(zmq_ports):
    """Start a Router bound to the allocated ZMQ ports."""
    p = zmq_ports
    r = Router(
        req_frontend_addr=f"tcp://127.0.0.1:{p[0]}",
        req_backend_addr=f"tcp://127.0.0.1:{p[1]}",
        res_frontend_addr=f"tcp://127.0.0.1:{p[2]}",
        res_backend_addr=f"tcp://127.0.0.1:{p[3]}",
        strategy=RoundRobinStrategy(),
        http_port=p[4],
    )
    r.start()
    time.sleep(0.2)  # let threads spin up and sockets bind
    yield r
    r.stop()


@pytest_asyncio.fixture
async def gateway_client(zmq_ports, router):
    """Start a Gateway connected to the Router, yield ``(client, zmq_ports)``."""
    p = zmq_ports
    app = create_app(
        req_frontend_addr=f"tcp://127.0.0.1:{p[0]}",
        res_backend_addr=f"tcp://127.0.0.1:{p[3]}",
        task_timeout=10.0,
        result_ttl=60.0,
    )
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client, zmq_ports


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _poll_until_done(
    client: AsyncClient,
    task_id: str,
    max_iters: int = 30,
    interval: float = 0.1,
) -> dict:
    """Poll ``GET /result/{task_id}`` until status != pending or timeout."""
    body: dict = {}
    for _ in range(max_iters):
        await asyncio.sleep(interval)
        resp = await client.get(f"/result/{task_id}")
        assert resp.status_code == 200
        body = resp.json()
        if body["status"] != "pending":
            return body
    return body


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFullFlow:
    """Integration tests for the complete Gateway → Router → Worker pipeline."""

    @pytest.mark.asyncio
    async def test_full_flow_submit_to_result(self, gateway_client):
        """Test full round-trip: submit → route → worker → result."""
        client, zmq_ports = gateway_client

        # Start a simulated worker that returns 42.0
        worker_thread = threading.Thread(
            target=_run_simulated_worker,
            args=(zmq_ports, "worker-1"),
            kwargs={"result_value": 42.0},
            daemon=True,
        )
        worker_thread.start()
        time.sleep(0.3)  # let Router register the worker

        # Submit a task via Gateway HTTP
        resp = await client.post("/submit", json=_EPISODE_BODY)
        assert resp.status_code == 200
        task_id = resp.json()["task_id"]
        assert task_id  # non-empty

        # Poll until the result arrives
        body = await _poll_until_done(client, task_id)

        assert body["status"] == "completed"
        assert body["result"] == 42.0

        worker_thread.join(timeout=5)

    @pytest.mark.asyncio
    async def test_multiple_workers_round_robin(self, gateway_client):
        """Test that 4 tasks are distributed across 2 workers via round-robin."""
        client, zmq_ports = gateway_client
        received_tasks: list[tuple[str, str]] = []

        # Start 2 simulated workers, each handles up to 2 tasks
        w1 = threading.Thread(
            target=_run_simulated_worker,
            args=(zmq_ports, "w1"),
            kwargs={
                "max_tasks": 2,
                "result_value": 10.0,
                "received_tasks": received_tasks,
            },
            daemon=True,
        )
        w2 = threading.Thread(
            target=_run_simulated_worker,
            args=(zmq_ports, "w2"),
            kwargs={
                "max_tasks": 2,
                "result_value": 20.0,
                "received_tasks": received_tasks,
            },
            daemon=True,
        )
        w1.start()
        w2.start()
        time.sleep(0.5)  # let both workers register with Router

        # Submit 4 tasks
        task_ids: list[str] = []
        for _ in range(4):
            resp = await client.post("/submit", json=_EPISODE_BODY)
            assert resp.status_code == 200
            task_ids.append(resp.json()["task_id"])

        # Wait for worker threads to finish processing
        w1.join(timeout=10)
        w2.join(timeout=10)

        # Poll all results until completed
        for tid in task_ids:
            body = await _poll_until_done(client, tid)
            assert body["status"] == "completed", f"task {tid} status={body['status']}"

        # Verify round-robin distribution: each worker got exactly 2 tasks
        w1_tasks = [t for w, t in received_tasks if w == "w1"]
        w2_tasks = [t for w, t in received_tasks if w == "w2"]
        assert len(w1_tasks) == 2, f"w1 got {len(w1_tasks)} tasks, expected 2"
        assert len(w2_tasks) == 2, f"w2 got {len(w2_tasks)} tasks, expected 2"

    @pytest.mark.asyncio
    async def test_error_propagation(self, gateway_client):
        """Test that a worker error propagates back to the Gateway caller."""
        client, zmq_ports = gateway_client

        # Start a worker that returns errors
        worker_thread = threading.Thread(
            target=_run_simulated_worker,
            args=(zmq_ports, "error-worker"),
            kwargs={"error_msg": "intentional test error"},
            daemon=True,
        )
        worker_thread.start()
        time.sleep(0.3)  # let Router register the worker

        # Submit a task
        resp = await client.post("/submit", json=_EPISODE_BODY)
        assert resp.status_code == 200
        task_id = resp.json()["task_id"]

        # Poll until the error result arrives
        body = await _poll_until_done(client, task_id)

        assert body["status"] == "error"
        assert body["error"] == "intentional test error"

        worker_thread.join(timeout=5)
