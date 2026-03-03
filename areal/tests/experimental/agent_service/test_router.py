"""Tests for the ZMQ Router module."""

from __future__ import annotations

import json
import time

import pytest
import zmq

from areal.experimental.agent_service.router import RoundRobinStrategy, Router
from areal.utils.network import find_free_ports

# ---------------------------------------------------------------------------
# Unit tests for RoutingStrategy
# ---------------------------------------------------------------------------


class TestRoundRobinStrategy:
    """Tests for the RoundRobinStrategy worker selector."""

    def test_round_robin_strategy_cycles_through_workers(self):
        """Test that select_worker cycles w1, w2, w1, w2 with 2 workers."""
        strategy = RoundRobinStrategy()
        workers = ["w1", "w2"]
        task = {"task_id": "t1"}

        results = [strategy.select_worker(task, workers) for _ in range(4)]

        assert results == ["w1", "w2", "w1", "w2"]

    def test_round_robin_no_workers_raises(self):
        """Test that select_worker raises RuntimeError with empty list."""
        strategy = RoundRobinStrategy()

        with pytest.raises(RuntimeError, match="No workers registered"):
            strategy.select_worker({"task_id": "t1"}, [])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def zmq_ctx():
    """Create a ZMQ context for test sockets, cleaned up after use."""
    ctx = zmq.Context()
    yield ctx
    ctx.term()


@pytest.fixture
def router_addrs():
    """Allocate 4 free ports and return Router bind addresses."""
    ports = find_free_ports(4)
    return {
        "req_frontend_addr": f"tcp://127.0.0.1:{ports[0]}",
        "req_backend_addr": f"tcp://127.0.0.1:{ports[1]}",
        "res_frontend_addr": f"tcp://127.0.0.1:{ports[2]}",
        "res_backend_addr": f"tcp://127.0.0.1:{ports[3]}",
    }


@pytest.fixture
def router(router_addrs):
    """Start a Router and yield it; stop on teardown."""
    r = Router(**router_addrs)
    r.start()
    yield r
    r.stop()


# ---------------------------------------------------------------------------
# Router lifecycle tests
# ---------------------------------------------------------------------------


class TestRouterLifecycle:
    """Tests for Router start/stop lifecycle."""

    def test_router_start_stop(self, router_addrs):
        """Test that Router starts and stops cleanly."""
        r = Router(**router_addrs)

        r.start()
        assert r._running is True

        r.stop()
        assert r._running is False


# ---------------------------------------------------------------------------
# Router message forwarding tests
# ---------------------------------------------------------------------------


class TestRouterForwarding:
    """Tests for Router request/result forwarding."""

    def test_router_forwards_requests_to_worker(self, router, router_addrs, zmq_ctx):
        """Test that a request from Gateway reaches a connected worker."""
        # Worker: DEALER connects to req_backend
        dealer = zmq_ctx.socket(zmq.DEALER)
        dealer.setsockopt(zmq.IDENTITY, b"worker-1")
        dealer.setsockopt(zmq.RCVTIMEO, 2000)
        dealer.setsockopt(zmq.LINGER, 0)
        dealer.connect(router_addrs["req_backend_addr"])

        # Send READY handshake
        dealer.send_multipart([b"", json.dumps({"type": "READY"}).encode()])
        time.sleep(0.2)

        # Gateway: PUSH connects to req_frontend
        push = zmq_ctx.socket(zmq.PUSH)
        push.setsockopt(zmq.LINGER, 0)
        push.connect(router_addrs["req_frontend_addr"])

        # Send a task
        push.send_json({"task_id": "test-1", "data": {}})

        # Worker should receive the task
        frames = dealer.recv_multipart()
        # DEALER receives [b"", payload]
        assert len(frames) >= 2
        payload = json.loads(frames[-1])
        assert payload["task_id"] == "test-1"

        push.close()
        dealer.close()

    def test_router_forwards_results_to_gateway(self, router, router_addrs, zmq_ctx):
        """Test that a result from Worker reaches the Gateway result receiver."""
        # Worker result sender: PUSH connects to res_frontend
        push = zmq_ctx.socket(zmq.PUSH)
        push.setsockopt(zmq.LINGER, 0)
        push.connect(router_addrs["res_frontend_addr"])

        # Gateway result receiver: PULL connects to res_backend
        pull = zmq_ctx.socket(zmq.PULL)
        pull.setsockopt(zmq.RCVTIMEO, 2000)
        pull.setsockopt(zmq.LINGER, 0)
        pull.connect(router_addrs["res_backend_addr"])

        # Allow sockets to connect
        time.sleep(0.1)

        # Send result
        push.send_json({"task_id": "test-1", "result": 1.0})

        # Gateway should receive the result
        data = pull.recv_json()
        assert data["task_id"] == "test-1"
        assert data["result"] == 1.0

        push.close()
        pull.close()

    def test_router_multiple_workers_round_robin(self, router, router_addrs, zmq_ctx):
        """Test that tasks are distributed round-robin across 2 workers."""
        # Connect 2 DEALER workers
        dealers = []
        for i in range(2):
            d = zmq_ctx.socket(zmq.DEALER)
            d.setsockopt(zmq.IDENTITY, f"worker-{i}".encode())
            d.setsockopt(zmq.RCVTIMEO, 2000)
            d.setsockopt(zmq.LINGER, 0)
            d.connect(router_addrs["req_backend_addr"])
            d.send_multipart([b"", json.dumps({"type": "READY"}).encode()])
            dealers.append(d)

        time.sleep(0.3)

        # Gateway: PUSH
        push = zmq_ctx.socket(zmq.PUSH)
        push.setsockopt(zmq.LINGER, 0)
        push.connect(router_addrs["req_frontend_addr"])

        # Send 4 tasks
        for i in range(4):
            push.send_json({"task_id": f"task-{i}", "data": {}})

        time.sleep(0.3)

        # Collect received tasks per worker
        received = {0: [], 1: []}
        for idx, d in enumerate(dealers):
            while True:
                try:
                    frames = d.recv_multipart(flags=zmq.NOBLOCK)
                    payload = json.loads(frames[-1])
                    received[idx].append(payload["task_id"])
                except zmq.Again:
                    break

        # Each worker should get exactly 2 tasks (round-robin)
        assert len(received[0]) == 2, f"worker-0 got {received[0]}"
        assert len(received[1]) == 2, f"worker-1 got {received[1]}"

        push.close()
        for d in dealers:
            d.close()

    def test_task_queued_before_workers(self, router_addrs, zmq_ctx):
        """Test that tasks sent before any worker READY are queued and drained."""
        r = Router(**router_addrs)
        r.start()
        try:
            # Gateway: PUSH connects to req_frontend
            push = zmq_ctx.socket(zmq.PUSH)
            push.setsockopt(zmq.LINGER, 0)
            push.connect(router_addrs["req_frontend_addr"])
            time.sleep(0.1)

            # Send a task BEFORE any worker sends READY
            push.send_json({"task_id": "early-1", "data": {}})
            time.sleep(0.3)

            # Task should be in the pending queue
            assert len(r._pending_queue) == 1, (
                f"Expected 1 pending task, got {len(r._pending_queue)}"
            )
            assert r._pending_queue[0]["task_id"] == "early-1"

            # Worker: DEALER connects and sends READY
            dealer = zmq_ctx.socket(zmq.DEALER)
            dealer.setsockopt(zmq.IDENTITY, b"worker-late")
            dealer.setsockopt(zmq.RCVTIMEO, 2000)
            dealer.setsockopt(zmq.LINGER, 0)
            dealer.connect(router_addrs["req_backend_addr"])
            dealer.send_multipart([b"", json.dumps({"type": "READY"}).encode()])
            time.sleep(0.5)

            # Pending queue should now be drained
            assert len(r._pending_queue) == 0, (
                f"Expected 0 pending tasks after drain, got {len(r._pending_queue)}"
            )

            # Worker should have received the queued task
            frames = dealer.recv_multipart()
            assert len(frames) >= 2
            payload = json.loads(frames[-1])
            assert payload["task_id"] == "early-1"

            push.close()
            dealer.close()
        finally:
            r.stop()
