"""Integration tests for multiple WorkerZMQ instances with a real Router.

Unlike test_integration.py (which uses simulated raw-ZMQ workers), these
tests instantiate *real* :class:`WorkerZMQ` objects backed by a stub
AgentService, verifying multi-worker coordination through the Router.

Architecture under test::

    Test PUSH → Router PULL → ROUTER/DEALER → WorkerZMQ (DEALER)
                                                    ↓
    Test PULL ← Router PUSH ← PULL ← WorkerZMQ (PUSH)

ZMQ port mapping (5 ports from ``find_free_ports``):
  [0] req_frontend  — Test PUSH  → Router PULL
  [1] req_backend   — Router ROUTER → WorkerZMQ DEALER
  [2] res_frontend  — WorkerZMQ PUSH → Router PULL
  [3] res_backend   — Router PUSH   → Test PULL
  [4] Router HTTP health port (required by constructor)
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest
import zmq

from areal.experimental.agent_service.router import RoundRobinStrategy, Router
from areal.experimental.agent_service.worker_server import WorkerZMQ
from areal.utils.network import find_free_ports

# ---------------------------------------------------------------------------
# Stub AgentService
# ---------------------------------------------------------------------------


class _MockService:
    """Stub AgentService for testing multi-worker scenarios."""

    def __init__(
        self,
        return_value: float = 1.0,
        raise_error: Exception | None = None,
        delay: float = 0.0,
    ) -> None:
        self.return_value = return_value
        self.raise_error = raise_error
        self.delay = delay
        self.call_count = 0

    async def run_episode(
        self,
        data,
        session_url,
        agent_kwargs=None,
        agent_import_path=None,
    ):
        self.call_count += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.raise_error:
            raise self.raise_error
        return self.return_value


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
    time.sleep(0.3)  # let threads spin up and sockets bind
    yield r
    r.stop()


def _start_worker(
    zmq_ports: list[int],
    service: _MockService,
    worker_id: str,
) -> tuple[WorkerZMQ, threading.Thread]:
    """Create a WorkerZMQ and run it in a daemon thread.

    Returns the worker instance and its thread so the caller can
    stop the worker and join the thread during teardown.
    """
    worker = WorkerZMQ(
        task_addr=f"tcp://127.0.0.1:{zmq_ports[1]}",
        result_addr=f"tcp://127.0.0.1:{zmq_ports[2]}",
        service=service,
        worker_id=worker_id,
    )
    thread = threading.Thread(target=worker.start, daemon=True, name=f"t-{worker_id}")
    thread.start()
    return worker, thread


def _inject_tasks(zmq_ports: list[int], count: int) -> list[str]:
    """Push *count* tasks into the Router's req_frontend and return task IDs."""
    ctx = zmq.Context()
    push = ctx.socket(zmq.PUSH)
    push.setsockopt(zmq.LINGER, 0)
    push.connect(f"tcp://127.0.0.1:{zmq_ports[0]}")
    time.sleep(0.1)  # let socket connect

    task_ids: list[str] = []
    for i in range(count):
        tid = f"task-{i}"
        push.send_json(
            {
                "task_id": tid,
                "data": {"question": f"q{i}"},
                "session_url": "http://localhost:8000",
            }
        )
        task_ids.append(tid)

    push.close()
    ctx.term()
    return task_ids


def _collect_results(
    zmq_ports: list[int], expected: int, timeout: float = 10.0
) -> list[dict]:
    """Pull up to *expected* results from the Router's res_backend."""
    ctx = zmq.Context()
    pull = ctx.socket(zmq.PULL)
    pull.setsockopt(zmq.LINGER, 0)
    pull.setsockopt(zmq.RCVTIMEO, int(timeout * 1000))
    pull.connect(f"tcp://127.0.0.1:{zmq_ports[3]}")
    time.sleep(0.1)  # let socket connect

    results: list[dict] = []
    for _ in range(expected):
        try:
            data = pull.recv_json()
            results.append(data)
        except zmq.Again:
            break

    pull.close()
    ctx.term()
    return results


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMultiWorkerIntegration:
    """Integration tests for multiple WorkerZMQ instances with a real Router."""

    def test_two_workers_each_process_tasks(self, zmq_ports, router):
        """Test that 2 WorkerZMQ instances process 4 tasks between them."""
        svc_a = _MockService(return_value=1.0)
        svc_b = _MockService(return_value=2.0)

        w_a, t_a = _start_worker(zmq_ports, svc_a, "worker-a")
        w_b, t_b = _start_worker(zmq_ports, svc_b, "worker-b")
        time.sleep(0.5)  # let workers register with Router

        task_ids = _inject_tasks(zmq_ports, 4)
        results = _collect_results(zmq_ports, expected=4)

        # All 4 tasks completed
        assert len(results) == 4
        returned_task_ids = sorted(r["task_id"] for r in results)
        assert returned_task_ids == sorted(task_ids)
        # Every result is a success
        for r in results:
            assert r["status"] == "success"

        # Both workers were called
        assert svc_a.call_count + svc_b.call_count == 4
        assert svc_a.call_count > 0
        assert svc_b.call_count > 0

        # Teardown
        w_a.stop()
        w_b.stop()
        t_a.join(timeout=5)
        t_b.join(timeout=5)

    def test_three_workers_process_six_tasks(self, zmq_ports, router):
        """Test that 3 workers share 6 tasks successfully."""
        services = [_MockService(return_value=float(i)) for i in range(3)]

        workers_threads = [
            _start_worker(zmq_ports, services[i], f"worker-{i}") for i in range(3)
        ]
        time.sleep(0.5)

        task_ids = _inject_tasks(zmq_ports, 6)
        results = _collect_results(zmq_ports, expected=6)

        assert len(results) == 6
        returned_ids = sorted(r["task_id"] for r in results)
        assert returned_ids == sorted(task_ids)

        total_calls = sum(s.call_count for s in services)
        assert total_calls == 6
        # With round-robin each of the 3 workers should get exactly 2 tasks
        for s in services:
            assert s.call_count == 2

        for w, t in workers_threads:
            w.stop()
            t.join(timeout=5)

    def test_tasks_distributed_across_workers_by_return_value(self, zmq_ports, router):
        """Test round-robin distribution by checking distinct return values."""
        svc_a = _MockService(return_value=10.0)
        svc_b = _MockService(return_value=20.0)

        w_a, t_a = _start_worker(zmq_ports, svc_a, "worker-a")
        w_b, t_b = _start_worker(zmq_ports, svc_b, "worker-b")
        time.sleep(0.5)

        _inject_tasks(zmq_ports, 4)
        results = _collect_results(zmq_ports, expected=4)

        assert len(results) == 4
        result_values = sorted(set(r["result"] for r in results))
        # Both return values must appear — proving distribution
        assert 10.0 in result_values
        assert 20.0 in result_values

        # With round-robin, each worker should get exactly 2 tasks
        count_10 = sum(1 for r in results if r["result"] == 10.0)
        count_20 = sum(1 for r in results if r["result"] == 20.0)
        assert count_10 == 2
        assert count_20 == 2

        w_a.stop()
        w_b.stop()
        t_a.join(timeout=5)
        t_b.join(timeout=5)

    def test_worker_error_does_not_block_other_workers(self, zmq_ports, router):
        """Test that one worker's errors don't prevent others from succeeding."""
        svc_ok = _MockService(return_value=42.0)
        svc_err = _MockService(raise_error=RuntimeError("boom"))

        w_ok, t_ok = _start_worker(zmq_ports, svc_ok, "worker-ok")
        w_err, t_err = _start_worker(zmq_ports, svc_err, "worker-err")
        time.sleep(0.5)

        _inject_tasks(zmq_ports, 2)
        results = _collect_results(zmq_ports, expected=2)

        assert len(results) == 2

        statuses = {r["status"] for r in results}
        assert "success" in statuses
        assert "error" in statuses

        success_results = [r for r in results if r["status"] == "success"]
        error_results = [r for r in results if r["status"] == "error"]

        assert len(success_results) == 1
        assert success_results[0]["result"] == 42.0

        assert len(error_results) == 1
        assert "boom" in error_results[0]["error"]

        w_ok.stop()
        w_err.stop()
        t_ok.join(timeout=5)
        t_err.join(timeout=5)

    def test_workers_stop_cleanly_router_survives(self, zmq_ports, router):
        """Test that workers stop cleanly and the Router remains alive."""
        svc = _MockService(return_value=99.0)

        w1, t1 = _start_worker(zmq_ports, svc, "worker-lifecycle")
        time.sleep(0.5)

        # Process one task
        _inject_tasks(zmq_ports, 1)
        results = _collect_results(zmq_ports, expected=1)

        assert len(results) == 1
        assert results[0]["status"] == "success"
        assert results[0]["result"] == 99.0

        # Stop the worker
        w1.stop()
        t1.join(timeout=5)
        assert not t1.is_alive(), "Worker thread did not exit cleanly"

        # Router should still be running
        assert router._running is True

        # Start a fresh worker and process another task
        svc2 = _MockService(return_value=100.0)
        w2, t2 = _start_worker(zmq_ports, svc2, "worker-lifecycle-2")
        time.sleep(0.5)

        _inject_tasks(zmq_ports, 1)
        results2 = _collect_results(zmq_ports, expected=1)

        assert len(results2) == 1
        assert results2[0]["status"] == "success"
        assert results2[0]["result"] == 100.0

        w2.stop()
        t2.join(timeout=5)
