"""Unit tests for the Router service (Plan 2, Task 3).

Tests worker registry, session registry, routing strategies,
and all router endpoints.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
import pytest_asyncio

from areal.experimental.gateway.router.app import create_app
from areal.experimental.gateway.router.config import RouterConfig
from areal.experimental.gateway.router.state import (
    CapacityManager,
    SessionRegistry,
    WorkerInfo,
    WorkerRegistry,
)
from areal.experimental.gateway.router.strategies import (
    LeastBusyStrategy,
    RoundRobinStrategy,
    get_strategy,
)

# =============================================================================
# Constants
# =============================================================================

ADMIN_KEY = "test-admin-key"
WORKER_1 = "http://worker-1:18082"
WORKER_2 = "http://worker-2:18082"
WORKER_3 = "http://worker-3:18082"


# =============================================================================
# WorkerRegistry unit tests
# =============================================================================


class TestWorkerRegistry:
    @pytest.mark.asyncio
    async def test_register_worker(self):
        reg = WorkerRegistry()
        await reg.register(WORKER_1)
        workers = await reg.get_all_workers()
        assert len(workers) == 1
        assert workers[0].worker_addr == WORKER_1
        assert workers[0].is_healthy is True

    @pytest.mark.asyncio
    async def test_register_duplicate_noop(self):
        reg = WorkerRegistry()
        await reg.register(WORKER_1)
        await reg.register(WORKER_1)
        workers = await reg.get_all_workers()
        assert len(workers) == 1

    @pytest.mark.asyncio
    async def test_deregister_worker(self):
        reg = WorkerRegistry()
        await reg.register(WORKER_1)
        await reg.deregister(WORKER_1)
        workers = await reg.get_all_workers()
        assert len(workers) == 0

    @pytest.mark.asyncio
    async def test_deregister_unknown_noop(self):
        reg = WorkerRegistry()
        await reg.deregister("http://unknown:9999")
        assert len(await reg.get_all_workers()) == 0

    @pytest.mark.asyncio
    async def test_health_update(self):
        reg = WorkerRegistry()
        await reg.register(WORKER_1)
        await reg.update_health(WORKER_1, False)
        workers = await reg.get_all_workers()
        assert workers[0].is_healthy is False

    @pytest.mark.asyncio
    async def test_get_healthy_workers(self):
        reg = WorkerRegistry()
        await reg.register(WORKER_1)
        await reg.register(WORKER_2)
        await reg.update_health(WORKER_1, False)
        healthy = await reg.get_healthy_workers()
        assert len(healthy) == 1
        assert healthy[0].worker_addr == WORKER_2

    @pytest.mark.asyncio
    async def test_get_all_workers(self):
        reg = WorkerRegistry()
        await reg.register(WORKER_1)
        await reg.register(WORKER_2)
        await reg.update_health(WORKER_1, False)
        all_w = await reg.get_all_workers()
        assert len(all_w) == 2

    @pytest.mark.asyncio
    async def test_list_worker_addrs(self):
        reg = WorkerRegistry()
        await reg.register(WORKER_1)
        await reg.register(WORKER_2)
        addrs = await reg.list_worker_addrs()
        assert set(addrs) == {WORKER_1, WORKER_2}


# =============================================================================
# SessionRegistry unit tests
# =============================================================================


class TestSessionRegistry:
    @pytest.mark.asyncio
    async def test_register_session(self):
        reg = SessionRegistry()
        await reg.register_session("key-1", "id-1", WORKER_1)
        assert await reg.lookup_by_key("key-1") == WORKER_1
        assert await reg.lookup_by_id("id-1") == WORKER_1

    @pytest.mark.asyncio
    async def test_lookup_by_key(self):
        reg = SessionRegistry()
        await reg.register_session("key-1", "id-1", WORKER_1)
        assert await reg.lookup_by_key("key-1") == WORKER_1

    @pytest.mark.asyncio
    async def test_lookup_by_id(self):
        reg = SessionRegistry()
        await reg.register_session("key-1", "id-1", WORKER_1)
        assert await reg.lookup_by_id("id-1") == WORKER_1

    @pytest.mark.asyncio
    async def test_lookup_unknown_key(self):
        reg = SessionRegistry()
        assert await reg.lookup_by_key("nonexistent") is None

    @pytest.mark.asyncio
    async def test_lookup_unknown_id(self):
        reg = SessionRegistry()
        assert await reg.lookup_by_id("nonexistent") is None

    @pytest.mark.asyncio
    async def test_revoke_by_worker(self):
        reg = SessionRegistry()
        await reg.register_session("key-1", "id-1", WORKER_1)
        await reg.register_session("key-2", "id-2", WORKER_1)
        await reg.register_session("key-3", "id-3", WORKER_2)  # different worker
        count = await reg.revoke_by_worker(WORKER_1)
        assert count == 2
        assert await reg.lookup_by_key("key-1") is None
        assert await reg.lookup_by_key("key-2") is None
        assert await reg.lookup_by_id("id-1") is None
        assert await reg.lookup_by_id("id-2") is None
        # Worker 2 sessions untouched
        assert await reg.lookup_by_key("key-3") == WORKER_2

    @pytest.mark.asyncio
    async def test_count(self):
        reg = SessionRegistry()
        assert await reg.count() == 0
        await reg.register_session("key-1", "id-1", WORKER_1)
        assert await reg.count() == 1
        await reg.register_session("key-2", "id-2", WORKER_2)
        assert await reg.count() == 2

    @pytest.mark.asyncio
    async def test_upsert_semantics(self):
        """Re-registering a session key updates the worker."""
        reg = SessionRegistry()
        await reg.register_session("key-1", "id-1", WORKER_1)
        await reg.register_session("key-1", "id-1", WORKER_2)
        assert await reg.lookup_by_key("key-1") == WORKER_2


# =============================================================================
# Routing strategies unit tests
# =============================================================================


class TestRoutingStrategies:
    def test_round_robin_cycling(self):
        s = RoundRobinStrategy()
        w1 = WorkerInfo(worker_addr=WORKER_1)
        w2 = WorkerInfo(worker_addr=WORKER_2)
        w3 = WorkerInfo(worker_addr=WORKER_3)
        workers = [w1, w2, w3]
        picks = [s.pick(workers).worker_addr for _ in range(6)]  # type: ignore[union-attr]
        assert picks == [
            WORKER_1,
            WORKER_2,
            WORKER_3,
            WORKER_1,
            WORKER_2,
            WORKER_3,
        ]

    def test_round_robin_empty(self):
        s = RoundRobinStrategy()
        assert s.pick([]) is None

    def test_least_busy_picks_minimum(self):
        s = LeastBusyStrategy()
        w1 = WorkerInfo(worker_addr=WORKER_1, active_requests=5)
        w2 = WorkerInfo(worker_addr=WORKER_2, active_requests=1)
        w3 = WorkerInfo(worker_addr=WORKER_3, active_requests=3)
        result = s.pick([w1, w2, w3])
        assert result is not None
        assert result.worker_addr == WORKER_2

    def test_least_busy_empty(self):
        s = LeastBusyStrategy()
        assert s.pick([]) is None

    def test_get_strategy_round_robin(self):
        s = get_strategy("round_robin")
        assert isinstance(s, RoundRobinStrategy)

    def test_get_strategy_least_busy(self):
        s = get_strategy("least_busy")
        assert isinstance(s, LeastBusyStrategy)

    def test_get_strategy_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown routing strategy"):
            get_strategy("random")


# =============================================================================
# Router endpoint tests
# =============================================================================


@pytest.fixture
def config():
    return RouterConfig(
        host="127.0.0.1",
        port=18081,
        admin_api_key=ADMIN_KEY,
        poll_interval=999,  # effectively disable polling in tests
        routing_strategy="round_robin",
    )


@pytest_asyncio.fixture
async def client(config):
    """Create router app and yield an httpx async client.
    Bypasses lifespan (no background health poller) by setting state directly.
    """
    app = create_app(config)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def admin_headers():
    return {"Authorization": f"Bearer {ADMIN_KEY}"}


class TestRouterEndpoints:
    # ----- Health -----

    @pytest.mark.asyncio
    async def test_health_200(self, client):
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["workers"] == 0
        assert data["sessions"] == 0
        assert data["strategy"] == "round_robin"

    # ----- Worker registration -----

    @pytest.mark.asyncio
    async def test_register_worker_admin_key(self, client):
        resp = await client.post(
            "/register_worker",
            json={"worker_addr": WORKER_1},
            headers=admin_headers(),
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

        # Verify via /health
        health = (await client.get("/health")).json()
        assert health["workers"] == 1

    @pytest.mark.asyncio
    async def test_register_worker_no_auth_401(self, client):
        resp = await client.post("/register_worker", json={"worker_addr": WORKER_1})
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_register_worker_wrong_key_403(self, client):
        resp = await client.post(
            "/register_worker",
            json={"worker_addr": WORKER_1},
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert resp.status_code == 403

    # ----- Worker deletion (with cascade) -----

    @pytest.mark.asyncio
    async def test_delete_worker_cascades(self, client):
        # Register worker + session pinned to it
        await client.post(
            "/register_worker",
            json={"worker_addr": WORKER_1},
            headers=admin_headers(),
        )
        await client.post(
            "/register_session",
            json={
                "session_api_key": "sess-key-1",
                "session_id": "task-0-0",
                "worker_addr": WORKER_1,
            },
        )

        # Delete the worker
        resp = await client.request(
            "DELETE",
            "/delete_worker",
            json={"worker_addr": WORKER_1},
            headers=admin_headers(),
        )
        assert resp.status_code == 200
        assert resp.json()["sessions_revoked"] == 1

        # Session key should no longer route
        resp = await client.post(
            "/route",
            json={"api_key": "sess-key-1", "path": "/chat/completions"},
        )
        assert resp.status_code == 404

    # ----- /route — admin key -----

    @pytest.mark.asyncio
    async def test_route_admin_key_round_robin(self, client):
        # Register two workers
        await client.post(
            "/register_worker",
            json={"worker_addr": WORKER_1},
            headers=admin_headers(),
        )
        await client.post(
            "/register_worker",
            json={"worker_addr": WORKER_2},
            headers=admin_headers(),
        )

        # Route with admin key — should round-robin
        resp1 = await client.post(
            "/route", json={"api_key": ADMIN_KEY, "path": "/generate"}
        )
        assert resp1.status_code == 200
        addr1 = resp1.json()["worker_addr"]

        resp2 = await client.post(
            "/route", json={"api_key": ADMIN_KEY, "path": "/generate"}
        )
        assert resp2.status_code == 200
        addr2 = resp2.json()["worker_addr"]

        # Two calls should hit different workers (round-robin)
        assert {addr1, addr2} == {WORKER_1, WORKER_2}

    # ----- /route — session key -----

    @pytest.mark.asyncio
    async def test_route_session_key_pinned(self, client):
        # Register worker + session
        await client.post(
            "/register_worker",
            json={"worker_addr": WORKER_1},
            headers=admin_headers(),
        )
        await client.post(
            "/register_session",
            json={
                "session_api_key": "sess-key-1",
                "session_id": "task-0-0",
                "worker_addr": WORKER_1,
            },
        )

        # Route with session key → pinned to WORKER_1
        resp = await client.post(
            "/route",
            json={"api_key": "sess-key-1", "path": "/chat/completions"},
        )
        assert resp.status_code == 200
        assert resp.json()["worker_addr"] == WORKER_1

    # ----- /route — unknown key -----

    @pytest.mark.asyncio
    async def test_route_unknown_key_404(self, client):
        resp = await client.post(
            "/route", json={"api_key": "unknown-key", "path": "/generate"}
        )
        assert resp.status_code == 404
        assert "Unknown API key" in resp.json()["detail"]

    # ----- /route — no healthy workers -----

    @pytest.mark.asyncio
    async def test_route_no_healthy_workers_503(self, client):
        """Admin key but no workers registered → 503."""
        resp = await client.post(
            "/route", json={"api_key": ADMIN_KEY, "path": "/generate"}
        )
        assert resp.status_code == 503
        assert "No healthy workers" in resp.json()["detail"]

    # ----- /route — pinned worker unhealthy -----

    @pytest.mark.asyncio
    async def test_route_pinned_worker_unhealthy_503(self, client):
        # Register worker + session
        await client.post(
            "/register_worker",
            json={"worker_addr": WORKER_1},
            headers=admin_headers(),
        )
        await client.post(
            "/register_session",
            json={
                "session_api_key": "sess-key-1",
                "session_id": "task-0-0",
                "worker_addr": WORKER_1,
            },
        )

        # Mark worker unhealthy
        wr: WorkerRegistry = client._transport.app.state.worker_registry  # type: ignore[attr-defined]
        await wr.update_health(WORKER_1, False)

        resp = await client.post(
            "/route",
            json={"api_key": "sess-key-1", "path": "/chat/completions"},
        )
        assert resp.status_code == 503
        assert "Pinned worker unhealthy" in resp.json()["detail"]

    # ----- /route — session_id lookup -----

    @pytest.mark.asyncio
    async def test_route_by_session_id(self, client):
        await client.post(
            "/register_session",
            json={
                "session_api_key": "sess-key-1",
                "session_id": "task-0-0",
                "worker_addr": WORKER_1,
            },
        )
        resp = await client.post("/route", json={"session_id": "task-0-0"})
        assert resp.status_code == 200
        assert resp.json()["worker_addr"] == WORKER_1

    @pytest.mark.asyncio
    async def test_route_by_session_id_unknown_404(self, client):
        resp = await client.post("/route", json={"session_id": "nonexistent"})
        assert resp.status_code == 404

    # ----- /route — missing both api_key and session_id -----

    @pytest.mark.asyncio
    async def test_route_missing_both_keys_422(self, client):
        resp = await client.post("/route", json={})
        assert resp.status_code == 422

    # ----- /register_session -----

    @pytest.mark.asyncio
    async def test_register_session(self, client):
        """Register a session, then verify pinned routing works."""
        await client.post(
            "/register_worker",
            json={"worker_addr": WORKER_1},
            headers=admin_headers(),
        )

        resp = await client.post(
            "/register_session",
            json={
                "session_api_key": "sess-key-1",
                "session_id": "task-0-0",
                "worker_addr": WORKER_1,
            },
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

        # Verify: /route with session key returns pinned worker
        route_resp = await client.post(
            "/route",
            json={"api_key": "sess-key-1", "path": "/chat/completions"},
        )
        assert route_resp.status_code == 200
        assert route_resp.json()["worker_addr"] == WORKER_1

    # ----- /workers -----

    @pytest.mark.asyncio
    async def test_workers_list_admin_key(self, client):
        await client.post(
            "/register_worker",
            json={"worker_addr": WORKER_1},
            headers=admin_headers(),
        )
        await client.post(
            "/register_worker",
            json={"worker_addr": WORKER_2},
            headers=admin_headers(),
        )

        resp = await client.get("/workers", headers=admin_headers())
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["workers"]) == 2
        addrs = {w["addr"] for w in data["workers"]}
        assert addrs == {WORKER_1, WORKER_2}
        assert all(w["healthy"] is True for w in data["workers"])

    @pytest.mark.asyncio
    async def test_workers_list_no_auth_401(self, client):
        resp = await client.get("/workers")
        assert resp.status_code == 401


# =============================================================================
# CapacityManager unit tests
# =============================================================================


class TestCapacityManager:
    @pytest.mark.asyncio
    async def test_initial_capacity_zero(self):
        cm = CapacityManager()
        assert await cm.get_capacity() == 0

    @pytest.mark.asyncio
    async def test_grant_increments(self):
        cm = CapacityManager()
        result = await cm.grant()
        assert result == 1
        assert await cm.get_capacity() == 1

    @pytest.mark.asyncio
    async def test_grant_multiple(self):
        cm = CapacityManager()
        await cm.grant()
        await cm.grant()
        result = await cm.grant()
        assert result == 3
        assert await cm.get_capacity() == 3

    @pytest.mark.asyncio
    async def test_try_acquire_success(self):
        cm = CapacityManager()
        await cm.grant()
        assert await cm.try_acquire() is True
        assert await cm.get_capacity() == 0

    @pytest.mark.asyncio
    async def test_try_acquire_empty_fails(self):
        cm = CapacityManager()
        assert await cm.try_acquire() is False
        assert await cm.get_capacity() == 0

    @pytest.mark.asyncio
    async def test_grant_then_acquire_then_empty(self):
        """Grant 2, acquire 2, then acquire again fails."""
        cm = CapacityManager()
        await cm.grant()
        await cm.grant()
        assert await cm.try_acquire() is True
        assert await cm.try_acquire() is True
        assert await cm.try_acquire() is False
        assert await cm.get_capacity() == 0

    @pytest.mark.asyncio
    async def test_interleaved_grant_acquire(self):
        """Interleaved grants and acquires track correctly."""
        cm = CapacityManager()
        await cm.grant()  # 1
        assert await cm.try_acquire() is True  # 0
        assert await cm.try_acquire() is False  # still 0
        await cm.grant()  # 1
        await cm.grant()  # 2
        assert await cm.try_acquire() is True  # 1
        assert await cm.get_capacity() == 1

    @pytest.mark.asyncio
    async def test_concurrent_grants(self):
        """Multiple concurrent grants all succeed."""
        cm = CapacityManager()
        results = await asyncio.gather(*[cm.grant() for _ in range(10)])
        # Results should be 1..10 in some order (each grant increments atomically)
        assert sorted(results) == list(range(1, 11))
        assert await cm.get_capacity() == 10

    @pytest.mark.asyncio
    async def test_concurrent_acquires(self):
        """Grant N, then N+M concurrent acquires → exactly N succeed."""
        cm = CapacityManager()
        for _ in range(5):
            await cm.grant()
        results = await asyncio.gather(*[cm.try_acquire() for _ in range(8)])
        assert sum(results) == 5  # exactly 5 succeed
        assert results.count(False) == 3  # 3 fail
        assert await cm.get_capacity() == 0


# =============================================================================
# Router capacity endpoint tests
# =============================================================================


class TestRouterCapacityEndpoints:
    @pytest.mark.asyncio
    async def test_health_includes_capacity(self, client):
        """Health response includes capacity field."""
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "capacity" in data
        assert data["capacity"] == 0

    # ----- /grant_capacity -----

    @pytest.mark.asyncio
    async def test_grant_capacity_200(self, client):
        """Admin key → /grant_capacity → capacity incremented."""
        resp = await client.post("/grant_capacity", headers=admin_headers())
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["capacity"] == 1

    @pytest.mark.asyncio
    async def test_grant_capacity_increments(self, client):
        """Multiple grants increment capacity."""
        await client.post("/grant_capacity", headers=admin_headers())
        resp = await client.post("/grant_capacity", headers=admin_headers())
        assert resp.json()["capacity"] == 2

        # Verify via /health
        health = (await client.get("/health")).json()
        assert health["capacity"] == 2

    @pytest.mark.asyncio
    async def test_grant_capacity_no_auth_401(self, client):
        """/grant_capacity without auth → 401."""
        resp = await client.post("/grant_capacity")
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_grant_capacity_wrong_key_403(self, client):
        """/grant_capacity with wrong key → 403."""
        resp = await client.post(
            "/grant_capacity",
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert resp.status_code == 403

    # ----- /acquire_capacity -----

    @pytest.mark.asyncio
    async def test_acquire_capacity_success(self, client):
        """Grant then acquire → 200."""
        await client.post("/grant_capacity", headers=admin_headers())
        resp = await client.post("/acquire_capacity", headers=admin_headers())
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

        # Capacity should be back to 0
        health = (await client.get("/health")).json()
        assert health["capacity"] == 0

    @pytest.mark.asyncio
    async def test_acquire_capacity_empty_429(self, client):
        """Acquire with no capacity → 429."""
        resp = await client.post("/acquire_capacity", headers=admin_headers())
        assert resp.status_code == 429
        assert "capacity" in resp.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_acquire_capacity_depletes(self, client):
        """Grant 2, acquire 2, then acquire → 429."""
        await client.post("/grant_capacity", headers=admin_headers())
        await client.post("/grant_capacity", headers=admin_headers())

        resp1 = await client.post("/acquire_capacity", headers=admin_headers())
        assert resp1.status_code == 200
        resp2 = await client.post("/acquire_capacity", headers=admin_headers())
        assert resp2.status_code == 200
        resp3 = await client.post("/acquire_capacity", headers=admin_headers())
        assert resp3.status_code == 429

    @pytest.mark.asyncio
    async def test_acquire_capacity_no_auth_401(self, client):
        """/acquire_capacity without auth → 401."""
        resp = await client.post("/acquire_capacity")
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_acquire_capacity_wrong_key_403(self, client):
        """/acquire_capacity with wrong key → 403."""
        resp = await client.post(
            "/acquire_capacity",
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert resp.status_code == 403

    # ----- Grant + Acquire interleaved -----

    @pytest.mark.asyncio
    async def test_grant_acquire_interleaved(self, client):
        """Grant, acquire, grant again, acquire → all succeed."""
        await client.post("/grant_capacity", headers=admin_headers())
        resp = await client.post("/acquire_capacity", headers=admin_headers())
        assert resp.status_code == 200

        # No capacity left
        resp = await client.post("/acquire_capacity", headers=admin_headers())
        assert resp.status_code == 429

        # Grant again
        await client.post("/grant_capacity", headers=admin_headers())
        resp = await client.post("/acquire_capacity", headers=admin_headers())
        assert resp.status_code == 200
