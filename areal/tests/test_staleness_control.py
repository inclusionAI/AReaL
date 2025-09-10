"""Unit tests for centralized staleness management system."""

import threading
import time

import pytest
import requests

from areal.api.workflow_api import (
    StalenessManager,
    StalenessManagerServer,
    create_capacity_app,
)
from areal.utils import network


class TestStalenessManager:
    """Test suite for CapacityManager."""

    def test_basic_capacity_allocation(self):
        """Test basic capacity request and release operations."""
        manager = StalenessManager(
            max_capacity=10,
            max_head_offpolicyness=5,
            consumer_batch_size=4,
            enable_rollout_tracing=False,
        )

        # Test initial capacity request
        response = manager.request_capacity(3)
        assert response["granted"] == 3
        assert response["running"] == 3
        assert response["submitted"] == 3

        # Test capacity constraint
        response = manager.request_capacity(8)
        assert response["granted"] == 7  # Only 7 remaining out of 10

        # Test release
        manager.release_capacity(completed=5, accepted=5)
        assert manager.rollout_stat.running == 5

    def test_staleness_control(self):
        """Test staleness control with version updates."""
        manager = StalenessManager(
            max_capacity=100,  # Large capacity to test staleness limits
            max_head_offpolicyness=2,
            consumer_batch_size=4,
            enable_rollout_tracing=False,
        )

        # Initial version 0, should allow (2+0+1)*4 = 12 samples
        response = manager.request_capacity(20)
        assert response["granted"] == 12

        # Update version to 1, should allow (2+1+1)*4 = 16 total samples
        manager.update_version(1)
        response = manager.request_capacity(10)
        assert response["granted"] == 4  # 16 - 12 = 4

        # Update version to 3, should allow (2+3+1)*4 = 24 total samples
        manager.update_version(3)
        response = manager.request_capacity(20)
        assert response["granted"] == 8  # 24 - 16 = 8

    def test_concurrent_access(self):
        """Test thread-safe concurrent access to capacity manager."""
        manager = StalenessManager(
            max_capacity=20,
            max_head_offpolicyness=10,
            consumer_batch_size=4,
            enable_rollout_tracing=False,
        )

        results = []

        def worker(worker_id):
            for i in range(5):
                response = manager.request_capacity(2)
                results.append((worker_id, i, response["granted"]))
                time.sleep(0.01)  # Small delay to increase contention
                if response["granted"] > 0:
                    manager.release_capacity(
                        completed=response["granted"], accepted=response["granted"]
                    )

        # Start multiple threads
        threads = []
        for i in range(4):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        # Wait for completion
        for t in threads:
            t.join()

        # Verify all operations completed successfully
        assert len(results) == 20  # 4 workers * 5 operations each
        assert manager.rollout_stat.running == 0  # All capacity released

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        manager = StalenessManager(
            max_capacity=5,
            max_head_offpolicyness=1,
            consumer_batch_size=2,
            enable_rollout_tracing=False,
        )

        # Test zero/negative requests
        response = manager.request_capacity(0)
        assert response["granted"] == 0

        # Test over-capacity request
        response = manager.request_capacity(10)
        assert response["granted"] <= 5

        # Test that release validates parameters with fresh manager
        fresh_manager = StalenessManager(
            max_capacity=5,
            max_head_offpolicyness=1,
            consumer_batch_size=2,
            enable_rollout_tracing=False,
        )
        response = fresh_manager.request_capacity(3)
        granted = response["granted"]
        # Should reject release with more completed than running
        try:
            fresh_manager.release_capacity(completed=granted + 10, accepted=granted)
            assert False, "Should have failed with assertion"
        except AssertionError:
            pass  # Expected
        # Normal release should work
        fresh_manager.release_capacity(completed=granted, accepted=granted)
        assert fresh_manager.rollout_stat.running == 0

    def test_completed_greater_than_accepted(self):
        """Test release_capacity when completed > accepted (rejected samples)."""
        manager = StalenessManager(
            max_capacity=10,
            max_head_offpolicyness=2,
            consumer_batch_size=4,
            enable_rollout_tracing=False,
        )

        # Request some capacity
        response = manager.request_capacity(6)
        assert response["granted"] == 6
        assert manager.rollout_stat.running == 6
        assert manager.rollout_stat.accepted == 0

        # Release with some rejections: completed=6, accepted=4 (2 rejected)
        manager.release_capacity(completed=6, accepted=4)

        # Check final state
        assert manager.rollout_stat.running == 0  # All completed
        assert (
            manager.rollout_stat.accepted == 4
        )  # Only accepted count toward staleness

        # Test that staleness control properly accounts for rejected samples
        # With version 0, max_head_offpolicyness=2, consumer_batch_size=4
        # Max allowed: (2+0+1)*4 = 12 samples
        # Current accepted + running: 4 + 0 = 4
        # Should allow 12-4 = 8 more samples
        response = manager.request_capacity(10)
        assert response["granted"] == 8  # Limited by staleness, not capacity

    def test_staleness_control_with_rejected_samples(self):
        """Test that rejected samples don't count toward staleness limits."""
        manager = StalenessManager(
            max_capacity=20,
            max_head_offpolicyness=1,
            consumer_batch_size=3,
            enable_rollout_tracing=False,
        )

        # Version 0: max allowed = (1+0+1)*3 = 6 samples
        response = manager.request_capacity(6)
        assert response["granted"] == 6

        # Complete all with high rejection rate: only 2 accepted out of 6
        manager.release_capacity(completed=6, accepted=2)
        assert manager.rollout_stat.accepted == 2
        assert manager.rollout_stat.running == 0

        # Should be able to request more since rejected samples don't count
        # Current: accepted=2, running=0, total=2
        # Max allowed: 6, so can request 6-2=4 more
        response = manager.request_capacity(10)
        assert response["granted"] == 4

        # Complete with different acceptance rate: 3 accepted out of 4
        manager.release_capacity(completed=4, accepted=3)
        assert manager.rollout_stat.accepted == 5  # 2+3
        assert manager.rollout_stat.running == 0

        # Now close to limit: 5 accepted, max 6, so only 1 more allowed
        response = manager.request_capacity(10)
        assert response["granted"] == 1

        # Complete with acceptance: 1 accepted out of 1
        manager.release_capacity(completed=1, accepted=1)
        assert manager.rollout_stat.accepted == 6
        assert manager.rollout_stat.running == 0

        # Now at limit: should not allow any more at version 0
        response = manager.request_capacity(10)
        assert response["granted"] == 0

        # Update version to 1: new limit = (1+1+1)*3 = 9
        manager.update_version(1)
        # Can now request 9-6=3 more
        response = manager.request_capacity(10)
        assert response["granted"] == 3


class TestStalenessManagerServer:
    """Test suite for StalenessManagerServer."""

    @pytest.fixture
    def staleness_manager(self):
        """Create a staleness manager for testing."""
        return StalenessManager(
            max_capacity=10,
            max_head_offpolicyness=3,
            consumer_batch_size=2,
            enable_rollout_tracing=False,
        )

    @pytest.fixture
    def server_port(self):
        """Get a free port for testing."""
        return network.find_free_ports(1)[0]

    def test_fastapi_app_creation(self, staleness_manager):
        """Test FastAPI app creation and route registration."""
        app = create_capacity_app(staleness_manager)

        # Check that routes are registered
        routes = [route.path for route in app.routes]
        assert "/request_capacity" in routes
        assert "/release_capacity" in routes
        assert "/health" in routes

    def test_server_lifecycle(self, staleness_manager, server_port):
        """Test server start and stop lifecycle."""
        server = StalenessManagerServer("localhost", server_port, staleness_manager)

        # Start server
        server.start()
        time.sleep(0.5)  # Wait for server to start

        # Test health endpoint
        response = requests.get(f"http://localhost:{server_port}/health", timeout=2)
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

        # Stop server
        server.stop()

    def test_capacity_endpoints(self, staleness_manager, server_port):
        """Test capacity request and release endpoints."""
        server = StalenessManagerServer("localhost", server_port, staleness_manager)
        server.start()
        time.sleep(0.5)

        try:
            base_url = f"http://localhost:{server_port}"

            # Test capacity request
            response = requests.post(
                f"{base_url}/request_capacity", json={"requested": 5}, timeout=2
            )
            assert response.status_code == 200
            result = response.json()
            assert result["granted"] == 5
            assert result["running"] == 5

            # Test capacity release
            response = requests.post(
                f"{base_url}/release_capacity",
                json={"completed": 2, "accepted": 2},
                timeout=2,
            )
            assert response.status_code == 200
            assert response.json()["status"] == "ok"

            # Verify capacity was released
            assert staleness_manager.rollout_stat.running == 3

        finally:
            server.stop()

    def test_concurrent_server_requests(self, staleness_manager, server_port):
        """Test concurrent requests to the staleness server."""
        server = StalenessManagerServer("localhost", server_port, staleness_manager)
        server.start()
        time.sleep(0.5)

        try:
            base_url = f"http://localhost:{server_port}"
            results = []

            def make_requests():
                try:
                    # Request capacity
                    response = requests.post(
                        f"{base_url}/request_capacity", json={"requested": 2}, timeout=2
                    )
                    if response.status_code == 200:
                        granted = response.json()["granted"]
                        results.append(granted)

                        # Release if we got any
                        if granted > 0:
                            requests.post(
                                f"{base_url}/release_capacity",
                                json={"completed": granted, "accepted": granted},
                                timeout=2,
                            )
                except Exception as e:
                    results.append(f"error: {e}")

            # Start concurrent requests
            threads = []
            for _ in range(8):
                t = threading.Thread(target=make_requests)
                threads.append(t)
                t.start()

            # Wait for completion
            for t in threads:
                t.join()

            # Verify results
            assert len(results) == 8
            total_granted = sum(r for r in results if isinstance(r, int))
            assert total_granted > 0  # Some requests should succeed
            assert staleness_manager.rollout_stat.running == 0  # All released

        finally:
            server.stop()

    def test_invalid_requests(self, staleness_manager, server_port):
        """Test handling of invalid requests."""
        server = StalenessManagerServer("localhost", server_port, staleness_manager)
        server.start()
        time.sleep(0.5)

        try:
            base_url = f"http://localhost:{server_port}"

            # Test invalid JSON
            response = requests.post(
                f"{base_url}/request_capacity", data="invalid json", timeout=2
            )
            # Flask should handle this gracefully (usually 400)
            assert response.status_code != 200

            # Test missing fields (should use defaults)
            response = requests.post(
                f"{base_url}/request_capacity", json={}, timeout=2  # Empty request
            )
            assert response.status_code == 200
            result = response.json()
            assert result["requested"] == 1  # Default value

        finally:
            server.stop()

    def test_completed_greater_than_accepted_via_server(
        self, staleness_manager, server_port
    ):
        """Test server endpoints handle completed > accepted properly."""
        server = StalenessManagerServer("localhost", server_port, staleness_manager)
        server.start()
        time.sleep(0.5)

        try:
            base_url = f"http://localhost:{server_port}"

            # Request capacity
            response = requests.post(
                f"{base_url}/request_capacity", json={"requested": 5}, timeout=2
            )
            assert response.status_code == 200
            result = response.json()
            granted = result["granted"]
            assert granted == 5

            # Release with rejections: completed > accepted
            response = requests.post(
                f"{base_url}/release_capacity",
                json={"completed": 5, "accepted": 3},
                timeout=2,
            )
            assert response.status_code == 200
            assert response.json()["status"] == "ok"

            # Verify state: only accepted samples count toward staleness
            assert staleness_manager.rollout_stat.running == 0
            assert staleness_manager.rollout_stat.accepted == 3

            # Test staleness behavior with rejected samples
            # Current setup: max_head_offpolicyness=3, consumer_batch_size=2, version=0
            # Max allowed: (3+0+1)*2 = 8, current accepted: 3, so 5 more allowed
            response = requests.post(
                f"{base_url}/request_capacity", json={"requested": 10}, timeout=2
            )
            assert response.status_code == 200
            result = response.json()
            assert result["granted"] == 5  # Limited by staleness control

        finally:
            server.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
