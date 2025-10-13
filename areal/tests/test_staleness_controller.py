"""Unit tests for StalenessController.

This module provides comprehensive test coverage for the StalenessController class,
including capacity calculations, thread safety, and state transitions.
"""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from areal.core.staleness_controller import StalenessController


class TestStalenessControllerBasics:
    """Test basic functionality of StalenessController."""

    def test_initialization(self):
        """Test controller initialization with various parameters."""
        controller = StalenessController(
            max_concurrent_rollouts=10,
            consumer_batch_size=4,
            max_staleness=2,
        )

        assert controller.max_concurrent_rollouts == 10
        assert controller.consumer_batch_size == 4
        assert controller.max_staleness == 2

        stats = controller.get_stats()
        assert stats.submitted == 0
        assert stats.accepted == 0
        assert stats.running == 0

    def test_initial_capacity_full(self):
        """Test that initial capacity equals max_concurrent_rollouts."""
        controller = StalenessController(
            max_concurrent_rollouts=10,
            consumer_batch_size=4,
            max_staleness=2,
        )

        # At version 0, with no running tasks, capacity should be limited by concurrency
        capacity = controller.get_capacity(current_version=0)
        assert capacity == 10  # max_concurrent_rollouts

    def test_initial_capacity_with_large_staleness(self):
        """Test capacity with large staleness allowance."""
        controller = StalenessController(
            max_concurrent_rollouts=100,
            consumer_batch_size=32,
            max_staleness=10,
        )

        # Staleness capacity: (10 + 0 + 1) * 32 = 352
        # Concurrency capacity: 100
        # Should be limited by concurrency
        capacity = controller.get_capacity(current_version=0)
        assert capacity == 100


class TestCapacityCalculations:
    """Test capacity calculation logic under various scenarios."""

    def test_concurrency_limit(self):
        """Test that concurrency limit is enforced."""
        controller = StalenessController(
            max_concurrent_rollouts=5,
            consumer_batch_size=2,
            max_staleness=10,  # Very high, won't be limiting factor
        )

        # Submit 3 rollouts
        for _ in range(3):
            controller.on_rollout_submitted()

        capacity = controller.get_capacity(current_version=0)
        assert capacity == 2  # 5 - 3 = 2 remaining

    def test_staleness_limit(self):
        """Test that staleness limit is enforced."""
        controller = StalenessController(
            max_concurrent_rollouts=100,  # Very high, won't be limiting factor
            consumer_batch_size=4,
            max_staleness=2,
        )

        # At version 0: max_samples = (2 + 0 + 1) * 4 = 12
        # With 0 submitted/accepted, capacity should be 12
        capacity = controller.get_capacity(current_version=0)
        assert capacity == 12

    def test_staleness_increases_with_version(self):
        """Test that allowed capacity increases with version."""
        controller = StalenessController(
            max_concurrent_rollouts=1000,
            consumer_batch_size=4,
            max_staleness=2,
        )

        # At version 0: (2 + 0 + 1) * 4 = 12
        capacity_v0 = controller.get_capacity(current_version=0)
        assert capacity_v0 == 12

        # At version 5: (2 + 5 + 1) * 4 = 32
        capacity_v5 = controller.get_capacity(current_version=5)
        assert capacity_v5 == 32

        # Capacity should increase with version
        assert capacity_v5 > capacity_v0

    def test_capacity_with_running_rollouts(self):
        """Test capacity calculation with running rollouts."""
        controller = StalenessController(
            max_concurrent_rollouts=10,
            consumer_batch_size=4,
            max_staleness=2,
        )

        # Submit 3 rollouts
        for _ in range(3):
            controller.on_rollout_submitted()

        # Concurrency capacity: 10 - 3 = 7
        # Staleness capacity: (2 + 0 + 1) * 4 - 3 = 12 - 3 = 9
        # Should be limited by concurrency
        capacity = controller.get_capacity(current_version=0)
        assert capacity == 7

    def test_capacity_with_accepted_rollouts(self):
        """Test capacity calculation with accepted rollouts."""
        controller = StalenessController(
            max_concurrent_rollouts=20,
            consumer_batch_size=4,
            max_staleness=2,
        )

        # Submit and accept 5 rollouts
        for _ in range(5):
            controller.on_rollout_submitted()
            controller.on_rollout_accepted()

        # running = 0, accepted = 5
        # Concurrency capacity: 20 - 0 = 20
        # Staleness capacity: (2 + 0 + 1) * 4 - 5 = 12 - 5 = 7
        # Should be limited by staleness
        capacity = controller.get_capacity(current_version=0)
        assert capacity == 7

    def test_capacity_at_limit(self):
        """Test capacity when at exactly the limit."""
        controller = StalenessController(
            max_concurrent_rollouts=5,
            consumer_batch_size=2,
            max_staleness=5,  # Make staleness not limiting
        )

        # Submit 5 rollouts (at concurrency limit)
        for _ in range(5):
            controller.on_rollout_submitted()

        # Concurrency: 5 - 5 = 0
        # Staleness: (5 + 0 + 1) * 2 - 5 = 12 - 5 = 7 (not limiting)
        capacity = controller.get_capacity(current_version=0)
        assert capacity == 0  # Limited by concurrency

    def test_capacity_can_be_negative(self):
        """Test that capacity can be negative when over limit."""
        controller = StalenessController(
            max_concurrent_rollouts=3,
            consumer_batch_size=2,
            max_staleness=1,
        )

        # Submit 10 rollouts (way over limit)
        for _ in range(10):
            controller.on_rollout_submitted()

        capacity = controller.get_capacity(current_version=0)
        assert capacity < 0

    def test_min_values_are_enforced(self):
        """Test that minimum values of 1 are enforced."""
        controller = StalenessController(
            max_concurrent_rollouts=0,  # Should become 1
            consumer_batch_size=0,  # Should become 1
            max_staleness=0,
        )

        # With max(1, 0) = 1 for both params:
        # Capacity should still work correctly
        capacity = controller.get_capacity(current_version=0)
        assert capacity >= 0  # Should not crash


class TestRolloutLifecycle:
    """Test rollout state transitions through their lifecycle."""

    def test_submit_increments_counters(self):
        """Test that submitting a rollout increments both submitted and running."""
        controller = StalenessController(
            max_concurrent_rollouts=10,
            consumer_batch_size=4,
            max_staleness=2,
        )

        controller.on_rollout_submitted()

        stats = controller.get_stats()
        assert stats.submitted == 1
        assert stats.running == 1
        assert stats.accepted == 0

    def test_accept_updates_counters(self):
        """Test that accepting a rollout updates counters correctly."""
        controller = StalenessController(
            max_concurrent_rollouts=10,
            consumer_batch_size=4,
            max_staleness=2,
        )

        controller.on_rollout_submitted()
        controller.on_rollout_accepted()

        stats = controller.get_stats()
        assert stats.submitted == 1
        assert stats.running == 0  # Decremented
        assert stats.accepted == 1  # Incremented

    def test_reject_updates_counters(self):
        """Test that rejecting a rollout updates counters correctly."""
        controller = StalenessController(
            max_concurrent_rollouts=10,
            consumer_batch_size=4,
            max_staleness=2,
        )

        controller.on_rollout_submitted()
        controller.on_rollout_rejected()

        stats = controller.get_stats()
        assert stats.submitted == 1
        assert stats.running == 0  # Decremented
        assert stats.accepted == 0  # NOT incremented (key difference from accept)

    def test_multiple_rollouts_lifecycle(self):
        """Test multiple rollouts going through their lifecycle."""
        controller = StalenessController(
            max_concurrent_rollouts=10,
            consumer_batch_size=4,
            max_staleness=2,
        )

        # Submit 5 rollouts
        for _ in range(5):
            controller.on_rollout_submitted()

        # Accept 3 of them
        for _ in range(3):
            controller.on_rollout_accepted()

        # Reject 2 of them
        for _ in range(2):
            controller.on_rollout_rejected()

        stats = controller.get_stats()
        assert stats.submitted == 5
        assert stats.running == 0  # All completed
        assert stats.accepted == 3  # Only 3 accepted

    def test_accept_without_submit_is_invalid(self):
        """Test that accepting without submitting leads to incorrect state."""
        controller = StalenessController(
            max_concurrent_rollouts=10,
            consumer_batch_size=4,
            max_staleness=2,
        )

        # This is invalid usage, but shouldn't crash
        controller.on_rollout_accepted()

        stats = controller.get_stats()
        assert stats.submitted == 0
        assert stats.running == -1  # Incorrect state!
        assert stats.accepted == 1


class TestThreadSafety:
    """Test thread safety of StalenessController operations."""

    def test_concurrent_submissions(self):
        """Test that concurrent submissions are thread-safe."""
        controller = StalenessController(
            max_concurrent_rollouts=1000,
            consumer_batch_size=32,
            max_staleness=10,
        )

        num_threads = 10
        submissions_per_thread = 100

        def submit_many():
            for _ in range(submissions_per_thread):
                controller.on_rollout_submitted()

        threads = [threading.Thread(target=submit_many) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        stats = controller.get_stats()
        expected_total = num_threads * submissions_per_thread
        assert stats.submitted == expected_total
        assert stats.running == expected_total

    def test_concurrent_mixed_operations(self):
        """Test concurrent mixed operations (submit, accept, reject)."""
        controller = StalenessController(
            max_concurrent_rollouts=1000,
            consumer_batch_size=32,
            max_staleness=10,
        )

        num_operations = 100

        def submit_operations():
            for _ in range(num_operations):
                controller.on_rollout_submitted()

        def accept_operations():
            for _ in range(num_operations // 2):
                controller.on_rollout_accepted()

        def reject_operations():
            for _ in range(num_operations // 2):
                controller.on_rollout_rejected()

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(submit_operations),
                executor.submit(accept_operations),
                executor.submit(reject_operations),
            ]
            for f in futures:
                f.result()

        stats = controller.get_stats()
        assert stats.submitted == num_operations
        # running should be 0 (100 submitted, 50 accepted, 50 rejected)
        assert stats.running == 0
        assert stats.accepted == num_operations // 2

    def test_concurrent_capacity_checks(self):
        """Test that concurrent capacity checks don't cause race conditions."""
        controller = StalenessController(
            max_concurrent_rollouts=100,
            consumer_batch_size=32,
            max_staleness=10,
        )

        results = []

        def check_capacity():
            for _ in range(100):
                capacity = controller.get_capacity(current_version=0)
                results.append(capacity)

        threads = [threading.Thread(target=check_capacity) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All capacity checks should return the same value (no submissions)
        assert all(c == results[0] for c in results)

    def test_concurrent_get_stats(self):
        """Test that concurrent get_stats calls are thread-safe."""
        controller = StalenessController(
            max_concurrent_rollouts=100,
            consumer_batch_size=32,
            max_staleness=10,
        )

        # Submit some rollouts in background
        for _ in range(10):
            controller.on_rollout_submitted()

        results = []

        def get_stats_many():
            for _ in range(100):
                stats = controller.get_stats()
                results.append((stats.submitted, stats.running, stats.accepted))

        threads = [threading.Thread(target=get_stats_many) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All stats should be consistent
        for submitted, running, accepted in results:
            assert submitted >= 0
            assert running >= 0
            assert accepted >= 0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_max_staleness(self):
        """Test with zero staleness (immediate consumption required)."""
        controller = StalenessController(
            max_concurrent_rollouts=100,
            consumer_batch_size=8,
            max_staleness=0,
        )

        # At version 0: (0 + 0 + 1) * 8 = 8
        capacity = controller.get_capacity(current_version=0)
        assert capacity == 8

        # At version 5: (0 + 5 + 1) * 8 = 48
        capacity = controller.get_capacity(current_version=5)
        assert capacity == 48

    def test_very_large_version(self):
        """Test with very large version numbers."""
        controller = StalenessController(
            max_concurrent_rollouts=10000,
            consumer_batch_size=64,
            max_staleness=10,
        )

        # At version 1000000: (10 + 1000000 + 1) * 64 = 64000704
        # Should be limited by concurrency
        capacity = controller.get_capacity(current_version=1000000)
        assert capacity == 10000  # Limited by max_concurrent_rollouts

    def test_single_rollout_batch_size(self):
        """Test with batch size of 1."""
        controller = StalenessController(
            max_concurrent_rollouts=10,
            consumer_batch_size=1,
            max_staleness=2,
        )

        # At version 0: (2 + 0 + 1) * 1 = 3
        capacity = controller.get_capacity(current_version=0)
        assert capacity == 3

    def test_all_rollouts_rejected(self):
        """Test scenario where all rollouts are rejected."""
        controller = StalenessController(
            max_concurrent_rollouts=10,
            consumer_batch_size=4,
            max_staleness=2,
        )

        # Submit and reject 10 rollouts
        for _ in range(10):
            controller.on_rollout_submitted()
            controller.on_rollout_rejected()

        stats = controller.get_stats()
        assert stats.submitted == 10
        assert stats.running == 0
        assert stats.accepted == 0

        # Should have full capacity again
        capacity = controller.get_capacity(current_version=0)
        assert capacity == 10  # Back to max_concurrent_rollouts

    def test_mixed_acceptance_rate(self):
        """Test with a realistic mixed acceptance rate."""
        controller = StalenessController(
            max_concurrent_rollouts=20,
            consumer_batch_size=8,
            max_staleness=3,
        )

        # Simulate 20 rollouts: 15 accepted, 5 rejected
        for _ in range(20):
            controller.on_rollout_submitted()

        for _ in range(15):
            controller.on_rollout_accepted()

        for _ in range(5):
            controller.on_rollout_rejected()

        stats = controller.get_stats()
        assert stats.submitted == 20
        assert stats.running == 0
        assert stats.accepted == 15

        # Check capacity with the accepted rollouts
        # Staleness capacity: (3 + 0 + 1) * 8 - 15 = 32 - 15 = 17
        # Concurrency capacity: 20 - 0 = 20
        # Should be limited by staleness
        capacity = controller.get_capacity(current_version=0)
        assert capacity == 17


class TestRealWorldScenarios:
    """Test realistic scenarios that might occur in production."""

    def test_typical_training_scenario(self):
        """Test a typical training scenario with version progression."""
        controller = StalenessController(
            max_concurrent_rollouts=32,
            consumer_batch_size=16,
            max_staleness=2,
        )

        # Version 0: Submit initial batch
        for _ in range(16):
            controller.on_rollout_submitted()

        # Complete 14 successfully, 2 rejected
        for _ in range(14):
            controller.on_rollout_accepted()
        for _ in range(2):
            controller.on_rollout_rejected()

        # Check capacity at version 1 (after training step)
        # Staleness: (2 + 1 + 1) * 16 - 14 = 64 - 14 = 50
        # Concurrency: 32 - 0 = 32
        capacity_v1 = controller.get_capacity(current_version=1)
        assert capacity_v1 == 32  # Limited by concurrency

        # Submit another batch
        for _ in range(16):
            controller.on_rollout_submitted()

        stats = controller.get_stats()
        assert stats.running == 16
        assert stats.accepted == 14

    def test_burst_load_scenario(self):
        """Test handling burst load of rollouts."""
        controller = StalenessController(
            max_concurrent_rollouts=50,
            consumer_batch_size=32,
            max_staleness=5,
        )

        # Burst: Submit 100 rollouts quickly
        for _ in range(100):
            controller.on_rollout_submitted()

        stats = controller.get_stats()
        assert stats.running == 100

        # Capacity should be negative (over limit)
        capacity = controller.get_capacity(current_version=0)
        assert capacity < 0

        # Process them gradually
        for _ in range(80):
            controller.on_rollout_accepted()
        for _ in range(20):
            controller.on_rollout_rejected()

        # Should have capacity again
        capacity = controller.get_capacity(current_version=0)
        assert capacity > 0

    def test_slow_consumption_scenario(self):
        """Test scenario where rollouts are consumed slower than generated."""
        controller = StalenessController(
            max_concurrent_rollouts=100,
            consumer_batch_size=8,
            max_staleness=3,  # Limited staleness
        )

        # Generate rollouts and accept them without version progression
        # Staleness limit: (3 + 0 + 1) * 8 = 32

        # Accept 30 rollouts (approaching staleness limit)
        for _ in range(30):
            controller.on_rollout_submitted()
            controller.on_rollout_accepted()

        # Capacity should be very limited now
        # Staleness: (3 + 0 + 1) * 8 - 30 = 32 - 30 = 2
        capacity = controller.get_capacity(current_version=0)
        assert capacity == 2  # Very constrained by staleness


# Parametrized tests for broader coverage
@pytest.mark.parametrize(
    "max_concurrent_rollouts,consumer_batch_size,max_staleness",
    [
        (10, 4, 2),
        (100, 32, 5),
        (1, 1, 0),
        (1000, 128, 10),
        (50, 16, 3),
    ],
)
def test_parametrized_initialization(
    max_concurrent_rollouts, consumer_batch_size, max_staleness
):
    """Test initialization with various parameter combinations."""
    controller = StalenessController(
        max_concurrent_rollouts=max_concurrent_rollouts,
        consumer_batch_size=consumer_batch_size,
        max_staleness=max_staleness,
    )

    assert controller.max_concurrent_rollouts == max_concurrent_rollouts
    assert controller.consumer_batch_size == consumer_batch_size
    assert controller.max_staleness == max_staleness

    # Should always be able to get capacity without crashing
    capacity = controller.get_capacity(current_version=0)
    assert isinstance(capacity, int)


@pytest.mark.parametrize("version", [0, 1, 10, 100, 1000])
def test_parametrized_version_progression(version):
    """Test capacity calculation across different versions."""
    controller = StalenessController(
        max_concurrent_rollouts=1000,
        consumer_batch_size=32,
        max_staleness=5,
    )

    capacity = controller.get_capacity(current_version=version)
    expected_staleness_capacity = (5 + version + 1) * 32

    # Capacity should be limited by concurrency (1000)
    assert capacity == min(1000, expected_staleness_capacity)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
