"""Test suite for ProxyRouter with Strategy Pattern."""

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


@dataclass
class MockWorker:
    """Mock Worker for testing."""

    id: str
    ip: str = "127.0.0.1"


class RoutingStrategy(str, Enum):
    """Enumeration of available worker routing strategies."""

    ROUND_ROBIN = "round-robin"
    RANDOM = "random"


class WorkerSelector(ABC):
    """Abstract base class for worker selection strategies."""

    @abstractmethod
    def select(self, workers: list[MockWorker]) -> int:
        """Select a worker and return its rank."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the selector state."""
        pass


class RoundRobinSelector(WorkerSelector):
    """Round-robin worker selection strategy."""

    def __init__(self):
        self._current_idx = 0

    def select(self, workers: list[MockWorker]) -> int:
        """Select the next worker in round-robin order."""
        if not workers:
            raise RuntimeError("No workers available to choose from.")

        rank = self._current_idx
        self._current_idx = (self._current_idx + 1) % len(workers)
        return rank

    def reset(self) -> None:
        """Reset the round-robin index to 0."""
        self._current_idx = 0


class RandomSelector(WorkerSelector):
    """Random worker selection strategy."""

    def select(self, workers: list[MockWorker]) -> int:
        """Randomly select a worker."""
        if not workers:
            raise RuntimeError("No workers available to choose from.")

        return random.randint(0, len(workers) - 1)

    def reset(self) -> None:
        """No-op for stateless random strategy."""
        pass


class ProxyRouter:
    """Router for choosing workers and managing proxy addresses using Strategy Pattern."""

    # Strategy factory: maps RoutingStrategy enum to WorkerSelector classes
    _SELECTOR_FACTORY: dict[RoutingStrategy, type[WorkerSelector]] = {
        RoutingStrategy.ROUND_ROBIN: RoundRobinSelector,
        RoutingStrategy.RANDOM: RandomSelector,
    }

    def __init__(
        self,
        workers: list[MockWorker],
        proxy_addrs: list[str] | None = None,
        routing_strategy: RoutingStrategy | str = RoutingStrategy.ROUND_ROBIN,
    ):
        """Initialize the ProxyRouter."""
        self.workers = workers
        self.proxy_addrs = proxy_addrs or []
        self._proxy_enabled = bool(proxy_addrs)

        # Convert string to enum if necessary
        if isinstance(routing_strategy, str):
            try:
                self.strategy = RoutingStrategy(routing_strategy)
            except ValueError:
                valid_strategies = ", ".join([s.value for s in RoutingStrategy])
                raise ValueError(
                    f"Invalid routing_strategy: {routing_strategy}. "
                    f"Must be one of: {valid_strategies}"
                )
        else:
            self.strategy = routing_strategy

        # Create the appropriate selector using the factory
        selector_class = self._SELECTOR_FACTORY.get(self.strategy)
        if selector_class is None:
            raise ValueError(
                f"No selector implementation found for strategy: {self.strategy}"
            )
        self.selector = selector_class()

    def route(self) -> tuple[MockWorker, int, str | None]:
        """Choose a worker and get its proxy address."""
        # Delegate selection to the strategy
        rank = self.selector.select(self.workers)
        worker = self.workers[rank]

        # Get proxy address if available
        proxy_addr = (
            self.proxy_addrs[rank]
            if self._proxy_enabled and rank < len(self.proxy_addrs)
            else None
        )

        return worker, rank, proxy_addr

    def get_proxy_addr(self, rank: int) -> str | None:
        """Get the proxy server address for a specific worker rank."""
        if not self._proxy_enabled or rank >= len(self.proxy_addrs):
            return None
        return self.proxy_addrs[rank]

    def update_proxy_addrs(self, proxy_addrs: list[str]) -> None:
        """Update the proxy addresses."""
        self.proxy_addrs = proxy_addrs
        self._proxy_enabled = bool(proxy_addrs)

    def reset(self) -> None:
        """Reset the routing strategy's internal state."""
        self.selector.reset()


def test_proxy_router_without_proxy():
    """Test ProxyRouter without proxy addresses."""
    workers = [MockWorker(id=f"worker-{i}") for i in range(3)]
    router = ProxyRouter(workers=workers)

    # Test round-robin selection without proxy
    for i in range(6):
        worker, rank, proxy_addr = router.route()
        expected_rank = i % 3
        assert rank == expected_rank, f"Expected rank {expected_rank}, got {rank}"
        assert worker.id == f"worker-{expected_rank}"
        assert proxy_addr is None, "Proxy address should be None when not enabled"

    print("✓ Test without proxy passed")


def test_proxy_router_with_proxy():
    """Test ProxyRouter with proxy addresses using round-robin."""
    workers = [MockWorker(id=f"worker-{i}") for i in range(3)]
    proxy_addrs = [f"http://127.0.0.1:800{i}" for i in range(3)]
    router = ProxyRouter(workers=workers, proxy_addrs=proxy_addrs)

    # Test round-robin selection with proxy
    for i in range(6):
        worker, rank, proxy_addr = router.route()
        expected_rank = i % 3
        assert rank == expected_rank, f"Expected rank {expected_rank}, got {rank}"
        assert worker.id == f"worker-{expected_rank}"
        assert proxy_addr == f"http://127.0.0.1:800{expected_rank}"

    print("✓ Test with proxy passed")


def test_proxy_router_get_proxy_addr():
    """Test get_proxy_addr method."""
    workers = [MockWorker(id=f"worker-{i}") for i in range(3)]
    proxy_addrs = [f"http://127.0.0.1:800{i}" for i in range(3)]
    router = ProxyRouter(workers=workers, proxy_addrs=proxy_addrs)

    # Test getting specific proxy addresses
    for i in range(3):
        addr = router.get_proxy_addr(i)
        assert addr == f"http://127.0.0.1:800{i}"

    # Test invalid rank
    addr = router.get_proxy_addr(10)
    assert addr is None

    print("✓ Test get_proxy_addr passed")


def test_proxy_router_update_proxy_addrs():
    """Test update_proxy_addrs method."""
    workers = [MockWorker(id=f"worker-{i}") for i in range(3)]
    router = ProxyRouter(workers=workers)

    # Initially no proxy
    worker, rank, proxy_addr = router.route()
    assert proxy_addr is None

    # Update with proxy addresses
    new_proxy_addrs = [f"http://127.0.0.1:900{i}" for i in range(3)]
    router.update_proxy_addrs(new_proxy_addrs)

    # Now should have proxy addresses
    worker, rank, proxy_addr = router.route()
    assert proxy_addr is not None
    assert proxy_addr.startswith("http://127.0.0.1:900")

    print("✓ Test update_proxy_addrs passed")


def test_proxy_router_empty_workers():
    """Test ProxyRouter with no workers."""
    router = ProxyRouter(workers=[])

    try:
        router.route()
        assert False, "Should raise RuntimeError"
    except RuntimeError as e:
        assert "No workers available" in str(e)

    print("✓ Test empty workers passed")


def test_proxy_router_random_strategy():
    """Test ProxyRouter with random routing strategy using enum."""
    workers = [MockWorker(id=f"worker-{i}") for i in range(3)]
    proxy_addrs = [f"http://127.0.0.1:800{i}" for i in range(3)]
    router = ProxyRouter(
        workers=workers,
        proxy_addrs=proxy_addrs,
        routing_strategy=RoutingStrategy.RANDOM,
    )

    # Test random selection - collect statistics
    selections = {0: 0, 1: 0, 2: 0}
    num_samples = 100

    for _ in range(num_samples):
        worker, rank, proxy_addr = router.route()
        selections[rank] += 1
        # Verify worker and proxy_addr match the rank
        assert worker.id == f"worker-{rank}"
        assert proxy_addr == f"http://127.0.0.1:800{rank}"

    # Verify all workers were selected at least once
    for rank, count in selections.items():
        assert count > 0, f"Worker {rank} was never selected in {num_samples} samples"

    print(f"✓ Test random strategy passed (distribution: {selections})")


def test_proxy_router_string_strategy():
    """Test ProxyRouter accepts string strategy and converts to enum."""
    workers = [MockWorker(id=f"worker-{i}") for i in range(3)]

    # Test with string "random"
    router_random = ProxyRouter(workers=workers, routing_strategy="random")
    assert router_random.strategy == RoutingStrategy.RANDOM

    # Test with string "round-robin"
    router_rr = ProxyRouter(workers=workers, routing_strategy="round-robin")
    assert router_rr.strategy == RoutingStrategy.ROUND_ROBIN

    print("✓ Test string strategy conversion passed")


def test_proxy_router_invalid_strategy():
    """Test ProxyRouter with invalid routing strategy."""
    workers = [MockWorker(id=f"worker-{i}") for i in range(3)]

    try:
        ProxyRouter(workers=workers, routing_strategy="invalid")
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "Invalid routing_strategy" in str(e)
        assert "round-robin" in str(e) or "random" in str(e)

    print("✓ Test invalid strategy passed")


def test_proxy_router_reset():
    """Test ProxyRouter reset functionality."""
    workers = [MockWorker(id=f"worker-{i}") for i in range(3)]
    router = ProxyRouter(workers=workers, routing_strategy=RoutingStrategy.ROUND_ROBIN)

    # Select a few workers
    _, rank1, _ = router.route()
    _, rank2, _ = router.route()
    assert rank1 == 0
    assert rank2 == 1

    # Reset and verify it starts from 0 again
    router.reset()
    _, rank3, _ = router.route()
    assert rank3 == 0

    print("✓ Test reset passed")


def test_strategy_pattern():
    """Test that the strategy pattern is properly implemented."""
    workers = [MockWorker(id=f"worker-{i}") for i in range(3)]

    # Create routers with different strategies
    router_rr = ProxyRouter(
        workers=workers, routing_strategy=RoutingStrategy.ROUND_ROBIN
    )
    router_random = ProxyRouter(
        workers=workers, routing_strategy=RoutingStrategy.RANDOM
    )

    # Verify they use different selector instances
    assert isinstance(router_rr.selector, RoundRobinSelector)
    assert isinstance(router_random.selector, RandomSelector)

    # Verify they behave differently
    # Round-robin should be deterministic
    router_rr.reset()
    ranks_rr = [router_rr.route()[1] for _ in range(6)]
    assert ranks_rr == [0, 1, 2, 0, 1, 2]

    print("✓ Test strategy pattern passed")


if __name__ == "__main__":
    print("=" * 70)
    print("ProxyRouter Test Suite - Strategy Pattern Implementation")
    print("=" * 70)
    print()

    test_proxy_router_without_proxy()
    test_proxy_router_with_proxy()
    test_proxy_router_get_proxy_addr()
    test_proxy_router_update_proxy_addrs()
    test_proxy_router_empty_workers()
    test_proxy_router_random_strategy()
    test_proxy_router_string_strategy()
    test_proxy_router_invalid_strategy()
    test_proxy_router_reset()
    test_strategy_pattern()

    print()
    print("=" * 70)
    print("All tests passed! ✓")
    print("=" * 70)
