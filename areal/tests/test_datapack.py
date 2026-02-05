"""Tests for datapack allocation functions."""

from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
import torch

from areal.api.cli_args import SchedulingSpec, TrainEngineConfig
from areal.api.engine_api import TrainEngine
from areal.api.io_struct import AllocationMode
from areal.infra import TrainController
from areal.scheduler.rpc.rtensor import RTensor, TensorShardInfo
from areal.utils.datapack import balanced_greedy_partition, ffd_allocate

# =============================================================================
# Test Data Generators
# =============================================================================


def generate_bimodal_seqlens(
    n_long: int, n_short: int, long_range: tuple, short_range: tuple, seed: int = 42
):
    """Generate bimodal distribution of sequence lengths (common in RL with varied prompts)."""
    rng = np.random.default_rng(seed)
    long_seqs = rng.integers(long_range[0], long_range[1], size=n_long).tolist()
    short_seqs = rng.integers(short_range[0], short_range[1], size=n_short).tolist()
    all_seqs = long_seqs + short_seqs
    rng.shuffle(all_seqs)
    return list(all_seqs)


def generate_uniform_seqlens(n: int, low: int, high: int, seed: int = 42):
    """Generate uniformly distributed sequence lengths."""
    rng = np.random.default_rng(seed)
    return rng.integers(low, high, size=n).tolist()


def generate_skewed_seqlens(n: int, max_len: int, skew: float = 2.0, seed: int = 42):
    """Generate skewed distribution (many short, few long - typical for chat/code)."""
    rng = np.random.default_rng(seed)
    # Use beta distribution to create skew
    samples = rng.beta(1, skew, size=n)
    return [int(s * max_len) + 1 for s in samples]


def generate_exponential_seqlens(n: int, scale: float = 500.0, seed: int = 42):
    """Generate exponentially distributed sequence lengths (common in NLP tasks)."""
    rng = np.random.default_rng(seed)
    samples = rng.exponential(scale=scale, size=n)
    # Clip to reasonable range and convert to int
    return [max(1, min(int(s), 8192)) for s in samples]


def generate_multimodal_seqlens(
    n: int, modes: list[tuple[int, int, float]], seed: int = 42
):
    """Generate multimodal distribution with multiple peaks.

    Args:
        n: Total number of sequences
        modes: List of (mean, std, weight) tuples for each mode
        seed: Random seed
    """
    rng = np.random.default_rng(seed)
    total_weight = sum(w for _, _, w in modes)
    seqlens = []

    for mean, std, weight in modes:
        count = int(n * weight / total_weight)
        samples = rng.normal(mean, std, size=count)
        seqlens.extend([max(1, int(s)) for s in samples])

    # Fill remaining with first mode
    while len(seqlens) < n:
        mean, std, _ = modes[0]
        seqlens.append(max(1, int(rng.normal(mean, std))))

    rng.shuffle(seqlens)
    return seqlens[:n]


def generate_power_law_seqlens(n: int, alpha: float = 2.0, seed: int = 42):
    """Generate power-law distributed sequence lengths (Zipf-like)."""
    rng = np.random.default_rng(seed)
    # Use Pareto distribution (power law)
    samples = (rng.pareto(alpha, size=n) + 1) * 100
    return [max(1, min(int(s), 8192)) for s in samples]


def generate_batch_realistic_seqlens(
    batch_size: int, prompt_range: tuple, response_range: tuple, seed: int = 42
):
    """Generate realistic batch with prompt+response patterns (typical in RL training).

    In RL training, each sample has a prompt (input) and response (generated).
    The total sequence length varies based on both components.
    """
    rng = np.random.default_rng(seed)
    prompts = rng.integers(prompt_range[0], prompt_range[1], size=batch_size)
    responses = rng.integers(response_range[0], response_range[1], size=batch_size)
    return [int(p + r) for p, r in zip(prompts, responses)]


def generate_code_seqlens(n: int, seed: int = 42):
    """Generate sequence lengths typical for code generation tasks.

    Code has characteristic length distribution:
    - Many short snippets (1-100 tokens)
    - Medium functions (100-500 tokens)
    - Fewer long functions (500-2000 tokens)
    - Rare very long files (2000+ tokens)
    """
    return generate_multimodal_seqlens(
        n=n,
        modes=[
            (50, 30, 0.4),  # Short snippets
            (250, 100, 0.35),  # Medium functions
            (800, 300, 0.2),  # Long functions
            (2000, 500, 0.05),  # Very long files
        ],
        seed=seed,
    )


def generate_chat_seqlens(n: int, seed: int = 42):
    """Generate sequence lengths typical for chat/conversation data.

    Chat has characteristic length distribution:
    - Many short messages (10-100 tokens)
    - Medium responses (100-500 tokens)
    - Fewer long explanations (500-1500 tokens)
    """
    return generate_multimodal_seqlens(
        n=n,
        modes=[
            (50, 30, 0.5),  # Short messages
            (200, 80, 0.35),  # Medium responses
            (800, 300, 0.15),  # Long explanations
        ],
        seed=seed,
    )


def generate_math_seqlens(n: int, seed: int = 42):
    """Generate sequence lengths typical for math problem solving.

    Math problems have:
    - Short problem statements (50-200 tokens)
    - Variable solution lengths (100-1000 tokens)
    """
    return generate_batch_realistic_seqlens(
        batch_size=n,
        prompt_range=(50, 200),
        response_range=(100, 800),
        seed=seed,
    )


class TestBalancedGreedyPartition:
    """Tests for balanced_greedy_partition function."""

    @pytest.mark.parametrize("K", [2, 4, 8])
    def test_basic_partition(self, K):
        """Test basic partition returns correct structure."""
        n = K * 10  # 10 items per group
        nums = list(range(100, 100 + n))

        groups = balanced_greedy_partition(nums, K)

        assert len(groups) == K
        # Each group should have n/K items
        for g in groups:
            assert len(g) == n // K

    def test_returns_indices_not_values(self):
        """Test that function returns indices, not values."""
        nums = [100, 200, 300, 400]
        K = 2

        groups = balanced_greedy_partition(nums, K)

        # Groups should contain indices (0-3), not values (100-400)
        all_indices = sorted(i for g in groups for i in g)
        assert all_indices == [0, 1, 2, 3]

        # Verify we can use indices to get original values
        for g in groups:
            values = [nums[i] for i in g]
            assert all(100 <= v <= 400 for v in values)

    def test_preserves_all_indices(self):
        """Test that all indices are assigned exactly once."""
        nums = [50, 100, 150, 200, 250, 300]
        K = 2

        groups = balanced_greedy_partition(nums, K)

        all_indices = sorted(i for g in groups for i in g)
        assert all_indices == list(range(len(nums)))

    def test_equal_group_sizes(self):
        """Test that all groups have equal size."""
        nums = list(range(24))
        K = 4

        groups = balanced_greedy_partition(nums, K)

        expected_size = len(nums) // K
        for g in groups:
            assert len(g) == expected_size

    def test_raises_on_non_divisible(self):
        """Test error when n is not divisible by K."""
        nums = [1, 2, 3, 4, 5]
        K = 2

        with pytest.raises(ValueError, match="must be divisible by K"):
            balanced_greedy_partition(nums, K)

    def test_raises_on_too_few_items(self):
        """Test error when n < K."""
        nums = [1, 2, 3]
        K = 5

        with pytest.raises(ValueError, match="must be >= K"):
            balanced_greedy_partition(nums, K)

    def test_raises_on_empty_input(self):
        """Test error when input is empty."""
        nums = []
        K = 4

        with pytest.raises(ValueError, match="must be >= K"):
            balanced_greedy_partition(nums, K)

    def test_balances_sums(self):
        """Test that group sums are well balanced."""
        # Create values with high variance
        nums = [1000, 900, 800, 700, 100, 200, 300, 400]
        K = 2

        groups = balanced_greedy_partition(nums, K)

        sums = [sum(nums[i] for i in g) for g in groups]

        # Sums should be reasonably balanced
        total = sum(nums)
        expected_avg = total / K
        # Each group should be within 20% of average
        for s in sums:
            assert abs(s - expected_avg) / expected_avg < 0.3

    @pytest.mark.parametrize("seed", [42, 123, 456, 789, 1000])
    def test_bimodal_distribution(self, seed):
        """Test with bimodal sequence lengths (typical for RL with varied prompts)."""
        n_long, n_short = 8, 24
        values = generate_bimodal_seqlens(
            n_long=n_long,
            n_short=n_short,
            long_range=(1000, 2000),
            short_range=(100, 400),
            seed=seed,
        )
        K = 4

        groups = balanced_greedy_partition(values, K)

        # Verify structure
        assert len(groups) == K
        all_indices = sorted(i for g in groups for i in g)
        assert all_indices == list(range(len(values)))

        # Verify balance
        sums = [sum(values[i] for i in g) for g in groups]
        max_diff = max(sums) - min(sums)
        avg_sum = sum(sums) / K
        # Difference should be reasonable
        assert max_diff / avg_sum < 0.5

    @pytest.mark.parametrize("seed", [42, 123, 456])
    def test_uniform_distribution(self, seed):
        """Test with uniformly distributed sequence lengths."""
        values = generate_uniform_seqlens(n=200, low=512, high=2048, seed=seed)
        K = 8

        groups = balanced_greedy_partition(values, K)

        # Verify structure
        assert len(groups) == K
        for g in groups:
            assert len(g) == 25  # 200 / 8

        # Verify balance
        sums = [sum(values[i] for i in g) for g in groups]
        max_diff = max(sums) - min(sums)
        avg_sum = sum(sums) / K
        assert max_diff / avg_sum < 0.2

    @pytest.mark.parametrize("seed", [42, 123, 456])
    def test_skewed_distribution(self, seed):
        """Test with skewed distribution (many short, few long - like chat data)."""
        values = generate_skewed_seqlens(n=160, max_len=4096, skew=3.0, seed=seed)
        K = 4

        groups = balanced_greedy_partition(values, K)

        # Verify structure
        assert len(groups) == K
        all_indices = sorted(i for g in groups for i in g)
        assert all_indices == list(range(len(values)))

    def test_edge_case_identical_values(self):
        """Test when all values are identical."""
        nums = [100] * 20
        K = 4

        groups = balanced_greedy_partition(nums, K)

        # All groups should have equal size and equal sum
        assert len(groups) == K
        sums = [sum(nums[i] for i in g) for g in groups]
        assert all(s == sums[0] for s in sums)

    def test_edge_case_two_values(self):
        """Test with extreme two-value distribution."""
        nums = [1000] * 4 + [1] * 4
        K = 2

        groups = balanced_greedy_partition(nums, K)

        # Each group should ideally have 2 large and 2 small
        sums = [sum(nums[i] for i in g) for g in groups]
        # Both sums should be close
        assert abs(sums[0] - sums[1]) <= 2  # Small difference due to 1s

    @pytest.mark.parametrize("dp_size", [2, 4, 8])
    def test_realistic_dp_sizes(self, dp_size):
        """Test with realistic data parallel sizes."""
        n_seqs = dp_size * 16  # 16 sequences per DP rank
        values = generate_bimodal_seqlens(
            n_long=n_seqs // 4,
            n_short=n_seqs - n_seqs // 4,
            long_range=(1000, 2000),
            short_range=(200, 600),
            seed=42,
        )

        groups = balanced_greedy_partition(values, dp_size)

        assert len(groups) == dp_size
        for g in groups:
            assert len(g) == 16

        # Check balance
        sums = [sum(values[i] for i in g) for g in groups]
        max_diff = max(sums) - min(sums)
        avg_sum = sum(sums) / dp_size
        assert max_diff / avg_sum < 0.3

    def test_large_scale(self):
        """Test with larger number of items."""
        n = 1000
        K = 10
        values = generate_uniform_seqlens(n=n, low=100, high=1000, seed=42)

        groups = balanced_greedy_partition(values, K)

        assert len(groups) == K
        all_indices = sorted(i for g in groups for i in g)
        assert all_indices == list(range(n))

        # Check balance
        sums = [sum(values[i] for i in g) for g in groups]
        max_diff = max(sums) - min(sums)
        avg_sum = sum(sums) / K
        assert max_diff / avg_sum < 0.1  # Should be very balanced

    def test_single_item_per_group(self):
        """Test edge case where each group gets exactly one item."""
        nums = [100, 200, 300, 400]
        K = 4

        groups = balanced_greedy_partition(nums, K)

        assert len(groups) == K
        for g in groups:
            assert len(g) == 1

    def test_deterministic(self):
        """Test that function is deterministic."""
        nums = [300, 100, 400, 200, 500, 600]
        K = 2

        groups1 = balanced_greedy_partition(nums, K)
        groups2 = balanced_greedy_partition(nums, K)

        # Should produce same result
        assert groups1 == groups2


class TestFFDAllocate:
    """Tests for existing ffd_allocate function to ensure no regression."""

    def test_basic_allocation(self):
        """Test basic FFD allocation."""
        values = [100, 200, 300, 150, 250]
        capacity = 500
        min_groups = 2

        groups = ffd_allocate(values, capacity, min_groups)

        assert len(groups) >= min_groups
        # All indices should be present
        all_indices = sorted(i for g in groups for i in g)
        assert all_indices == list(range(len(values)))
        # Each group should respect capacity
        for g in groups:
            total = sum(values[i] for i in g)
            assert total <= capacity

    def test_respects_min_groups(self):
        """Test that min_groups constraint is respected."""
        values = [50] * 10
        capacity = 1000
        min_groups = 4

        groups = ffd_allocate(values, capacity, min_groups)

        assert len(groups) >= min_groups

    def test_raises_on_value_exceeds_capacity(self):
        """Test error when a value exceeds capacity."""
        values = [100, 600, 200]
        capacity = 500

        with pytest.raises(RuntimeError, match="larger than capacity"):
            ffd_allocate(values, capacity, min_groups=1)

    def test_raises_on_insufficient_values(self):
        """Test error when not enough values for min_groups."""
        values = [100, 200]
        capacity = 500
        min_groups = 5

        with pytest.raises(RuntimeError, match="smaller than min_groups"):
            ffd_allocate(values, capacity, min_groups)


# =============================================================================
# Integration Tests: RTensor Data Parallel Dispatch
# =============================================================================


class TestRTensorDataParallelDispatchIntegration:
    """Integration tests for RTensor.data_parallel_dispatch with balanced_greedy_partition.

    These tests verify that global batches are split into equal sizes for different
    DP ranks when processed through the RTensor dispatch mechanism.
    """

    def _create_rtensor_with_seqlens(self, seqlens: list[int]):
        """Helper to create an RTensor with specified sequence lengths."""
        shards = [
            TensorShardInfo(size=1, seqlens=[slen], shard_id=str(i), node_addr="")
            for i, slen in enumerate(seqlens)
        ]
        max_len = max(seqlens) if seqlens else 1
        data = torch.zeros(len(seqlens), max_len)
        return RTensor(shards=shards, data=data)

    @pytest.mark.parametrize("dp_size", [2, 4, 8])
    def test_equal_split_uniform_distribution(self, dp_size):
        """Test that uniform distribution splits into equal-size groups."""
        n_seqs = dp_size * 16  # 16 sequences per DP rank
        seqlens = generate_uniform_seqlens(n=n_seqs, low=100, high=1000, seed=42)
        rtensor = self._create_rtensor_with_seqlens(seqlens)

        split_rtensors, group_indices = RTensor.data_parallel_dispatch(
            rtensor, dp_size=dp_size
        )

        # Verify equal split sizes
        assert group_indices is not None
        assert len(split_rtensors) == dp_size
        expected_size = n_seqs // dp_size
        for i, rt in enumerate(split_rtensors):
            assert len(rt.shards) == expected_size, (
                f"DP rank {i} got {len(rt.shards)} shards, expected {expected_size}"
            )
            assert len(group_indices[i]) == expected_size

    @pytest.mark.parametrize("dp_size", [2, 4, 8])
    def test_equal_split_bimodal_distribution(self, dp_size):
        """Test that bimodal distribution splits into equal-size groups."""
        n_seqs = dp_size * 20
        seqlens = generate_bimodal_seqlens(
            n_long=n_seqs // 4,
            n_short=n_seqs - n_seqs // 4,
            long_range=(1000, 2000),
            short_range=(100, 400),
            seed=42,
        )
        rtensor = self._create_rtensor_with_seqlens(seqlens)

        split_rtensors, group_indices = RTensor.data_parallel_dispatch(
            rtensor, dp_size=dp_size
        )

        # Verify equal split sizes
        expected_size = n_seqs // dp_size
        for i, rt in enumerate(split_rtensors):
            assert len(rt.shards) == expected_size

    @pytest.mark.parametrize("dp_size", [2, 4, 8])
    def test_equal_split_skewed_distribution(self, dp_size):
        """Test that skewed distribution splits into equal-size groups."""
        n_seqs = dp_size * 24
        seqlens = generate_skewed_seqlens(n=n_seqs, max_len=2000, skew=3.0, seed=42)
        rtensor = self._create_rtensor_with_seqlens(seqlens)

        split_rtensors, group_indices = RTensor.data_parallel_dispatch(
            rtensor, dp_size=dp_size
        )

        expected_size = n_seqs // dp_size
        for i, rt in enumerate(split_rtensors):
            assert len(rt.shards) == expected_size

    @pytest.mark.parametrize("dp_size", [2, 4, 8])
    def test_equal_split_exponential_distribution(self, dp_size):
        """Test that exponential distribution splits into equal-size groups."""
        n_seqs = dp_size * 16
        seqlens = generate_exponential_seqlens(n=n_seqs, scale=500.0, seed=42)
        rtensor = self._create_rtensor_with_seqlens(seqlens)

        split_rtensors, group_indices = RTensor.data_parallel_dispatch(
            rtensor, dp_size=dp_size
        )

        expected_size = n_seqs // dp_size
        for i, rt in enumerate(split_rtensors):
            assert len(rt.shards) == expected_size

    @pytest.mark.parametrize("dp_size", [2, 4, 8])
    def test_equal_split_code_distribution(self, dp_size):
        """Test that code-like distribution splits into equal-size groups."""
        n_seqs = dp_size * 32
        seqlens = generate_code_seqlens(n=n_seqs, seed=42)
        rtensor = self._create_rtensor_with_seqlens(seqlens)

        split_rtensors, group_indices = RTensor.data_parallel_dispatch(
            rtensor, dp_size=dp_size
        )

        expected_size = n_seqs // dp_size
        for i, rt in enumerate(split_rtensors):
            assert len(rt.shards) == expected_size

    @pytest.mark.parametrize("dp_size", [2, 4, 8])
    def test_equal_split_chat_distribution(self, dp_size):
        """Test that chat-like distribution splits into equal-size groups."""
        n_seqs = dp_size * 24
        seqlens = generate_chat_seqlens(n=n_seqs, seed=42)
        rtensor = self._create_rtensor_with_seqlens(seqlens)

        split_rtensors, group_indices = RTensor.data_parallel_dispatch(
            rtensor, dp_size=dp_size
        )

        expected_size = n_seqs // dp_size
        for i, rt in enumerate(split_rtensors):
            assert len(rt.shards) == expected_size

    @pytest.mark.parametrize("dp_size", [2, 4, 8])
    def test_equal_split_math_distribution(self, dp_size):
        """Test that math problem distribution splits into equal-size groups."""
        n_seqs = dp_size * 16
        seqlens = generate_math_seqlens(n=n_seqs, seed=42)
        rtensor = self._create_rtensor_with_seqlens(seqlens)

        split_rtensors, group_indices = RTensor.data_parallel_dispatch(
            rtensor, dp_size=dp_size
        )

        expected_size = n_seqs // dp_size
        for i, rt in enumerate(split_rtensors):
            assert len(rt.shards) == expected_size

    @pytest.mark.parametrize("dp_size", [2, 4, 8])
    def test_equal_split_power_law_distribution(self, dp_size):
        """Test that power-law distribution splits into equal-size groups."""
        n_seqs = dp_size * 20
        seqlens = generate_power_law_seqlens(n=n_seqs, alpha=2.0, seed=42)
        rtensor = self._create_rtensor_with_seqlens(seqlens)

        split_rtensors, group_indices = RTensor.data_parallel_dispatch(
            rtensor, dp_size=dp_size
        )

        expected_size = n_seqs // dp_size
        for i, rt in enumerate(split_rtensors):
            assert len(rt.shards) == expected_size

    @pytest.mark.parametrize("seed", [42, 123, 456, 789, 1000, 2024, 3141, 9999])
    def test_equal_split_various_seeds(self, seed):
        """Test equal split consistency across many random seeds."""
        dp_size = 4
        n_seqs = 64
        seqlens = generate_bimodal_seqlens(
            n_long=16,
            n_short=48,
            long_range=(500, 1500),
            short_range=(50, 300),
            seed=seed,
        )
        rtensor = self._create_rtensor_with_seqlens(seqlens)

        split_rtensors, group_indices = RTensor.data_parallel_dispatch(
            rtensor, dp_size=dp_size
        )

        expected_size = n_seqs // dp_size
        for i, rt in enumerate(split_rtensors):
            assert len(rt.shards) == expected_size

    def test_all_indices_preserved_after_split(self):
        """Test that all original indices are preserved after dispatch."""
        dp_size = 4
        n_seqs = 100
        seqlens = generate_uniform_seqlens(n=n_seqs, low=100, high=500, seed=42)
        rtensor = self._create_rtensor_with_seqlens(seqlens)

        _, group_indices = RTensor.data_parallel_dispatch(rtensor, dp_size=dp_size)
        assert group_indices is not None

        # All indices should be present exactly once
        all_indices = sorted(i for g in group_indices for i in g)
        assert all_indices == list(range(n_seqs))

    def test_token_balance_across_dp_ranks(self):
        """Test that total tokens are reasonably balanced across DP ranks."""
        dp_size = 4
        seqlens = generate_bimodal_seqlens(
            n_long=20,
            n_short=60,
            long_range=(1000, 2000),
            short_range=(100, 300),
            seed=42,
        )
        rtensor = self._create_rtensor_with_seqlens(seqlens)

        _, group_indices = RTensor.data_parallel_dispatch(rtensor, dp_size=dp_size)
        assert group_indices is not None

        # Compute total tokens per rank
        tokens_per_rank = [sum(seqlens[i] for i in g) for g in group_indices]
        avg_tokens = sum(tokens_per_rank) / dp_size
        max_diff = max(tokens_per_rank) - min(tokens_per_rank)

        # Token imbalance should be reasonable (< 30% of average)
        assert max_diff / avg_tokens < 0.3, (
            f"Token imbalance too high: {max_diff / avg_tokens:.2%}"
        )

    @pytest.mark.parametrize(
        "batch_size,dp_size",
        [
            (32, 2),
            (64, 4),
            (128, 8),
            (256, 8),
            (512, 8),
            (1024, 8),
        ],
    )
    def test_large_batch_equal_split(self, batch_size, dp_size):
        """Test equal split for various large batch sizes."""
        seqlens = generate_uniform_seqlens(n=batch_size, low=100, high=2000, seed=42)
        rtensor = self._create_rtensor_with_seqlens(seqlens)

        split_rtensors, group_indices = RTensor.data_parallel_dispatch(
            rtensor, dp_size=dp_size
        )

        expected_size = batch_size // dp_size
        for i, rt in enumerate(split_rtensors):
            assert len(rt.shards) == expected_size

    def test_dispatch_with_nested_dict(self):
        """Test equal split when RTensor is in a nested dict structure."""
        dp_size = 4
        n_seqs = 64
        seqlens = generate_uniform_seqlens(n=n_seqs, low=100, high=500, seed=42)
        rtensor = self._create_rtensor_with_seqlens(seqlens)

        # Wrap in nested dict structure
        nested_data = {
            "input_ids": rtensor,
            "metadata": {"batch_size": n_seqs, "dp_size": dp_size},
        }

        split_results, group_indices = RTensor.data_parallel_dispatch(
            nested_data, dp_size=dp_size
        )

        expected_size = n_seqs // dp_size
        for i, result in enumerate(split_results):
            assert len(result["input_ids"].shards) == expected_size
            # Scalar values should be replicated
            assert result["metadata"]["batch_size"] == n_seqs

    def test_dispatch_with_multiple_rtensors(self):
        """Test equal split when multiple RTensors share the same layout."""
        dp_size = 4
        n_seqs = 48
        seqlens = generate_uniform_seqlens(n=n_seqs, low=100, high=500, seed=42)

        input_rtensor = self._create_rtensor_with_seqlens(seqlens)
        # Create another RTensor with same shard structure
        labels_rtensor = self._create_rtensor_with_seqlens(seqlens)

        data = {
            "input_ids": input_rtensor,
            "labels": labels_rtensor,
        }

        split_results, group_indices = RTensor.data_parallel_dispatch(
            data, dp_size=dp_size
        )

        expected_size = n_seqs // dp_size
        for i, result in enumerate(split_results):
            assert len(result["input_ids"].shards) == expected_size
            assert len(result["labels"].shards) == expected_size


class TestTrainControllerDispatchIntegration:
    """Integration tests for TrainController._dispatch_inputs with equal-size splits.

    These tests simulate the full dispatch flow from TrainController to verify
    that batches are split into equal sizes for different DP ranks.
    """

    @pytest.fixture
    def mock_scheduler(self):
        """Create a mock scheduler for testing."""
        scheduler = Mock()
        scheduler.async_call_engine = AsyncMock(return_value=None)
        return scheduler

    @pytest.fixture
    def train_config(self):
        """Create a TrainEngineConfig for testing."""
        return TrainEngineConfig(
            scheduling_spec=(
                SchedulingSpec(cpu=4, gpu=1, mem=16000, port_count=2, cmd="dummy"),
            )
        )

    def _create_rtensor_with_seqlens(self, seqlens: list[int]):
        """Helper to create an RTensor with specified sequence lengths."""
        shards = [
            TensorShardInfo(size=1, seqlens=[slen], shard_id=str(i), node_addr="")
            for i, slen in enumerate(seqlens)
        ]
        max_len = max(seqlens) if seqlens else 1
        data = torch.zeros(len(seqlens), max_len)
        return RTensor(shards=shards, data=data)

    @pytest.mark.parametrize("dp_size", [2, 4, 8])
    def test_train_controller_dispatch_equal_split(
        self, mock_scheduler, train_config, dp_size
    ):
        """Test TrainController dispatches batches to equal-size DP groups."""

        class MockTrainEngine(TrainEngine):
            pass

        controller = TrainController(
            train_engine=MockTrainEngine,
            config=train_config,
            scheduler=mock_scheduler,
        )

        alloc_mode = AllocationMode.from_str(f"d{dp_size}p1t1")
        controller.parallel_strategy = alloc_mode.train

        # Create batch with sequences that should split equally
        n_seqs = dp_size * 16
        seqlens = generate_bimodal_seqlens(
            n_long=n_seqs // 4,
            n_short=n_seqs - n_seqs // 4,
            long_range=(500, 1500),
            short_range=(100, 300),
            seed=42,
        )
        rtensor = self._create_rtensor_with_seqlens(seqlens)

        split_args, split_kwargs, group_indices = controller._dispatch_inputs(rtensor)

        # Verify equal split
        expected_size = n_seqs // dp_size
        assert len(group_indices) == dp_size
        for i, g in enumerate(group_indices):
            assert len(g) == expected_size, (
                f"DP rank {i} got {len(g)} sequences, expected {expected_size}"
            )

    @pytest.mark.parametrize(
        "generator_name,generator_func",
        [
            ("uniform", lambda n, seed: generate_uniform_seqlens(n, 100, 1000, seed)),
            (
                "bimodal",
                lambda n, seed: generate_bimodal_seqlens(
                    n // 4, n - n // 4, (800, 1500), (100, 300), seed
                ),
            ),
            ("skewed", lambda n, seed: generate_skewed_seqlens(n, 2000, 3.0, seed)),
            (
                "exponential",
                lambda n, seed: generate_exponential_seqlens(n, 500.0, seed),
            ),
            ("code", lambda n, seed: generate_code_seqlens(n, seed)),
            ("chat", lambda n, seed: generate_chat_seqlens(n, seed)),
            ("math", lambda n, seed: generate_math_seqlens(n, seed)),
            ("power_law", lambda n, seed: generate_power_law_seqlens(n, 2.0, seed)),
        ],
    )
    def test_train_controller_dispatch_all_distributions(
        self, mock_scheduler, train_config, generator_name, generator_func
    ):
        """Test TrainController dispatch with all distribution types."""

        class MockTrainEngine(TrainEngine):
            pass

        controller = TrainController(
            train_engine=MockTrainEngine,
            config=train_config,
            scheduler=mock_scheduler,
        )

        dp_size = 4
        alloc_mode = AllocationMode.from_str(f"d{dp_size}p1t1")
        controller.parallel_strategy = alloc_mode.train

        n_seqs = dp_size * 20
        seqlens = generator_func(n_seqs, seed=42)
        rtensor = self._create_rtensor_with_seqlens(seqlens)

        _, _, group_indices = controller._dispatch_inputs(rtensor)

        expected_size = n_seqs // dp_size
        for i, g in enumerate(group_indices):
            assert len(g) == expected_size, (
                f"Distribution '{generator_name}': DP rank {i} got {len(g)} "
                f"sequences, expected {expected_size}"
            )

    def test_train_controller_dispatch_preserves_indices(
        self, mock_scheduler, train_config
    ):
        """Test that all sequence indices are preserved after dispatch."""

        class MockTrainEngine(TrainEngine):
            pass

        controller = TrainController(
            train_engine=MockTrainEngine,
            config=train_config,
            scheduler=mock_scheduler,
        )

        dp_size = 4
        alloc_mode = AllocationMode.from_str(f"d{dp_size}p1t1")
        controller.parallel_strategy = alloc_mode.train

        n_seqs = 100
        seqlens = generate_uniform_seqlens(n=n_seqs, low=100, high=500, seed=42)
        rtensor = self._create_rtensor_with_seqlens(seqlens)

        _, _, group_indices = controller._dispatch_inputs(rtensor)

        all_indices = sorted(i for g in group_indices for i in g)
        assert all_indices == list(range(n_seqs))

    @pytest.mark.parametrize("seed", [42, 123, 456, 789, 1000])
    def test_train_controller_dispatch_deterministic(
        self, mock_scheduler, train_config, seed
    ):
        """Test that dispatch is deterministic for the same input."""

        class MockTrainEngine(TrainEngine):
            pass

        controller = TrainController(
            train_engine=MockTrainEngine,
            config=train_config,
            scheduler=mock_scheduler,
        )

        dp_size = 4
        alloc_mode = AllocationMode.from_str(f"d{dp_size}p1t1")
        controller.parallel_strategy = alloc_mode.train

        seqlens = generate_bimodal_seqlens(
            n_long=16,
            n_short=48,
            long_range=(500, 1500),
            short_range=(100, 300),
            seed=seed,
        )
        rtensor = self._create_rtensor_with_seqlens(seqlens)

        _, _, group_indices1 = controller._dispatch_inputs(rtensor)

        # Dispatch again with same data
        rtensor2 = self._create_rtensor_with_seqlens(seqlens)
        _, _, group_indices2 = controller._dispatch_inputs(rtensor2)

        assert group_indices1 == group_indices2
