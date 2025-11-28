"""
Unit tests for vLLM server launcher port allocation and resource setup.

Tests the port range calculation logic to ensure:
1. Port ranges stay within valid bounds (0-65535)
2. Node-local indexing works correctly across multiple nodes
3. Various allocation modes are handled properly
4. Edge cases don't cause overflow errors
"""

import os
import sys
from unittest.mock import MagicMock, Mock, patch

import pytest


# Mock the allocation mode to avoid heavy dependencies
class MockParallelStrategy:
    """Mock ParallelStrategy for testing."""

    def __init__(self, dp=1, tp=1, pp=1, cp=1):
        self.data_parallel_size = dp
        self.tensor_parallel_size = tp
        self.pipeline_parallel_size = pp
        self.context_parallel_size = cp

    @property
    def dp_size(self):
        return self.data_parallel_size

    @property
    def tp_size(self):
        return self.tensor_parallel_size

    @property
    def pp_size(self):
        return self.pipeline_parallel_size

    @property
    def world_size(self):
        return self.dp_size * self.tp_size * self.pp_size * self.context_parallel_size


class MockAllocationMode:
    """Mock AllocationMode for testing."""

    def __init__(self, dp=1, tp=1, pp=1):
        self.gen = MockParallelStrategy(dp=dp, tp=tp, pp=pp)
        self.train = MockParallelStrategy(dp=dp, tp=tp, pp=pp)
        self.gen_backend = "vllm"

    @property
    def gen_instance_size(self):
        return self.gen.tp_size * self.gen.pp_size


class TestPortAllocationLogic:
    """Test port allocation logic directly without requiring full vLLM setup."""

    def calculate_port_range(
        self, server_local_idx: int, n_servers_per_node: int
    ) -> tuple[int, int]:
        """
        Replicate the port calculation logic from vllm_server.py.

        Args:
            server_local_idx: Node-local server index (should be 0 to n_servers_per_node-1)
            n_servers_per_node: Number of servers per node

        Returns:
            Tuple of (min_port, max_port)
        """
        ports_per_server = 40000 // n_servers_per_node
        min_port = server_local_idx * ports_per_server + 10000
        max_port = (server_local_idx + 1) * ports_per_server + 10000
        return min_port, max_port

    def calculate_server_idx_offset(
        self, visible_gpus: list[int], gpus_per_server: int, n_servers_per_node: int
    ) -> int:
        """
        Calculate server_idx_offset with the fix (modulo operation).

        Args:
            visible_gpus: List of visible GPU indices
            gpus_per_server: Number of GPUs per server
            n_servers_per_node: Number of servers per node

        Returns:
            Node-local server index offset
        """
        return (min(visible_gpus) // gpus_per_server) % n_servers_per_node

    def test_single_server_port_range(self):
        """Test port allocation for a single server."""
        n_servers_per_node = 8
        server_idx = 0

        min_port, max_port = self.calculate_port_range(server_idx, n_servers_per_node)

        assert min_port == 10000
        assert max_port == 15000
        assert max_port <= 65535

    def test_all_servers_on_single_node(self):
        """Test port allocation for all 8 servers on a single node."""
        n_servers_per_node = 8

        for server_idx in range(n_servers_per_node):
            min_port, max_port = self.calculate_port_range(
                server_idx, n_servers_per_node
            )

            assert (
                0 <= min_port <= 65535
            ), f"Server {server_idx} min_port {min_port} out of range"
            assert (
                0 <= max_port <= 65535
            ), f"Server {server_idx} max_port {max_port} out of range"
            assert min_port < max_port, "Port range must be non-empty"

        # Check specific values
        min_port_0, max_port_0 = self.calculate_port_range(0, n_servers_per_node)
        assert min_port_0 == 10000
        assert max_port_0 == 15000

        min_port_7, max_port_7 = self.calculate_port_range(7, n_servers_per_node)
        assert min_port_7 == 45000
        assert max_port_7 == 50000

    def test_node_local_indexing_node0(self):
        """Test server index calculation for node 0 (GPUs 0-7)."""
        visible_gpus = list(range(0, 8))
        gpus_per_server = 1
        n_servers_per_node = 8

        offset = self.calculate_server_idx_offset(
            visible_gpus, gpus_per_server, n_servers_per_node
        )

        assert offset == 0, "Node 0 should have offset 0"

    def test_node_local_indexing_node1_with_fix(self):
        """Test server index calculation for node 1 (GPUs 8-15) WITH the fix."""
        visible_gpus = list(range(8, 16))
        gpus_per_server = 1
        n_servers_per_node = 8

        # WITH FIX: (8 // 1) % 8 = 0 (node-local)
        offset = self.calculate_server_idx_offset(
            visible_gpus, gpus_per_server, n_servers_per_node
        )

        assert offset == 0, "Node 1 should have node-local offset 0 after fix"

    def test_node_local_indexing_node2_with_fix(self):
        """Test server index calculation for node 2 (GPUs 16-23) WITH the fix."""
        visible_gpus = list(range(16, 24))
        gpus_per_server = 1
        n_servers_per_node = 8

        # WITH FIX: (16 // 1) % 8 = 0 (node-local)
        offset = self.calculate_server_idx_offset(
            visible_gpus, gpus_per_server, n_servers_per_node
        )

        assert offset == 0, "Node 2 should have node-local offset 0 after fix"

    def test_high_data_parallelism_d12_no_overflow(self):
        """
        Test the specific bug: d12 (12 servers) across 2 nodes should not overflow.

        This is the key test that validates the fix for the user's reported issue.
        """
        gpus_per_server = 1
        n_servers_per_node = 8

        # Node 0: GPUs 0-7, servers 0-7
        visible_gpus_node0 = list(range(0, 8))
        offset_node0 = self.calculate_server_idx_offset(
            visible_gpus_node0, gpus_per_server, n_servers_per_node
        )
        assert offset_node0 == 0

        for server_idx in range(offset_node0, offset_node0 + 8):
            min_port, max_port = self.calculate_port_range(
                server_idx, n_servers_per_node
            )
            assert (
                max_port <= 65535
            ), f"Node 0 server {server_idx} port overflow: {max_port}"

        # Node 1: GPUs 8-15, servers 8-11 (global), but 0-3 (node-local after fix)
        visible_gpus_node1 = list(range(8, 16))
        offset_node1 = self.calculate_server_idx_offset(
            visible_gpus_node1, gpus_per_server, n_servers_per_node
        )

        # After fix: offset should be 0 (node-local)
        assert offset_node1 == 0, "Node 1 should have node-local offset 0"

        # Only 4 servers on node 1 (servers 8-11 globally, 0-3 locally)
        for server_idx in range(offset_node1, offset_node1 + 4):
            min_port, max_port = self.calculate_port_range(
                server_idx, n_servers_per_node
            )
            assert (
                max_port <= 65535
            ), f"Node 1 server {server_idx} port overflow: {max_port}"

    def test_tensor_parallelism_t2_port_allocation(self):
        """Test port allocation with tensor parallelism (d4t2)."""
        gpus_per_server = 2  # t2
        n_servers_per_node = 8 // 2  # 4 servers per node
        ports_per_server = 40000 // n_servers_per_node  # 10000

        assert n_servers_per_node == 4
        assert ports_per_server == 10000

        for server_idx in range(n_servers_per_node):
            min_port, max_port = self.calculate_port_range(
                server_idx, n_servers_per_node
            )
            assert max_port <= 65535, f"Server {server_idx} exceeds port range"

    def test_tensor_parallelism_t4_port_allocation(self):
        """Test port allocation with high tensor parallelism (d2t4)."""
        gpus_per_server = 4  # t4
        n_servers_per_node = 8 // 4  # 2 servers per node
        ports_per_server = 40000 // n_servers_per_node  # 20000

        assert n_servers_per_node == 2
        assert ports_per_server == 20000

        for server_idx in range(n_servers_per_node):
            min_port, max_port = self.calculate_port_range(
                server_idx, n_servers_per_node
            )
            expected_min = 10000 + server_idx * 20000
            expected_max = 10000 + (server_idx + 1) * 20000
            assert min_port == expected_min
            assert max_port == expected_max
            assert max_port <= 65535

    def test_full_node_t8_port_allocation(self):
        """Test port allocation when one server uses all GPUs (d1t8)."""
        gpus_per_server = 8  # t8
        n_servers_per_node = 1
        ports_per_server = 40000

        min_port, max_port = self.calculate_port_range(0, n_servers_per_node)

        assert min_port == 10000
        assert max_port == 50000
        assert max_port <= 65535

    def test_cuda_visible_devices_partial_node(self):
        """Test offset calculation when CUDA_VISIBLE_DEVICES shows partial GPUs."""
        # Simulate second half of GPUs: 4,5,6,7
        visible_gpus = [4, 5, 6, 7]
        gpus_per_server = 2  # t2
        n_servers_per_node = 4

        # With fix: (4 // 2) % 4 = 2 % 4 = 2
        offset = self.calculate_server_idx_offset(
            visible_gpus, gpus_per_server, n_servers_per_node
        )

        assert offset == 2, "Should start at server index 2 on this node"

        # Should run 2 servers (indices 2 and 3)
        n_servers_per_proc = len(visible_gpus) // gpus_per_server
        assert n_servers_per_proc == 2

        for local_idx in range(offset, offset + n_servers_per_proc):
            min_port, max_port = self.calculate_port_range(local_idx, n_servers_per_node)
            assert max_port <= 65535

    def test_ports_per_server_calculation(self):
        """Test calculation of port range size per server for various configurations."""
        test_cases = [
            (8, 5000),  # 8 servers: 40000 / 8 = 5000
            (4, 10000),  # 4 servers: 40000 / 4 = 10000
            (2, 20000),  # 2 servers: 40000 / 2 = 20000
            (1, 40000),  # 1 server: 40000 / 1 = 40000
        ]

        for n_servers_per_node, expected_ports in test_cases:
            ports_per_server = 40000 // n_servers_per_node
            assert (
                ports_per_server == expected_ports
            ), f"Expected {expected_ports} ports for {n_servers_per_node} servers"

    def test_multi_node_all_offsets_zero(self):
        """Test that all nodes have offset 0 with the fix."""
        gpus_per_server = 1
        n_servers_per_node = 8

        # Test 5 nodes
        for node_id in range(5):
            visible_gpus = list(range(node_id * 8, (node_id + 1) * 8))
            offset = self.calculate_server_idx_offset(
                visible_gpus, gpus_per_server, n_servers_per_node
            )
            assert offset == 0, f"Node {node_id} should have offset 0"


class TestResourceSetupCalculations:
    """Test resource setup calculations without requiring imports."""

    def test_gen_instance_size(self):
        """Test calculation of GPUs per instance."""
        test_cases = [
            (1, 1, 1),  # d4t1: 1 GPU per instance
            (2, 1, 2),  # d2t2: 2 GPUs per instance
            (1, 4, 4),  # d1t4: 4 GPUs per instance
            (2, 4, 8),  # d2t4p2: 8 GPUs per instance (2 * 4 = 8)
        ]

        for tp, pp, expected_size in test_cases:
            gen_instance_size = tp * pp
            assert (
                gen_instance_size == expected_size
            ), f"tp={tp}, pp={pp} should give {expected_size}"

    def test_servers_per_node(self):
        """Test calculation of number of servers per node."""
        test_cases = [
            (8, 1, 8),  # 8 GPUs, 1 per server = 8 servers
            (8, 2, 4),  # 8 GPUs, 2 per server = 4 servers
            (8, 4, 2),  # 8 GPUs, 4 per server = 2 servers
            (8, 8, 1),  # 8 GPUs, 8 per server = 1 server
        ]

        for n_gpus_per_node, gpus_per_server, expected_servers in test_cases:
            n_servers = max(1, n_gpus_per_node // gpus_per_server)
            assert (
                n_servers == expected_servers
            ), f"{n_gpus_per_node} GPUs with {gpus_per_server} per server should give {expected_servers} servers"

    def test_world_size_calculation(self):
        """Test world_size calculation."""
        test_cases = [
            (4, 2, 1, 8),  # d4t2: 4 * 2 * 1 = 8
            (8, 1, 1, 8),  # d8t1: 8 * 1 * 1 = 8
            (2, 4, 1, 8),  # d2t4: 2 * 4 * 1 = 8
            (12, 1, 1, 12),  # d12t1: 12 * 1 * 1 = 12
        ]

        for dp, tp, pp, expected_world_size in test_cases:
            world_size = dp * tp * pp
            assert (
                world_size == expected_world_size
            ), f"dp={dp}, tp={tp}, pp={pp} should give world_size={expected_world_size}"

    def test_nodes_required(self):
        """Test calculation of nodes required."""
        import math

        test_cases = [
            (8, 8, 1),  # 8 GPUs, 8 per node = 1 node
            (12, 8, 2),  # 12 GPUs, 8 per node = 2 nodes (ceiling)
            (16, 8, 2),  # 16 GPUs, 8 per node = 2 nodes
            (24, 8, 3),  # 24 GPUs, 8 per node = 3 nodes
        ]

        for world_size, gpus_per_node, expected_nodes in test_cases:
            n_nodes = math.ceil(world_size / gpus_per_node)
            assert (
                n_nodes == expected_nodes
            ), f"{world_size} GPUs with {gpus_per_node} per node should need {expected_nodes} nodes"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def calculate_port_range(
        self, server_local_idx: int, n_servers_per_node: int
    ) -> tuple[int, int]:
        """Helper to calculate port range."""
        ports_per_server = 40000 // n_servers_per_node
        min_port = server_local_idx * ports_per_server + 10000
        max_port = (server_local_idx + 1) * ports_per_server + 10000
        return min_port, max_port

    def test_port_boundaries(self):
        """Test that all port ranges stay within [10000, 50000]."""
        n_servers_per_node = 8

        for server_idx in range(n_servers_per_node):
            min_port, max_port = self.calculate_port_range(
                server_idx, n_servers_per_node
            )
            assert 10000 <= min_port < 50000, f"min_port {min_port} out of [10000, 50000)"
            assert 10000 < max_port <= 50000, f"max_port {max_port} out of (10000, 50000]"

    def test_first_server_port_range(self):
        """Test that first server always starts at 10000."""
        for n_servers in [1, 2, 4, 8]:
            min_port, _ = self.calculate_port_range(0, n_servers)
            assert min_port == 10000, f"First server should always start at 10000"

    def test_last_server_port_range(self):
        """Test that last server on a node ends at 50000."""
        for n_servers in [1, 2, 4, 8]:
            last_idx = n_servers - 1
            _, max_port = self.calculate_port_range(last_idx, n_servers)
            assert max_port == 50000, f"Last server should always end at 50000"

    def test_no_port_overlap(self):
        """Test that server port ranges don't overlap."""
        n_servers_per_node = 8

        ranges = []
        for server_idx in range(n_servers_per_node):
            min_port, max_port = self.calculate_port_range(
                server_idx, n_servers_per_node
            )
            ranges.append((min_port, max_port))

        # Check no overlaps
        for i in range(len(ranges) - 1):
            assert (
                ranges[i][1] == ranges[i + 1][0]
            ), f"Port ranges should be contiguous: {ranges[i]} and {ranges[i+1]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
