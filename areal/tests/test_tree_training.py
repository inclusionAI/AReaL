import pytest
import torch

from areal.utils.tree_training import greedy_build_tree


class TestGreedyBuildTree:
    """Test cases for greedy_build_tree function."""

    # ==================== Fixtures ====================

    @pytest.fixture
    def simple_sequences(self):
        r"""Simple test sequences with some overlap.

        Sequences:
        [1, 2, 3] - shares [1, 2] with seq2
        [1, 2, 4] - shares [1, 2] with seq1
        [1, 5, 6] - shares [1] with seq1 and seq2

        Expected tree structure:
              1
             / \
            2   5
           / \   \
          3   4   6
        """
        return [[1, 2, 3], [1, 2, 4], [1, 5, 6]]

    @pytest.fixture
    def identical_sequences(self):
        """Sequences that are all identical."""
        return [[1, 2, 3], [1, 2, 3], [1, 2, 3]]

    @pytest.fixture
    def no_overlap_sequences(self):
        """Sequences with no overlap at all."""
        return [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    @pytest.fixture
    def varying_length_sequences(self):
        """Sequences of varying lengths."""
        return [[1, 2], [1, 2, 3, 4], [1, 2, 3], [5, 6]]

    @pytest.fixture
    def deep_tree_sequences(self):
        """Sequences with long shared prefix (deep tree)."""
        return [
            [1, 2, 3, 4, 5, 6],
            [1, 2, 3, 4, 5, 7],
            [1, 2, 3, 4, 8, 9],
        ]

    @pytest.fixture
    def wide_tree_sequences(self):
        """Sequences with many branches at root (wide tree)."""
        return [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
        ]

    # ==================== Basic Functionality Tests ====================

    def test_single_sequence(self):
        """Test with a single sequence."""
        sequences = [[1, 2, 3, 4]]
        trees = greedy_build_tree(sequences, max_tokens_per_tree=10)

        assert len(trees) == 1
        tree = trees[0]
        assert tree.batch_size == 1
        assert tree.n_tree_tokens == 4  # 4 unique tokens
        assert tree.n_total_tokens == 4  # 4 total tokens
        assert len(tree.to_tree_indices) == 4
        assert len(tree.from_tree_indices) == 4

    def test_multiple_sequences_shared_prefix(self, simple_sequences):
        """Test sequences with shared prefixes create efficient trie."""
        trees = greedy_build_tree(simple_sequences, max_tokens_per_tree=20)

        assert len(trees) == 1
        tree = trees[0]
        assert tree.batch_size == 3
        # Tree structure: 1 shared, then 2 branches (2,5), then 3 leaves (3,4,6)
        # Total unique nodes: 6 (1, 2, 5, 3, 4, 6)
        assert tree.n_tree_tokens == 6
        # Total tokens: seq1(3) + seq2(3) + seq3(3) = 9
        assert tree.n_total_tokens == 9
        assert len(tree.from_tree_indices) == 9

    def test_multiple_sequences_no_overlap(self, no_overlap_sequences):
        """Test completely different sequences with no sharing."""
        trees = greedy_build_tree(no_overlap_sequences, max_tokens_per_tree=20)

        assert len(trees) == 1
        tree = trees[0]
        assert tree.batch_size == 3
        # No sharing, so n_tree_tokens = n_total_tokens
        assert tree.n_tree_tokens == 9  # 3 sequences × 3 tokens each
        assert tree.n_total_tokens == 9

    def test_identical_sequences(self, identical_sequences):
        """Test all identical sequences (maximum sharing)."""
        trees = greedy_build_tree(identical_sequences, max_tokens_per_tree=20)

        assert len(trees) == 1
        tree = trees[0]
        assert tree.batch_size == 3
        # All sequences share the same path, so only 3 unique nodes
        assert tree.n_tree_tokens == 3
        # But total tokens counts each sequence traversal
        assert tree.n_total_tokens == 9  # 3 sequences × 3 tokens each
        assert len(tree.from_tree_indices) == 9

    # ==================== Edge Cases ====================

    def test_empty_sequences_list(self):
        """Test with empty input list."""
        sequences = []
        trees = greedy_build_tree(sequences, max_tokens_per_tree=10)

        assert len(trees) == 0

    def test_single_token_sequences(self):
        """Test sequences with only one token each."""
        sequences = [[1], [2], [3]]
        trees = greedy_build_tree(sequences, max_tokens_per_tree=10)

        assert len(trees) == 1
        tree = trees[0]
        assert tree.batch_size == 3
        assert tree.n_tree_tokens == 3  # 3 unique single-token sequences
        assert tree.n_total_tokens == 3

    def test_empty_individual_sequence(self):
        """Test list containing an empty sequence."""
        sequences = [[1, 2, 3], [], [4, 5, 6]]
        trees = greedy_build_tree(sequences, max_tokens_per_tree=20)

        # Empty sequence is included in batch_size but adds no tokens
        assert len(trees) >= 1
        # Verify all sequences are counted (including empty one)
        total_batch_size = sum(tree.batch_size for tree in trees)
        assert total_batch_size == 3  # All 3 sequences including empty one
        # But total tokens should only be from non-empty sequences
        total_tokens = sum(tree.n_total_tokens for tree in trees)
        assert total_tokens == 6  # 3 + 3 tokens from non-empty sequences

    def test_very_long_sequence(self):
        """Test sequence longer than max_tokens_per_tree raises error."""
        sequences = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]

        # Should raise ValueError since sequence is longer than max_tokens_per_tree
        with pytest.raises(
            ValueError,
            match="Sequence at index 0 has 10 tokens, which exceeds max_tokens_per_tree=5",
        ):
            greedy_build_tree(sequences, max_tokens_per_tree=5)

    # ==================== Tree Splitting Behavior ====================

    @pytest.mark.parametrize("max_tokens", [5, 10, 15, 20, 100])
    def test_max_tokens_splitting(self, simple_sequences, max_tokens):
        """Test that max_tokens_per_tree correctly limits tree size."""
        trees = greedy_build_tree(simple_sequences, max_tokens_per_tree=max_tokens)

        for tree in trees:
            assert tree.n_total_tokens <= max_tokens

        # Verify all sequences are included across all trees
        total_batch_size = sum(tree.batch_size for tree in trees)
        assert total_batch_size == len(simple_sequences)

    def test_exact_boundary_fit(self):
        """Test sequences that exactly fill max_tokens_per_tree."""
        sequences = [[1, 2], [3, 4], [5, 6]]  # Total: 6 tokens
        trees = greedy_build_tree(sequences, max_tokens_per_tree=6)

        # Should fit exactly in one tree (no overlap, so 6 unique tokens)
        assert len(trees) == 1
        assert trees[0].n_total_tokens == 6

    def test_boundary_overflow(self):
        """Test sequences that just exceed max_tokens_per_tree by 1 token."""
        sequences = [[1, 2, 3], [4, 5, 6], [7, 8]]  # 8 tokens total
        trees = greedy_build_tree(sequences, max_tokens_per_tree=7)

        # Should create 2 trees since we exceed by 1
        assert len(trees) == 2
        assert trees[0].n_total_tokens <= 7
        assert trees[1].n_total_tokens <= 7

    def test_multiple_trees_created(self):
        """Test that correct number of trees are created when splitting needed."""
        sequences = [[i, i + 1, i + 2] for i in range(1, 21, 3)]  # 7 sequences
        trees = greedy_build_tree(sequences, max_tokens_per_tree=10)

        # With 7 sequences of 3 tokens each (21 total), should need multiple trees
        assert len(trees) >= 2

        # Verify all sequences processed
        total_batch_size = sum(tree.batch_size for tree in trees)
        assert total_batch_size == 7

        # Verify total token count
        total_tokens = sum(tree.n_total_tokens for tree in trees)
        assert total_tokens == 21

    # ==================== Tree Structure Validation ====================

    def test_node_counts(self, simple_sequences):
        """Test n_tree_tokens vs n_total_tokens relationship."""
        trees = greedy_build_tree(simple_sequences, max_tokens_per_tree=20)

        tree = trees[0]
        # With sharing, n_tree_tokens should be less than n_total_tokens
        assert tree.n_tree_tokens <= tree.n_total_tokens
        # Specifically for simple_sequences: 6 unique nodes, 9 total tokens
        assert tree.n_tree_tokens == 6
        assert tree.n_total_tokens == 9

    def test_shared_prefix_reduces_nodes(self):
        """Test that shared prefixes reduce unique node count."""
        # Two sequences with shared prefix
        shared = [[1, 2, 3], [1, 2, 4]]
        trees_shared = greedy_build_tree(shared, max_tokens_per_tree=20)

        # Two sequences with no sharing
        no_share = [[1, 2, 3], [4, 5, 6]]
        trees_no_share = greedy_build_tree(no_share, max_tokens_per_tree=20)

        # Shared prefix should result in fewer unique nodes
        assert trees_shared[0].n_tree_tokens < trees_no_share[0].n_tree_tokens

    def test_index_mappings_length(self, simple_sequences):
        """Test that index mappings have correct lengths."""
        trees = greedy_build_tree(simple_sequences, max_tokens_per_tree=20)

        tree = trees[0]
        # to_tree_indices maps from tree nodes to positions
        assert len(tree.to_tree_indices) == tree.n_tree_tokens
        # from_tree_indices maps from sequence positions to tree nodes
        assert len(tree.from_tree_indices) == tree.n_total_tokens

    def test_index_mappings_validity(self, simple_sequences):
        """Test that indices are within valid ranges."""
        trees = greedy_build_tree(simple_sequences, max_tokens_per_tree=20)

        tree = trees[0]
        # All to_tree_indices should be valid positions (0 to n_total_tokens-1)
        assert all(0 <= idx < tree.n_total_tokens for idx in tree.to_tree_indices)
        # All from_tree_indices should be valid node indices (0 to n_tree_tokens-1)
        assert all(0 <= idx < tree.n_tree_tokens for idx in tree.from_tree_indices)

    # ==================== Attention Mask Properties ====================

    def test_attention_mask_shape(self, simple_sequences):
        """Test that attention mask has correct shape."""
        trees = greedy_build_tree(simple_sequences, max_tokens_per_tree=20)

        tree = trees[0]
        expected_shape = (tree.n_tree_tokens, tree.n_tree_tokens)
        assert tree.attn_mask.shape == expected_shape
        assert tree.attn_mask.dtype == torch.bool

    def test_attention_mask_device(self, simple_sequences):
        """Test that attention mask is created on CUDA device."""
        trees = greedy_build_tree(simple_sequences, max_tokens_per_tree=20)

        tree = trees[0]
        assert tree.attn_mask.device.type == "cuda"

    def test_attention_mask_diagonal(self, simple_sequences):
        """Test that diagonal of attention mask is all True (self-attention)."""
        trees = greedy_build_tree(simple_sequences, max_tokens_per_tree=20)

        tree = trees[0]
        # Extract diagonal
        diagonal = torch.diagonal(tree.attn_mask)
        assert diagonal.all(), "All diagonal elements should be True for self-attention"

    def test_attention_mask_causal(self):
        """Test that parent nodes can attend to children."""
        # Simple parent-child sequence
        sequences = [[1, 2, 3]]
        trees = greedy_build_tree(sequences, max_tokens_per_tree=10)

        tree = trees[0]
        # In a linear sequence: node 0 -> node 1 -> node 2
        # node 0 should be able to attend to node 1 (parent to child)
        assert tree.attn_mask[0, 1]
        # node 1 should be able to attend to node 2
        assert tree.attn_mask[1, 2]

    def test_attention_mask_ancestors(self, simple_sequences):
        """Test that ancestor nodes can attend to all descendants."""
        trees = greedy_build_tree(simple_sequences, max_tokens_per_tree=20)

        tree = trees[0]
        # For the tree structure with node 1 at root:
        # Root should be able to attend to all other nodes
        # (Since we're using 0-indexed, root is at index 0)
        # The first node (root) should attend to multiple children/descendants
        root_attention = tree.attn_mask[0, :]
        # Root should attend to itself and its descendants
        assert root_attention.sum() > 1, "Root should attend to itself and children"

    # ==================== Complex Scenarios ====================

    def test_partial_overlap_sequences(self):
        """Test mix of shared and unique prefixes."""
        sequences = [
            [1, 2, 3, 4],  # unique path
            [1, 2, 5, 6],  # shares [1,2] with seq1
            [7, 8, 9, 10],  # completely unique
        ]
        trees = greedy_build_tree(sequences, max_tokens_per_tree=20)

        tree = trees[0]
        assert tree.batch_size == 3
        # Unique nodes: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 = 10 nodes
        assert tree.n_tree_tokens == 10
        # Total tokens: 4 + 4 + 4 = 12
        assert tree.n_total_tokens == 12

    def test_varying_sequence_lengths(self, varying_length_sequences):
        """Test sequences of different lengths."""
        trees = greedy_build_tree(varying_length_sequences, max_tokens_per_tree=20)

        tree = trees[0]
        # Should handle different lengths gracefully
        assert tree.batch_size == 4
        # Total tokens: 2 + 4 + 3 + 2 = 11
        assert tree.n_total_tokens == 11

    def test_deep_tree_structure(self, deep_tree_sequences):
        """Test sequences creating deep trie (long shared prefix)."""
        trees = greedy_build_tree(deep_tree_sequences, max_tokens_per_tree=30)

        tree = trees[0]
        assert tree.batch_size == 3
        # Shared: [1,2,3,4], then branches at position 5
        # Unique nodes: 1,2,3,4,5,6,7,8,9 = 9 nodes
        assert tree.n_tree_tokens == 9
        # Total tokens: 6 + 6 + 6 = 18
        assert tree.n_total_tokens == 18

    def test_wide_tree_structure(self, wide_tree_sequences):
        """Test sequences creating wide trie (many branches at root)."""
        trees = greedy_build_tree(wide_tree_sequences, max_tokens_per_tree=20)

        tree = trees[0]
        assert tree.batch_size == 4
        # No sharing, 4 sequences × 3 tokens = 12 unique nodes
        assert tree.n_tree_tokens == 12
        assert tree.n_total_tokens == 12

    @pytest.mark.parametrize(
        "max_tokens,expected_trees",
        [
            (5, 3),  # Each sequence fits separately (3 tokens each)
            (10, 1),  # All sequences fit together (9 total tokens, 6 unique)
            (20, 1),  # Large enough for all in one tree
        ],
    )
    def test_parametrized_splitting(self, simple_sequences, max_tokens, expected_trees):
        """Test various max_tokens values produce expected tree counts."""
        trees = greedy_build_tree(simple_sequences, max_tokens_per_tree=max_tokens)

        assert len(trees) == expected_trees
        # Verify all sequences processed
        total_batch_size = sum(tree.batch_size for tree in trees)
        assert total_batch_size == len(simple_sequences)

    # ==================== Additional Corner Cases ====================

    def test_all_empty_sequences(self):
        """Test with all sequences being empty."""
        sequences = [[], [], []]
        trees = greedy_build_tree(sequences, max_tokens_per_tree=10)

        # Should create one tree with all empty sequences
        assert len(trees) == 1
        assert trees[0].batch_size == 3
        assert trees[0].n_tree_tokens == 0
        assert trees[0].n_total_tokens == 0

    def test_mix_of_long_and_normal_sequences(self):
        """Test mix of very long sequences and normal ones raises error."""
        sequences = [
            [1, 2, 3],  # Normal
            [4, 5, 6, 7, 8, 9, 10, 11, 12, 13],  # Very long (10 tokens)
            [14, 15, 16],  # Normal
        ]

        # Should raise ValueError since one sequence is too long
        with pytest.raises(
            ValueError,
            match="Sequence at index 1 has 10 tokens, which exceeds max_tokens_per_tree=5",
        ):
            greedy_build_tree(sequences, max_tokens_per_tree=5)

    def test_zero_max_tokens(self):
        """Test with max_tokens_per_tree = 0 raises error."""
        sequences = [[1, 2, 3]]

        # Should raise ValueError for non-positive max_tokens_per_tree
        with pytest.raises(
            ValueError, match="max_tokens_per_tree must be positive, got 0"
        ):
            greedy_build_tree(sequences, max_tokens_per_tree=0)

    def test_negative_max_tokens(self):
        """Test with negative max_tokens_per_tree raises error."""
        sequences = [[1, 2, 3]]

        # Should raise ValueError for non-positive max_tokens_per_tree
        with pytest.raises(
            ValueError, match="max_tokens_per_tree must be positive, got -1"
        ):
            greedy_build_tree(sequences, max_tokens_per_tree=-1)

    def test_very_large_batch(self):
        """Test with a very large number of sequences."""
        sequences = [[i, i + 1] for i in range(100)]  # 100 sequences
        trees = greedy_build_tree(sequences, max_tokens_per_tree=50)

        # Should create multiple trees
        assert len(trees) >= 2
        # All sequences should be processed
        total_batch_size = sum(tree.batch_size for tree in trees)
        assert total_batch_size == 100

    def test_sequences_with_repeated_divergence(self):
        """Test sequences that diverge at multiple levels."""
        sequences = [
            [1, 2, 3, 4, 5],  # Baseline
            [1, 2, 3, 4, 6],  # Diverges at last token
            [1, 2, 3, 7, 8],  # Diverges at 4th token
            [1, 2, 9, 10, 11],  # Diverges at 3rd token
            [1, 12, 13, 14, 15],  # Diverges at 2nd token
        ]
        trees = greedy_build_tree(sequences, max_tokens_per_tree=30)

        assert len(trees) == 1
        tree = trees[0]
        assert tree.batch_size == 5
        # Shared: 1 (all), 2 (first 4), 3 (first 3), 4 (first 2)
        # Then unique branches
        assert tree.n_total_tokens == 25  # 5 sequences × 5 tokens

    def test_single_sequence_exactly_at_limit(self):
        """Test single sequence with exactly max_tokens_per_tree tokens."""
        sequences = [[1, 2, 3, 4, 5]]
        trees = greedy_build_tree(sequences, max_tokens_per_tree=5)

        assert len(trees) == 1
        assert trees[0].n_total_tokens == 5
        assert trees[0].batch_size == 1

    def test_only_long_sequences_small_limit(self):
        """Test when all sequences are longer than max_tokens_per_tree raises error."""
        sequences = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]]

        # Each sequence is 5 tokens but limit is 3 - should raise ValueError
        with pytest.raises(
            ValueError,
            match="Sequence at index 0 has 5 tokens, which exceeds max_tokens_per_tree=3",
        ):
            greedy_build_tree(sequences, max_tokens_per_tree=3)

    def test_index_mapping_consistency(self):
        """Test that to_tree_indices and from_tree_indices are consistent."""
        sequences = [[1, 2, 3], [1, 2, 4], [1, 5, 6]]
        trees = greedy_build_tree(sequences, max_tokens_per_tree=20)

        tree = trees[0]
        # from_tree_indices should map sequence positions to tree node indices
        # to_tree_indices should map tree node indices to sequence positions

        # Verify round-trip consistency
        for seq_pos in range(tree.n_total_tokens):
            tree_idx = tree.from_tree_indices[seq_pos]
            # tree_idx should be valid
            assert 0 <= tree_idx < tree.n_tree_tokens

        # to_tree_indices should have unique values representing original positions
        assert len(tree.to_tree_indices) == tree.n_tree_tokens

    def test_attention_mask_symmetry(self):
        """Test attention mask properties for symmetric sequences."""
        sequences = [[1, 2], [3, 4]]  # Two independent sequences
        trees = greedy_build_tree(sequences, max_tokens_per_tree=10)

        tree = trees[0]
        # Verify mask is not all True or all False (has some structure)
        assert tree.attn_mask.any()
        assert not tree.attn_mask.all()
        # Verify it's a square matrix
        assert tree.attn_mask.shape[0] == tree.attn_mask.shape[1]
        # Verify it has the expected size
        assert tree.attn_mask.shape[0] == tree.n_tree_tokens

    def test_batch_boundary_splitting(self):
        """Test splitting exactly at batch boundaries."""
        # Create sequences that when split should have clean boundaries
        sequences = [
            [1, 2],
            [3, 4],
            [5, 6],
            [7, 8],
        ]  # 8 tokens total
        trees = greedy_build_tree(sequences, max_tokens_per_tree=4)

        # Should split into 2 trees of 2 sequences each
        assert len(trees) == 2
        assert trees[0].batch_size == 2
        assert trees[1].batch_size == 2
