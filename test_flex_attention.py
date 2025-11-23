from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention


@dataclass
class TrieNode:
    idx: int
    value: int
    pos: int
    children: dict[int, "TrieNode"]


@dataclass
class TokenTree:
    n_nodes: int
    n_tokens: int
    to_tree_indices: list[int]
    from_tree_indices: list[int]
    attn_mask: torch.Tensor


def build_tree_for_flex_attn(sequences: list[list[int]], max_tokens_per_tree: int) -> list[TokenTree]:
    batch_idx = 0

    def _build_mb_tree():
        root = TrieNode(idx=-1, value=-1, children={}, pos=-1)
        nodes: list[TrieNode] = []
        from_tree_indices: list[int] = []
        pos = 0

        nonlocal batch_idx
        # Build a token trie from the sequences
        for seq in sequences[batch_idx:]:
            if pos + len(seq) > max_tokens_per_tree:
                break
            node = root
            for token in seq:
                if token not in node.children:
                    child = TrieNode(idx=len(nodes), value=token, children={}, pos=pos)
                    node.children[token] = child
                    nodes.append(child)
                node = node.children[token]
                pos += 1
                from_tree_indices.append(node.idx)
            batch_idx += 1
            if batch_idx >= len(sequences):
                break

        # Create attention mask by DFS
        attn_mask = torch.eye(len(nodes), len(nodes), dtype=torch.bool, device="cuda")

        def dfs(node: TrieNode, ancestors: list[int]):
            for child in node.children.values():
                attn_mask[node.idx, child.idx] = 1
                for idx in ancestors:
                    attn_mask[idx, child.idx] = 1
                dfs(child, ancestors + [node.idx])

        for node in root.children.values():
            dfs(node, [])

        return TokenTree(
            n_nodes=len(nodes),
            n_tokens=pos,
            to_tree_indices=[node.pos for node in nodes],
            from_tree_indices=from_tree_indices,
            attn_mask=attn_mask,
        )

    trees = []
    while batch_idx < len(sequences):
        tree = _build_mb_tree()
        trees.append(tree)
    return trees


def tree_flex_attention(
    q: torch.Tensor,  # [total_seqlen, num_heads, head_dim]
    k: torch.Tensor,  # [total_seqlen, num_heads, head_dim]
    v: torch.Tensor,  # [total_seqlen, num_heads, head_dim]
    tree: TokenTree,
) -> torch.Tensor:
    q_tree = q[tree.to_tree_indices]  # [num_tree_nodes, num_heads, head_dim]
    k_tree = k[tree.to_tree_indices]
    v_tree = v[tree.to_tree_indices]

    # Transpose to [num_tree_nodes, num_heads, head_dim] -> [num_heads, num_tree_nodes, head_dim]
    # Then add batch dimension for flex_attention (expects [B, H, Q_LEN, head_dim])
    q_tree = q_tree.transpose(0, 1).unsqueeze(0)  # [1, num_heads, num_tree_nodes, head_dim]
    k_tree = k_tree.transpose(0, 1).unsqueeze(0)
    v_tree = v_tree.transpose(0, 1).unsqueeze(0)

    # Create mask function
    def mask_mod(batch_idx, head_idx, q_idx, k_idx):
        is_causal = k_idx <= q_idx
        return is_causal & tree.attn_mask[k_idx, q_idx]

    # Create block mask
    block_mask = create_block_mask(
        mask_mod,
        B=None,
        H=None,
        Q_LEN=tree.n_nodes,
        KV_LEN=tree.n_nodes,
    )

    # Apply flex attention
    output_tree = flex_attention(q_tree, k_tree, v_tree, block_mask=block_mask, enable_gqa=True)

    # Remove batch dimension and transpose back: [1, num_heads, num_tree_nodes, head_dim] -> [num_tree_nodes, num_heads, head_dim]
    output_tree = output_tree.squeeze(0).transpose(0, 1)  # [num_tree_nodes, num_heads, head_dim]

    # Expand back to original packed format
    # Each original position maps to its corresponding tree position
    output = output_tree[tree.from_tree_indices]  # [total_seqlen, num_heads, head_dim]

    return output


def create_sequence_ids(sequence_lengths: list[int], device: str = "cuda") -> torch.Tensor:
    """Convert sequence lengths to a tensor mapping each position to its sequence index.

    Args:
        sequence_lengths: List of lengths for each sequence
        device: Device to create the tensor on

    Returns:
        Tensor of shape [total_seqlen] where each element is the sequence index

    Example:
        >>> create_sequence_ids([3, 3, 2, 3])
        tensor([0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 3])
    """
    sequence_ids = []
    for seq_idx, length in enumerate(sequence_lengths):
        sequence_ids.extend([seq_idx] * length)
    return torch.tensor(sequence_ids, dtype=torch.long, device=device)


def causal_flex_attention(
    q: torch.Tensor,  # [total_seqlen, num_heads, head_dim]
    k: torch.Tensor,  # [total_seqlen, num_heads, head_dim]
    v: torch.Tensor,  # [total_seqlen, num_heads, head_dim]
    sequence_lengths: list[int],
) -> torch.Tensor:
    """Standard causal attention with sequence boundary awareness.

    Prevents cross-contamination between different sequences in the packed batch.
    Each token can only attend to previous tokens within the same sequence.

    Args:
        q, k, v: Query, key, value tensors in packed format
        sequence_lengths: List of lengths for each sequence in the batch

    Returns:
        Output tensor of shape [total_seqlen, num_heads, head_dim]
    """
    total_seqlen = q.shape[0]

    # Create sequence ID mapping
    sequence_ids = create_sequence_ids(sequence_lengths, device=q.device)

    def mask_mod(batch_idx, head_idx, q_idx, k_idx):
        # Causal constraint: can only attend to previous or current positions
        is_causal = k_idx <= q_idx
        # Same-sequence constraint: prevent cross-contamination
        same_sequence = sequence_ids[k_idx] == sequence_ids[q_idx]
        return is_causal & same_sequence

    # Reshape to [1, num_heads, total_seqlen, head_dim] for flex_attention
    q_batched = q.transpose(0, 1).unsqueeze(0)
    k_batched = k.transpose(0, 1).unsqueeze(0)
    v_batched = v.transpose(0, 1).unsqueeze(0)

    # Create block mask
    block_mask = create_block_mask(
        mask_mod,
        B=None,
        H=None,
        Q_LEN=total_seqlen,
        KV_LEN=total_seqlen,
    )

    # Apply flex attention
    output = flex_attention(q_batched, k_batched, v_batched, block_mask=block_mask, enable_gqa=True)

    # Reshape back: [1, num_heads, total_seqlen, head_dim] -> [total_seqlen, num_heads, head_dim]
    output = output.squeeze(0).transpose(0, 1)

    return output


def flatten_list(lis):
    return sum(lis, [])


# ============================================================================
# Example usage and validation
# ============================================================================

print("=" * 80)
print("VALIDATION 1: Linear Tree (Single Sequence)")
print("=" * 80)
print("For a single sequence, tree attention should equal causal attention")
print()

# Single sequence - creates a linear tree (no branching)
linear_sequences = [[1, 2, 3, 4]]
linear_tree = build_tree_for_flex_attn(linear_sequences)

linear_seqlen = len(flatten_list(linear_sequences))
linear_lengths = [len(seq) for seq in linear_sequences]

# Create random Q, K, V tensors
q_linear = torch.randn((linear_seqlen, 4, 16), dtype=torch.bfloat16, device="cuda")
k_linear = torch.randn((linear_seqlen, 2, 16), dtype=torch.bfloat16, device="cuda")
v_linear = torch.randn((linear_seqlen, 2, 16), dtype=torch.bfloat16, device="cuda")

# Run both attention mechanisms
tree_output_linear = tree_flex_attention(q_linear, k_linear, v_linear, linear_tree)
causal_output_linear = causal_flex_attention(q_linear, k_linear, v_linear, linear_lengths)

# Validate equivalence
is_close = torch.allclose(tree_output_linear, causal_output_linear, rtol=1e-3, atol=1e-3)
max_diff = (tree_output_linear - causal_output_linear).abs().max().item()

print(f"Tree output shape: {tree_output_linear.shape}")
print(f"Causal output shape: {causal_output_linear.shape}")
print(f"Outputs are close: {is_close}")
print(f"Max absolute difference: {max_diff:.6e}")

if is_close:
    print("✓ VALIDATION PASSED: Tree attention equals causal attention for linear tree!")
else:
    print("✗ VALIDATION FAILED: Outputs differ unexpectedly")

print()
print("=" * 80)
print("VALIDATION 2: Branching Tree (Multiple Sequences)")
print("=" * 80)
print("For branching trees, outputs should differ due to shared prefix structure")
print()

# Multiple sequences - creates a branching tree
sequences = [[1, 2, 3], [1, 2, 4], [1, 5], [1, 6, 7]]
tree = build_tree_for_flex_attn(sequences)

total_seqlen = len(flatten_list(sequences))
sequence_lengths = [len(seq) for seq in sequences]

print(f"Sequences: {sequences}")
print(f"Sequence lengths: {sequence_lengths}")
print(f"Total sequence length: {total_seqlen}")
print()

# Create random Q, K, V tensors
q = torch.randn((total_seqlen, 4, 16), dtype=torch.bfloat16, device="cuda")
k = torch.randn((total_seqlen, 2, 16), dtype=torch.bfloat16, device="cuda")
v = torch.randn((total_seqlen, 2, 16), dtype=torch.bfloat16, device="cuda")
q = q[tree.to_tree_indices][tree.from_tree_indices]
k = k[tree.to_tree_indices][tree.from_tree_indices]
v = v[tree.to_tree_indices][tree.from_tree_indices]

# Run both attention mechanisms
tree_output = tree_flex_attention(q, k, v, tree)
causal_output = causal_flex_attention(q, k, v, sequence_lengths)

# Compute differences
diff = (tree_output - causal_output).abs()
max_diff = diff.max().item()
mean_diff = diff.mean().item()

print(f"Tree output shape: {tree_output.shape}")
print(f"Causal output shape: {causal_output.shape}")
print(f"Max absolute difference: {max_diff:.6e}")
print(f"Mean absolute difference: {mean_diff:.6e}")
print()
print("Why they differ:")
print("  - Tree attention: Tokens attend to ancestors in the shared prefix tree")
print("    Example: sequences [1,2,3] and [1,2,4] share nodes [1,2]")
print("  - Causal attention: Each sequence is independent, no shared computation")
print("    Example: position 5 only attends within its own sequence")
print()
print("This demonstrates the efficiency gain of tree attention for prefix sharing!")
print("=" * 80)
