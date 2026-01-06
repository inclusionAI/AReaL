"""Tree-based sequence packing for efficient training with shared prefixes.

This module provides utilities for building trie structures from sequences,
packing them into tree-structured inputs for efficient training with shared
prefix computation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from areal.utils import logging, stats_tracker
from areal.utils.data import MicroBatchList
from areal.utils.perf_tracer import trace_perf, trace_scope
from areal.utils.tree_training.module import BLOCK_SIZE, USE_BLOCK_MASK

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class TrieNode:
    """A node in a compressed trie (prefix tree) for token sequences.

    Each node represents a contiguous run of tokens that share the same set of
    sequences. When sequences diverge, child nodes are created.

    For root nodes, start_idx and end_idx are -1, and the node contains no tokens.
    Root nodes use the `nodes` list to track all descendant nodes in pre-order.

    Attributes:
        tree_id: Identifier of the tree this node belongs to.
        start_idx: Starting index in the flattened tree representation (-1 for root).
        end_idx: Ending index (inclusive) in the flattened tree representation (-1 for root).
        tokens: List of token IDs stored in this node (empty for root).
        sequence_ids: IDs of sequences that pass through this node.
        children: Child nodes keyed by the first diverging token.
        ancestors: List of ancestor nodes from root to parent (empty for root).
        nodes: All descendant nodes in pre-order traversal (only used by root).
    """

    tree_id: int
    start_idx: int = -1
    end_idx: int = -1
    tokens: list[int] = field(default_factory=list)
    sequence_ids: list[int] = field(default_factory=list)
    children: dict[int, TrieNode] = field(default_factory=dict)
    ancestors: list[TrieNode] = field(default_factory=list)
    nodes: list[TrieNode] = field(default_factory=list)

    @property
    def is_root(self) -> bool:
        """Check if this is a root node."""
        return self.start_idx == -1 and self.end_idx == -1

    @property
    def num_tokens(self) -> int:
        """Number of tokens stored in this node (or total for root)."""
        if self.is_root:
            return sum(node.num_tokens for node in self.nodes)
        return len(self.tokens)

    @property
    def tree_indices(self) -> tuple[int, int]:
        """Return (start_idx, end_idx) tuple for this node's position in tree."""
        return (self.start_idx, self.end_idx)

    @property
    def all_sequence_ids(self) -> list[int]:
        """Get sorted list of all sequence IDs in this tree (for root nodes)."""
        if self.is_root:
            seq_ids: set[int] = set()
            for node in self.nodes:
                seq_ids.update(node.sequence_ids)
            return sorted(seq_ids)
        return sorted(set(self.sequence_ids))

    def get_all_tree_indices(self) -> list[tuple[int, int]]:
        """Get tree indices for this node and all ancestors."""
        indices = [ancestor.tree_indices for ancestor in self.ancestors]
        if not self.is_root:
            indices.append(self.tree_indices)
        return indices

    def get_sequence_tree_indices(self, seq_id: int) -> list[tuple[int, int]]:
        """Get all tree index ranges for a given sequence ID (for root nodes)."""
        indices = []
        for node in self.nodes:
            if seq_id in node.sequence_ids:
                indices.append(node.tree_indices)
        return indices


# =============================================================================
# Internal Trie Building Helpers
# =============================================================================


class _BuildNode:
    """Temporary node structure used during trie construction."""

    __slots__ = ("tree_id", "token_id", "node_id", "children", "is_end", "sequence_ids")

    def __init__(self, tree_id: int, token_id: int, node_id: int):
        self.tree_id = tree_id
        self.token_id = token_id
        self.node_id = node_id
        self.children: dict[int, _BuildNode] = {}
        self.is_end = False
        self.sequence_ids: list[int] = []


def _count_additional_nodes(root: _BuildNode, sequence: list[int]) -> int:
    """Count how many new nodes are needed to insert sequence into root."""
    current = root
    for idx, token in enumerate(sequence):
        child = current.children.get(token)
        if child is None:
            return len(sequence) - idx
        current = child
    return 0


def _insert_sequence(
    root: _BuildNode,
    all_nodes: list[_BuildNode],
    sequence: list[int],
    tree_id: int,
    sequence_id: int,
) -> None:
    """Insert a sequence into the trie, mutating it in-place."""
    current = root
    for token in sequence:
        if token not in current.children:
            node_id = len(all_nodes)
            current.children[token] = _BuildNode(tree_id, token, node_id)
            all_nodes.append(current.children[token])
        current.children[token].sequence_ids.append(sequence_id)
        current = current.children[token]
    current.is_end = True


@trace_perf("tree_training._compress_trie")
def _compress_trie(root: _BuildNode) -> TrieNode:
    """Compress a trie by merging linear chains into single TrieNodes."""
    trie_root = TrieNode(tree_id=root.tree_id)

    def _compress_chain(
        node: _BuildNode,
        ancestors: list[TrieNode],
    ) -> TrieNode:
        """Compress a chain of nodes with single children into one TrieNode."""
        tokens: list[int] = []
        current = node
        start_id = node.node_id

        # Follow single-child chains
        while True:
            tokens.append(current.token_id)
            if len(current.children) != 1 or current.is_end:
                break
            # Get the single child
            next_child = next(iter(current.children.values()))
            # Verify consistency
            if current.sequence_ids != next_child.sequence_ids:
                raise ValueError(
                    f"Sequence IDs mismatch along chain: "
                    f"{current.sequence_ids} vs {next_child.sequence_ids}"
                )
            if next_child.node_id != current.node_id + 1:
                raise ValueError("Node IDs not consecutive along chain.")
            current = next_child

        # Create compressed node
        trie_node = TrieNode(
            tree_id=root.tree_id,
            start_idx=start_id,
            end_idx=current.node_id,
            tokens=tokens,
            sequence_ids=current.sequence_ids.copy(),
            ancestors=ancestors.copy(),
        )
        trie_root.nodes.append(trie_node)

        # Recursively compress children
        if current.children:
            for token, child in sorted(current.children.items()):
                trie_node.children[token] = _compress_chain(
                    child,
                    ancestors + [trie_node],
                )

        return trie_node

    # Compress all children of root
    if root.children:
        for token, child in sorted(root.children.items()):
            trie_root.children[token] = _compress_chain(child, [])

    return trie_root


# =============================================================================
# Public API
# =============================================================================


@trace_perf("tree_training.build_packed_tree_batch")
def build_packed_tree_batch(
    data: dict[str, Any],
    max_tokens_per_tree: int,
    pad_to_maximum: bool = True,
    pad_to_multiple_of: int = 1,
) -> MicroBatchList:
    """Build a MicroBatchList from input data using greedy trie packing.

    This function constructs tries from input sequences using a greedy packing
    strategy, then converts them into tree-packed format suitable for training.

    Args:
        data: Dictionary containing 'input_ids' and 'attention_mask' tensors
            describing the batch of sequences. Shape: [batch_size, seq_len].
        max_tokens_per_tree: Maximum number of tokens allowed per tree.
            Must be a multiple of BLOCK_SIZE when pad_to_maximum=True.
        pad_to_maximum: If True, pad all trees to max_tokens_per_tree.
            If False, padding is determined by pad_to_multiple_of.
        pad_to_multiple_of: When pad_to_maximum=False, pad to the nearest
            multiple of this value. If <= 1, no padding is applied.
            No padding raises error if USE_BLOCK_MASK=True.

    Returns:
        MicroBatchList containing all packed tree data.

    Raises:
        ValueError: If max_tokens_per_tree is not positive, or if padding
            constraints are violated, or if a sequence exceeds max_tokens_per_tree.
    """
    if max_tokens_per_tree <= 0:
        raise ValueError("max_tokens_per_tree must be positive")

    # Validate padding constraints when using block masks
    if USE_BLOCK_MASK:
        no_padding = not pad_to_maximum and pad_to_multiple_of <= 1
        if no_padding:
            raise ValueError(
                "No padding is not supported when USE_BLOCK_MASK=True. "
                "Block masks require padded sequences for efficient computation. "
                "Set pad_to_maximum=True or pad_to_multiple_of > 1."
            )
        if pad_to_maximum and max_tokens_per_tree % BLOCK_SIZE != 0:
            raise ValueError(
                f"max_tokens_per_tree must be a multiple of BLOCK_SIZE ({BLOCK_SIZE}) "
                f"when pad_to_maximum=True and USE_BLOCK_MASK=True"
            )
        if not pad_to_maximum and pad_to_multiple_of % BLOCK_SIZE != 0:
            raise ValueError(
                f"pad_to_multiple_of must be a multiple of BLOCK_SIZE ({BLOCK_SIZE}) "
                f"when USE_BLOCK_MASK=True"
            )

    # Build tries using greedy packing
    tries, num_tokens_list = _greedy_build_tries(data, max_tokens_per_tree)

    # Prepare templates and metadata
    input_template: torch.Tensor = data["input_ids"]
    mask_template: torch.Tensor = data["attention_mask"]

    # Directly track tree token ratio statistic
    original_num_tokens = mask_template.sum()
    total_tree_tokens = sum(num_tokens_list)
    ratio = total_tree_tokens / original_num_tokens
    stats_tracker.scalar(tree_token_ratio=ratio)
    
    sequence_lens = mask_template.sum(dim=1, dtype=torch.int32)

    # Identify packable keys (same shape as input_ids)
    packable_keys = {
        key
        for key, value in data.items()
        if key not in {"input_ids", "attention_mask"}
        and torch.is_tensor(value)
        and value.shape == input_template.shape
    }
    non_packable_keys = (
        set(data.keys()) - packable_keys - {"input_ids", "attention_mask"}
    )

    # Build packed outputs for each tree
    mbs: list[dict[str, Any]] = []
    padding_lengths: list[int] = []
    padded_to_lengths: list[int] = []

    for trie, num_tokens in zip(tries, num_tokens_list):
        # Compute padded size based on padding options
        padded_size = _compute_padded_size(
            num_tokens, max_tokens_per_tree, pad_to_maximum, pad_to_multiple_of
        )

        # Pack input_ids
        with trace_scope("tree_training.pack_input_ids"):
            input_ids = _pack_input_ids(
                trie,
                input_template,
                padded_size,
            )

        # Build attention mask
        with trace_scope("tree_training.build_attention_mask"):
            attention_mask = _build_attention_mask(
                trie,
                padded_size,
                mask_template.device,
            )

        # Amend position_ids
        with trace_scope("tree_training.get_position_ids"):
            position_ids = get_packed_tree_position_ids(
                input_ids,
                attention_mask,
            )

        # Pack extra data
        with trace_scope("tree_training.pack_extra_data"):
            extra_data = _pack_extra_data(
                trie,
                data,
                sequence_lens,
                packable_keys,
                non_packable_keys,
            )

        # Build micro-batch dict
        mb = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "trie_node": trie,
            **extra_data,
        }
        mbs.append(mb)
        padding_lengths.append(padded_size - num_tokens)
        padded_to_lengths.append(padded_size)

    # NOTE: mbs is padded data instead of original data 
    # to avoid duplicate attention mask memory consumption
    batch = MicroBatchList(
        data=data,
        mb_spec=None,  # type: ignore[arg-type]
        mbs=mbs,
        group_lens=[num for num in num_tokens_list],
        padded_mbs=mbs,
        padding_lengths=padding_lengths,
        padded_to_lengths=padded_to_lengths,
        _max_seqlen=max(padded_to_lengths),
    )
    return batch


def _compute_padded_size(
    num_tokens: int,
    max_tokens_per_tree: int,
    pad_to_maximum: bool,
    pad_to_multiple_of: int,
) -> int:
    """Compute the padded size for a tree based on padding options.

    Args:
        num_tokens: Actual number of tokens in the tree.
        max_tokens_per_tree: Maximum tokens allowed per tree.
        pad_to_maximum: If True, return max_tokens_per_tree.
        pad_to_multiple_of: If pad_to_maximum=False and this is > 1,
            round up to the nearest multiple.

    Returns:
        The padded size for the tree.
    """
    if pad_to_maximum:
        return max_tokens_per_tree
    elif pad_to_multiple_of > 1:
        # Round up to nearest multiple
        return (
            (num_tokens + pad_to_multiple_of - 1) // pad_to_multiple_of
        ) * pad_to_multiple_of
    else:
        # No padding
        return num_tokens


@trace_perf("tree_training._greedy_build_tries")
def _greedy_build_tries(
    data: dict[str, Any],
    max_tokens_per_tree: int,
) -> tuple[list[TrieNode], list[int]]:
    """Build tries using greedy packing strategy."""
    sequences = _extract_sequences(data)
    forests: list[dict[str, Any]] = []

    for seq_id, seq in enumerate(sequences):
        inserted = False

        # Try to insert into existing trees
        for tree_id, tree in enumerate(forests):
            additional = _count_additional_nodes(tree["root"], seq)
            if tree["nodes"] + additional <= max_tokens_per_tree:
                _insert_sequence(
                    tree["root"],
                    tree["all_nodes"],
                    seq,
                    tree_id,
                    seq_id,
                )
                tree["nodes"] += additional
                inserted = True
                break

        if inserted:
            continue

        # Create new tree
        if len(seq) > max_tokens_per_tree:
            raise ValueError(
                f"Sequence length {len(seq)} exceeds max_tokens_per_tree "
                f"{max_tokens_per_tree}; adjust limit or split sequences."
            )

        new_tree_id = len(forests)
        new_root = _BuildNode(new_tree_id, -1, -1)
        all_nodes: list[_BuildNode] = []
        _insert_sequence(new_root, all_nodes, seq, new_tree_id, seq_id)
        forests.append({"root": new_root, "all_nodes": all_nodes, "nodes": len(seq)})

    # Compress all trees
    tries = [_compress_trie(f["root"]) for f in forests]
    num_tokens_list = [f["nodes"] for f in forests]

    return tries, num_tokens_list


def _extract_sequences(data: dict[str, Any]) -> list[list[int]]:
    """Extract token sequences from padded input data."""
    assert "input_ids" in data, "Input data must contain 'input_ids'"
    assert "attention_mask" in data, "Input data must contain 'attention_mask'"

    input_ids = data["input_ids"]
    attention_mask = data["attention_mask"]

    sequences = []
    for ids, mask in zip(input_ids, attention_mask):
        seq = ids[mask.bool()].tolist()
        sequences.append(seq)
    return sequences


def _pack_input_ids(
    trie: TrieNode,
    input_template: torch.Tensor,
    max_tokens: int,
) -> torch.Tensor:
    """Pack input_ids from trie structure into a flat tensor."""
    input_ids = torch.zeros(
        (max_tokens,),
        dtype=input_template.dtype,
        device=input_template.device,
    )

    for node in trie.nodes:
        # Find the first sequence that owns this node to get source tokens
        seq_id = node.sequence_ids[0]
        # Calculate position in original sequence
        seq_pos = sum(ancestor.num_tokens for ancestor in node.ancestors)
        # Copy tokens from source sequence
        tree_start, tree_end = node.tree_indices
        input_ids[tree_start : tree_end + 1] = input_template[seq_id][
            seq_pos : seq_pos + node.num_tokens
        ]

    return input_ids.unsqueeze(0)


def _build_attention_mask(
    trie: TrieNode,
    max_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    """Build 2D attention mask from trie structure."""
    mask = torch.zeros((max_tokens, max_tokens), dtype=torch.bool, device=device)
    tril_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    for seq_id in trie.all_sequence_ids:
        # Get all tree index ranges for this sequence
        indices = trie.get_sequence_tree_indices(seq_id)
        if not indices:
            continue

        # Build position tensor from all segments
        position_chunks = [
            torch.arange(start, end + 1, device=device)
            for start, end in indices
            if end >= start
        ]
        if not position_chunks:
            continue

        positions = torch.cat(position_chunks, dim=0)
        seq_len = positions.numel()
        if seq_len == 0:
            continue

        # Get or create lower triangular indices
        rows_cols = tril_cache.get(seq_len)
        if rows_cols is None:
            rows_cols = torch.tril_indices(seq_len, seq_len, device=device)
            tril_cache[seq_len] = rows_cols
        rows, cols = rows_cols

        # Set causal mask entries
        mask[positions[rows], positions[cols]] = True
    return mask


def _pack_extra_data(
    trie: TrieNode,
    data: dict[str, Any],
    sequence_lens: torch.Tensor,
    packable_keys: set[str],
    non_packable_keys: set[str],
) -> dict[str, Any]:
    """Pack additional tensor data according to trie structure."""
    extra_data: dict[str, Any] = {}
    seq_ids = trie.all_sequence_ids
    lens = [sequence_lens[sid].item() for sid in seq_ids]

    # Pack tensors according to the order in trie.all_sequence_ids
    for key in packable_keys:
        value = data[key]
        packed = torch.empty(
            (sum(lens), *value.shape[2:]),
            dtype=value.dtype,
            device=value.device,
        )
        cursor = 0
        for length, seq_id in zip(lens, seq_ids):
            packed[cursor : cursor + length] = value[seq_id][:length]
            cursor += length
        extra_data[key] = packed

    # Copy non-packable data as-is
    for key in non_packable_keys:
        extra_data[key] = data[key]

    return extra_data


def get_packed_tree_position_ids(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> dict[str, Any]:
    """Generate position IDs for packed tree inputs.

    Position IDs are computed from the attention mask by counting the number
    of ancestors each token can attend to (minus 1 for 0-indexing).

    Args:
        input_ids: 1D tensor of input IDs for the packed tree.
        attention_mask: 2D attention mask tensor for the packed tree.

    Returns:
        position_ids: Tensor of position IDs aligned with input_ids.
    """
    input_ids = input_ids.squeeze()
    if input_ids.ndim != 1:
        raise ValueError("Packed tree 'input_ids' must be a 1D tensor after squeezing.")
    if attention_mask.ndim != 2 or attention_mask.shape[0] != attention_mask.shape[1]:
        raise ValueError("Packed tree attention_mask must be a square matrix.")
    if attention_mask.shape[0] != input_ids.shape[0]:
        raise ValueError("Packed tree attention_mask must align with input_ids length.")

    if attention_mask.shape[0] == 0:
        position_ids = torch.empty(0, dtype=torch.long, device=attention_mask.device)
    else:
        ancestor_counts = attention_mask.bool().sum(dim=-1, dtype=torch.long)
        position_ids = torch.clamp_min(ancestor_counts - 1, 0)

    return position_ids
