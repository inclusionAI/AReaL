"""Tree-based sequence packing for efficient training with shared prefixes.

This module provides utilities for building trie structures from sequences,
packing them into tree-structured inputs for efficient training with shared
prefix computation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from areal.api.cli_args import MicroBatchSpec
from areal.models.tree_attn.module import BLOCK_SIZE, USE_BLOCK_MASK
from areal.utils import logging, stats_tracker
from areal.utils.data import MicroBatchList
from areal.utils.perf_tracer import trace_perf, trace_scope

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

    Attributes
    ----------
    tree_id : int
        Identifier of the tree this node belongs to.
    start_idx : int
        Starting index in the flattened tree representation (-1 for root).
    end_idx : int
        Ending index (inclusive) in the flattened tree representation (-1 for root).
    tokens : list[int]
        List of token IDs stored in this node (empty for root).
    sequence_ids : list[int]
        IDs of sequences that pass through this node.
    children : dict[int, TrieNode]
        Child nodes keyed by the first diverging token.
    ancestors : list[TrieNode]
        List of ancestor nodes from root to parent (empty for root).
    nodes : list[TrieNode]
        All descendant nodes in pre-order traversal (only used by root).
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


@trace_perf("tree_attn._compress_trie")
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


@trace_perf("tree_attn.build_packed_tree_batch")
def build_packed_tree_batch(
    data: dict[str, Any],
    mb_spec: MicroBatchSpec,
    pad_to_maximum: bool = True,
    pad_to_multiple_of: int = 1,
) -> MicroBatchList:
    """Build a MicroBatchList from input data using greedy trie packing.

    This function constructs tries from input sequences using a greedy packing
    strategy, then converts them into tree-packed format suitable for training.

    Parameters
    ----------
    data : dict[str, Any]
        Dictionary containing 'input_ids' and 'attention_mask' tensors
        describing the batch of sequences. Shape: [batch_size, seq_len].
    mb_spec : MicroBatchSpec
        MicroBatchSpec containing:
        - max_tokens_per_mb: Maximum tokens per tree.
        - n_mbs: Minimum number of trees to produce. Trees will be split
          if needed to meet this requirement.
        - n_mbs_divisor: Final number of trees must be divisible by this value.
        Note: granularity is not used in tree training and will trigger
        a warning if set to non-default values.
    pad_to_maximum : bool, default=True
        If True, pad all trees to max_tokens_per_mb.
        If False, padding is determined by pad_to_multiple_of.
    pad_to_multiple_of : int, default=1
        When pad_to_maximum=False, pad to the nearest multiple of this value.
        If <= 1, no padding is applied. No padding raises error if
        USE_BLOCK_MASK=True.

    Returns
    -------
    MicroBatchList
        MicroBatchList containing all packed tree data.

    Raises
    ------
    ValueError
        If max_tokens_per_mb is None or not positive, or if padding
        constraints are violated, or if a sequence exceeds max_tokens_per_mb.
    """
    # Warn about non-effective attributes (only granularity now)
    if mb_spec.granularity != 1:
        logger.warning(
            "`granularity` is currently not effective for tree packing."
        )

    max_tokens_per_tree = mb_spec.max_tokens_per_mb
    if max_tokens_per_tree is None or max_tokens_per_tree <= 0:
        raise ValueError(
            "MicroBatchSpec.max_tokens_per_mb must be a positive value for tree training."
        )

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
    tries, num_tokens_list = _greedy_build_tries(
        data,
        max_tokens_per_tree,
        min_trees=mb_spec.n_mbs if mb_spec.n_mbs is not None else 1,
        n_trees_divisor=mb_spec.n_mbs_divisor if mb_spec.n_mbs_divisor is not None else 1,
    )

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
        with trace_scope("tree_attn.pack_input_ids"):
            input_ids = _pack_input_ids(
                trie,
                input_template,
                padded_size,
            )

        # Build attention mask
        with trace_scope("tree_attn.build_attention_mask"):
            attention_mask = _build_attention_mask(
                trie,
                padded_size,
                mask_template.device,
            )

        # Amend position_ids
        with trace_scope("tree_attn.get_position_ids"):
            position_ids = get_packed_tree_position_ids(
                input_ids,
                attention_mask,
            )

        # Pack extra data
        with trace_scope("tree_attn.pack_extra_data"):
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
        mb_spec=mb_spec,
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
    """Compute the padded size for a tree based on padding options."""
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


def _get_tree_sequence_ids(tree: dict[str, Any]) -> set[int]:
    """Get all sequence IDs in a tree."""
    seq_ids: set[int] = set()
    for node in tree["all_nodes"]:
        seq_ids.update(node.sequence_ids)
    return seq_ids


def _find_splittable_tree_idx(forests: list[dict[str, Any]]) -> int | None:
    """Find the index of a tree that can be split.

    A tree can be split if it has at least 2 sequences.
    Returns the index of the tree with the most sequences, or None if no tree can be split.
    """
    best_idx = None
    best_seq_count = 1  # Must have at least 2 sequences to split
    for i, tree in enumerate(forests):
        seq_ids = _get_tree_sequence_ids(tree)
        if len(seq_ids) > best_seq_count:
            best_seq_count = len(seq_ids)
            best_idx = i
    return best_idx


def _extract_sequences_from_tree(
    node: _BuildNode,
    current_tokens: list[int],
    seq_tokens: dict[int, list[int]],
) -> None:
    """Recursively extract all sequences from a tree."""
    if node.token_id != -1:  # Not root
        current_tokens = current_tokens + [node.token_id]

    if node.is_end:
        # This node marks the end of at least one sequence
        for seq_id in node.sequence_ids:
            if seq_id not in seq_tokens:
                seq_tokens[seq_id] = current_tokens.copy()

    for child in node.children.values():
        _extract_sequences_from_tree(child, current_tokens, seq_tokens)


def _split_tree_by_sequences(
    forests: list[dict[str, Any]],
    tree_idx: int,
) -> list[dict[str, Any]]:
    """Split a tree into two by moving half of its sequences to a new tree.

    This rebuilds both trees from scratch using the sequence data.
    """
    tree = forests[tree_idx]
    seq_ids = sorted(_get_tree_sequence_ids(tree))

    if len(seq_ids) < 2:
        return forests  # Cannot split

    # Split sequences in half
    mid = len(seq_ids) // 2
    seq_ids_first = set(seq_ids[:mid])
    seq_ids_second = set(seq_ids[mid:])

    # Rebuild first tree with first half of sequences
    new_tree_id_1 = tree["root"].tree_id
    new_root_1 = _BuildNode(new_tree_id_1, -1, -1)
    all_nodes_1: list[_BuildNode] = []

    # Rebuild second tree with second half of sequences
    new_tree_id_2 = len(forests)  # New tree ID
    new_root_2 = _BuildNode(new_tree_id_2, -1, -1)
    all_nodes_2: list[_BuildNode] = []

    # Extract sequences from original tree nodes and re-insert
    seq_tokens: dict[int, list[int]] = {}
    _extract_sequences_from_tree(tree["root"], [], seq_tokens)

    nodes_1 = 0
    nodes_2 = 0
    for seq_id, tokens in seq_tokens.items():
        if seq_id in seq_ids_first:
            _insert_sequence(new_root_1, all_nodes_1, tokens, new_tree_id_1, seq_id)
            nodes_1 = len(all_nodes_1)
        else:
            _insert_sequence(new_root_2, all_nodes_2, tokens, new_tree_id_2, seq_id)
            nodes_2 = len(all_nodes_2)

    # Replace the original tree with the first new tree
    forests[tree_idx] = {
        "root": new_root_1,
        "all_nodes": all_nodes_1,
        "nodes": nodes_1,
    }

    # Append the second new tree
    forests.append({
        "root": new_root_2,
        "all_nodes": all_nodes_2,
        "nodes": nodes_2,
    })

    return forests


@trace_perf("tree_attn._greedy_build_tries")
def _greedy_build_tries(
    data: dict[str, Any],
    max_tokens_per_tree: int,
    min_trees: int = 1,
    n_trees_divisor: int = 1,
) -> tuple[list[TrieNode], list[int]]:
    """Build tries using greedy packing strategy.

    Parameters
    ----------
    data : dict[str, Any]
        Dictionary containing 'input_ids' and 'attention_mask' tensors.
    max_tokens_per_tree : int
        Maximum tokens allowed per tree.
    min_trees : int, default=1
        Minimum number of trees to produce. Trees will be split if needed.
    n_trees_divisor : int, default=1
        Final number of trees must be divisible by this value.

    Returns
    -------
    tuple[list[TrieNode], list[int]]
        List of compressed TrieNodes and list of token counts per tree.
    """
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

    # Ensure min_trees is at least n_trees_divisor
    if min_trees < n_trees_divisor:
        min_trees = n_trees_divisor

    # Split trees until min count and divisor requirements are met
    while len(forests) < min_trees or len(forests) % n_trees_divisor != 0:
        # Find tree with most sequences (can be split)
        best_idx = _find_splittable_tree_idx(forests)
        if best_idx is None:
            # Cannot split further - raise error
            raise RuntimeError(
                f"Cannot split trees to meet n_mbs={min_trees} or n_mbs_divisor={n_trees_divisor}. "
                f"Current tree count: {len(forests)}. "
                f"Each tree has only 1 sequence and cannot be split further. "
                f"Consider reducing n_mbs/n_mbs_divisor or increasing the number of sequences."
            )
        # Split the tree
        forests = _split_tree_by_sequences(forests, best_idx)

        # Update min_trees to next valid divisor multiple if needed
        if len(forests) >= min_trees and len(forests) % n_trees_divisor != 0:
            min_trees = ((len(forests) // n_trees_divisor) + 1) * n_trees_divisor

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
) -> torch.Tensor:
    """Generate position IDs for packed tree inputs.
    Position IDs are computed from the attention mask by counting the number
    of ancestors each token can attend to (minus 1 for 0-indexing).
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
