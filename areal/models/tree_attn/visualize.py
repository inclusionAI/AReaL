import torch

from areal.utils import logging
from areal.models.tree_attn.tree import TrieNode

logger = logging.getLogger(__name__)


# Helper function for visualizing attention masks
# Used for debugging and performance optimization
def visualize_attention_mask(mask_tensor: torch.Tensor, granularity: int = 128) -> None:
    """Visualize an attention mask as a text grid with configurable granularity.

    Parameters
    ----------
    mask_tensor : torch.Tensor
        A 2D boolean or numeric tensor representing the attention mask.
    granularity : int, default=128
        Maximum number of cells to display in each dimension.
        If the mask is larger than this, it will be downsampled by aggregating
        blocks. Each cell shows the density of attention in that block.
    """
    mask = mask_tensor.bool().cpu().float()
    n = mask.shape[0]

    if n == 0:
        logger.info("Attention Mask Visualization: empty mask (0x0)")
        return

    # Determine block size for downsampling
    if n <= granularity:
        # No downsampling needed, show full resolution
        display_size = n
        block_size = 1
        downsampled = mask
    else:
        # Downsample by averaging blocks
        display_size = granularity
        block_size = (n + granularity - 1) // granularity  # ceiling division

        # Pad mask to be evenly divisible by block_size
        padded_size = block_size * display_size
        if padded_size > n:
            padded_mask = torch.zeros((padded_size, padded_size), dtype=mask.dtype)
            padded_mask[:n, :n] = mask
            mask = padded_mask

        # Reshape and compute mean for each block
        mask_reshaped = mask[: block_size * display_size, : block_size * display_size]
        mask_reshaped = mask_reshaped.view(
            display_size, block_size, display_size, block_size
        )
        downsampled = mask_reshaped.mean(dim=(1, 3))

    # Define density characters (from empty to full)
    density_chars = " ·░▒▓█"

    # Build visualization
    lines = []
    lines.append(
        f"Attention Mask ({n}x{n}), displayed at {display_size}x{display_size} (block size: {block_size}x{block_size})"
    )

    # Header with column indices (every 10 columns)
    header_line1 = "     "
    header_line2 = "     "
    for i in range(display_size):
        if i % 10 == 0:
            header_line1 += (
                f"{(i * block_size) // 1000 % 10}" if i * block_size >= 1000 else " "
            )
            header_line2 += (
                f"{(i * block_size) // 100 % 10}" if i * block_size >= 100 else " "
            )
        else:
            header_line1 += " "
            header_line2 += " "
    lines.append(header_line1)
    lines.append(header_line2)

    header_line3 = "     " + "".join(
        f"{(i * block_size) // 10 % 10}" for i in range(display_size)
    )
    header_line4 = "     " + "".join(
        f"{(i * block_size) % 10}" for i in range(display_size)
    )
    lines.append(header_line3)
    lines.append(header_line4)
    lines.append("     " + "-" * display_size)

    for row_idx in range(display_size):
        row_chars = []
        for col_idx in range(display_size):
            density = downsampled[row_idx, col_idx].item()
            # Map density [0, 1] to character index [0, 5]
            char_idx = min(int(density * len(density_chars)), len(density_chars) - 1)
            if density > 0 and char_idx == 0:
                char_idx = (
                    1  # Ensure non-zero density shows at least the lightest character
                )
            row_chars.append(density_chars[char_idx])
        row_str = "".join(row_chars)
        actual_row = row_idx * block_size
        lines.append(f"{actual_row:4d}|{row_str}")

    visualization = "\n".join(lines)
    logger.info("Attention Mask Visualization:\n%s", visualization)

def visualize_trie(
    trie: TrieNode,
    max_tokens_display: int = 5,
    indent: str = "  ",
) -> str:
    """Visualize a compressed trie structure as a text tree.

    Parameters
    ----------
    trie : TrieNode
        The root TrieNode to visualize.
    max_tokens_display : int, default=5
        Maximum number of tokens to display per node. If a node has more tokens,
        they will be truncated with "...".
    indent : str, default="  "
        String used for indentation at each level.

    Returns
    -------
    str
        A string representation of the trie structure.

    Example
    -------
    >>> print(visualize_trie(trie))
    Tree 0 (total tokens: 150, sequences: [0, 1, 2, 3])
    Shared prefix savings: 120 tokens (compression ratio: 1.80x)
    └── [0-49] 50 toks × 4 seqs (shared 150) tokens=[1, 2, 3, ...] seqs=[0, 1, 2, 3]
        ├── [50-79] 30 toks × 2 seqs (shared 30) tokens=[51, 52, ...] seqs=[0, 1]
        │   ├── [80-99] 20 toks × 1 seq tokens=[81, ...] seqs=[0]
        │   └── [100-119] 20 toks × 1 seq tokens=[101, ...] seqs=[1]
        └── [120-149] 30 toks × 2 seqs (shared 30) tokens=[121, ...] seqs=[2, 3]
    """
    lines: list[str] = []

    # Calculate shared prefix statistics
    total_tokens = trie.num_tokens
    all_seqs = trie.all_sequence_ids

    # Calculate total tokens if no sharing (sum of all sequence lengths)
    total_unshared_tokens = 0
    shared_prefix_savings = 0
    for node in trie.nodes:
        num_tokens = len(node.tokens)
        num_seqs_using = len(node.sequence_ids)
        # Without sharing, each sequence would have its own copy
        total_unshared_tokens += num_tokens * num_seqs_using
        # Shared tokens: tokens that are reused (count - 1 for each extra sequence)
        if num_seqs_using > 1:
            shared_prefix_savings += num_tokens * (num_seqs_using - 1)

    # Header with tree summary
    lines.append(
        f"Tree {trie.tree_id} (total tokens: {total_tokens}, sequences: {all_seqs})"
    )
    if shared_prefix_savings > 0:
        compression_ratio = total_unshared_tokens / total_tokens if total_tokens > 0 else 1.0
        lines.append(
            f"Shared prefix savings: {shared_prefix_savings} tokens "
            f"(compression ratio: {compression_ratio:.2f}x)"
        )
    else:
        lines.append("Shared prefix savings: 0 tokens (no prefix sharing)")

    def _format_tokens(tokens: list[int]) -> str:
        """Format token list with truncation."""
        if len(tokens) <= max_tokens_display:
            return str(tokens)
        truncated = tokens[:max_tokens_display]
        return f"[{', '.join(map(str, truncated))}, ...]"

    def _format_seqs(seq_ids: list[int]) -> str:
        """Format sequence IDs."""
        if len(seq_ids) <= 5:
            return str(seq_ids)
        return f"[{', '.join(map(str, seq_ids[:5]))}, ...({len(seq_ids)} total)]"

    def _visualize_node(
        node: TrieNode,
        prefix: str,
        is_last: bool,
        depth: int,
    ) -> None:
        """Recursively visualize a node and its children."""
        # Determine the connector
        connector = "└── " if is_last else "├── "

        # Calculate node statistics
        num_tokens = len(node.tokens)
        num_seqs = len(node.sequence_ids)
        shared_count = num_tokens * (num_seqs - 1) if num_seqs > 1 else 0

        # Format node info
        idx_range = f"[{node.start_idx}-{node.end_idx}]"
        tokens_str = _format_tokens(node.tokens)
        seqs_str = _format_seqs(node.sequence_ids)

        # Build node info string with token count and sharing info
        if num_seqs > 1:
            sharing_info = f" (shared {shared_count})"
            node_info = (
                f"{idx_range} {num_tokens} toks × {num_seqs} seqs{sharing_info} "
                f"tokens={tokens_str} seqs={seqs_str}"
            )
        else:
            node_info = (
                f"{idx_range} {num_tokens} toks × {num_seqs} seq "
                f"tokens={tokens_str} seqs={seqs_str}"
            )

        lines.append(f"{prefix}{connector}{node_info}")

        # Prepare prefix for children
        child_prefix = prefix + ("    " if is_last else "│   ")

        # Recursively visualize children
        children = list(node.children.values())
        for i, child in enumerate(children):
            _visualize_node(
                child,
                child_prefix,
                is_last=(i == len(children) - 1),
                depth=depth + 1,
            )

    # Visualize children of the root
    children = list(trie.children.values())
    for i, child in enumerate(children):
        _visualize_node(
            child,
            prefix="",
            is_last=(i == len(children) - 1),
            depth=0,
        )

    return "\n".join(lines)


def visualize_forest(
    tries: list[TrieNode],
    max_tokens_display: int = 5,
) -> str:
    """Visualize a list of tries (forest) produced by _greedy_build_tries.

    Parameters
    ----------
    tries : list[TrieNode]
        List of TrieNode roots to visualize.
    max_tokens_display : int, default=5
        Maximum number of tokens to display per node.

    Returns
    -------
    str
        A string representation of all tries in the forest.

    Example
    -------
    >>> tries, num_tokens = _greedy_build_tries(data, max_tokens_per_tree=100)
    >>> print(visualize_forest(tries))
    ============================================================
    Forest Summary: 3 trees, 450 total tokens, 12 sequences
    Total shared prefix savings: 180 tokens (compression ratio: 1.40x)
    ============================================================

    Tree 0 (total tokens: 150, sequences: [0, 1, 2, 3])
    Shared prefix savings: 60 tokens (compression ratio: 1.40x)
    └── [0-49] 50 toks × 4 seqs (shared 150) tokens=[1, 2, 3, ...] seqs=[0, 1, 2, 3]
        ...
    """
    lines: list[str] = []

    # Calculate forest-wide statistics
    total_trees = len(tries)
    total_tokens = sum(t.num_tokens for t in tries)
    total_seqs = sum(len(t.all_sequence_ids) for t in tries)

    # Calculate total shared prefix savings across all trees
    total_unshared_tokens = 0
    total_shared_savings = 0
    for trie in tries:
        for node in trie.nodes:
            num_tokens = len(node.tokens)
            num_seqs_using = len(node.sequence_ids)
            total_unshared_tokens += num_tokens * num_seqs_using
            if num_seqs_using > 1:
                total_shared_savings += num_tokens * (num_seqs_using - 1)

    # Forest summary
    lines.append("=" * 60)
    lines.append(
        f"Forest Summary: {total_trees} trees, {total_tokens} total tokens, "
        f"{total_seqs} sequences"
    )
    if total_shared_savings > 0:
        compression_ratio = total_unshared_tokens / total_tokens if total_tokens > 0 else 1.0
        lines.append(
            f"Total shared prefix savings: {total_shared_savings} tokens "
            f"(compression ratio: {compression_ratio:.2f}x)"
        )
    else:
        lines.append("Total shared prefix savings: 0 tokens (no prefix sharing)")
    lines.append("=" * 60)

    # Visualize each tree
    for i, trie in enumerate(tries):
        lines.append("")
        lines.append(visualize_trie(trie, max_tokens_display))

    return "\n".join(lines)