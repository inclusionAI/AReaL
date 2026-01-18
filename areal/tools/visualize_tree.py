"""Visualize dumped tree pack data from build_packed_tree_batch.

This script loads and visualizes the data dumped by build_packed_tree_batch
when TREE_PACK_DUMP_PATH environment variable is set.

Usage:
    python -m areal.tools.visualize_tree --dump-dir /path/to/dump --call 1 --rank 0
    python -m areal.tools.visualize_tree --dump-dir /path/to/dump --call 1 --rank 0 --mask-granularity 64
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import torch

from areal.models.tree_attn.visualize import (
    visualize_attention_mask,
    visualize_forest,
    visualize_trie,
)
from areal.utils import logging

logger = logging.getLogger(__name__)


def load_dump_data(dump_dir: Path, call: int, rank: int) -> tuple[dict, list]:
    """Load dumped data and trie nodes from files.

    Parameters
    ----------
    dump_dir : Path
        Directory containing the dump files.
    call : int
        Call count number.
    rank : int
        Rank number.

    Returns
    -------
    tuple[dict, list]
        Tuple of (tensor_data, trie_nodes).
    """
    base_filename = f"call{call}_rank{rank}"
    tensor_filepath = dump_dir / f"{base_filename}.pt"
    trie_filepath = dump_dir / f"{base_filename}_trie.pkl"

    if not tensor_filepath.exists():
        raise FileNotFoundError(f"Tensor data file not found: {tensor_filepath}")
    if not trie_filepath.exists():
        raise FileNotFoundError(f"Trie data file not found: {trie_filepath}")

    # Load tensor data
    tensor_data = torch.load(tensor_filepath, map_location="cpu")

    # Load trie nodes
    with open(trie_filepath, "rb") as f:
        trie_nodes = pickle.load(f)

    return tensor_data, trie_nodes


def visualize_dump(
    dump_dir: Path,
    call: int,
    rank: int,
    mask_granularity: int = 128,
    max_tokens_display: int = 5,
    show_input_data: bool = False,
) -> None:
    """Visualize dumped tree pack data.

    Parameters
    ----------
    dump_dir : Path
        Directory containing the dump files.
    call : int
        Call count number.
    rank : int
        Rank number.
    mask_granularity : int, default=128
        Granularity for attention mask visualization.
    max_tokens_display : int, default=5
        Maximum number of tokens to display per trie node.
    show_input_data : bool, default=False
        Whether to show input data summary.
    """
    logger.info(f"Loading dump data from {dump_dir} (call={call}, rank={rank})")

    tensor_data, trie_nodes = load_dump_data(dump_dir, call, rank)

    # Show input data summary
    if show_input_data:
        logger.info("=" * 60)
        logger.info("Input Data Summary")
        logger.info("=" * 60)
        input_data = tensor_data.get("input_data", {})
        for key, value in input_data.items():
            if torch.is_tensor(value):
                logger.info(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                logger.info(f"  {key}: {type(value).__name__}")

    # Show output mbs summary
    output_mbs = tensor_data.get("output_mbs", [])
    logger.info("=" * 60)
    logger.info(f"Output MicroBatches Summary: {len(output_mbs)} micro-batches")
    logger.info("=" * 60)
    for i, mb in enumerate(output_mbs):
        logger.info(f"  MicroBatch {i}:")
        for key, value in mb.items():
            if torch.is_tensor(value):
                logger.info(f"    {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                logger.info(f"    {key}: {type(value).__name__}")

    # Visualize trie nodes
    logger.info("")
    logger.info("=" * 60)
    logger.info("Trie Forest Visualization")
    logger.info("=" * 60)
    if trie_nodes:
        forest_viz = visualize_forest(trie_nodes, max_tokens_display=max_tokens_display)
        logger.info("\n%s", forest_viz)
    else:
        logger.info("No trie nodes found.")

    # Visualize attention masks
    # logger.info("")
    # logger.info("=" * 60)
    # logger.info("Attention Mask Visualizations")
    # logger.info("=" * 60)
    # for i, mb in enumerate(output_mbs):
    #     attention_mask = mb.get("attention_mask")
    #     if attention_mask is not None:
    #         logger.info(f"\nMicroBatch {i} Attention Mask:")
    #         visualize_attention_mask(attention_mask, granularity=mask_granularity)


def list_available_dumps(dump_dir: Path) -> list[tuple[int, int]]:
    """List all available dump files in a directory.

    Parameters
    ----------
    dump_dir : Path
        Directory containing dump files.

    Returns
    -------
    list[tuple[int, int]]
        List of (call, rank) tuples for available dumps.
    """
    available = []
    for pt_file in dump_dir.glob("call*_rank*.pt"):
        # Parse filename: call{N}_rank{M}.pt
        name = pt_file.stem
        if "_trie" in name:
            continue
        try:
            parts = name.split("_")
            call = int(parts[0].replace("call", ""))
            rank = int(parts[1].replace("rank", ""))
            # Check if corresponding trie file exists
            trie_file = dump_dir / f"call{call}_rank{rank}_trie.pkl"
            if trie_file.exists():
                available.append((call, rank))
        except (ValueError, IndexError):
            continue
    return sorted(available)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize dumped tree pack data from build_packed_tree_batch."
    )
    parser.add_argument(
        "--dump-dir",
        type=Path,
        required=True,
        help="Directory containing the dump files.",
    )
    parser.add_argument(
        "--call",
        type=int,
        default=None,
        help="Call count number to visualize. If not specified, lists available dumps.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help="Rank number to visualize (default: 0).",
    )
    parser.add_argument(
        "--mask-granularity",
        type=int,
        default=128,
        help="Granularity for attention mask visualization (default: 128).",
    )
    parser.add_argument(
        "--max-tokens-display",
        type=int,
        default=5,
        help="Maximum tokens to display per trie node (default: 5).",
    )
    parser.add_argument(
        "--show-input-data",
        action="store_true",
        help="Show input data summary.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_dumps",
        help="List all available dumps in the directory.",
    )

    args = parser.parse_args()

    if not args.dump_dir.exists():
        logger.error(f"Dump directory does not exist: {args.dump_dir}")
        sys.exit(1)

    if args.list_dumps or args.call is None:
        available = list_available_dumps(args.dump_dir)
        if not available:
            logger.info(f"No dump files found in {args.dump_dir}")
        else:
            logger.info(f"Available dumps in {args.dump_dir}:")
            for call, rank in available:
                logger.info(f"  call={call}, rank={rank}")
        if args.call is None and not args.list_dumps:
            logger.info("\nUse --call N to visualize a specific dump.")
        sys.exit(0)

    try:
        visualize_dump(
            dump_dir=args.dump_dir,
            call=args.call,
            rank=args.rank,
            mask_granularity=args.mask_granularity,
            max_tokens_display=args.max_tokens_display,
            show_input_data=args.show_input_data,
        )
    except FileNotFoundError as e:
        logger.error(str(e))
        logger.info("\nAvailable dumps:")
        available = list_available_dumps(args.dump_dir)
        for call, rank in available:
            logger.info(f"  call={call}, rank={rank}")
        sys.exit(1)


if __name__ == "__main__":
    main()
