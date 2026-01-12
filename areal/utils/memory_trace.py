"""Memory profiling utilities using PyTorch's CUDA memory snapshot APIs."""

import os
from contextlib import contextmanager
from typing import Optional

import torch

# Global directory for saving memory snapshots
MEMORY_SNAPSHOT_DIR: str = "/storage/openpsi/users/meizhiyu.mzy/zeta/memory_profile/"
GLOBAL_ENABLE_PROFILE: bool = os.getenv("AREAL_ENABLE_MEMORY_PROFILE", "0") == "1"

@contextmanager
def memory_profile(
    file_name: str,
    enable_profile: bool = True,
    max_entries: int = 100000,
    snapshot_dir: Optional[str] = None,
):
    """Context manager to dump CUDA memory profile for wrapped code block.

    Args:
        file_name: Name of the snapshot file (without directory path).
        enable_profile: Whether to enable memory profiling. If False, the context
            manager is a no-op.
        max_entries: Maximum number of entries to record in memory history.
        snapshot_dir: Directory to save snapshots. If None, uses MEMORY_SNAPSHOT_DIR.

    Example:
        >>> with memory_profile("my_snapshot.pickle", enable_profile=True):
        ...     # Your code here
        ...     model(input)
    """
    enable_profile = enable_profile and GLOBAL_ENABLE_PROFILE
    if not enable_profile:
        yield
        return

    # Determine output directory
    output_dir = snapshot_dir if snapshot_dir is not None else MEMORY_SNAPSHOT_DIR

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Construct full file path
    file_path = os.path.join(output_dir, file_name)

    try:
        # Start recording memory history
        torch.cuda.memory._record_memory_history(max_entries=max_entries)
        yield
    finally:
        # Save the snapshot
        torch.cuda.memory._dump_snapshot(file_path)
        # Stop recording memory history
        torch.cuda.memory._record_memory_history(enabled=None)
