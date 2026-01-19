"""GPU memory profiling utilities controlled by environment variables.

This module provides a context manager for recording CUDA memory history
and dumping snapshots for analysis. The profiling is controlled by the
CUDA_MEMORY_PROFILE_PATH environment variable.

Usage:
    # Set environment variable to enable profiling
    export CUDA_MEMORY_PROFILE_PATH=/path/to/dump/memory_snapshot.pickle

    # In code
    from areal.utils.profile_memory import profile_memory

    with profile_memory("my_operation"):
        # Code to profile
        ...

    # Or as a decorator
    @profile_memory("my_function")
    def my_function():
        ...
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Callable, Generator, TypeVar

import torch
import torch.distributed as dist

# Environment variable to control memory profiling
CUDA_MEMORY_PROFILE_PATH_ENV = os.environ.get(
    "AREAL_CUDA_MEMORY_PROFILE_PATH", 
    None,
)

# Global counter for unique snapshot names
_snapshot_counter = 0


def _get_rank() -> int:
    """Get the distributed rank, or 0 if not in distributed mode."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


@contextmanager
def profile_memory(
    name: str = "snapshot",
    max_entries: int = 100000,
) -> Generator[None, None, None]:
    """Context manager for profiling CUDA memory usage.

    Records memory allocation history and dumps a snapshot when the context
    exits. Profiling is only enabled when CUDA_MEMORY_PROFILE_PATH environment
    variable is set. In distributed settings, each rank dumps to a separate file.

    Args:
        name: Name prefix for the snapshot file. The final filename will be
            "{name}_rank{rank}_{counter}.pickle" appended to the path from env var.
        max_entries: Maximum number of entries to record in memory history.

    Example:
        export CUDA_MEMORY_PROFILE_PATH=/tmp/memory_profiles/

        with profile_memory("attention"):
            # Memory allocations here will be recorded
            output = model(input)

        # Snapshot saved to /tmp/memory_profiles/attention_rank0_0.pickle
    """
    global _snapshot_counter
    global CUDA_MEMORY_PROFILE_PATH_ENV
    
    dump_path = CUDA_MEMORY_PROFILE_PATH_ENV

    if not dump_path or not torch.cuda.is_available():
        # Profiling disabled, just yield
        yield
        return

    # Ensure directory exists
    os.makedirs(dump_path, exist_ok=True)

    # Generate unique filename with rank for distributed runs
    rank = _get_rank()
    filename = f"{name}_rank{rank}_{_snapshot_counter}.pickle"
    full_path = os.path.join(dump_path, filename)
    _snapshot_counter += 1

    # Start recording
    torch.cuda.memory._record_memory_history(max_entries=max_entries)

    oom_error = None
    try:
        yield
    except torch.cuda.OutOfMemoryError as e:
        oom_error = e
    except RuntimeError as e:
        # Catch legacy OOM errors (older PyTorch versions)
        if "CUDA out of memory" in str(e) or "out of memory" in str(e).lower():
            oom_error = e
        else:
            raise
    finally:
        # Always attempt to dump snapshot, even on OOM
        try:
            torch.cuda.memory._dump_snapshot(full_path)
        except Exception as dump_error:
            print(f"Warning: Failed to dump memory snapshot to {full_path}: {dump_error}")

        # Stop recording
        torch.cuda.memory._record_memory_history(enabled=None)

    # Re-raise OOM error after snapshot is saved
    if oom_error is not None:
        raise oom_error


F = TypeVar("F", bound=Callable)


def profile_memory_decorator(
    name: str = "snapshot",
    max_entries: int = 100000,
) -> Callable[[F], F]:
    """Decorator version of profile_memory.

    Args:
        name: Name prefix for the snapshot file.
        max_entries: Maximum number of entries to record.

    Example:
        @profile_memory_decorator("forward_pass")
        def forward(self, x):
            ...
    """

    def decorator(func: F) -> F:
        def wrapper(*args, **kwargs):
            with profile_memory(name, max_entries):
                return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator
