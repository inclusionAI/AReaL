"""Utilities for model and optimizer offloading to CPU."""

import gc

from platforms import current_platform
from utils.logging import getLogger

logger = getLogger(__name__)


def clear_memory():
    """Clear CUDA memory cache."""
    current_platform.synchronize()
    gc.collect()
    current_platform.empty_cache()


def print_memory(msg: str, clear_before_print: bool = False):
    """Print memory usage information."""
    if clear_before_print:
        clear_memory()

    device = current_platform.current_device()
    free, total = current_platform.mem_get_info(device)
    allocated = current_platform.memory_allocated(device)
    reserved = current_platform.memory_reserved(device)

    def _byte_to_gb(n: int) -> float:
        return round(n / (1024**3), 2)

    memory_info = {
        "gpu": str(device),
        "total_GB": _byte_to_gb(total),
        "free_GB": _byte_to_gb(free),
        "used_GB": _byte_to_gb(total - free),
        "allocated_GB": _byte_to_gb(allocated),
        "reserved_GB": _byte_to_gb(reserved),
    }

    logger.info(
        f"Memory-Usage {msg}{' (cleared before print)' if clear_before_print else ''}:",
        memory_info,
    )
    return memory_info
