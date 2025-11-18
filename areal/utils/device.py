import gc

import torch.distributed as dist

from areal.platforms import current_platform
from areal.utils import logging

logger = logging.getLogger(__file__)


def _get_current_mem_info(unit: str = "GB", precision: int = 2) -> tuple[str]:
    """Get current memory usage."""
    assert unit in ["GB", "MB", "KB"]
    divisor = 1024**3 if unit == "GB" else 1024**2 if unit == "MB" else 1024
    mem_allocated = current_platform.memory_allocated()
    mem_reserved = current_platform.memory_reserved()
    mem_free, mem_total = current_platform.mem_get_info()
    mem_used = mem_total - mem_free
    mem_allocated = f"{mem_allocated / divisor:.{precision}f}"
    mem_reserved = f"{mem_reserved / divisor:.{precision}f}"
    mem_used = f"{mem_used / divisor:.{precision}f}"
    mem_total = f"{mem_total / divisor:.{precision}f}"
    return mem_allocated, mem_reserved, mem_used, mem_total


# Adapted from verl
def log_gpu_stats(head: str, rank: int = 0):
    if (not dist.is_initialized()) or (rank is None) or (dist.get_rank() == rank):
        mem_allocated, mem_reserved, mem_used, mem_total = _get_current_mem_info()
        message = f"{head}, memory allocated (GB): {mem_allocated}, memory reserved (GB): {mem_reserved}, device memory used/total (GB): {mem_used}/{mem_total}"
        logger.info(msg=message)


def clear_memory():
    """Clear device memory cache and run garbage collection."""
    current_platform.synchronize()
    gc.collect()
    current_platform.empty_cache()


def print_memory(msg: str, clear_before_print: bool = False):
    """Print detailed memory usage information."""
    if clear_before_print:
        clear_memory()

    mem_allocated, mem_reserved, mem_used, mem_total = _get_current_mem_info()

    logger.info(
        f"Memory-Usage {msg}{' (cleared before print)' if clear_before_print else ''}: "
        f"memory allocated (GB): {mem_allocated}, "
        f"memory reserved (GB): {mem_reserved}, "
        f"device memory used/total (GB): {mem_used}/{mem_total}"
    )
