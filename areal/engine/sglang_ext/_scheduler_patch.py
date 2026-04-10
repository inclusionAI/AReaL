"""Pickle-safe scheduler process wrapper for multiprocessing spawn compatibility.

This module provides the spawn target function for SGLang scheduler in a form
that is compatible with Python's multiprocessing spawn mechanism (used on macOS
and explicitly on Linux).

The key design:
- Module-level or callable class instances pickle as just (name, {})
- All heavy imports deferred to inside __call__() or function body
- Unpickle resolves in subprocess with fresh imports, avoiding circular deps
"""

from __future__ import annotations


class AwexSchedulerLauncher:
    """Pickle-safe scheduler launcher for SGLang Engine under multiprocessing spawn.

    This callable class wraps SGLang's run_scheduler_process with AWEX patching.
    Unlike @staticmethod, this pickles safely on all platforms:

    - Pickle stores: (AwexSchedulerLauncher, {})  [minimal footprint]
    - Unpickle: Creates instance, calls __call__() with args
    - All imports happen inside __call__ = resolved in subprocess context

    This is the pattern used by PyTorch Distributed and Ray Tune.
    """

    def __call__(self, *args, **kwargs):
        """Execute scheduler process with AWEX patches applied.

        Args:
            *args: Positional args for sglang.srt.managers.scheduler.run_scheduler_process
            **kwargs: Keyword args for sglang.srt.managers.scheduler.run_scheduler_process

        Returns:
            Return value from run_scheduler_process (typically None or int)

        Note:
            All imports are inside this method (not at module level) to ensure
            they resolve in the subprocess context after unpickling, avoiding
            circular import issues that plague spawn mode.
        """
        # CRITICAL: Import after unpickle in subprocess = avoid circular deps
        from sglang.srt.managers.scheduler import run_scheduler_process

        # Apply AReaL-specific patches to Scheduler class
        from areal.engine.sglang_ext.sglang_worker_extension import (
            patch_scheduler_for_awex,
        )

        patch_scheduler_for_awex()

        # Delegate to original SGLang scheduler
        return run_scheduler_process(*args, **kwargs)


# Module-level instance for direct use as spawn target
_AWEX_SCHEDULER_LAUNCHER = AwexSchedulerLauncher()

__all__ = ["AwexSchedulerLauncher", "_AWEX_SCHEDULER_LAUNCHER"]
