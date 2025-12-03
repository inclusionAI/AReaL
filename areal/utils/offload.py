"""Utilities for torch_memory_saver (TMS) configuration and setup.

This module handles the environment variable setup required for TMS to work
properly with LD_PRELOAD hooks.
"""

import os

import torch_memory_saver


def get_tms_env_vars() -> dict[str, str]:
    """Get environment variables for torch_memory_saver (TMS)."""

    # Locate the LD_PRELOAD shared library
    dynlib_path = os.path.join(
        os.path.dirname(os.path.dirname(torch_memory_saver.__file__)),
        "torch_memory_saver_hook_mode_preload.abi3.so",
    )

    if not os.path.exists(dynlib_path):
        raise RuntimeError(f"LD_PRELOAD so file {dynlib_path} does not exist.")

    env_vars = {
        "LD_PRELOAD": dynlib_path,
        "TMS_INIT_ENABLE": "1",
        "TMS_INIT_ENABLE_CPU_BACKUP": "1",
    }
    return env_vars


def is_tms_enabled() -> bool:
    return os.environ.get("TMS_INIT_ENABLE", "0") == "1"
