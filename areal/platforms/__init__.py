from __future__ import annotations

import torch

import areal.utils.logging as logging

from .cpu import CpuPlatform
from .cuda import CudaPlatform
from .platform import Platform
from .unknown import UnknownPlatform

logger = logging.getLogger("Platform init")


def _init_platform() -> Platform:
    """
    Detect and initialize the appropriate platform based on available devices.
    Priority:
    1. CUDA (NVIDIA)
    2. TODO: NPU (if torch_npu is installed)
    3. CPU (fallback)
    Returns:
        An instance of a subclass of Platform corresponding to the detected hardware.
    """
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name().upper()
        logger.info(f"Detected CUDA device: {device_name}")
        if "NVIDIA" in device_name:
            logger.info("Initializing CUDA platform (NVIDIA).")
            return CudaPlatform()
        logger.warning("Unrecognized CUDA device. Falling back to UnknownPlatform.")
        return UnknownPlatform()
    else:
        logger.info("No supported accelerator detected. Initializing CPU platform.")
        return CpuPlatform()


class _LazyPlatform:
    """
    Lazy initialization wrapper for platform detection.

    This class defers platform detection and initialization until the first
    attribute access, avoiding CUDA initialization during import.
    """

    def __init__(self):
        self._platform: Platform | None = None
        self._initialized = False

    def _ensure_initialized(self) -> Platform:
        """Ensure the platform is initialized and return it."""
        if not self._initialized:
            self._platform = _init_platform()
            self._initialized = True
        return self._platform

    def __getattr__(self, name: str):
        """Delegate attribute access to the underlying platform."""
        platform = self._ensure_initialized()
        return getattr(platform, name)

    def __setattr__(self, name: str, value):
        """Handle setting attributes on the wrapper."""
        if name.startswith("_"):
            # Allow setting private attributes on the wrapper itself
            super().__setattr__(name, value)
        else:
            # Delegate to the underlying platform
            platform = self._ensure_initialized()
            setattr(platform, name, value)

    def __repr__(self) -> str:
        """Return string representation."""
        if self._initialized:
            return f"LazyPlatform({self._platform!r})"
        else:
            return "LazyPlatform(uninitialized)"


# Global singleton representing the current platform in use.
# Platform detection and initialization is deferred until first access.
current_platform: Platform = _LazyPlatform()

__all__ = [
    "Platform",
    "current_platform",
]
