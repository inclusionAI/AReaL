import torch

import areal.utils.logging as logging

from .cpu import CpuPlatform
from .cuda import CudaPlatform
from .npu import NPUPlatform
from .platform import Platform
from .unknown import UnknownPlatform

logger = logging.getLogger("Platform init")


def is_torch_npu_available() -> bool:
    """Check the availability of NPU."""
    try:
        import torch_npu

        return True
    except ImportError:
        return False


is_npu_available = is_torch_npu_available()


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
    elif is_npu_available:
        # device_name = torch.npu.get_device_name().upper()
        # logger.info(f"Detected NPU device: {device_name}")
        # if "NPU" in device_name:
        #     logger.info("Initializing NPU platform (NPU).")
        #     return NPUPlatform()
        # logger.warning("Unrecognized NPU device. Falling back to UnknownPlatform.")
        return NPUPlatform()
    else:
        logger.info("No supported accelerator detected. Initializing CPU platform.")
        return CpuPlatform()


# Global singleton representing the current platform in use.
current_platform: Platform = _init_platform()

__all__ = [
    "Platform",
    "current_platform",
    "is_npu_available"
]
