"""Base Tool Model Actor - Abstract base class for all tool agents."""

import os
from abc import ABC, abstractmethod
from typing import List, Optional, Union

import ray

from geo_edit.utils.logger import setup_logger

logger = setup_logger(__name__)


class BaseToolModelActor(ABC):
    """Abstract base class for Tool Model Actors.

    All tool agents must inherit from this class and implement:
    - __init__: Initialize model and resources
    - analyze: Process image and question
    - health_check: Return actor health status

    GPU Assignment:
        Ray assigns GPUs to actors via num_gpus parameter. Subclasses should call
        setup_gpu() at the beginning of __init__ to configure GPU visibility.

        For transformers: use self.device or self.device_map
        For vLLM: CUDA_VISIBLE_DEVICES is automatically set
    """

    # GPU configuration set by setup_gpu()
    device: str = "cuda:0"
    device_map: Union[str, int] = 0
    gpu_ids: List[int] = [0]

    def setup_gpu(self) -> None:
        """Configure GPU visibility based on Ray assignment.

        Call this at the beginning of __init__ before loading any models.
        Sets:
            - self.gpu_ids: List of assigned GPU IDs
            - self.device: torch device string (e.g., "cuda:0")
            - self.device_map: For transformers (GPU index or device string)
            - CUDA_VISIBLE_DEVICES: For vLLM and other frameworks
        """
        ray_gpu_ids = ray.get_gpu_ids()
        self.gpu_ids = [int(g) for g in ray_gpu_ids] if ray_gpu_ids else [0]

        logger.info(
            "setup_gpu: Ray assigned GPUs: %s, setting CUDA_VISIBLE_DEVICES=%s",
            ray_gpu_ids,
            ",".join(str(g) for g in self.gpu_ids),
        )

        # Set CUDA_VISIBLE_DEVICES for vLLM and other frameworks
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in self.gpu_ids)

        # For transformers: after setting CUDA_VISIBLE_DEVICES, device 0 maps to first visible GPU
        self.device = "cuda:0"
        self.device_map = "cuda:0"

    @abstractmethod
    def __init__(self, **kwargs):
        """Initialize the tool model actor with flexible parameters.

        Subclasses can define their own required parameters.
        Common parameters:
            model_name (str): Path or name of the model to load.
            max_model_len (int): Maximum sequence length (for VLM agents).
            gpu_memory_utilization (float): Fraction of GPU memory to use (for vLLM agents).
            system_prompt (Optional[str]): System prompt for the model (for VLM agents).

        Note:
            Subclasses should call self.setup_gpu() at the beginning of __init__
            before loading any models.
        """
        pass

    @abstractmethod
    def analyze(
        self,
        image_b64: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        **kwargs,
    ) -> str:
        """Analyze an image with tool-specific parameters.

        Args:
            image_b64: Base64-encoded image string.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            **kwargs: Tool-specific parameters (e.g., question, text_prompt, mode, bbox, etc.).

        Returns:
            Analysis result as string.
        """
        pass

    @abstractmethod
    def health_check(self) -> dict:
        """Return health status of the actor.

        Returns:
            Dict with at least 'model' and 'initialized' keys.
        """
        pass
