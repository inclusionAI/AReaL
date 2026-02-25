"""Base Tool Model Actor - Abstract base class for all tool agents."""

import os
from abc import ABC, abstractmethod
from typing import List, Optional, Union

import ray


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
        self.gpu_ids = [int(g) for g in ray.get_gpu_ids()] or [0]

        # Set CUDA_VISIBLE_DEVICES for vLLM and other frameworks
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in self.gpu_ids)

        # For transformers: after setting CUDA_VISIBLE_DEVICES, device 0 maps to first visible GPU
        self.device = "cuda:0"
        self.device_map = "cuda:0"

    @abstractmethod
    def __init__(
        self,
        model_name: str,
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.8,
        system_prompt: Optional[str] = None,
    ):
        """Initialize the tool model actor.

        Args:
            model_name: Path or name of the model to load.
            max_model_len: Maximum sequence length.
            gpu_memory_utilization: Fraction of GPU memory to use.
            system_prompt: Optional system prompt for the model.

        Note:
            Subclasses should call self.setup_gpu() at the beginning of __init__
            before loading any models.
        """
        pass

    @abstractmethod
    def analyze(
        self,
        image_b64: str,
        question: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        """Analyze an image and answer the question.

        Args:
            image_b64: Base64-encoded image string.
            question: Question to ask about the image.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

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
