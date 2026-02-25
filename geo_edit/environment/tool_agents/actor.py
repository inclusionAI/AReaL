"""Base Tool Model Actor - Abstract base class for all tool agents."""

from abc import ABC, abstractmethod
from typing import Optional


class BaseToolModelActor(ABC):
    """Abstract base class for Tool Model Actors.

    All tool agents must inherit from this class and implement:
    - __init__: Initialize model and resources
    - analyze: Process image and question
    - health_check: Return actor health status
    """

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
