
#!/usr/bin/env python3
"""Base Agent class for VLM Gym"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Optional, List
import time
from dataclasses import dataclass, field
import logging

# Setup logger
logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for VLM agents"""
    # Model configuration
    model_type: str = "HuggingFace"  # HuggingFace, vLLM, OpenAI, Azure
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    model_path: Optional[str] = None  # For local models

    # Generation parameters
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.95
    do_sample: bool = True

    # Device configuration
    device_map: str = "auto"
    torch_dtype: str = "bfloat16"  # float16, bfloat16, float32
    trust_remote_code: bool = True

    # Retry configuration
    n_retry: int = 3
    retry_delay: float = 1.0

    # Advanced features
    return_logprobs: bool = False  # For external RL training
    return_attention: bool = False  # For analysis

    # API specific (for OpenAI/Azure/vLLM)
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    deployment_name: Optional[str] = None
    port: Optional[int] = None  # For vLLM
    generate_config: Optional[Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration after initialization"""
        valid_model_types = ["HuggingFace", "vLLM", "OpenAI", "Azure", "Google"]
        if self.model_type not in valid_model_types:
            raise ValueError(f"model_type must be one of {valid_model_types}, got {self.model_type}")

        valid_dtypes = ["float16", "bfloat16", "float32"]
        if self.torch_dtype not in valid_dtypes:
            raise ValueError(f"torch_dtype must be one of {valid_dtypes}, got {self.torch_dtype}")


class BaseAgent(ABC):
    """Abstract base class for all VLM agents"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize agent with configuration

        Args:
            config: Dictionary containing agent configuration
        """
        if isinstance(config, AgentConfig):
            self.config = config
        else:
            self.config = AgentConfig(**config)

        # Initialize state
        self.conversation_history: List[Dict[str, Any]] = []
        self.step_count: int = 0
        self.total_tokens_used: int = 0
        self.cost: List[float] = []  # For API-based models

        # Model/client placeholder
        self.model = None
        self.client = None
        self._model_loaded = False
        self.processor = None
        self.tokenizer = None

        logger.info("Initialized %s with model: %s", self.__class__.__name__, self.config.model_name)

    @abstractmethod
    def load_model(self):
        """Load the model - must be implemented by subclasses"""
        pass

    @abstractmethod
    def _generate_response(self, model_input: Any) -> Tuple[Any, Dict[str, int | float | str | None]]:
        """Generate response from input

        Args:
            model_input: Input for the model/API

        Returns:
            response: Generated response
            extra_info: Additional information (tokens used, etc.)
        """
        pass

    @abstractmethod
    def _validate_response(self, response: Any) -> None:
        """Validate the response, raise ValueError if invalid"""
        pass

    def act(self, observation: Any) -> Tuple[Any, Dict[str, int | float | str | None]]:
        """Main interface to get action from agent

        Args:
            observation: Input observation (format depends on subclass)

        Returns:
            action: The agent's response
            extra_info: Dictionary with additional information
        """
        if not self._model_loaded:
            self.load_model()

        for attempt in range(self.config.n_retry):
            try:
                content, extra_info = self._generate_response(observation)
                self._validate_response(content)

                extra_info.update({
                    "model_name": self.config.model_name,
                    "attempt": attempt + 1,
                    "step_count": self.step_count,
                })

                self.step_count += 1
                if "tokens_used" in extra_info:
                    self.total_tokens_used += extra_info["tokens_used"]

                logger.info("Step %s: Generated response in attempt %s", self.step_count, attempt + 1)
                return content, extra_info

            except Exception as e:
                logger.warning("Attempt %s failed: %s", attempt + 1, e)
                if attempt < self.config.n_retry - 1:
                    time.sleep(self.config.retry_delay)
                else:
                    error_msg = f"Failed after {self.config.n_retry} attempts: {e}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)

    def reset(self):
        """Reset agent state"""
        self.conversation_history = []
        self.step_count = 0
        self.total_tokens_used = 0
        self.cost = []
        self.client = None
        self._model_loaded = False
        logger.info("Agent state reset")

    def get_state(self) -> Dict[str, Any]:
        """Get current agent state

        Returns:
            Dictionary containing agent state
        """
        return {
            "step_count": self.step_count,
            "total_tokens_used": self.total_tokens_used,
            "total_cost": sum(self.cost),
            "config": self.config.__dict__,
        }

