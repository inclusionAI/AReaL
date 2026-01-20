
#!/usr/bin/env python3
"""Base Agent class for VLM Gym"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Optional, List
from google.genai import types
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
        valid_model_types = ["HuggingFace", "vLLM", "OpenAI", "Azure","Google"]
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
        # Parse configuration
        # if isinstance(config.get('agent'), dict):
        #     self.config = AgentConfig(**config['agent'])
        # else:
        if isinstance(config, AgentConfig):
            self.config = config
        else:
            self.config = AgentConfig(**config)
        
        # Initialize state
        self.conversation_history: List[Dict[str, Any]] = []
        self.step_count: int = 0
        self.total_tokens_used: int = 0
        self.cost: List[float] = []  # For API-based models
        
        # Model placeholder
        self.model = None
        self.processor = None
        self.tokenizer = None
        
        logger.info(f"Initialized {self.__class__.__name__} with model: {self.config.model_name}")
    
    @abstractmethod
    def load_model(self):
        """Load the model - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def _prepare_input(self, observation: Dict[str, Any]) -> Any:
        """Prepare input for the model
        
        Args:
            observation: Dictionary containing image_path, question, etc.
            
        Returns:
            Model-specific input format
        """
        pass
    
    @abstractmethod
    def _generate_response(self, model_input: Any) -> Tuple[str, Dict[str, Any]]:
        """Generate response from prepared input
        
        Args:
            model_input: Prepared input from _prepare_input
            
        Returns:
            response: Generated text response
            extra_info: Additional information (logprobs, tokens used, etc.)
        """
        pass
    
    @abstractmethod
    def _parse_response(self, raw_response: str, observation: Dict[str, Any]) -> str:
        """Parse/clean the model response
        
        Args:
            raw_response: Raw text from model
            observation: Original observation (for context)
            
        Returns:
            Parsed/cleaned response
        """
        pass
    
    def act(self, observation: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Main interface to get action from agent
        
        Args:
            observation: Dictionary containing:
                - image_path: Path to image
                - question: Question text
                - choices: Optional list of choices for multiple choice
                - other task-specific fields
                
        Returns:
            action: The agent's response/answer
            extra_info: Dictionary with additional information:
                - model: Model name used
                - temperature: Temperature used
                - tokens_used: Number of tokens
                - logprobs: Log probabilities (if enabled)
                - error: Error message (if any)
        """
        # Ensure model is loaded
        if self.model is None:
            self.load_model()
        
        # Retry logic
        for attempt in range(self.config.n_retry):
            try:
                # Prepare input
                model_input = self._prepare_input(observation)
                
                # Generate response
                raw_response, extra_info = self._generate_response(model_input)
                
                # Parse response
                action = self._parse_response(raw_response, observation)
                
                # Add metadata
                extra_info.update({
                    'model': self.config.model_name,
                    'temperature': self.config.temperature,
                    'attempt': attempt + 1,
                    'step_count': self.step_count,
                })
                
                # Update state
                self.step_count += 1
                if 'tokens_used' in extra_info:
                    self.total_tokens_used += extra_info['tokens_used']
                
                # Log successful action
                logger.debug(f"Step {self.step_count}: Generated response in attempt {attempt + 1}")
                
                return action, extra_info
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                
                if attempt < self.config.n_retry - 1:
                    time.sleep(self.config.retry_delay)
                else:
                    # Final attempt failed
                    error_msg = f"Failed after {self.config.n_retry} attempts: {str(e)}"
                    logger.error(error_msg)
                    
                    return f"Error: {str(e)}", {
                        "error": str(e),
                        "attempt": self.config.n_retry,
                        "step_count": self.step_count
                    }
    
    def reset(self):
        """Reset agent state"""
        self.conversation_history = []
        self.step_count = 0
        self.total_tokens_used = 0
        self.cost = []
        logger.info("Agent state reset")
    
    def get_state(self) -> Dict[str, Any]:
        """Get current agent state
        
        Returns:
            Dictionary containing agent state
        """
        return {
            'step_count': self.step_count,
            'total_tokens_used': self.total_tokens_used,
            'total_cost': sum(self.cost),
            'config': self.config.__dict__,
            'conversation_history_length': len(self.conversation_history)
        }
    
    def save_checkpoint(self, path: str):
        """Save agent checkpoint (optional implementation by subclasses)"""
        logger.info(f"save_checkpoint not implemented for {self.__class__.__name__}")
    
    def load_checkpoint(self, path: str):
        """Load agent checkpoint (optional implementation by subclasses)"""
        logger.info(f"load_checkpoint not implemented for {self.__class__.__name__}")

