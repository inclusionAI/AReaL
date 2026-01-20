
#!/usr/bin/env python3
"""VLM Agents module"""

from .base import BaseAgent, AgentConfig
from .api_agent import APIBasedAgent

from .utils import (
    load_image_safely,
    parse_vlm_response,
    extract_choice_letter,
    clean_response,
    format_prompt_with_choices,
    calculate_confidence_score
)

__all__ = [
    # Base classes
    'BaseAgent',
    'AgentConfig',
    
    # Concrete implementations
    "APIBasedAgent",
    
    # Utility functions
    'load_image_safely',
    'parse_vlm_response',
    'extract_choice_letter',
    'clean_response',
    'format_prompt_with_choices',
    'calculate_confidence_score',
]
