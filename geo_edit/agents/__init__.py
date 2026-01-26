
#!/usr/bin/env python3
"""VLM Agents module"""

from geo_edit.agents.base import BaseAgent, AgentConfig
from geo_edit.agents.api_agent import APIBasedAgent
from geo_edit.agents.vllm_agent import VLLMBasedAgent

from geo_edit.agents.utils import (
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
    "VLLMBasedAgent",
    
    # Utility functions
    'load_image_safely',
    'parse_vlm_response',
    'extract_choice_letter',
    'clean_response',
    'format_prompt_with_choices',
    'calculate_confidence_score',
]
