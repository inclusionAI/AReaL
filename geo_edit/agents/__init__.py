#!/usr/bin/env python3
"""VLM Agents module"""

from geo_edit.agents.api_agent import APIBasedAgent
from geo_edit.agents.base import AgentConfig, BaseAgent
from geo_edit.utils.image_utils import load_image_safely
from geo_edit.utils.text_utils import (
    calculate_confidence_score,
    clean_response,
    extract_choice_letter,
    format_prompt_with_choices,
    parse_vlm_response,
)

__all__ = [
    # Base classes
    "BaseAgent",
    "AgentConfig",
    # Concrete implementations
    "APIBasedAgent",
    # Utility functions
    "load_image_safely",
    "parse_vlm_response",
    "extract_choice_letter",
    "clean_response",
    "format_prompt_with_choices",
    "calculate_confidence_score",
]
