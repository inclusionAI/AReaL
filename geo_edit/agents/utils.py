#!/usr/bin/env python3
"""Utility functions for VLM agents"""

import os
import re
import math
from PIL import Image
from typing import List, Optional, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


def load_image_safely(image_path: str, default_size: Tuple[int, int] = (336, 336)) -> Image.Image:
    """Safely load an image with multiple fallback options
    
    Args:
        image_path: Path to the image file
        default_size: Size for placeholder image if loading fails
        
    Returns:
        PIL Image object (either loaded image or placeholder)
    """
    # List of possible path variations to try
    possible_paths = [
        image_path,  # Original path
        os.path.abspath(image_path),  # Absolute path
        os.path.join('data', image_path),  # In data directory
        os.path.join('../', image_path),  # Parent directory
        os.path.join('../../', image_path),  # Two levels up
    ]
    
    # Add paths from environment variable if set
    if 'VLM_DATA_DIR' in os.environ:
        data_dir = os.environ['VLM_DATA_DIR']
        possible_paths.extend([
            os.path.join(data_dir, image_path),
            os.path.join(data_dir, os.path.basename(image_path))
        ])
    
    # Common dataset paths
    dataset_dirs = [
        'llava_cot_images',
        'data/llava_cot_images',
        '../llava_cot_images',
        'mulberry_images',
        'data/mulberry_images'
    ]
    
    # If the path contains a known dataset directory, try variations
    for dataset_dir in dataset_dirs:
        if dataset_dir in image_path:
            # Try replacing the dataset directory part
            base_name = image_path.split(dataset_dir)[-1].lstrip('/')
            for dir_variant in dataset_dirs:
                possible_paths.append(os.path.join(dir_variant, base_name))
    
    # Try to load from each path
    for path in possible_paths:
        try:
            if path and os.path.exists(path):
                img = Image.open(path).convert('RGB')
                logger.debug(f"Successfully loaded image from: {path}")
                return img
        except Exception as e:
            logger.debug(f"Failed to load from {path}: {e}")
            continue
    
    # If all attempts fail, create a placeholder
    logger.warning(f"Could not load image: {image_path}, using placeholder")
    placeholder = Image.new('RGB', default_size, color='gray')
    
    # Draw a simple pattern to indicate it's a placeholder
    from PIL import ImageDraw
    draw = ImageDraw.Draw(placeholder)
    draw.text((10, 10), "IMAGE NOT FOUND", fill='white')
    draw.rectangle([5, 5, default_size[0]-5, default_size[1]-5], outline='white', width=2)
    
    return placeholder


def parse_vlm_response(response: str, choices: Optional[List[str]] = None) -> str:
    """Parse VLM response to extract the answer
    
    Args:
        response: Raw response from VLM
        choices: List of multiple choice options (if applicable)
        
    Returns:
        Parsed answer - either the selected choice or cleaned response
    """
    if not response:
        return choices[0] if choices else ""
    
    response = response.strip()
    
    if choices:
        # Multiple choice question - try to extract answer
        answer_letter = extract_choice_letter(response)
        
        if answer_letter:
            # Map letter to choice
            idx = ord(answer_letter.upper()) - ord('A')
            if 0 <= idx < len(choices):
                logger.debug(f"Extracted choice {answer_letter} -> {choices[idx]}")
                return choices[idx]
        
        # Fallback: check if any choice text appears in response
        response_lower = response.lower()
        for i, choice in enumerate(choices):
            if choice.lower() in response_lower:
                logger.debug(f"Found choice text match: {choice}")
                return choice
        
        # If we can't extract an answer, default to first choice
        logger.warning(f"Could not parse choice from response: {response[:100]}...")
        return choices[0]
    
    else:
        # Open-ended question - clean up response
        return clean_response(response)


def extract_choice_letter(text: str) -> Optional[str]:
    """Extract multiple choice answer letter from text
    
    Args:
        text: Response text
        
    Returns:
        Letter (A, B, C, D, etc.) or None
    """
    # Common patterns for expressing choices
    patterns = [
        r'^([A-Z])[\.\)]\s',           # "A. " or "A) " at start
        r'^([A-Z])$',                  # Just the letter
        r'answer is ([A-Z])',          # "answer is X"
        r'correct answer is ([A-Z])',   # "correct answer is X"
        r'choose ([A-Z])',             # "choose X"
        r'select ([A-Z])',             # "select X"
        r'option ([A-Z])',             # "option X"
        r'\(([A-Z])\)',               # "(X)"
        r'^\s*([A-Z])\s*[\:\.\)]\s*', # "X:" or "X." with whitespace
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text.strip(), re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).upper()
    
    # Look for standalone letters in the text
    words = text.split()
    for word in words[:5]:  # Check first 5 words
        cleaned = word.strip('.,;:()[]{}')
        if len(cleaned) == 1 and cleaned.upper() in 'ABCDEFGHIJ':
            return cleaned.upper()
    
    return None


def clean_response(text: str, max_length: int = 200) -> str:
    """Clean up VLM response text
    
    Args:
        text: Raw response text
        max_length: Maximum length for response
        
    Returns:
        Cleaned text
    """
    # Remove common prefixes that don't add information
    prefixes_to_remove = [
        "Based on the image,",
        "Looking at the image,",
        "The image shows",
        "In the image,",
        "According to the image,",
        "From the image,",
        "I can see that",
        "The answer is:",
        "Answer:",
    ]
    
    text = text.strip()
    text_lower = text.lower()
    
    for prefix in prefixes_to_remove:
        if text_lower.startswith(prefix.lower()):
            text = text[len(prefix):].strip()
            text_lower = text.lower()
    
    # Remove excess whitespace
    text = ' '.join(text.split())
    
    # Truncate if too long
    if len(text) > max_length:
        # Try to cut at sentence boundary
        sentences = re.split(r'[.!?]+', text)
        if sentences and len(sentences[0]) <= max_length:
            text = sentences[0].strip() + '.'
        else:
            # Cut at word boundary
            words = text[:max_length].split()
            text = ' '.join(words[:-1]) + '...'
    
    return text


def format_prompt_with_choices(question: str, choices: List[str]) -> str:
    """Format a multiple choice question prompt
    
    Args:
        question: The question text
        choices: List of answer choices
        
    Returns:
        Formatted prompt
    """
    # Ensure question ends with proper punctuation
    question = question.strip()
    if not question.endswith(('?', '.', '!')):
        question += '?'
    
    # Format choices
    formatted_choices = "\n".join([
        f"{chr(65+i)}. {choice}" 
        for i, choice in enumerate(choices)
    ])
    
    # Build prompt
    prompt = f"""{question}

Choose from the following options:
{formatted_choices}

Please select the correct answer by stating the letter (A, B, C, etc.) of your choice."""
    
    return prompt


def calculate_confidence_score(logprobs: List[float]) -> float:
    """Calculate confidence score from log probabilities
    
    Args:
        logprobs: List of log probabilities for each token
        
    Returns:
        Confidence score between 0 and 1
    """
    if not logprobs:
        return 0.5  # Neutral confidence if no logprobs
    
    # Calculate average log probability
    avg_logprob = sum(logprobs) / len(logprobs)
    
    # Convert to probability space
    # More negative logprobs = lower confidence
    # Typical range: -5 (very low) to 0 (very high)
    
    # Sigmoid-like transformation to map to [0, 1]
    # Center around -2.5 (moderate confidence)
    confidence = 1 / (1 + math.exp(-(avg_logprob + 2.5)))
    
    return max(0.0, min(1.0, confidence))


def get_image_info(image: Image.Image) -> Dict[str, Any]:
    """Get information about an image
    
    Args:
        image: PIL Image object
        
    Returns:
        Dictionary with image information
    """
    return {
        'size': image.size,
        'mode': image.mode,
        'format': image.format,
        'width': image.width,
        'height': image.height,
        'aspect_ratio': image.width / image.height if image.height > 0 else 0
    }


def validate_observation(observation: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validate that observation has required fields
    
    Args:
        observation: Observation dictionary
        
    Returns:
        is_valid: Whether observation is valid
        error_msg: Error message if invalid
    """
    required_fields = ['image_path', 'question']
    
    for field in required_fields:
        if field not in observation:
            return False, f"Missing required field: {field}"
    
    if not isinstance(observation['question'], str):
        return False, "Question must be a string"
    
    if not observation['question'].strip():
        return False, "Question cannot be empty"
    
    return True, None
