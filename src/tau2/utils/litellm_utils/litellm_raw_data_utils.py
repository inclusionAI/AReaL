#!/usr/bin/env python3
"""
Utility functions to access raw response data from LiteLLM.

This module provides helper functions to extract and analyze raw response data
from LiteLLM completion responses, helping you understand what LiteLLM actually
sends and receives.
"""

import json
from typing import Any, Dict, Optional, Union


def get_raw_response_headers(response: Any) -> Optional[Dict]:
    """
    Get the raw HTTP response headers from a LiteLLM response.

    Args:
        response: A LiteLLM response object (ModelResponse, etc.)

    Returns:
        Dictionary of response headers, or None if not available
    """
    if hasattr(response, "_response_headers") and response._response_headers:
        return response._response_headers
    return None


def get_hidden_params(response: Any) -> Optional[Dict]:
    """
    Get the hidden parameters from a LiteLLM response.

    These contain LiteLLM internal data like custom_llm_provider, region_name, etc.

    Args:
        response: A LiteLLM response object

    Returns:
        Dictionary of hidden parameters, or None if not available
    """
    if hasattr(response, "_hidden_params") and response._hidden_params:
        return response._hidden_params
    return None


def get_complete_response_dict(response: Any) -> Optional[Dict]:
    """
    Get the complete response as a dictionary.

    Args:
        response: A LiteLLM response object

    Returns:
        Complete response as dictionary, or None if serialization fails
    """
    try:
        return response.model_dump()
    except:
        try:
            return response.to_dict()
        except:
            return None


def get_provider_info(response: Any) -> Dict[str, str]:
    """
    Get provider-specific information from the response.

    Args:
        response: A LiteLLM response object

    Returns:
        Dictionary with provider information
    """
    info = {}

    # Get custom provider
    hidden_params = get_hidden_params(response)
    if hidden_params:
        info["custom_llm_provider"] = hidden_params.get(
            "custom_llm_provider", "unknown"
        )
        info["region_name"] = hidden_params.get("region_name", "unknown")

    # Get model info
    if hasattr(response, "model"):
        info["model"] = response.model

    return info


def get_rate_limit_info(response: Any) -> Optional[Dict]:
    """
    Extract rate limiting information from response headers.

    Args:
        response: A LiteLLM response object

    Returns:
        Dictionary with rate limit info, or None if not available
    """
    headers = get_raw_response_headers(response)
    if not headers:
        return None

    rate_limit_info = {}

    # Common rate limit headers
    rate_limit_headers = [
        "x-ratelimit-limit-requests",
        "x-ratelimit-limit-tokens",
        "x-ratelimit-remaining-requests",
        "x-ratelimit-remaining-tokens",
        "x-ratelimit-reset-requests",
        "x-ratelimit-reset-tokens",
        "retry-after",
    ]

    for header in rate_limit_headers:
        if header in headers:
            rate_limit_info[header] = headers[header]

    return rate_limit_info if rate_limit_info else None


def analyze_response_transformation(
    original_response: Dict, litellm_response: Any
) -> Dict:
    """
    Analyze how LiteLLM transformed the original response.

    Args:
        original_response: The raw response from the LLM provider
        litellm_response: The processed LiteLLM response object

    Returns:
        Dictionary with transformation analysis
    """
    analysis = {
        "transformations": [],
        "preserved_fields": [],
        "added_fields": [],
        "modified_fields": [],
    }

    # Get LiteLLM response as dict
    litellm_dict = get_complete_response_dict(litellm_response)
    if not litellm_dict:
        return analysis

    # Check for common transformations
    if "choices" in original_response and "choices" in litellm_dict:
        analysis["preserved_fields"].append("choices")

        # Check message transformation
        if (
            litellm_response.choices
            and litellm_response.choices[0].message
            and "message" in litellm_response.choices[0].__dict__
        ):
            analysis["transformations"].append("message_object_created")

    # Check usage transformation
    if "usage" in original_response and "usage" in litellm_dict:
        analysis["preserved_fields"].append("usage")

    # Check model name transformation
    if "model" in original_response and hasattr(litellm_response, "model"):
        original_model = original_response["model"]
        litellm_model = litellm_response.model
        if original_model != litellm_model:
            analysis["modified_fields"].append(
                f"model: {original_model} -> {litellm_model}"
            )

    # Check for added fields
    litellm_fields = set(litellm_dict.keys())
    original_fields = set(original_response.keys())
    added_fields = litellm_fields - original_fields
    if added_fields:
        analysis["added_fields"].extend(list(added_fields))

    return analysis


def print_response_analysis(response: Any, title: str = "Response Analysis"):
    """
    Print a comprehensive analysis of a LiteLLM response.

    Args:
        response: A LiteLLM response object
        title: Title for the analysis output
    """
    print(f"\n{'=' * 60}")
    print(title)
    print(f"{'=' * 60}")

    # Basic response info
    print("\n1. Basic Response Info:")
    if hasattr(response, "id"):
        print(f"   - ID: {response.id}")
    if hasattr(response, "model"):
        print(f"   - Model: {response.model}")
    if hasattr(response, "object"):
        print(f"   - Object: {response.object}")
    if hasattr(response, "created"):
        print(f"   - Created: {response.created}")

    # Content info
    if hasattr(response, "choices") and response.choices:
        choice = response.choices[0]
        print(
            f"   - Content: {choice.message.content[:100]}{'...' if len(choice.message.content) > 100 else ''}"
        )
        print(f"   - Finish Reason: {choice.finish_reason}")
        if choice.message.tool_calls:
            print(f"   - Tool Calls: {len(choice.message.tool_calls)}")

    # Usage info
    if hasattr(response, "usage") and response.usage:
        print(f"   - Usage: {response.usage}")

    # Provider info
    print("\n2. Provider Information:")
    provider_info = get_provider_info(response)
    for key, value in provider_info.items():
        print(f"   - {key}: {value}")

    # Rate limit info
    print("\n3. Rate Limit Information:")
    rate_limit_info = get_rate_limit_info(response)
    if rate_limit_info:
        for key, value in rate_limit_info.items():
            print(f"   - {key}: {value}")
    else:
        print("   - No rate limit information available")

    # Raw headers
    print("\n4. Raw Response Headers:")
    headers = get_raw_response_headers(response)
    if headers:
        print(f"   - Headers: {json.dumps(headers, indent=2)}")
    else:
        print("   - No raw headers available")

    # Hidden params
    print("\n5. Hidden Parameters:")
    hidden_params = get_hidden_params(response)
    if hidden_params:
        print(f"   - Params: {json.dumps(hidden_params, indent=2)}")
    else:
        print("   - No hidden parameters available")


def save_response_debug_info(
    response: Any, filename: str = "litellm_response_debug.json"
):
    """
    Save comprehensive debug information about a response to a JSON file.

    Args:
        response: A LiteLLM response object
        filename: Output filename
    """
    debug_info = {
        "basic_info": {
            "id": getattr(response, "id", None),
            "model": getattr(response, "model", None),
            "object": getattr(response, "object", None),
            "created": getattr(response, "created", None),
        },
        "provider_info": get_provider_info(response),
        "rate_limit_info": get_rate_limit_info(response),
        "raw_headers": get_raw_response_headers(response),
        "hidden_params": get_hidden_params(response),
        "complete_response": get_complete_response_dict(response),
    }

    # Add content info
    if hasattr(response, "choices") and response.choices:
        choice = response.choices[0]
        debug_info["content_info"] = {
            "content": choice.message.content,
            "finish_reason": choice.finish_reason,
            "tool_calls": choice.message.tool_calls
            if choice.message.tool_calls
            else None,
        }

    # Add usage info
    if hasattr(response, "usage") and response.usage:
        debug_info["usage_info"] = {
            "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
            "completion_tokens": getattr(response.usage, "completion_tokens", None),
            "total_tokens": getattr(response.usage, "total_tokens", None),
        }

    with open(filename, "w") as f:
        json.dump(debug_info, f, indent=2, default=str)

    print(f"Debug information saved to {filename}")


# Convenience function for quick debugging
def debug_response(response: Any, save_to_file: bool = False, filename: str = None):
    """
    Quick debug function to analyze a LiteLLM response.

    Args:
        response: A LiteLLM response object
        save_to_file: Whether to save debug info to file
        filename: Optional filename for debug output
    """
    print_response_analysis(response)

    if save_to_file:
        if filename is None:
            filename = f"litellm_debug_{getattr(response, 'id', 'unknown')}.json"
        save_response_debug_info(response, filename)
