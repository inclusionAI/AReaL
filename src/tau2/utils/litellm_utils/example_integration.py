#!/usr/bin/env python3
"""
Example integration of LiteLLM debugging tools into existing code.

This shows how to add debugging capabilities to your existing LiteLLM usage
without changing your core logic.
"""

import json
import os

from litellm import completion

from litellm_raw_data_utils import (debug_response, get_provider_info,
                                    get_raw_response_headers)


def your_existing_completion_function():
    """
    This represents your existing LiteLLM completion code.
    We'll add debugging around it without changing the core logic.
    """
    # Your existing completion call
    response = completion(
        model="gpt-3.5-turbo",  # or any other model
        messages=[
            {"role": "user", "content": "What is 2+2?"}
        ],
        temperature=0.1
    )
    
    return response


def your_existing_completion_with_tools():
    """
    Example with tool calling - your existing code.
    """
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ]
    
    response = completion(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "What's the weather like in New York?"}
        ],
        tools=tools,
        tool_choice="auto",
        temperature=0.1
    )
    
    return response


def enhanced_completion_with_debugging():
    """
    Same completion function but with debugging added.
    """
    try:
        # Your existing completion call
        response = completion(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "What is 2+2?"}
            ],
            temperature=0.1
        )
        
        # Add debugging (this is the new part)
        print("\n" + "="*60)
        print("DEBUGGING INFO")
        print("="*60)
        
        # Quick debug analysis
        debug_response(response)
        
        # Check provider info
        provider_info = get_provider_info(response)
        print(f"\nProvider used: {provider_info.get('custom_llm_provider', 'unknown')}")
        
        # Check for rate limiting
        headers = get_raw_response_headers(response)
        if headers:
            rate_limit_headers = [k for k in headers.keys() if 'ratelimit' in k.lower()]
            if rate_limit_headers:
                print(f"Rate limit headers found: {rate_limit_headers}")
        
        return response
        
    except Exception as e:
        print(f"Error in completion: {e}")
        
        # Even on error, try to get debug info if we have a partial response
        if hasattr(e, 'response') and e.response:
            print("\nDebug info from error response:")
            debug_response(e.response)
        
        raise


def completion_with_custom_logging():
    """
    Example using custom logging callback.
    """
    def my_logger_fn(model, messages, optional_params, litellm_params, result, start_time, end_time):
        """Custom logging function that captures raw data."""
        print(f"\n{'='*40}")
        print("CUSTOM LOGGER CALLBACK")
        print(f"{'='*40}")
        print(f"Model: {model}")
        print(f"Duration: {end_time - start_time:.2f}s")
        print(f"Messages count: {len(messages)}")
        
        # Access raw response data
        if hasattr(result, '_response_headers'):
            print(f"Raw headers available: {bool(result._response_headers)}")
        if hasattr(result, '_hidden_params'):
            print(f"Hidden params: {result._hidden_params}")
    
    response = completion(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "What is 2+2?"}
        ],
        temperature=0.1,
        logger_fn=my_logger_fn
    )
    
    return response


def save_debug_info_for_analysis():
    """
    Save debug information to file for later analysis.
    """
    response = completion(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "What is 2+2?"}
        ],
        temperature=0.1
    )
    
    # Save comprehensive debug info
    debug_response(response, save_to_file=True, filename="my_completion_debug.json")
    
    return response


def main():
    """Run different examples."""
    print("LiteLLM Debugging Integration Examples")
    print("="*50)
    
    # Check if we have an API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set OPENAI_API_KEY environment variable to run these examples")
        print("You can also modify the examples to use other providers (Ollama, etc.)")
        return
    
    print("\n1. Basic completion with debugging:")
    try:
        enhanced_completion_with_debugging()
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n2. Completion with custom logging:")
    try:
        completion_with_custom_logging()
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n3. Save debug info to file:")
    try:
        save_debug_info_for_analysis()
        print("Debug info saved to my_completion_debug.json")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "="*50)
    print("INTEGRATION TIPS")
    print("="*50)
    print("""
To add debugging to your existing code:

1. Import the utilities:
   from litellm_raw_data_utils import debug_response, get_provider_info

2. Add after your completion call:
   debug_response(response)

3. Or save to file for later analysis:
   debug_response(response, save_to_file=True)

4. For custom logging, add logger_fn parameter:
   response = completion(..., logger_fn=my_logger_fn)

5. For HTTP interception, use the interceptor:
   from litellm_request_interceptor import LiteLLMInterceptor, setup_httpx_interceptor
   interceptor = LiteLLMInterceptor()
   setup_httpx_interceptor(interceptor)
    """)


if __name__ == "__main__":
    main() 