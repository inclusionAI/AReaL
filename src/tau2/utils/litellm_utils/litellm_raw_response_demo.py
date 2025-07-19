#!/usr/bin/env python3
"""
Demo script to show how to access raw response data from LiteLLM.

This script demonstrates how to see what LiteLLM actually sends to the LLM provider
and what raw response it receives, so you can understand the parsing/transformation
that LiteLLM does.
"""

import json
import os
from typing import Any, Dict, Optional

import litellm
from litellm import completion


def print_raw_response_data(response: Any, title: str = "Response Data"):
    """Print the raw response data from a LiteLLM response object."""
    print(f"\n{'=' * 60}")
    print(f"{title}")
    print(f"{'=' * 60}")

    # Print the standard response structure
    print("\n1. Standard Response Structure:")
    print(f"   - ID: {response.id}")
    print(f"   - Model: {response.model}")
    print(f"   - Object: {response.object}")
    print(f"   - Created: {response.created}")

    if hasattr(response, "choices") and response.choices:
        choice = response.choices[0]
        print(f"   - Content: {choice.message.content}")
        print(f"   - Finish Reason: {choice.finish_reason}")
        if choice.message.tool_calls:
            print(f"   - Tool Calls: {len(choice.message.tool_calls)}")

    if hasattr(response, "usage") and response.usage:
        print(f"   - Usage: {response.usage}")

    # Print hidden parameters (LiteLLM internal data)
    print("\n2. Hidden Parameters (LiteLLM Internal Data):")
    if hasattr(response, "_hidden_params"):
        hidden_params = response._hidden_params
        print(
            f"   - Custom LLM Provider: {hidden_params.get('custom_llm_provider', 'Not set')}"
        )
        print(f"   - Region Name: {hidden_params.get('region_name', 'Not set')}")
        print(
            f"   - All Hidden Params: {json.dumps(hidden_params, indent=2, default=str)}"
        )
    else:
        print("   - No hidden parameters found")

    # Print response headers (raw HTTP response headers)
    print("\n3. Raw Response Headers:")
    if hasattr(response, "_response_headers") and response._response_headers:
        headers = response._response_headers
        print(f"   - All Headers: {json.dumps(headers, indent=2, default=str)}")
    else:
        print("   - No response headers found")

    # Print the complete response as JSON
    print("\n4. Complete Response as JSON:")
    try:
        response_dict = response.model_dump()
        print(json.dumps(response_dict, indent=2, default=str))
    except Exception as e:
        print(f"   - Error serializing response: {e}")
        # Fallback to dict conversion
        try:
            response_dict = response.to_dict()
            print(json.dumps(response_dict, indent=2, default=str))
        except Exception as e2:
            print(f"   - Error with to_dict(): {e2}")


def demo_openai_completion():
    """Demonstrate with OpenAI completion."""
    print("\n" + "=" * 80)
    print("DEMO: OpenAI Completion")
    print("=" * 80)

    # You'll need to set your OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set OPENAI_API_KEY environment variable")
        return

    try:
        response = completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "What is 2+2?"}],
            api_key=api_key,
            temperature=0.1,
        )

        print_raw_response_data(response, "OpenAI Completion Response")

    except Exception as e:
        print(f"Error with OpenAI completion: {e}")


def demo_ollama_completion():
    """Demonstrate with Ollama completion."""
    print("\n" + "=" * 80)
    print("DEMO: Ollama Completion")
    print("=" * 80)

    try:
        response = completion(
            model="ollama_chat/qwen2.5:1.5b",
            messages=[{"role": "user", "content": "What is 2+2?"}],
            api_base="http://localhost:11434/v1",
            temperature=0.1,
        )

        print_raw_response_data(response, "Ollama Completion Response")

    except Exception as e:
        print(f"Error with Ollama completion: {e}")
        print("Make sure Ollama is running with: ollama serve")


def demo_with_tools():
    """Demonstrate with tool calls to see how LiteLLM handles them."""
    print("\n" + "=" * 80)
    print("DEMO: Completion with Tools")
    print("=" * 80)

    # Define a simple tool
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
                            "description": "The city and state, e.g. San Francisco, CA",
                        }
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    try:
        response = completion(
            model="ollama_chat/qwen2.5:1.5b",
            messages=[
                {"role": "user", "content": "What's the weather like in New York?"}
            ],
            tools=tools,
            tool_choice="auto",
            api_base="http://localhost:11434/v1",
            temperature=0.1,
        )

        print_raw_response_data(response, "Ollama Completion with Tools Response")

    except Exception as e:
        print(f"Error with tool completion: {e}")
        print("Make sure Ollama is running with: ollama serve")


def enable_litellm_debug():
    """Enable LiteLLM debug mode to see more detailed logs."""
    print("\nEnabling LiteLLM debug mode...")
    litellm.set_verbose = True
    print("Debug mode enabled. You'll see detailed logs of requests and responses.")


def main():
    """Main function to run all demos."""
    print("LiteLLM Raw Response Data Demo")
    print("This script shows how to access raw response data from LiteLLM")

    # Enable debug mode to see more details
    enable_litellm_debug()

    # Run demos
    demo_ollama_completion()
    demo_with_tools()

    # Uncomment the line below if you have OpenAI API key set
    demo_openai_completion()

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
To access raw response data from LiteLLM:

1. Response Headers: response._response_headers
   - Contains raw HTTP response headers from the LLM provider
   - Useful for debugging rate limits, authentication issues, etc.

2. Hidden Parameters: response._hidden_params
   - Contains LiteLLM internal data like custom_llm_provider, region_name
   - Shows how LiteLLM processed the request

3. Complete Response: response.model_dump() or response.to_dict()
   - The full response object as a dictionary
   - Shows the final parsed structure that LiteLLM returns

4. Debug Mode: litellm.set_verbose = True
   - Enables detailed logging of requests and responses
   - Shows what LiteLLM sends to the provider

This helps you understand:
- What LiteLLM actually sends to the LLM provider
- What raw response it receives
- How it transforms/parses the response
- Any provider-specific handling or modifications
    """)


if __name__ == "__main__":
    main()
