#!/usr/bin/env python3
"""
Advanced LiteLLM Request/Response Interceptor

This script demonstrates how to intercept and log the actual HTTP requests
and responses that LiteLLM makes to understand the raw communication.
"""

import json
import logging
import os
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import httpx
import litellm
from litellm import completion


class LiteLLMInterceptor:
    """Interceptor to capture and log LiteLLM HTTP requests and responses."""

    def __init__(self):
        self.requests_log = []
        self.responses_log = []

    def log_request(self, request: httpx.Request, **kwargs):
        """Log an outgoing HTTP request."""
        request_data = {
            "method": request.method,
            "url": str(request.url),
            "headers": dict(request.headers),
            "body": self._safe_decode_body(request.content),
            "timestamp": kwargs.get("timestamp"),
        }
        self.requests_log.append(request_data)

        print(f"\n{'=' * 80}")
        print("OUTGOING REQUEST")
        print(f"{'=' * 80}")
        print(f"Method: {request.method}")
        print(f"URL: {request.url}")
        print(f"Headers: {json.dumps(dict(request.headers), indent=2)}")
        print(
            f"Body: {json.dumps(request_data['body'], indent=2) if request_data['body'] else 'None'}"
        )

    def log_response(self, response: httpx.Response, **kwargs):
        """Log an incoming HTTP response."""
        try:
            response_body = response.json()
        except:
            response_body = response.text

        response_data = {
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "body": response_body,
            "timestamp": kwargs.get("timestamp"),
        }
        self.responses_log.append(response_data)

        print(f"\n{'=' * 80}")
        print("INCOMING RESPONSE")
        print(f"{'=' * 80}")
        print(f"Status: {response.status_code}")
        print(f"Headers: {json.dumps(dict(response.headers), indent=2)}")
        print(
            f"Body: {json.dumps(response_data['body'], indent=2) if isinstance(response_data['body'], dict) else response_data['body']}"
        )

    def _safe_decode_body(self, content: bytes) -> Optional[Dict]:
        """Safely decode request body."""
        if not content:
            return None
        try:
            return json.loads(content.decode("utf-8"))
        except:
            return {"raw_content": content.decode("utf-8", errors="ignore")}

    def get_logs(self) -> Dict[str, list]:
        """Get all captured logs."""
        return {"requests": self.requests_log, "responses": self.responses_log}

    def save_logs(self, filename: str = "litellm_logs.json"):
        """Save logs to a JSON file."""
        with open(filename, "w") as f:
            json.dump(self.get_logs(), f, indent=2, default=str)
        print(f"\nLogs saved to {filename}")


def setup_httpx_interceptor(interceptor: LiteLLMInterceptor):
    """Set up HTTPX interceptor to capture requests/responses."""

    # Monkey patch httpx.Client to capture requests
    original_request = httpx.Client.request

    def intercepted_request(self, method, url, **kwargs):
        # Log the request
        request = httpx.Request(method, url, **kwargs)
        interceptor.log_request(request, timestamp=kwargs.get("timestamp"))

        # Make the actual request
        response = original_request(self, method, url, **kwargs)

        # Log the response
        interceptor.log_response(response, timestamp=kwargs.get("timestamp"))

        return response

    httpx.Client.request = intercepted_request


def analyze_litellm_transformation(original_response: Dict, litellm_response: Any):
    """Analyze how LiteLLM transformed the original response."""
    print(f"\n{'=' * 80}")
    print("LITELLM TRANSFORMATION ANALYSIS")
    print(f"{'=' * 80}")

    print("\n1. Original Provider Response Structure:")
    print(json.dumps(original_response, indent=2, default=str))

    print("\n2. LiteLLM Processed Response Structure:")
    try:
        litellm_dict = litellm_response.model_dump()
        print(json.dumps(litellm_dict, indent=2, default=str))
    except:
        print("Could not serialize LiteLLM response")

    print("\n3. Key Transformations:")

    # Check for common transformations
    if "choices" in original_response and hasattr(litellm_response, "choices"):
        print("   ✓ Choices structure preserved")

        # Check message transformation
        if litellm_response.choices and litellm_response.choices[0].message:
            print("   ✓ Message object created")

            # Check tool calls transformation
            if hasattr(litellm_response.choices[0].message, "tool_calls"):
                print("   ✓ Tool calls structure added")

    # Check usage transformation
    if "usage" in original_response and hasattr(litellm_response, "usage"):
        print("   ✓ Usage statistics preserved")

    # Check model name transformation
    if "model" in original_response:
        print(
            f"   ✓ Model name: {original_response['model']} -> {litellm_response.model}"
        )


def demo_with_interceptor():
    """Demonstrate using the interceptor to capture raw requests/responses."""
    print("LiteLLM Request/Response Interceptor Demo")
    print("This will capture the actual HTTP communication")

    # Create interceptor
    interceptor = LiteLLMInterceptor()

    # Set up HTTPX interceptor
    setup_httpx_interceptor(interceptor)

    # Enable LiteLLM debug mode
    litellm.set_verbose = True

    try:
        print("\nMaking a completion request...")
        response = completion(
            model="ollama_chat/qwen2.5:1.5b",
            messages=[{"role": "user", "content": "What is 2+2?"}],
            api_base="http://localhost:11434/v1",
            temperature=0.1,
        )

        # Analyze the transformation
        if interceptor.responses_log:
            original_response = interceptor.responses_log[-1]["body"]
            analyze_litellm_transformation(original_response, response)

        # Save logs
        interceptor.save_logs()

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure Ollama is running with: ollama serve")


def demo_with_custom_logging():
    """Demonstrate custom logging callbacks."""
    print("\n" + "=" * 80)
    print("DEMO: Custom Logging Callbacks")
    print("=" * 80)

    def custom_logger_fn(
        model: str,
        messages: list,
        optional_params: dict,
        litellm_params: dict,
        result: Any,
        start_time: float,
        end_time: float,
    ):
        """Custom logging function to capture request/response details."""
        print(f"\n{'=' * 60}")
        print("CUSTOM LOGGER CALLBACK")
        print(f"{'=' * 60}")
        print(f"Model: {model}")
        print(f"Messages: {json.dumps(messages, indent=2)}")
        print(f"Optional Params: {json.dumps(optional_params, indent=2)}")
        print(f"LiteLLM Params: {json.dumps(litellm_params, indent=2)}")
        print(f"Result: {result}")
        print(f"Duration: {end_time - start_time:.2f}s")

        # Access raw response data
        if hasattr(result, "_response_headers"):
            print(f"Raw Headers: {result._response_headers}")
        if hasattr(result, "_hidden_params"):
            print(f"Hidden Params: {result._hidden_params}")

    try:
        response = completion(
            model="ollama_chat/qwen2.5:1.5b",
            messages=[{"role": "user", "content": "What is 2+2?"}],
            api_base="http://localhost:11434/v1",
            temperature=0.1,
            logger_fn=custom_logger_fn,
        )

    except Exception as e:
        print(f"Error: {e}")


def main():
    """Main function."""
    print("LiteLLM Request/Response Interceptor")
    print("This demonstrates how to capture raw HTTP communication")

    # Demo 1: HTTP interceptor
    demo_with_interceptor()

    # Demo 2: Custom logging
    demo_with_custom_logging()

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
This interceptor helps you understand:

1. Raw HTTP Requests:
   - Exact headers sent to the LLM provider
   - Request body format and content
   - Authentication and API versioning

2. Raw HTTP Responses:
   - Response status codes and headers
   - Original response format from the provider
   - Rate limiting and error information

3. LiteLLM Transformations:
   - How LiteLLM converts provider-specific formats to OpenAI format
   - What fields are added, removed, or modified
   - Provider-specific handling and edge cases

4. Debugging Benefits:
   - Identify provider-specific issues
   - Understand rate limiting and authentication problems
   - See exactly what LiteLLM sends vs. what you provide
   - Debug tool calling and function calling issues

Use this to:
- Debug provider-specific issues
- Understand LiteLLM's transformation logic
- Optimize your requests
- Troubleshoot authentication and rate limiting
    """)


if __name__ == "__main__":
    main()
