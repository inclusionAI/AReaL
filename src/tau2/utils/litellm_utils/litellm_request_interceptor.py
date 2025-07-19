#!/usr/bin/env python3
"""
Minimal LiteLLM HTTP Request/Response Interceptor

Captures and displays the raw HTTP requests and responses that LiteLLM makes.
"""

import json
import httpx
import litellm
from litellm import completion


def setup_http_interceptor():
    """Set up HTTP interceptor to capture raw requests/responses from LiteLLM."""
    
    import requests
    import urllib3
    
    # Store original methods
    original_httpx_client_request = httpx.Client.request
    original_httpx_async_client_request = httpx.AsyncClient.request
    original_requests_request = requests.Session.request
    
    # Also try to intercept at the urllib3 level
    original_urllib3_request = urllib3.PoolManager.request
    
    def log_request_response(method, url, request_kwargs, response):
        """Log the raw HTTP request and response."""
        print(f"\n{'=' * 80}")
        print("RAW HTTP REQUEST")
        print(f"{'=' * 80}")
        print(f"Method: {method}")
        print(f"URL: {url}")
        print(f"Headers:")
        headers = request_kwargs.get('headers', {})
        for key, value in headers.items():
            print(f"  {key}: {value}")
        
        # Log request body
        if 'json' in request_kwargs:
            print(f"Body: {json.dumps(request_kwargs['json'], indent=2)}")
        elif 'data' in request_kwargs:
            try:
                if isinstance(request_kwargs['data'], (str, bytes)):
                    content = request_kwargs['data']
                    if isinstance(content, bytes):
                        content = content.decode('utf-8')
                    body = json.loads(content)
                    print(f"Body: {json.dumps(body, indent=2)}")
                else:
                    print(f"Body: {json.dumps(request_kwargs['data'], indent=2)}")
            except:
                print(f"Body: {request_kwargs['data']}")
        elif 'content' in request_kwargs:
            try:
                content = request_kwargs['content']
                if isinstance(content, bytes):
                    content = content.decode('utf-8')
                body = json.loads(content)
                print(f"Body: {json.dumps(body, indent=2)}")
            except:
                print(f"Body: {request_kwargs['content']}")
        else:
            print("Body: None")
        
        print(f"\n{'=' * 80}")
        print("RAW HTTP RESPONSE")
        print(f"{'=' * 80}")
        print(f"Status: {response.status_code}")
        print(f"Headers:")
        for key, value in response.headers.items():
            print(f"  {key}: {value}")
        
        try:
            if hasattr(response, 'json'):
                body = response.json()
            else:
                body = json.loads(response.text)
            print(f"Body: {json.dumps(body, indent=2)}")
        except:
            print(f"Body: {response.text}")
    
    def should_log_url(url):
        """Check if this URL should be logged."""
        url_str = str(url)
        return any(pattern in url_str for pattern in [
            '/chat', '/completions', '/api/chat', '/v1/', 
            'openai', 'anthropic', 'ollama', ':11434'
        ])
    
    # Intercept httpx requests
    def intercepted_httpx_request(self, method, url, **kwargs):
        response = original_httpx_client_request(self, method, url, **kwargs)
        if should_log_url(url):
            log_request_response(method, url, kwargs, response)
        return response
    
    async def intercepted_httpx_async_request(self, method, url, **kwargs):
        response = await original_httpx_async_client_request(self, method, url, **kwargs)
        if should_log_url(url):
            log_request_response(method, url, kwargs, response)
        return response
    
    # Intercept requests library
    def intercepted_requests_request(self, method, url, **kwargs):
        response = original_requests_request(self, method, url, **kwargs)
        if should_log_url(url):
            log_request_response(method, url, kwargs, response)
        return response
    
    # Intercept urllib3
    def intercepted_urllib3_request(self, method, url, **kwargs):
        print(f"DEBUG: urllib3 request - {method} {url}")
        response = original_urllib3_request(self, method, url, **kwargs)
        if should_log_url(url):
            # Convert urllib3 response format
            response_dict = {
                'status_code': response.status,
                'headers': dict(response.headers),
                'text': response.data.decode('utf-8') if response.data else '',
                'json': lambda: json.loads(response.data.decode('utf-8')) if response.data else {}
            }
            log_request_response(method, url, kwargs, type('Response', (), response_dict)())
        return response
    
    # Apply monkey patches
    httpx.Client.request = intercepted_httpx_request
    httpx.AsyncClient.request = intercepted_httpx_async_request
    requests.Session.request = intercepted_requests_request
    urllib3.PoolManager.request = intercepted_urllib3_request
    
    print("🔧 HTTP interceptor setup complete - monitoring all HTTP libraries")
    
    # Also enable verbose logging in case we missed something
    import logging
    logging.basicConfig(level=logging.DEBUG)
    httpx_logger = logging.getLogger("httpx")
    httpx_logger.setLevel(logging.DEBUG)


def demo_litellm_http_capture():
    """Demonstrate capturing raw HTTP from LiteLLM calls."""
    print("LiteLLM Raw HTTP Request/Response Capture")
    print("=" * 50)
    
    # Set up the interceptor
    setup_http_interceptor()
    
    # Enable LiteLLM debug mode to see what it's doing
    import os
    os.environ["LITELLM_LOG"] = "DEBUG"
    litellm.set_verbose = True
    
    try:
        print("\n🔍 Making LiteLLM completion call...")
        print("DEBUG: LiteLLM will now make HTTP request...")
        
        response = completion(
            model="ollama_chat/qwen2.5:1.5b",
            messages=[{"role": "user", "content": "What is 2+2?"}],
            api_base="http://localhost:11434",
            temperature=0.1,
        )
        
        print(f"\n✅ LiteLLM Response: {response.choices[0].message.content}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure Ollama is running with: ollama serve")


def demo_with_tools():
    """Demonstrate capturing HTTP requests with tool calling."""
    print("\n" + "=" * 50)
    print("LiteLLM Tool Calling HTTP Capture")
    print("=" * 50)
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Perform mathematical calculations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Mathematical expression to evaluate"
                        }
                    },
                    "required": ["expression"]
                }
            }
        }
    ]
    
    try:
        print("\n🔧 Making LiteLLM call with tools...")
        response = completion(
            model="ollama_chat/qwen2.5:1.5b",
            messages=[{"role": "user", "content": "Calculate 15 * 23 + 7"}],
            tools=tools,
            tool_choice="auto",
            api_base="http://localhost:11434",
            temperature=0.1,
        )
        
        print(f"\n✅ LiteLLM Tool Response: {response.choices[0].message.content}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    demo_litellm_http_capture()
    demo_with_tools() 