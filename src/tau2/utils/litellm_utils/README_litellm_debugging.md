# LiteLLM Raw Response Debugging Tools

This collection of tools helps you understand what LiteLLM actually sends to LLM providers and what raw responses it receives, so you can see the parsing and transformation that LiteLLM does.

## Why This Matters

LiteLLM is a wrapper that provides a unified OpenAI-compatible API for many different LLM providers. However, this abstraction can make it hard to understand:

- What exact HTTP requests LiteLLM sends to the provider
- What raw response format the provider returns
- How LiteLLM transforms/parses the response
- Provider-specific handling and edge cases

These tools help you debug issues and understand the underlying communication.

## Files Overview

### 1. `litellm_raw_response_demo.py`
Basic demonstration script showing how to access raw response data from LiteLLM responses.

**Usage:**
```bash
python litellm_raw_response_demo.py
```

**What it shows:**
- How to access `_response_headers` (raw HTTP headers)
- How to access `_hidden_params` (LiteLLM internal data)
- How to get the complete response as JSON
- Debug mode logging

### 2. `litellm_request_interceptor.py`
Advanced script that intercepts and logs the actual HTTP requests and responses.

**Usage:**
```bash
python litellm_request_interceptor.py
```

**What it shows:**
- Raw HTTP requests sent to the provider
- Raw HTTP responses received from the provider
- Analysis of how LiteLLM transforms responses
- Custom logging callbacks

### 3. `litellm_raw_data_utils.py`
Utility module with helper functions to extract and analyze raw response data.

**Usage:**
```python
from litellm_raw_data_utils import debug_response, get_raw_response_headers

# In your existing code
response = completion(model="gpt-3.5-turbo", messages=[...])

# Quick debug
debug_response(response)

# Or access specific data
headers = get_raw_response_headers(response)
```

## Key Data Points You Can Access

### 1. Raw Response Headers (`_response_headers`)
```python
headers = response._response_headers
```
Contains the actual HTTP response headers from the LLM provider, including:
- Rate limiting information
- Authentication headers
- Provider-specific headers
- Error information

### 2. Hidden Parameters (`_hidden_params`)
```python
params = response._hidden_params
```
Contains LiteLLM internal data:
- `custom_llm_provider`: Which provider was used
- `region_name`: AWS region (for Bedrock/Azure)
- Other provider-specific metadata

### 3. Complete Response Structure
```python
# Get as dictionary
response_dict = response.model_dump()  # or response.to_dict()

# Get as JSON string
response_json = response.json()
```

### 4. Debug Mode Logging
```python
import litellm
litellm.set_verbose = True
```
Enables detailed logging of requests and responses.

## Common Use Cases

### 1. Debug Provider-Specific Issues
```python
from litellm_raw_data_utils import get_provider_info, get_rate_limit_info

response = completion(model="anthropic/claude-3", messages=[...])

# Check which provider was used
provider_info = get_provider_info(response)
print(f"Provider: {provider_info['custom_llm_provider']}")

# Check rate limiting
rate_limit = get_rate_limit_info(response)
if rate_limit:
    print(f"Remaining requests: {rate_limit.get('x-ratelimit-remaining-requests')}")
```

### 2. Understand Tool Calling Transformations
```python
# When using tools, see how LiteLLM handles them
response = completion(
    model="ollama/llama2",
    messages=[{"role": "user", "content": "Get weather for NYC"}],
    tools=[weather_tool],
    tool_choice="auto"
)

# Check if tools were transformed
if response.choices[0].message.tool_calls:
    print("Tools were successfully parsed")
    print(f"Tool calls: {response.choices[0].message.tool_calls}")
```

### 3. Debug Authentication Issues
```python
headers = get_raw_response_headers(response)
if headers and 'www-authenticate' in headers:
    print(f"Authentication error: {headers['www-authenticate']}")
```

### 4. Analyze Response Transformations
```python
from litellm_raw_data_utils import analyze_response_transformation

# If you have access to the original response
original_response = {...}  # Raw response from provider
analysis = analyze_response_transformation(original_response, litellm_response)
print(f"Transformations: {analysis['transformations']}")
```

## Integration with Your Existing Code

### Option 1: Quick Debug Function
```python
from litellm_raw_data_utils import debug_response

# Add this to your existing code
response = completion(model="gpt-4", messages=[...])
debug_response(response, save_to_file=True)  # Saves debug info to JSON file
```

### Option 2: Custom Logging Callback
```python
def my_logger_fn(model, messages, optional_params, litellm_params, result, start_time, end_time):
    print(f"Model: {model}")
    print(f"Duration: {end_time - start_time:.2f}s")
    
    # Access raw data
    if hasattr(result, '_response_headers'):
        print(f"Headers: {result._response_headers}")

response = completion(
    model="gpt-4",
    messages=[...],
    logger_fn=my_logger_fn
)
```

### Option 3: HTTP Interceptor
```python
from litellm_request_interceptor import LiteLLMInterceptor, setup_httpx_interceptor

# Set up interceptor
interceptor = LiteLLMInterceptor()
setup_httpx_interceptor(interceptor)

# Make your requests
response = completion(model="gpt-4", messages=[...])

# Analyze the captured data
logs = interceptor.get_logs()
print(f"Captured {len(logs['requests'])} requests and {len(logs['responses'])} responses")
```

## Provider-Specific Insights

### Ollama
- Uses `ollama` prefix for basic completion
- Uses `ollama_chat` prefix for tool calling
- May inject tools as system messages for some models

### Anthropic
- Handles Claude's thinking/reasoning features
- Transforms tool calls to function calls
- Manages different response formats

### Azure OpenAI
- Handles deployment IDs and API versions
- Manages Azure AD authentication
- Transforms Azure-specific response formats

### Bedrock
- Handles AWS region configuration
- Manages different model formats (Titan, Claude, etc.)
- Transforms AWS-specific response structures

## Troubleshooting Common Issues

### 1. Tool Calling Not Working
```python
# Check if tools were properly transformed
response = completion(model="ollama/llama2", messages=[...], tools=[...])
print(f"Tool calls: {response.choices[0].message.tool_calls}")

# Check provider-specific handling
provider_info = get_provider_info(response)
print(f"Provider: {provider_info['custom_llm_provider']}")
```

### 2. Rate Limiting Issues
```python
rate_limit = get_rate_limit_info(response)
if rate_limit:
    print(f"Rate limit info: {rate_limit}")
    print(f"Remaining: {rate_limit.get('x-ratelimit-remaining-requests')}")
```

### 3. Authentication Problems
```python
headers = get_raw_response_headers(response)
if headers:
    print(f"Response status headers: {headers}")
```

### 4. Model Not Found
```python
# Check what model LiteLLM actually sent
provider_info = get_provider_info(response)
print(f"Requested model: {provider_info['model']}")
```

## Best Practices

1. **Enable Debug Mode**: Use `litellm.set_verbose = True` during development
2. **Save Debug Info**: Use `debug_response(response, save_to_file=True)` for persistent debugging
3. **Check Provider Info**: Always verify which provider LiteLLM is using
4. **Monitor Rate Limits**: Check rate limit headers for quota issues
5. **Validate Transformations**: Ensure tool calls and other features are properly transformed

## Example Workflow

```python
import litellm
from litellm_raw_data_utils import debug_response, get_provider_info

# Enable debug mode
litellm.set_verbose = True

# Make your request
response = completion(
    model="ollama/llama2",
    messages=[{"role": "user", "content": "What is 2+2?"}],
    api_base="http://localhost:11434/v1"
)

# Quick analysis
debug_response(response)

# Check provider
provider_info = get_provider_info(response)
print(f"Used provider: {provider_info['custom_llm_provider']}")

# Save detailed debug info
debug_response(response, save_to_file=True, filename="debug_output.json")
```

This workflow helps you understand exactly what's happening under the hood with LiteLLM and troubleshoot any issues effectively. 