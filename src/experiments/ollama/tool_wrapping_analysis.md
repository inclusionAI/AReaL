# Tool Wrapping Analysis in LiteLLM Completion

## Overview

This document analyzes how tools are wrapped into messages by `litellm_completion` based on the demonstration results from `tool_wrapping_demo.py`.

## Key Findings

### 1. **Two Different Wrapping Mechanisms**

LiteLLM uses two different approaches to handle tools depending on model support:

#### **Native Function Calling Support**
When a model supports function calling natively (like OpenAI models), tools are passed as separate parameters in the API call.

#### **Prompt Injection for Unsupported Models**
When a model doesn't support function calling natively (like some Ollama models), LiteLLM injects the tool definitions into the system prompt.

### 2. **Evidence from the Demo**

From the debug output, we can see:

```
Final returned optional params: {
  'format': 'json', 
  'functions_unsupported_model': [{
    'type': 'function', 
    'function': {
      'name': 'get_weather', 
      'description': 'Get the current weather for a given location', 
      'parameters': {...}
    }
  }]
}
```

This shows that LiteLLM detected that the Ollama model doesn't support native function calling and moved the tools to `functions_unsupported_model`.

### 3. **Prompt Injection Process**

The actual prompt sent to Ollama shows the injection:

```
POST Request Sent from LiteLLM:
curl -X POST \
http://localhost:11434/api/generate \
-d '{
  'model': 'qwen2.5:latest', 
  'prompt': '### User:\nWhat\'s the weather like in New York?\n\n### System:\nProduce JSON OUTPUT ONLY! Adhere to this format {"name": "function_name", "arguments":{"argument_name": "argument_value"}} The following functions are available to you:\n{\'type\': \'function\', \'function\': {\'name\': \'get_weather\', \'description\': \'Get the current weather for a given location\', \'parameters\': {...}}}\n\n\n', 
  'options': {}, 
  'stream': False, 
  'format': 'json', 
  'images': []
}'
```

### 4. **Function Call Prompt Template**

The `function_call_prompt` function in LiteLLM creates this template:

```python
function_prompt = """Produce JSON OUTPUT ONLY! Adhere to this format {"name": "function_name", "arguments":{"argument_name": "argument_value"}} The following functions are available to you:"""
for function in functions:
    function_prompt += f"""\n{function}\n"""
```

### 5. **Message Structure Evolution**

#### **Initial Message (No Tools)**
```json
[
  {
    "role": "user",
    "content": "What is the capital of France?"
  }
]
```

#### **Message with Tool Injection**
```json
[
  {
    "role": "user", 
    "content": "What's the weather like in New York?"
  },
  {
    "role": "system",
    "content": "Produce JSON OUTPUT ONLY! Adhere to this format {\"name\": \"function_name\", \"arguments\":{\"argument_name\": \"argument_value\"}} The following functions are available to you:\n{...tool definition...}"
  }
]
```

## Implementation Details

### 1. **Tool Choice Handling**

LiteLLM supports several `tool_choice` options:

- `"auto"`: Let the model decide whether to call tools
- `"none"`: Force the model to not call any tools  
- `"required"`: Force the model to call at least one tool
- `{"type": "function", "function": {"name": "specific_tool"}}`: Force a specific tool

### 2. **Parameter Processing**

The completion function processes tools through these steps:

1. **Validation**: `validate_chat_completion_tool_choice(tool_choice)`
2. **Parameter Assembly**: Tools are added to `optional_params`
3. **Provider Detection**: Determines if model supports native function calling
4. **Prompt Injection**: If not supported, calls `function_call_prompt()`
5. **API Routing**: Routes to appropriate provider-specific handler

### 3. **Response Handling**

When tools are called, the response includes:

- `content`: The assistant's message content (may be null for tool calls)
- `tool_calls`: Array of tool call objects with:
  - `id`: Unique identifier for the tool call
  - `function.name`: Name of the function to call
  - `function.arguments`: JSON string of arguments

## Code Flow in LiteLLM

### 1. **Main Completion Function** (`main.py:872`)

```python
def completion(model: str, messages: List = [], tools: Optional[List] = None, tool_choice: Optional[Union[str, dict]] = None, **kwargs):
    # Validate tool_choice
    tool_choice = validate_chat_completion_tool_choice(tool_choice=tool_choice)
    
    # Add tools to optional_params
    optional_param_args = {
        "tools": tools,
        "tool_choice": tool_choice,
        # ... other params
    }
    optional_params = get_optional_params(**optional_param_args, **non_default_params)
    
    # Handle unsupported models
    if litellm.add_function_to_prompt and optional_params.get("functions_unsupported_model", None):
        functions_unsupported_model = optional_params.pop("functions_unsupported_model")
        messages = function_call_prompt(messages=messages, functions=functions_unsupported_model)
```

### 2. **Function Call Prompt** (`factory.py:3704`)

```python
def function_call_prompt(messages: list, functions: list):
    function_prompt = """Produce JSON OUTPUT ONLY! Adhere to this format {"name": "function_name", "arguments":{"argument_name": "argument_value"}} The following functions are available to you:"""
    for function in functions:
        function_prompt += f"""\n{function}\n"""

    function_added_to_prompt = False
    for message in messages:
        if "system" in message["role"]:
            message["content"] += f""" {function_prompt}"""
            function_added_to_prompt = True

    if function_added_to_prompt is False:
        messages.append({"role": "system", "content": f"""{function_prompt}"""})

    return messages
```

## Practical Implications

### 1. **Model Compatibility**
- Models with native function calling support get cleaner API calls
- Models without support get prompt injection, which may be less reliable

### 2. **Token Usage**
- Prompt injection increases token usage as tool definitions are included in every request
- Native function calling is more token-efficient

### 3. **Reliability**
- Native function calling is more reliable and follows standard formats
- Prompt injection depends on the model's ability to follow JSON formatting instructions

### 4. **Debugging**
- The debug output shows exactly how tools are being processed
- You can see whether native support or prompt injection is being used

## Conclusion

LiteLLM's tool wrapping mechanism is sophisticated and handles both native and non-native function calling models. The key insight is that when a model doesn't support function calling natively, LiteLLM automatically falls back to injecting tool definitions into the system prompt, ensuring compatibility across different model providers while maintaining a consistent API interface. 