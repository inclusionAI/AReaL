#!/usr/bin/env python3
"""
Demonstration of how tools are wrapped into messages by litellm_completion.

This script shows the different ways tools can be integrated into the message flow
and what the final message structure looks like when sent to the model.
"""

import json
import litellm
from litellm import completion as litellm_completion

# Enable debug mode to see what's happening
litellm._turn_on_debug()

# Configuration
BASE_URL = "http://localhost:11434"
MODEL = "qwen2.5:latest"

# Define a test tool
test_tool = {
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
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The temperature unit to use"
                }
            },
            "required": ["location"]
        }
    }
}

def print_separator(title):
    """Print a separator with title."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def print_messages_structure(messages, title="Message Structure"):
    """Print the structure of messages in a readable format."""
    print(f"\n{title}:")
    for i, msg in enumerate(messages):
        print(f"  Message {i+1}:")
        print(f"    Role: {msg['role']}")
        print(f"    Content: {msg['content']}")
        if 'tool_calls' in msg and msg['tool_calls']:
            print(f"    Tool Calls: {json.dumps(msg['tool_calls'], indent=6)}")
        print()

def demonstrate_tool_wrapping():
    """Demonstrate how tools are wrapped into messages by litellm_completion."""
    
    print_separator("TOOL WRAPPING DEMONSTRATION")
    print(f"Model: {MODEL}")
    print(f"Base URL: {BASE_URL}")
    
    # Test 1: Basic completion without tools
    print_separator("TEST 1: Basic Completion (No Tools)")
    
    messages_1 = [{"role": "user", "content": "What is the capital of France?"}]
    print("Input messages:")
    print_messages_structure(messages_1)
    
    print("Calling litellm_completion without tools...")
    try:
        response_1 = litellm_completion(
            model=f"ollama/{MODEL}",
            messages=messages_1,
            api_base=BASE_URL,
        )
        print(f"✅ Response: {response_1.choices[0].message.content}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 2: Tool call with auto tool choice
    print_separator("TEST 2: Tool Call with Auto Tool Choice")
    
    messages_2 = [{"role": "user", "content": "What's the weather like in New York?"}]
    print("Input messages:")
    print_messages_structure(messages_2)
    
    print("Calling litellm_completion with tools and tool_choice='auto'...")
    print(f"Tools provided: {json.dumps(test_tool, indent=2)}")
    
    try:
        response_2 = litellm_completion(
            model=f"ollama/{MODEL}",
            messages=messages_2,
            tools=[test_tool],
            tool_choice="auto",
            api_base=BASE_URL
        )
        
        print(f"✅ Response content: {response_2.choices[0].message.content}")
        if response_2.choices[0].message.tool_calls:
            print(f"✅ Tool calls: {json.dumps(response_2.choices[0].message.tool_calls, indent=2)}")
        else:
            print("ℹ️  No tool calls made")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 3: Forced tool usage
    print_separator("TEST 3: Forced Tool Usage")
    
    messages_3 = [{"role": "user", "content": "Get the weather for London"}]
    print("Input messages:")
    print_messages_structure(messages_3)
    
    print("Calling litellm_completion with forced tool usage...")
    print(f"Tool choice: {{'type': 'function', 'function': {{'name': 'get_weather'}}}}")
    
    try:
        response_3 = litellm_completion(
            model=f"ollama/{MODEL}",
            messages=messages_3,
            tools=[test_tool],
            tool_choice={"type": "function", "function": {"name": "get_weather"}},
            api_base=BASE_URL
        )
        
        print(f"✅ Response content: {response_3.choices[0].message.content}")
        if response_3.choices[0].message.tool_calls:
            print(f"✅ Tool calls: {json.dumps(response_3.choices[0].message.tool_calls, indent=2)}")
        else:
            print("ℹ️  No tool calls made")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 4: Tool call with tool response
    print_separator("TEST 4: Tool Call with Tool Response")
    
    # First, make a tool call
    messages_4 = [{"role": "user", "content": "What's the weather in Tokyo?"}]
    print("Step 1: Making tool call...")
    
    try:
        response_4 = litellm_completion(
            model=f"ollama/{MODEL}",
            messages=messages_4,
            tools=[test_tool],
            tool_choice="auto",
            api_base=BASE_URL
        )
        
        if response_4.choices[0].message.tool_calls:
            tool_call = response_4.choices[0].message.tool_calls[0]
            print(f"✅ Tool call made: {json.dumps(tool_call, indent=2)}")
            
            # Add the assistant's tool call to messages
            messages_4.append({
                "role": "assistant",
                "content": response_4.choices[0].message.content,
                "tool_calls": response_4.choices[0].message.tool_calls
            })
            
            # Simulate tool response
            tool_response = {
                "role": "tool",
                "content": "The weather in Tokyo is sunny with a temperature of 25°C",
                "tool_call_id": tool_call.id
            }
            messages_4.append(tool_response)
            
            print("\nStep 2: Adding tool response to conversation...")
            print("Updated message structure:")
            print_messages_structure(messages_4, "Messages with Tool Response")
            
            # Continue conversation with tool response
            print("\nStep 3: Continuing conversation with tool response...")
            response_4_continued = litellm_completion(
                model=f"ollama/{MODEL}",
                messages=messages_4,
                tools=[test_tool],
                tool_choice="auto",
                api_base=BASE_URL
            )
            
            print(f"✅ Final response: {response_4_continued.choices[0].message.content}")
            
        else:
            print("ℹ️  No tool call was made")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def demonstrate_function_call_prompt():
    """Demonstrate the function_call_prompt mechanism."""
    
    print_separator("FUNCTION CALL PROMPT MECHANISM")
    
    print("When a model doesn't support native function calling, litellm can")
    print("add function definitions to the system prompt. Here's how it works:")
    
    # Simulate the function_call_prompt logic
    functions = [test_tool]
    messages = [{"role": "user", "content": "What's the weather like?"}]
    
    function_prompt = """Produce JSON OUTPUT ONLY! Adhere to this format {"name": "function_name", "arguments":{"argument_name": "argument_value"}} The following functions are available to you:"""
    for function in functions:
        function_prompt += f"""\n{function}\n"""
    
    print(f"\nFunction prompt that would be added:")
    print(f"'{function_prompt}'")
    
    # Check if there's a system message
    function_added_to_prompt = False
    for message in messages:
        if "system" in message["role"]:
            message["content"] += f""" {function_prompt}"""
            function_added_to_prompt = True
    
    if function_added_to_prompt is False:
        messages.append({"role": "system", "content": f"""{function_prompt}"""})
    
    print(f"\nFinal message structure with function prompt:")
    print_messages_structure(messages, "Messages with Function Prompt")

def demonstrate_tool_choice_options():
    """Demonstrate different tool_choice options."""
    
    print_separator("TOOL CHOICE OPTIONS")
    
    tool_choice_options = [
        ("auto", "Let the model decide whether to call tools"),
        ("none", "Force the model to not call any tools"),
        ("required", "Force the model to call at least one tool"),
        ({"type": "function", "function": {"name": "get_weather"}}, "Force the model to call a specific tool")
    ]
    
    for choice, description in tool_choice_options:
        print(f"\n• {choice}: {description}")
        if isinstance(choice, dict):
            print(f"  Example: {json.dumps(choice, indent=2)}")

if __name__ == "__main__":
    try:
        demonstrate_tool_wrapping()
        demonstrate_function_call_prompt()
        demonstrate_tool_choice_options()
        
        print_separator("SUMMARY")
        print("""
Key points about tool wrapping in litellm_completion:

1. **Native Support**: When the model supports function calling natively, tools are passed as separate parameters
2. **Prompt Injection**: When the model doesn't support function calling, tools are injected into the system prompt
3. **Tool Choice**: Controls whether and which tools the model should call
4. **Message Flow**: Tool calls and responses are added to the message history for context
5. **Response Format**: Tool calls are returned in the response with function name and arguments

The actual wrapping happens in the litellm library's completion function, which:
- Validates tool_choice parameters
- Adds tools to optional_params
- Handles function_call_prompt for unsupported models
- Routes to appropriate provider-specific handlers
        """)
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc() 