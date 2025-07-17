import litellm
from litellm import completion as litellm_completion
import ollama
import os
# from openai import OpenAI
import dotenv
## requires openai==1.85.0

from langfuse.openai import OpenAI

dotenv.load_dotenv()

litellm.success_callback = ["langfuse"]
litellm.failure_callback = ["langfuse"]
# litellm._turn_on_debug()

# litellm_prefix = "ollama"
litellm_prefix = "ollama_chat"


BASE_URL = "http://localhost:11434"
# model = "qwen2.5:1.5b"
# model = "llama2:latest"
model="qwen2.5:latest"

client = OpenAI(api_key="ollama", base_url='http://localhost:11434/v1')
openai_completion = client.chat.completions.create


ollama_models = ollama.list()['models']
model_names = [model['model'] for model in ollama_models]
print(f"Using model: {model}")

assert model in model_names, f"Model {model} not found in available models"
print(f"Available models: {model_names}")


def check_litellm_tool_support(model):
    """
    Check if the model supports function calling.
    What I would want to know if whether it is going to pass the tools as they are provided or whether it is going to inject them in the prompt...
    """
    res = litellm.supports_function_calling(model=model)
    print(f"Litellm tool call support for {model}: {res}")
    return res

check_litellm_tool_support(f"{litellm_prefix}/{model}")


# Define a tool for testing
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

# Test 1: Basic completion without tools
print("\n=== Test 1: Basic completion ===")
messages_1 = [{"role": "user", "content": "What is the capital of France?"}]
print(f"Prompt sent: {messages_1}")
response = litellm_completion(
    model=f"{litellm_prefix}/{model}",
    messages=messages_1,
    api_base=BASE_URL,  
)
print(response.choices[0].message.content)
if response.choices[0].message.tool_calls:
    print("❌ Unexpected: Direct response without tool calls")
else:
    print("✅ Expected: Direct response without tool calls")

response = openai_completion(
    model=model,
    messages=messages_1,
)
print("OpenAI client response:")
print(response.choices[0].message.content)

# Test 2: Tool call test
print("\n=== Test 2: Tool call test ===")
messages_2 = [{"role": "user", "content": "What's the weather like in New York?"}]
print(f"Prompt sent: {messages_2}")
tool_response = litellm_completion(
    model=f"{litellm_prefix}/{model}",
    messages=messages_2,
    tools=[test_tool],
    tool_choice="auto",
    api_base=BASE_URL
)

print("Tool call response:")
print(f"Content: {tool_response.choices[0].message.content}")
print(f"Tool calls: {tool_response.choices[0].message.tool_calls}")

# Check if tool was called
if tool_response.choices[0].message.tool_calls:
    print("✅ Expected: Model should call get_weather tool for weather query")
else:
    print("❌ Unexpected: Model should have called get_weather tool for weather query")

# Test 3: Force tool usage
print("\n=== Test 3: Forced tool usage ===")
messages_3 = [{"role": "user", "content": "Get the weather for London"}]
print(f"Prompt sent: {messages_3}")
forced_tool_response = litellm_completion(
    model=f"{litellm_prefix}/{model}",
    messages=messages_3,
    tools=[test_tool],
    tool_choice={"type": "function", "function": {"name": "get_weather"}},
    api_base=BASE_URL
)

print("Forced tool response:")
print(f"Content: {forced_tool_response.choices[0].message.content}")
print(f"Tool calls: {forced_tool_response.choices[0].message.tool_calls}")

# Check if tool was called
if forced_tool_response.choices[0].message.tool_calls:
    tool_call = forced_tool_response.choices[0].message.tool_calls[0]
    if tool_call.function.name == "get_weather":
        print("✅ Expected: Model correctly called get_weather tool when forced")
    else:
        print(f"❌ Unexpected: Model called {tool_call.function.name} instead of get_weather")
else:
    print("❌ Unexpected: Model should have called get_weather tool when forced")

# Test 4: Natural tool selection (not forcing)
print("\n=== Test 4: Natural tool selection ===")
messages_4 = [{"role": "user", "content": "I need to know the current temperature in Tokyo"}]
print(f"Prompt sent: {messages_4}")
natural_tool_response = litellm_completion(
    model=f"{litellm_prefix}/{model}",
    messages=messages_4,
    tools=[test_tool],
    tool_choice="auto",
    api_base=BASE_URL
)

print("Natural tool selection response:")
print(f"Content: {natural_tool_response.choices[0].message.content}")
print(f"Tool calls: {natural_tool_response.choices[0].message.tool_calls}")

# Check if tool was called
if natural_tool_response.choices[0].message.tool_calls:
    tool_call = natural_tool_response.choices[0].message.tool_calls[0]
    if tool_call.function.name == "get_weather":
        print("✅ Expected: Model intelligently selected get_weather tool for temperature query")
    else:
        print(f"❌ Unexpected: Model called {tool_call.function.name} instead of get_weather")
else:
    print("❌ Unexpected: Model should have called get_weather tool for temperature query")

# Test 5: Context where tool is not needed
print("\n=== Test 5: Context where tool is not needed ===")
messages_5 = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What's 2 + 2?"}]
print(f"Prompt sent: {messages_5}")
no_tool_response = litellm_completion(
    model=f"{litellm_prefix}/{model}",
    messages=messages_5,
    tools=[test_tool],
    tool_choice="auto",
    api_base=BASE_URL,
)

print("No tool needed response:")
print(no_tool_response)
print(f"Content: {no_tool_response.choices[0].message.content}")
print(f"Tool calls: {no_tool_response.choices[0].message.tool_calls}")

# Check if tool was NOT called
if not no_tool_response.choices[0].message.tool_calls:
    print("✅ Expected: Model correctly avoided using weather tool for geography question")
else:
    print("❌ Unexpected: Model should not have called weather tool for geography question")


response = openai_completion(
    model=model,
    messages=messages_5,
    tool_choice="auto",
    tools=[test_tool],
)
print("OpenAI client response:")
print(response.choices[0].message.content)