import litellm
from litellm import completion
import requests
import json
from openai import OpenAI

BASE_URL = "http://localhost:8000"
LITELLM_PREFIX = "vllm"


# litellm._turn_on_debug()

model = "Qwen/Qwen2.5-1.5B-Instruct"  # Not mapped by litellm yet...
model = "Qwen/Qwen2.5-3B-Instruct"

# Check if VLLM server is running
try:
    response = requests.get(f"{BASE_URL}/v1/models")
    if response.status_code == 200:
        models = response.json()["data"]
        model_names = [model["id"] for model in models]
        print(f"Available models: {model_names}")
        model = model_names[0] if model_names else "default"
    else:
        print("❌ VLLM server not responding properly")
        exit(1)
except Exception as e:
    print(f"❌ Cannot connect to VLLM server: {e}")
    print("Please start VLLM server first with: vllm serve <model_name>")
    exit(1)

print(f"Using model: {model}")

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
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The temperature unit to use",
                },
            },
            "required": ["location"],
        },
    },
}

litellm_model_name = f"{LITELLM_PREFIX}/{model}"
print(f"Model name for litellm: {litellm_model_name}")

# Test 1: Basic completion without tools
print("\n=== Test 1: Basic completion ===")
try:
    response = completion(
        model=litellm_model_name,
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        api_base=BASE_URL,
    )
    print(response.choices[0].message.content)
    print("✅ Expected: Direct response without tool calls")
except Exception as e:
    print(f"❌ Error in basic completion: {e}")


# # Check OpenAI client
# openai_api_key = "EMPTY"
# openai_api_base = "http://localhost:8000/v1"
# client = OpenAI(
#     api_key=openai_api_key,
#     base_url=openai_api_base,
# )
# chat_response = client.chat.completions.create(
#     model="Qwen/Qwen2.5-1.5B-Instruct",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "Tell me a joke."},
#     ]
# )
# print("Chat response:", chat_response)

# # Test 2: Tool call test
# print("\n=== Test 2: Tool call test ===")
# try:
#     tool_response = completion(
#         model=litellm_model_name,
#         messages=[{"role": "user", "content": "What's the weather like in New York?"}],
#         tools=[test_tool],
#         tool_choice="auto",
#         api_base=BASE_URL
#     )

#     print("Tool call response:")
#     print(f"Content: {tool_response.choices[0].message.content}")
#     print(f"Tool calls: {tool_response.choices[0].message.tool_calls}")

#     # Check if tool was called
#     if tool_response.choices[0].message.tool_calls:
#         print("✅ Expected: Model should call get_weather tool for weather query")
#     else:
#         print("❌ Unexpected: Model should have called get_weather tool for weather query")
# except Exception as e:
#     print(f"❌ Error in tool call test: {e}")

# # Test 3: Force tool usage
# print("\n=== Test 3: Forced tool usage ===")
# try:
#     forced_tool_response = completion(
#         model=litellm_model_name,
#         messages=[{"role": "user", "content": "Get the weather for London"}],
#         tools=[test_tool],
#         tool_choice={"type": "function", "function": {"name": "get_weather"}},
#         api_base=BASE_URL
#     )

#     print("Forced tool response:")
#     print(f"Content: {forced_tool_response.choices[0].message.content}")
#     print(f"Tool calls: {forced_tool_response.choices[0].message.tool_calls}")

#     # Check if tool was called
#     if forced_tool_response.choices[0].message.tool_calls:
#         tool_call = forced_tool_response.choices[0].message.tool_calls[0]
#         if tool_call.function.name == "get_weather":
#             print("✅ Expected: Model correctly called get_weather tool when forced")
#         else:
#             print(f"❌ Unexpected: Model called {tool_call.function.name} instead of get_weather")
#     else:
#         print("❌ Unexpected: Model should have called get_weather tool when forced")
# except Exception as e:
#     print(f"❌ Error in forced tool usage test: {e}")

# # Test 4: Natural tool selection (not forcing)
# print("\n=== Test 4: Natural tool selection ===")
# try:
#     natural_tool_response = completion(
#         model=litellm_model_name,
#         messages=[{"role": "user", "content": "I need to know the current temperature in Tokyo"}],
#         tools=[test_tool],
#         tool_choice="auto",
#         api_base=BASE_URL
#     )

#     print("Natural tool selection response:")
#     print(f"Content: {natural_tool_response.choices[0].message.content}")
#     print(f"Tool calls: {natural_tool_response.choices[0].message.tool_calls}")

#     # Check if tool was called
#     if natural_tool_response.choices[0].message.tool_calls:
#         tool_call = natural_tool_response.choices[0].message.tool_calls[0]
#         if tool_call.function.name == "get_weather":
#             print("✅ Expected: Model intelligently selected get_weather tool for temperature query")
#         else:
#             print(f"❌ Unexpected: Model called {tool_call.function.name} instead of get_weather")
#     else:
#         print("❌ Unexpected: Model should have called get_weather tool for temperature query")
# except Exception as e:
#     print(f"❌ Error in natural tool selection test: {e}")

# # Test 5: Context where tool is not needed
# print("\n=== Test 5: Context where tool is not needed ===")
# try:
#     no_tool_response = completion(
#         model=litellm_model_name,
#         messages=[{"role": "user", "content": "What is the capital of France?"}],
#         tools=[test_tool],
#         tool_choice="auto",
#         api_base=BASE_URL
#     )

#     print("No tool needed response:")
#     print(f"Content: {no_tool_response.choices[0].message.content}")
#     print(f"Tool calls: {no_tool_response.choices[0].message.tool_calls}")

#     # Check if tool was NOT called
#     if not no_tool_response.choices[0].message.tool_calls:
#         print("✅ Expected: Model correctly avoided using weather tool for geography question")
#     else:
#         print("❌ Unexpected: Model should not have called weather tool for geography question")
# except Exception as e:
#     print(f"❌ Error in no tool needed test: {e}")


print("\n=== Test Summary ===")
print("VLLM integration tests completed!")
