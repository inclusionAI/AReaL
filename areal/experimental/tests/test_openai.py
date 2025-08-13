import os
import subprocess
import sys
import time

import pytest
import requests

from areal.api.cli_args import SGLangConfig
from areal.experimental.openai import AsyncOpenAI
from areal.utils import network
from realhf.api.core.data_api import load_hf_tokenizer

os.environ["OPENAI_API_KEY"] = "none"
os.environ["OPENAI_BASE_URL"] = "none"
EXPR_NAME = "test_openai"
TRIAL_NAME = "trial_0"
MODEL_PATH = "/storage/testing/models/Qwen__Qwen3-1.7B/"
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "Qwen/Qwen2-0.5B"
PORT, DIST_PORT = network.find_free_ports(2)
HOST = network.gethostip()
# set a large timeout since we may need to download the model from hub
RUN_SERVER_TIMEOUT = 180


def check_server_health(base_url):
    try:
        response = requests.get(f"{base_url}/health", timeout=30)
        return response.status_code == 200
    except requests.exceptions.RequestException as e:
        return False


@pytest.fixture(scope="module")
def sglang_server():
    from realhf.base import seeding

    seeding.set_random_seed(1, EXPR_NAME)
    cmd = SGLangConfig.build_cmd(
        sglang_config=SGLangConfig(
            skip_tokenizer_init=True,
            model_path=MODEL_PATH,
            mem_fraction_static=0.3,
        ),
        host=HOST,
        port=PORT,
        tp_size=1,
        base_gpu_id=0,
        dist_init_addr=f"{HOST}:{DIST_PORT}",
    )
    # Launch process
    cmd = cmd.replace("\\\n", " ").replace("\\", " ")
    process = subprocess.Popen(
        cmd.split(),
        text=True,
        stdout=sys.stdout,
        stderr=sys.stdout,
    )
    base_url = f"http://{HOST}:{PORT}"
    tik = time.time()
    while time.time() - tik < RUN_SERVER_TIMEOUT:
        if check_server_health(base_url):
            break
        time.sleep(1)
    if time.time() - tik > RUN_SERVER_TIMEOUT:
        raise RuntimeError("server launch failed")
    yield
    process.terminate()


@pytest.fixture(scope="module")
def tokenizer():
    return load_hf_tokenizer(MODEL_PATH)


@pytest.fixture(scope="module")
def openai_client(sglang_server, tokenizer):
    from areal.api.cli_args import InferenceEngineConfig
    from areal.engine.sglang_remote import RemoteSGLangEngine

    config = InferenceEngineConfig(
        experiment_name=EXPR_NAME,
        trial_name=TRIAL_NAME,
        max_concurrent_rollouts=2,
        consumer_batch_size=2,
    )
    os.environ["AREAL_LLM_SERVER_ADDRS"] = f"{HOST}:{PORT}"
    engine = RemoteSGLangEngine(config)
    engine.initialize(None, None)
    yield AsyncOpenAI(engine=engine, tokenizer=tokenizer, tool_call_parser="qwen25")
    engine.destroy()


@pytest.mark.asyncio
async def test_single_turn_rollout(openai_client):
    c = await openai_client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ]
    )
    openai_client.set_reward(c.id, reward=0.5)
    completions = openai_client.export_completions(final_reward=1.0)
    assert len(completions) == 1
    assert completions[c.id].reward == 1.5


@pytest.mark.asyncio
async def test_multi_round_conversation(openai_client):
    """Test multi-round conversation with reward backpropagation."""
    # Round 1
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ]
    c1 = await openai_client.chat.completions.create(messages=messages)

    # Round 2 - extends the conversation
    messages += [
        {"role": "assistant", "content": c1.choices[0].message.content},
        {"role": "user", "content": "What about Germany?"},
    ]
    c2 = await openai_client.chat.completions.create(messages=messages)

    # Round 3 - further extends the conversation
    messsages += [
        {"role": "assistant", "content": c2.choices[0].message.content},
        {"role": "user", "content": "And Italy?"},
    ]
    c3 = await openai_client.chat.completions.create(messages=messages)

    # Set rewards - only the final completion gets explicit reward
    openai_client.set_reward(c3.id, reward=2.0)

    # Export completions with reward backpropagation
    completions = openai_client.export_completions(final_reward=1.0, turn_discount=0.9)

    # Verify structure
    assert len(completions) == 3

    # Verify reward backpropagation: c3 is leaf (gets final_reward + set reward),
    # c2 gets discounted reward from c3, c1 gets discounted reward from c2
    assert completions[c3.id].reward == 3.0  # 2.0 + 1.0 (final_reward)
    assert completions[c2.id].reward == 0.9 * 3.0  # discounted from c3
    assert completions[c1.id].reward == 0.9 * (0.9 * 3.0)  # discounted from c2


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a location",
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
                        "description": "Temperature unit",
                    },
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform basic arithmetic calculations",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate, e.g. '2 + 2'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_fact",
            "description": "Get an interesting fact about a number",
            "parameters": {
                "type": "object",
                "properties": {
                    "number": {
                        "type": "integer",
                        "description": "The number to get a fact about",
                    }
                },
                "required": ["number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get current time in a timezone",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "Timezone, e.g. 'America/New_York'",
                    },
                },
                "required": ["timezone"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "translate",
            "description": "Translate text to another language",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to translate"},
                    "target_language": {
                        "type": "string",
                        "description": "Target language code",
                    },
                },
                "required": ["text", "target_language"],
            },
        },
    },
]


@pytest.mark.asyncio
async def test_single_round_tool_calling(openai_client):
    """Test single-round conversation with tool calling."""

    c = await openai_client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant with access to weather information.",
            },
            {"role": "user", "content": "What's the weather like in San Francisco?"},
        ],
        tools=tools,
        tool_choice="auto",
    )

    # Check if tool call was made (might depend on model capability)
    assert c.id is not None
    assert c.choices[0].message.role == "assistant"
    assert c.choices[0].message.tool_calls is not None

    openai_client.set_reward(c.id, reward=1.5)
    completions = openai_client.export_completions(final_reward=0.5)

    assert len(completions) == 1
    assert completions[c.id].reward == 2.0  # 1.5 + 0.5


@pytest.mark.asyncio
async def test_multi_round_tool_calling(openai_client):
    """Test multi-round conversation with tool calling across rounds."""

    # Round 1 - Initial tool call request
    c1 = await openai_client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful calculator assistant."},
            {"role": "user", "content": "Calculate 15 * 7"},
        ],
        tools=tools,
        tool_choice="auto",
    )

    # Simulate tool call response
    tool_response = "105"

    # Round 2 - Continue with tool result and new request
    c2 = await openai_client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful calculator assistant."},
            {"role": "user", "content": "Calculate 15 * 7"},
            {"role": "assistant", "content": c1.choices[0].message.content},
            {"role": "tool", "content": tool_response, "tool_call_id": "mock_call_id"},
            {
                "role": "user",
                "content": "Now get an interesting fact about this number",
            },
        ],
        tools=tools,
        tool_choice="auto",
    )

    # Round 3 - Final response with fact
    c3 = await openai_client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful calculator assistant."},
            {"role": "user", "content": "Calculate 15 * 7"},
            {"role": "assistant", "content": c1.choices[0].message.content},
            {"role": "tool", "content": tool_response, "tool_call_id": "mock_call_id"},
            {
                "role": "user",
                "content": "Now get an interesting fact about this number",
            },
            {"role": "assistant", "content": c2.choices[0].message.content},
            {
                "role": "tool",
                "content": "105 is divisible by 3, 5, 7, 15, 21, and 35!",
                "tool_call_id": "mock_call_id_2",
            },
            {"role": "user", "content": "Thank you!"},
        ]
    )

    # Set rewards
    openai_client.set_reward(c2.id, reward=1.0)
    openai_client.set_reward(c3.id, reward=2.0)

    completions = openai_client.export_completions(final_reward=0.5, turn_discount=0.8)

    assert len(completions) == 3
    # c3 is leaf: gets final_reward + explicit reward
    assert completions[c3.id].reward == 2.5  # 2.0 + 0.5
    # c2 gets explicit reward + discounted reward from c3
    assert completions[c2.id].reward == 1.0 + 0.8 * 2.5
    # c1 gets discounted reward from c2
    assert completions[c1.id].reward == 0.8 * (1.0 + 0.8 * 2.5)


@pytest.mark.asyncio
async def test_parallel_tool_calling(openai_client):
    """Test parallel tool calling within a single round."""

    # Single request that could trigger multiple tool calls
    c1 = await openai_client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that can check weather, time, and translate text.",
            },
            {
                "role": "user",
                "content": "I need you to check the weather in Tokyo, get the current time in Japan, and translate 'hello world' to Japanese. Please do all of these.",
            },
        ],
        tools=tools,
        tool_choice="auto",
    )

    # Check the response structure
    assert c1.id is not None
    assert c1.choices[0].message.role == "assistant"

    # Even if parallel tool calling isn't supported by the model,
    # we can test the caching and reward system
    openai_client.set_reward(c1.id, reward=3.0)

    # Test with tool responses in follow-up
    c2 = await openai_client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that can check weather, time, and translate text.",
            },
            {
                "role": "user",
                "content": "I need you to check the weather in Tokyo, get the current time in Japan, and translate 'hello world' to Japanese. Please do all of these.",
            },
            {"role": "assistant", "content": c1.choices[0].message.content},
            {"role": "tool", "content": "Sunny, 25°C", "tool_call_id": "weather_call"},
            {"role": "tool", "content": "14:30 JST", "tool_call_id": "time_call"},
            {
                "role": "tool",
                "content": "こんにちは世界",
                "tool_call_id": "translate_call",
            },
            {"role": "user", "content": "Perfect, thank you!"},
        ]
    )

    openai_client.set_reward(c2.id, reward=2.0)

    completions = openai_client.export_completions(final_reward=1.0, turn_discount=0.9)

    assert len(completions) == 2
    # c2 is leaf: gets final_reward + explicit reward
    assert completions[c2.id].reward == 3.0  # 2.0 + 1.0
    # c1 gets explicit reward + discounted reward from c2
    assert completions[c1.id].reward == 3.0 + 0.9 * 3.0


def strip_thinking_tags(content: str) -> str:
    """Remove thinking tags and their content from a message."""
    import re

    # Remove <think>...</think> blocks (including multi-line)
    pattern = r"<think>.*?</think>"
    cleaned = re.sub(pattern, "", content, flags=re.DOTALL)
    # Clean up extra whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


@pytest.mark.asyncio
async def test_multi_round_conversation_with_thinking(openai_client):
    """Test multi-round conversation where thinking content is excluded from subsequent rounds."""

    # Round 1 - Model generates response with thinking
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. When reasoning, use <think></think> tags for your internal thoughts.",
        },
        {"role": "user", "content": "What is 15 * 24? Please show your thinking."},
    ]
    c1 = await openai_client.chat.completions.create(messages=messages)

    # Simulate response with thinking content
    c1.choices[0].message.content
    # For testing, let's assume the model generated content with thinking
    thinking_content = "Let me work this out step by step. <think>15 * 24 = 15 * 20 + 15 * 4 = 300 + 60 = 360</think> The answer is 15 * 24 = 360."

    # Update the cached completion to simulate thinking content
    completion_data = openai_client.get_completions(c1.id)
    completion_data.completion.choices[0].message.content = thinking_content

    # Round 2 - Strip thinking from previous response
    cleaned_assistant_content = strip_thinking_tags(thinking_content)
    messages += [
        {"role": "assistant", "content": cleaned_assistant_content},
        {"role": "user", "content": "Now what is 360 divided by 12?"},
    ]
    c2 = await openai_client.chat.completions.create(messages=messages)

    # Simulate another response with thinking
    thinking_content_2 = "Let me calculate this. <think>360 ÷ 12 = 30 because 12 * 30 = 360</think> The answer is 360 ÷ 12 = 30."
    completion_data_2 = openai_client.get_completions(c2.id)
    completion_data_2.completion.choices[0].message.content = thinking_content_2

    # Round 3 - Continue conversation, stripping thinking from previous response
    cleaned_assistant_content_2 = strip_thinking_tags(thinking_content_2)
    messages += [
        {"role": "assistant", "content": cleaned_assistant_content_2},
        {
            "role": "user",
            "content": "Great! Can you explain why division by 12 gave us 30?",
        },
    ]
    c3 = await openai_client.chat.completions.create(messages=messages)

    # Verify the messages don't contain thinking tags in the conversation history
    stored_messages_c2 = openai_client.get_completions(c2.id).messages
    stored_messages_c3 = openai_client.get_completions(c3.id).messages

    # Check that assistant messages in stored history don't contain thinking tags
    for msg in stored_messages_c2:
        if msg.get("role") == "assistant":
            assert "<think>" not in msg.get("content", "")
            assert "</think>" not in msg.get("content", "")

    for msg in stored_messages_c3:
        if msg.get("role") == "assistant":
            assert "<think>" not in msg.get("content", "")
            assert "</think>" not in msg.get("content", "")

    # Test reward system
    openai_client.set_reward(c2.id, reward=1.5)
    openai_client.set_reward(c3.id, reward=2.5)

    completions = openai_client.export_completions(final_reward=1.0, turn_discount=0.85)

    assert len(completions) == 3
    # c3 is leaf: gets final_reward + explicit reward
    assert completions[c3.id].reward == 3.5  # 2.5 + 1.0
    # c2 gets explicit reward + discounted reward from c3
    assert completions[c2.id].reward == 1.5 + 0.85 * 3.5
    # c1 gets discounted reward from c2
    assert completions[c1.id].reward == 0.85 * (1.5 + 0.85 * 3.5)


@pytest.mark.asyncio
async def test_multi_round_conversation_with_thinking_and_tool_calling(openai_client):
    """Test multi-round conversation with both thinking and tool calling, ensuring thinking is stripped but tool calls are preserved."""

    # Round 1 - Model thinks before making a tool call
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant with calculation abilities. Use <think></think> for internal reasoning before tool calls.",
        },
        {
            "role": "user",
            "content": "I need to calculate the area of a rectangle that is 25 meters long and 18 meters wide.",
        },
    ]
    c1 = await openai_client.chat.completions.create(
        messages=messages, tools=tools, tool_choice="auto"
    )

    # Simulate response with thinking before tool call
    thinking_content_1 = "<think>I need to calculate area = length × width = 25 × 18. I'll use the calculator tool.</think>I'll calculate the area for you using length × width."
    completion_data_1 = openai_client.get_completions(c1.id)
    completion_data_1.completion.choices[0].message.content = thinking_content_1

    # Round 2 - Provide tool result and ask follow-up, stripping thinking from previous response
    cleaned_content_1 = strip_thinking_tags(thinking_content_1)
    messages += [
        {
            "role": "assistant",
            "content": cleaned_content_1,
            "tool_calls": c1.choices[0].message.tool_calls,
        },
        {"role": "tool", "content": "450", "tool_call_id": "calc_call_1"},
        {
            "role": "user",
            "content": "Perfect! Now what if I want to carpet this room and carpet costs $15 per square meter?",
        },
    ]
    c2 = await openai_client.chat.completions.create(
        messages=messages, tools=tools, tool_choice="auto"
    )

    # Simulate another response with thinking before tool call
    thinking_content_2 = "<think>I need to multiply the area (450 sq m) by the cost per square meter ($15). So 450 × 15.</think>I'll calculate the total carpet cost by multiplying the area by the price per square meter."
    completion_data_2 = openai_client.get_completions(c2.id)
    completion_data_2.completion.choices[0].message.content = thinking_content_2

    # Round 3 - Continue with tool result
    cleaned_content_2 = strip_thinking_tags(thinking_content_2)
    messages += [
        {
            "role": "assistant",
            "content": cleaned_content_2,
            "tool_calls": c2.choices[0].message.tool_calls,
        },
        {"role": "tool", "content": "6750", "tool_call_id": "calc_call_2"},
        {
            "role": "user",
            "content": "That's quite expensive! What would be the cost per square foot instead?",
        },
    ]
    c3 = await openai_client.chat.completions.create(messages=messages)

    # Verify conversation history
    stored_messages_c2 = openai_client.get_completions(c2.id).messages
    stored_messages_c3 = openai_client.get_completions(c3.id).messages

    # Verify thinking tags are stripped from assistant messages
    for msg_list in [stored_messages_c2, stored_messages_c3]:
        for msg in msg_list:
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                assert "<think>" not in content
                assert "</think>" not in content

    # Verify tool calls are preserved (check that tool messages exist in history)
    tool_messages_found = False
    for msg in stored_messages_c3:
        if msg.get("role") == "tool":
            tool_messages_found = True
            break
    assert (
        tool_messages_found
    ), "Tool messages should be preserved in conversation history"

    # Test reward system with thinking + tool calling
    openai_client.set_reward(c1.id, reward=1.0)
    openai_client.set_reward(c2.id, reward=2.0)
    openai_client.set_reward(c3.id, reward=1.5)

    completions = openai_client.export_completions(final_reward=0.8, turn_discount=0.9)

    assert len(completions) == 3
    # c3 is leaf: gets final_reward + explicit reward
    assert completions[c3.id].reward == 2.3  # 1.5 + 0.8
    # c2 gets explicit reward + discounted reward from c3
    assert completions[c2.id].reward == 2.0 + 0.9 * 2.3
    # c1 gets explicit reward + discounted reward from c2
    assert completions[c1.id].reward == 1.0 + 0.9 * (2.0 + 0.9 * 2.3)
