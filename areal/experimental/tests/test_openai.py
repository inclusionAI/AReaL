# This file tests the functionality of our customized OpenAI client.
# The client should be able to generate completions and correctly assign rewards with back-propagation.
import asyncio
import os
import subprocess
import sys
import time

import pytest
import requests

from areal.api.cli_args import SGLangConfig
from areal.experimental.openai import ArealOpenAI
from areal.utils import network, seeding
from areal.utils.hf_utils import load_hf_tokenizer

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


@pytest.fixture
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
    engine.initialize()
    yield ArealOpenAI(engine=engine, tokenizer=tokenizer, tool_call_parser="qwen25")
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
    completions = openai_client.export_completions()
    assert len(completions) == 1
    assert completions[c.id].reward == 0.5


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
    messages += [
        {"role": "assistant", "content": c2.choices[0].message.content},
        {"role": "user", "content": "And Italy?"},
    ]
    c3 = await openai_client.chat.completions.create(messages=messages)

    # Set rewards - only the final completion gets explicit reward
    openai_client.set_reward(c3.id, reward=2.0)

    # Export completions with reward backpropagation
    completions = openai_client.export_completions(turn_discount=0.9)

    # Verify structure
    assert len(completions) == 3

    # Verify reward backpropagation: c3 is leaf,
    # c2 gets discounted reward from c3, c1 gets discounted reward from c2
    assert completions[c3.id].reward == 2.0
    assert completions[c2.id].reward == 0.9 * 2.0  # discounted from c3
    assert completions[c1.id].reward == 0.9 * (0.9 * 2.0)  # discounted from c2


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
    assert c.choices[0].finish_reason == "tool_calls"

    openai_client.set_reward(c.id, reward=1.5)
    completions = openai_client.export_completions()

    assert len(completions) == 1
    assert completions[c.id].reward == 1.5


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

    completions = openai_client.export_completions(turn_discount=0.8)

    assert len(completions) == 3
    # c3 is leaf: gets explicit reward
    assert completions[c3.id].reward == 2.0
    # c2 gets explicit reward + discounted reward from c3
    assert completions[c2.id].reward == 1.0 + 0.8 * 2.0
    # c1 gets discounted reward from c2
    assert completions[c1.id].reward == 0.8 * (1.0 + 0.8 * 2.0)


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

    completions = openai_client.export_completions(turn_discount=0.9)

    assert len(completions) == 2
    # c2 is leaf
    assert completions[c2.id].reward == 2.0
    # c1 gets explicit reward + discounted reward from c2
    assert completions[c1.id].reward == 3.0 + 0.9 * 2.0


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
            "content": "You are a helpful assistant. Use <think></think> tags for your internal thoughts.",
        },
        {"role": "user", "content": "What is 15 * 24? Please think step-by-step."},
    ]
    c1 = await openai_client.chat.completions.create(messages=messages, max_tokens=1024)

    # Round 2 - Strip thinking from previous response
    cleaned_assistant_content = strip_thinking_tags(c1.choices[0].message.content)
    messages += [
        {"role": "assistant", "content": cleaned_assistant_content},
        {
            "role": "user",
            "content": "Now what is 360 divided by 12? Please think step-by-step.",
        },
    ]
    c2 = await openai_client.chat.completions.create(messages=messages, max_tokens=1024)

    # Round 3 - Continue conversation, stripping thinking from previous response
    cleaned_assistant_content_2 = strip_thinking_tags(c2.choices[0].message.content)
    messages += [
        {"role": "assistant", "content": cleaned_assistant_content_2},
        {
            "role": "user",
            "content": "Great! Can you explain why division by 12 gave us 30?  Please think step-by-step.",
        },
    ]
    c3 = await openai_client.chat.completions.create(messages=messages, max_tokens=1024)

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

    # Test reward system
    openai_client.set_reward(c2.id, reward=1.5)
    openai_client.set_reward(c3.id, reward=2.5)

    completions = openai_client.export_completions(turn_discount=0.85)

    assert len(completions) == 3
    # c3 is leaf
    assert completions[c3.id].reward == 2.5
    # c2 gets explicit reward + discounted reward from c3
    assert completions[c2.id].reward == 1.5 + 0.85 * 2.5
    # c1 gets discounted reward from c2
    assert completions[c1.id].reward == 0.85 * (1.5 + 0.85 * 2.5)


@pytest.mark.asyncio
async def test_multi_round_conversation_with_thinking_and_tool_calling(openai_client):
    """Test multi-round conversation with both thinking and tool calling, ensuring thinking is stripped but tool calls are preserved."""

    # Round 1 - Model thinks before making a tool call
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant with calculation abilities. Use <think></think> tags for your internal thoughts.",
        },
        {
            "role": "user",
            "content": "I need to calculate the area of a rectangle that is 25 meters long and 18 meters wide. Please think step-by-step.",
        },
    ]
    c1 = await openai_client.chat.completions.create(
        messages=messages, tools=tools, tool_choice="auto", max_tokens=1024
    )

    # Round 2 - Provide tool result and ask follow-up, stripping thinking from previous response
    cleaned_content_1 = strip_thinking_tags(c1.choices[0].message.content)
    messages += [
        {
            "role": "assistant",
            "content": cleaned_content_1,
            "tool_calls": c1.choices[0].message.tool_calls,
        },
        {"role": "tool", "content": "450", "tool_call_id": "calc_call_1"},
        {
            "role": "user",
            "content": "Perfect! Now what if I want to carpet this room and carpet costs $15 per square meter? Please think step-by-step.",
        },
    ]
    c2 = await openai_client.chat.completions.create(
        messages=messages, tools=tools, tool_choice="auto", max_tokens=1024
    )

    # Round 3 - Continue with tool result
    cleaned_content_2 = strip_thinking_tags(c2.choices[0].message.content)
    messages += [
        {
            "role": "assistant",
            "content": cleaned_content_2,
            "tool_calls": c2.choices[0].message.tool_calls,
        },
        {"role": "tool", "content": "6750", "tool_call_id": "calc_call_2"},
        {
            "role": "user",
            "content": "That's quite expensive! What would be the cost per square foot instead?  Please think step-by-step.",
        },
    ]
    c3 = await openai_client.chat.completions.create(messages=messages, max_tokens=1024)

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

    completions = openai_client.export_completions(turn_discount=0.9)

    assert len(completions) == 3
    # c3 is leaf
    assert completions[c3.id].reward == 1.5  # 1.5 + 0.8
    # c2 gets explicit reward + discounted reward from c3
    assert completions[c2.id].reward == 2.0 + 0.9 * 1.5
    # c1 gets explicit reward + discounted reward from c2
    assert completions[c1.id].reward == 1.0 + 0.9 * (2.0 + 0.9 * 1.5)


@pytest.mark.asyncio
async def test_parallel_responses_with_merged_results(openai_client):
    """Test collecting parallel LLM responses with same history but different questions, then merging them."""

    # Base conversation history
    base_messages = [
        {"role": "system", "content": "You are a helpful research assistant."},
        {
            "role": "user",
            "content": "I'm planning a trip to Japan. Can you help me with some information?",
        },
    ]

    c0 = await openai_client.chat.completions.create(messages=base_messages)

    base_messages = base_messages + [
        {
            "role": "assistant",
            "content": "I'd be happy to help you plan your trip to Japan! What specific information would you like to know?",
        }
    ]

    # Different parallel questions about the trip
    parallel_questions = [
        "What are the best months to visit for cherry blossoms?",
        "What are some must-try traditional Japanese foods?",
        "What cultural etiquette should I be aware of?",
        "What are the most popular tourist destinations?",
    ]

    # Create parallel completion tasks with the same history but different questions
    async def create_completion_with_question(question):
        messages = base_messages + [{"role": "user", "content": question}]
        return await openai_client.chat.completions.create(
            messages=messages, max_tokens=128
        )

    # Execute all completions in parallel
    parallel_tasks = [create_completion_with_question(q) for q in parallel_questions]
    parallel_completions = await asyncio.gather(*parallel_tasks)

    # Verify we got responses for all questions
    assert len(parallel_completions) == 4
    for i, completion in enumerate(parallel_completions):
        assert completion.id is not None
        assert completion.choices[0].message.role == "assistant"
        assert len(completion.choices[0].message.content) > 0
        # Set individual rewards for each parallel response
        openai_client.set_reward(completion.id, reward=1.0 + i * 0.5)

    # Now create a final completion that merges the insights from parallel responses
    merged_messages = base_messages
    for q, c in zip(parallel_questions, parallel_completions):
        merged_messages += [
            {"role": "user", "content": q},
            {"role": "assistant", "content": c.choices[0].message.content},
        ]
    merged_messages = merged_messages + [
        {
            "role": "user",
            "content": "Based on all the information about visiting Japan, please provide a comprehensive summary covering: cherry blossom timing, food recommendations, cultural etiquette, and top destinations.",
        }
    ]

    final_completion = await openai_client.chat.completions.create(
        messages=merged_messages, temperature=0.3, max_tokens=4096
    )

    # Set reward for the final merged response
    openai_client.set_reward(final_completion.id, reward=3.0)

    # Export completions to verify the reward structure
    completions = openai_client.export_completions(turn_discount=0.8)

    # Verify we have all completions (1 base + 4 parallel + 1 final)
    assert len(completions) == 6

    # Verify final completion gets its reward + final reward
    assert completions[final_completion.id].reward == 3.0  # 3.0 + 2.0

    # Verify parallel completions have their individual rewards + final reward
    expected_rewards = [1.0, 1.5, 2.0, 2.5]  # individual rewards
    r = 0.0
    for i, completion in enumerate(parallel_completions):
        assert completions[completion.id].reward == expected_rewards[i] + 3.0 * 0.8
        r += completions[completion.id].reward

    assert completions[c0.id].reward == 0.8 * r / len(
        parallel_completions
    )  # discounted from final completion


@pytest.mark.asyncio
async def test_tree_multi_round_conversation(openai_client):
    """Create a conversation tree using create() and verify parents and rewards.

    Rewards are explicitly set (no propagation). Export should return only leaves.
    """
    openai_client: ArealOpenAI
    # Base conversation
    base = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Start the session."},
    ]

    # Root
    c_root = await openai_client.chat.completions.create(messages=base)

    # Branch A1: root -> a -> a1
    msgs_a = base + [
        {"role": "assistant", "content": c_root.choices[0].message.content},
        {"role": "user", "content": "Question A"},
    ]
    c_a = await openai_client.chat.completions.create(messages=msgs_a)
    msgs_a1 = msgs_a + [
        {"role": "assistant", "content": c_a.choices[0].message.content},
        {"role": "user", "content": "Follow-up A1"},
    ]
    c_a1 = await openai_client.chat.completions.create(messages=msgs_a1)

    # Branch A2: root -> a -> a2
    msgs_a2 = msgs_a + [
        {"role": "assistant", "content": c_a.choices[0].message.content},
        {"role": "user", "content": "Follow-up A2"},
    ]
    c_a2 = await openai_client.chat.completions.create(messages=msgs_a2)

    # Branch B: root -> b -> b1
    msgs_b = base + [
        {"role": "assistant", "content": c_root.choices[0].message.content},
        {"role": "user", "content": "Question B"},
    ]
    c_b = await openai_client.chat.completions.create(messages=msgs_b)
    msgs_b1 = msgs_b + [
        {"role": "assistant", "content": c_b.choices[0].message.content},
        {"role": "user", "content": "Follow-up B1"},
    ]
    c_b1 = await openai_client.chat.completions.create(messages=msgs_b1)

    # Set rewards to leaf nodes only, which should be c_a1, c_a2, c_b1
    openai_client.set_reward(c_a1.id, 2)
    openai_client.set_reward(c_a2.id, 1.5)
    openai_client.set_reward(c_b1.id, 3)

    # Export completions of leaf nodes, check whether all leaves are present
    leaf_completions = openai_client.export_completions(
        turn_discount=0.9, reward_propagation_style="tree"
    )
    all_completions = openai_client.export_completions(
        turn_discount=0.0, reward_propagation_style="linear"
    )
    assert set(leaf_completions.keys()) == {c_a1.id, c_a2.id, c_b1.id}
    assert set(all_completions.keys()) == {
        c_root.id,
        c_a.id,
        c_a1.id,
        c_a2.id,
        c_b.id,
        c_b1.id,
    }

    def wrapped_completion(chat_completion):
        return all_completions[chat_completion.id]

    # Check tree structure
    assert wrapped_completion(c_b1).parent is wrapped_completion(c_b)
    assert wrapped_completion(c_b).parent is wrapped_completion(c_root)
    assert wrapped_completion(c_a2).parent is wrapped_completion(c_a)
    assert wrapped_completion(c_a1).parent is wrapped_completion(c_a)
    assert wrapped_completion(c_a).parent is wrapped_completion(c_root)

    # Reward is discounted by time sequence, check reward values
    assert wrapped_completion(c_b1).reward == 3
    assert wrapped_completion(c_b).reward == 3 * 0.9
    assert wrapped_completion(c_a2).reward == 1.5
    assert wrapped_completion(c_a1).reward == 2
    assert wrapped_completion(c_a).reward == 2 * 0.9
    assert wrapped_completion(c_root).reward == 2 * 0.9 * 0.9
