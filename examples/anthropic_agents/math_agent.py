"""Anthropic-based math agent implementation.

This module provides math agents using:
1. MathAgent - Simple proxy-based agent using AgentWorkflow pattern
2. MathToolAgent - claude_agent_sdk based agent with calculator tools via MCP server
"""

from typing import Any

import anthropic
from claude_agent_sdk import ClaudeAgentOptions, create_sdk_mcp_server, query, tool

from areal.api.reward_api import AsyncRewardWrapper
from areal.api.workflow_api import AgentWorkflow
from areal.reward import get_math_verify_worker


def gsm8k_reward_fn(result, answer):
    try:
        worker = get_math_verify_worker()
        return worker.verify(str(result), str(answer))
    except Exception:
        return 0.0


class MathAgent(AgentWorkflow):
    """Simple single-turn math agent using Anthropic Messages API."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    async def run(self, data: dict, **extra_kwargs) -> float:
        """Run the agent on a single problem.

        Args:
            data: Input data containing "messages" and "answer"
            **extra_kwargs: Contains base_url and http_client from proxy

        Returns:
            Reward value for this trajectory
        """
        http_client = extra_kwargs.get("http_client", None)
        base_url = extra_kwargs.get("base_url", None)

        client = anthropic.AsyncAnthropic(
            api_key="placeholder",
            base_url=base_url,
            http_client=http_client,
            max_retries=0,
        )

        # Convert OpenAI-style messages to Anthropic format
        messages = data["messages"]
        system_prompt = None
        anthropic_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                anthropic_messages.append(
                    {
                        "role": msg["role"],
                        "content": msg["content"],
                    }
                )

        # Call the Anthropic Messages API (via proxy)
        response = await client.messages.create(
            model="default",
            messages=anthropic_messages,
            system=system_prompt if system_prompt else anthropic.NOT_GIVEN,
            **self.kwargs,
        )

        # Extract response text
        completion_text = ""
        for block in response.content:
            if hasattr(block, "text"):
                completion_text += block.text

        # Calculate reward
        reward_fn = AsyncRewardWrapper(gsm8k_reward_fn)
        return await reward_fn(
            result=completion_text,
            answer=data["answer"],
        )


# ============================================================================
# claude_agent_sdk based implementation with calculator tools via MCP server
# ============================================================================


@tool("add", "Add two numbers", {"a": float, "b": float})
async def add(args: dict[str, Any]) -> dict[str, Any]:
    result = args["a"] + args["b"]
    return {"content": [{"type": "text", "text": str(result)}]}


@tool("subtract", "Subtract two numbers", {"a": float, "b": float})
async def subtract(args: dict[str, Any]) -> dict[str, Any]:
    result = args["a"] - args["b"]
    return {"content": [{"type": "text", "text": str(result)}]}


@tool("multiply", "Multiply two numbers", {"a": float, "b": float})
async def multiply(args: dict[str, Any]) -> dict[str, Any]:
    result = args["a"] * args["b"]
    return {"content": [{"type": "text", "text": str(result)}]}


@tool("divide", "Divide two numbers (b must not be zero)", {"a": float, "b": float})
async def divide(args: dict[str, Any]) -> dict[str, Any]:
    if args["b"] == 0:
        return {
            "content": [{"type": "text", "text": "Error: Division by zero"}],
            "is_error": True,
        }
    result = args["a"] / args["b"]
    return {"content": [{"type": "text", "text": str(result)}]}


@tool("power", "Raise base to the exponent power", {"base": float, "exponent": float})
async def power(args: dict[str, Any]) -> dict[str, Any]:
    result = args["base"] ** args["exponent"]
    return {"content": [{"type": "text", "text": str(result)}]}


@tool("sqrt", "Calculate square root of a non-negative number", {"n": float})
async def sqrt(args: dict[str, Any]) -> dict[str, Any]:
    if args["n"] < 0:
        return {
            "content": [{"type": "text", "text": "Error: Cannot sqrt negative number"}],
            "is_error": True,
        }
    import math

    result = math.sqrt(args["n"])
    return {"content": [{"type": "text", "text": str(result)}]}


# Create MCP server with calculator tools
calculator_server = create_sdk_mcp_server(
    name="calculator",
    version="1.0.0",
    tools=[add, subtract, multiply, divide, power, sqrt],
)

# Tool names in MCP format: mcp__<server_name>__<tool_name>
CALCULATOR_MCP_TOOLS = [
    "mcp__calc__add",
    "mcp__calc__subtract",
    "mcp__calc__multiply",
    "mcp__calc__divide",
    "mcp__calc__power",
    "mcp__calc__sqrt",
]


class MathToolAgent(AgentWorkflow):
    """Math agent with calculator tools using claude_agent_sdk.

    This agent uses the claude_agent_sdk query() function with custom tools
    via MCP server for mathematical operations. The agent communicates with
    Claude Code which can be configured to use a proxy via ANTHROPIC_BASE_URL.
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    async def run(self, data: dict, **extra_kwargs):
        """Run the agent on a math problem.

        Args:
            data: Input data containing "messages" and "answer"
            **extra_kwargs: Contains base_url and http_client from proxy

        Returns:
            Reward value for this trajectory
        """
        content = data["messages"][-1]["content"]
        base_url = extra_kwargs.get("base_url", None)

        # Build environment variables
        env = {}
        if base_url:
            env["ANTHROPIC_BASE_URL"] = base_url

        # Configure agent options with MCP server
        options = ClaudeAgentOptions(
            system_prompt="Answer the user's math questions using the available calculator tools. Don't give the answer directly, you must use tools to do the mathematical calculation.",
            mcp_servers={"calc": calculator_server},
            allowed_tools=["Read"],
            max_turns=self.kwargs.get("max_turns", 10),
            env=env,
        )

        # Run query and collect final output
        final_output = ""
        async for message in query(prompt=content, options=options):
            # Extract text content from the message
            if hasattr(message, "content"):
                for block in message.content:
                    if hasattr(block, "text"):
                        final_output += block.text

        reward_fn = AsyncRewardWrapper(gsm8k_reward_fn)
        reward = await reward_fn(result=final_output, answer=data["answer"])
        return reward
