import math
import os
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    TextBlock,
    create_sdk_mcp_server,
    tool,
)

from areal.api.reward_api import AsyncRewardWrapper
from areal.api.workflow_api import AgentWorkflow
from areal.reward import get_math_verify_worker


def gsm8k_reward_fn(result, answer):
    try:
        worker = get_math_verify_worker()
        return worker.verify(str(result), str(answer))
    except Exception:
        return 0.0


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
    result = math.sqrt(args["n"])
    return {"content": [{"type": "text", "text": str(result)}]}


# Create MCP server with calculator tools.
calculator_server = create_sdk_mcp_server(
    name="calc",
    version="1.0.0",
    tools=[add, subtract, multiply, divide, power, sqrt],
)

# List of allowed MCP tools: mcp__<server_name>__<tool_name>
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

    This agent uses ClaudeSDKClient for bidirectional, interactive conversations
    with custom tools via MCP server for mathematical operations. The agent
    communicates with Claude Code which can be configured to use a proxy via
    ANTHROPIC_BASE_URL.
    """

    def __init__(self, use_mcp_tools: bool = False, **kwargs):
        self.use_mcp_tools = use_mcp_tools
        self.kwargs = kwargs

    async def run(self, data: dict, **extra_kwargs) -> float:
        """Run the agent on a math problem.

        Args:
            data: Input data containing "messages" and "answer"
            **extra_kwargs: Contains base_url and http_client from proxy (inline mode)

        Returns:
            Reward value for this trajectory

        Notes:
            - inline mode: base_url comes from extra_kwargs
            - subproc mode: base_url comes from OPENAI_BASE_URL env var
        """

        content = data["messages"][-1]["content"]

        base_url = extra_kwargs.get("base_url", None) or os.environ.get(
            "OPENAI_BASE_URL"
        )

        # Build environment variables
        env = {}
        if base_url:
            env["ANTHROPIC_BASE_URL"] = base_url
            env["ANTHROPIC_API_KEY"] = os.environ.get("OPENAI_API_KEY", "placeholder")
            env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(
                extra_kwargs.get("max_tokens", 1024)
            )

        # Configure agent options
        if self.use_mcp_tools:
            # With MCP server for calculator tools
            options = ClaudeAgentOptions(
                system_prompt="Answer the user's math questions using the available calculator tools. Don't give the answer directly, you must use tools to do the mathematical calculation.",
                mcp_servers={"calc": calculator_server},
                allowed_tools=CALCULATOR_MCP_TOOLS,
                max_turns=self.kwargs.get("max_turns", 10),
                env=env,
            )
        else:
            # Without MCP tools - simpler setup for debugging
            options = ClaudeAgentOptions(
                system_prompt="Answer the user's math questions. Show your work step by step. Do not use tools.",
                allowed_tools=[],
                max_turns=self.kwargs.get("max_turns", 10),
                env=env,
            )

        # Use ClaudeSDKClient for bidirectional, interactive conversations
        final_output = ""
        async with ClaudeSDKClient(options=options) as client:
            await client.query(content)

            async for message in client.receive_response():
                # Extract text content from AssistantMessage
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            final_output += block.text

        reward_fn = AsyncRewardWrapper(gsm8k_reward_fn)
        reward = await reward_fn(result=final_output, answer=data["answer"])
        return reward
