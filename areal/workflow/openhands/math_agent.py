"""OpenHands SDK-based math agent implementation for AReaL training.

This module provides math agents using the OpenHands Software Agent SDK that
integrate with AReaL's proxy-based training infrastructure via the AgentWorkflow pattern.

The OpenHands SDK provides a powerful agent framework with built-in tools for
terminal commands, file editing, and task tracking. This implementation adapts
the SDK for use with AReaL's reinforcement learning training pipeline.

For more information about the OpenHands SDK, see:
- Documentation: https://docs.openhands.dev/sdk
- GitHub: https://github.com/OpenHands/software-agent-sdk
"""

import asyncio
import math
import tempfile
from collections.abc import Sequence
from typing import Any

from math_verify import parse, verify
from pydantic import Field, SecretStr

from areal.api.reward_api import AsyncRewardWrapper
from areal.api.workflow_api import AgentWorkflow


def math_reward_fn(completions: str, answer: str) -> float:
    """Calculate reward based on math answer correctness.

    Args:
        completions: The model's completion/response text
        answer: The ground truth answer

    Returns:
        1.0 if the answer is correct, 0.0 otherwise
    """
    ans = parse(completions)
    gold = parse(answer)
    return float(verify(ans, gold))


# Lazy import for OpenHands SDK to avoid import errors when SDK is not installed
def _get_openhands_sdk():
    """Lazily import OpenHands SDK components."""
    try:
        from openhands.sdk import (
            LLM,
            Action,
            Agent,
            Conversation,
            Observation,
            TextContent,
            Tool,
            ToolDefinition,
            get_logger,
        )
        from openhands.sdk.tool import ToolExecutor, register_tool
        from openhands.tools.task_tracker import TaskTrackerTool

        return {
            "LLM": LLM,
            "Agent": Agent,
            "Conversation": Conversation,
            "Tool": Tool,
            "ToolDefinition": ToolDefinition,
            "Action": Action,
            "Observation": Observation,
            "TextContent": TextContent,
            "ToolExecutor": ToolExecutor,
            "register_tool": register_tool,
            "TaskTrackerTool": TaskTrackerTool,
            "get_logger": get_logger,
        }
    except ImportError as e:
        raise ImportError(
            "OpenHands SDK is not installed. Please install it with:\n"
            "  pip install openhands-sdk openhands-tools\n"
            "Or from the local path:\n"
            "  pip install -e ./software-agent-sdk/openhands-sdk\n"
            "  pip install -e ./software-agent-sdk/openhands-tools"
        ) from e


def _create_calculator_tools():
    """Create calculator tools for the OpenHands agent.

    Returns a list of tool definitions and a registration function for the tools.
    """
    sdk = _get_openhands_sdk()
    Action = sdk["Action"]
    Observation = sdk["Observation"]
    TextContent = sdk["TextContent"]
    ToolDefinition = sdk["ToolDefinition"]
    ToolExecutor = sdk["ToolExecutor"]
    register_tool = sdk["register_tool"]

    # Define calculator actions
    class AddAction(Action):
        a: float = Field(description="First number to add")
        b: float = Field(description="Second number to add")

    class SubtractAction(Action):
        a: float = Field(description="Number to subtract from")
        b: float = Field(description="Number to subtract")

    class MultiplyAction(Action):
        a: float = Field(description="First number to multiply")
        b: float = Field(description="Second number to multiply")

    class DivideAction(Action):
        a: float = Field(description="Dividend (number to be divided)")
        b: float = Field(description="Divisor (number to divide by)")

    class PowerAction(Action):
        base: float = Field(description="Base number")
        exponent: float = Field(description="Exponent")

    class SqrtAction(Action):
        n: float = Field(description="Number to take square root of")

    # Define observation class for calculator results
    class CalculatorObservation(Observation):
        result: float = Field(description="The calculation result")

        @property
        def to_llm_content(self) -> Sequence[TextContent]:
            return [TextContent(text=f"Result: {self.result}")]

    # Define executors
    class AddExecutor(ToolExecutor):
        def __call__(
            self, action: AddAction, conversation=None
        ) -> CalculatorObservation:
            return CalculatorObservation(result=action.a + action.b)

    class SubtractExecutor(ToolExecutor):
        def __call__(
            self, action: SubtractAction, conversation=None
        ) -> CalculatorObservation:
            return CalculatorObservation(result=action.a - action.b)

    class MultiplyExecutor(ToolExecutor):
        def __call__(
            self, action: MultiplyAction, conversation=None
        ) -> CalculatorObservation:
            return CalculatorObservation(result=action.a * action.b)

    class DivideExecutor(ToolExecutor):
        def __call__(
            self, action: DivideAction, conversation=None
        ) -> CalculatorObservation:
            if action.b == 0:
                raise ValueError("Division by zero is not allowed.")
            return CalculatorObservation(result=action.a / action.b)

    class PowerExecutor(ToolExecutor):
        def __call__(
            self, action: PowerAction, conversation=None
        ) -> CalculatorObservation:
            return CalculatorObservation(result=action.base**action.exponent)

    class SqrtExecutor(ToolExecutor):
        def __call__(
            self, action: SqrtAction, conversation=None
        ) -> CalculatorObservation:
            if action.n < 0:
                raise ValueError("Cannot compute square root of a negative number.")
            return CalculatorObservation(result=math.sqrt(action.n))

    # Create tool definitions
    class AddTool(ToolDefinition):
        @classmethod
        def create(cls, conv_state) -> Sequence[ToolDefinition]:
            return [
                cls(
                    description="Add two numbers together.",
                    action_type=AddAction,
                    observation_type=CalculatorObservation,
                    executor=AddExecutor(),
                )
            ]

    class SubtractTool(ToolDefinition):
        @classmethod
        def create(cls, conv_state) -> Sequence[ToolDefinition]:
            return [
                cls(
                    description="Subtract the second number from the first.",
                    action_type=SubtractAction,
                    observation_type=CalculatorObservation,
                    executor=SubtractExecutor(),
                )
            ]

    class MultiplyTool(ToolDefinition):
        @classmethod
        def create(cls, conv_state) -> Sequence[ToolDefinition]:
            return [
                cls(
                    description="Multiply two numbers together.",
                    action_type=MultiplyAction,
                    observation_type=CalculatorObservation,
                    executor=MultiplyExecutor(),
                )
            ]

    class DivideTool(ToolDefinition):
        @classmethod
        def create(cls, conv_state) -> Sequence[ToolDefinition]:
            return [
                cls(
                    description="Divide the first number by the second.",
                    action_type=DivideAction,
                    observation_type=CalculatorObservation,
                    executor=DivideExecutor(),
                )
            ]

    class PowerTool(ToolDefinition):
        @classmethod
        def create(cls, conv_state) -> Sequence[ToolDefinition]:
            return [
                cls(
                    description="Raise base to the power of exponent.",
                    action_type=PowerAction,
                    observation_type=CalculatorObservation,
                    executor=PowerExecutor(),
                )
            ]

    class SqrtTool(ToolDefinition):
        @classmethod
        def create(cls, conv_state) -> Sequence[ToolDefinition]:
            return [
                cls(
                    description="Calculate the square root of a number.",
                    action_type=SqrtAction,
                    observation_type=CalculatorObservation,
                    executor=SqrtExecutor(),
                )
            ]

    # Register all tools
    register_tool("AddTool", AddTool.create)
    register_tool("SubtractTool", SubtractTool.create)
    register_tool("MultiplyTool", MultiplyTool.create)
    register_tool("DivideTool", DivideTool.create)
    register_tool("PowerTool", PowerTool.create)
    register_tool("SqrtTool", SqrtTool.create)

    return [
        "AddTool",
        "SubtractTool",
        "MultiplyTool",
        "DivideTool",
        "PowerTool",
        "SqrtTool",
    ]


# Register calculator tools on module load
_CALCULATOR_TOOL_NAMES: list[str] | None = None


def _ensure_calculator_tools_registered() -> list[str]:
    """Ensure calculator tools are registered and return their names."""
    global _CALCULATOR_TOOL_NAMES
    if _CALCULATOR_TOOL_NAMES is None:
        _CALCULATOR_TOOL_NAMES = _create_calculator_tools()
    return _CALCULATOR_TOOL_NAMES


class MathAgent(AgentWorkflow):
    """Simple single-turn math agent using OpenHands SDK.

    This agent uses the AReaL proxy to generate responses and calculates
    rewards based on math answer correctness. It provides a minimal
    implementation that demonstrates how to integrate OpenHands SDK
    with AReaL's training infrastructure.

    The agent creates an LLM instance pointing to the AReaL proxy server,
    sends the math problem as a user message, and evaluates the response
    using math verification.

    Example:
        >>> agent = MathAgent(temperature=1.0, max_tokens=1024)
        >>> reward = await agent.run(
        ...     data={"messages": [...], "answer": "42"},
        ...     base_url="http://localhost:8000/v1",
        ...     http_client=http_client
        ... )
    """

    def __init__(self, **kwargs):
        """Initialize the MathAgent.

        Args:
            **kwargs: Configuration options including:
                - temperature: Sampling temperature (default: 1.0)
                - top_p: Top-p sampling parameter (default: 1.0)
                - max_tokens: Maximum tokens for completion (default: 1024)
                - max_completion_tokens: Alternative name for max_tokens
        """
        self.kwargs = kwargs.copy()

    async def run(self, data: dict[str, Any], **extra_kwargs) -> float:
        """Run the agent on a single math problem.

        Args:
            data: Input data containing:
                - messages: List of message dicts with "role" and "content"
                - answer: Ground truth answer string
            **extra_kwargs: Contains:
                - base_url: URL of the AReaL proxy server
                - http_client: httpx.AsyncClient for making requests

        Returns:
            Reward value (0.0 or 1.0) based on answer correctness
        """
        sdk = _get_openhands_sdk()
        LLM = sdk["LLM"]
        Agent = sdk["Agent"]
        Conversation = sdk["Conversation"]

        base_url = extra_kwargs.get("base_url", None)
        # http_client is not directly used by OpenHands SDK LLM
        # but we can configure it if needed

        # Create LLM instance pointing to the AReaL proxy
        llm = LLM(
            model="default",  # Proxy handles actual model selection
            base_url=base_url,
            api_key=SecretStr("placeholder"),  # Proxy handles authentication
            temperature=self.kwargs.get("temperature", 1.0),
            top_p=self.kwargs.get("top_p", 1.0),
            max_output_tokens=self.kwargs.get(
                "max_completion_tokens", self.kwargs.get("max_tokens", 1024)
            ),
        )

        # Create a simple agent without tools for single-turn completion
        agent = Agent(llm=llm, tools=[])

        # Extract user content from messages
        user_content = ""
        for msg in data.get("messages", []):
            if msg.get("role") == "user":
                user_content = msg.get("content", "")
                break

        # Create a temporary workspace for the conversation
        with tempfile.TemporaryDirectory() as workspace:
            conversation = Conversation(agent=agent, workspace=workspace)

            # Send the math problem
            conversation.send_message(user_content)

            # Run the conversation synchronously in a thread pool
            # to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, conversation.run)

            # Extract the response from the conversation events
            completion_text = ""
            for event in conversation.state.events:
                # Look for agent message events
                if hasattr(event, "source") and event.source == "agent":
                    if hasattr(event, "llm_message") and event.llm_message:
                        for content in event.llm_message.content:
                            if hasattr(content, "text"):
                                completion_text += content.text

        # Calculate reward
        reward_fn = AsyncRewardWrapper(math_reward_fn)
        return await reward_fn(completions=completion_text, answer=data["answer"])


class MathToolAgent(AgentWorkflow):
    """Math agent with calculator tools using OpenHands SDK.

    This agent extends the basic MathAgent with calculator tools,
    allowing the LLM to perform mathematical operations step by step.
    It demonstrates how to use OpenHands SDK's tool framework with
    AReaL's training infrastructure.

    The agent registers custom calculator tools (add, subtract, multiply,
    divide, power, sqrt) and creates a conversation with these tools
    available. The LLM can call these tools to perform calculations
    and build up to the final answer.

    Example:
        >>> agent = MathToolAgent(temperature=1.0, max_tokens=2048)
        >>> reward = await agent.run(
        ...     data={"messages": [...], "answer": "42"},
        ...     base_url="http://localhost:8000/v1",
        ...     http_client=http_client
        ... )
    """

    def __init__(self, **kwargs):
        """Initialize the MathToolAgent.

        Args:
            **kwargs: Configuration options including:
                - temperature: Sampling temperature (default: 1.0)
                - top_p: Top-p sampling parameter (default: 1.0)
                - max_tokens: Maximum tokens for completion (default: 2048)
                - max_completion_tokens: Alternative name for max_tokens
        """
        self.kwargs = kwargs.copy()

    async def run(self, data: dict[str, Any], **extra_kwargs) -> float:
        """Run the agent with tool-calling capabilities.

        Args:
            data: Input data containing:
                - messages: List of message dicts with "role" and "content"
                - answer: Ground truth answer string
            **extra_kwargs: Contains:
                - base_url: URL of the AReaL proxy server
                - http_client: httpx.AsyncClient for making requests

        Returns:
            Reward value (0.0 or 1.0) based on answer correctness
        """
        sdk = _get_openhands_sdk()
        LLM = sdk["LLM"]
        Agent = sdk["Agent"]
        Conversation = sdk["Conversation"]
        Tool = sdk["Tool"]

        base_url = extra_kwargs.get("base_url", None)

        # Ensure calculator tools are registered
        tool_names = _ensure_calculator_tools_registered()

        # Create LLM instance pointing to the AReaL proxy
        llm = LLM(
            model="default",  # Proxy handles actual model selection
            base_url=base_url,
            api_key=SecretStr("placeholder"),  # Proxy handles authentication
            temperature=self.kwargs.get("temperature", 1.0),
            top_p=self.kwargs.get("top_p", 1.0),
            max_output_tokens=self.kwargs.get(
                "max_completion_tokens", self.kwargs.get("max_tokens", 2048)
            ),
        )

        # Create tools list from registered tool names
        tools = [Tool(name=name) for name in tool_names]

        # Create agent with calculator tools and custom system prompt
        system_prompt = (
            "You are a math assistant with calculator tools. "
            "You MUST use the provided calculator tools (AddTool, SubtractTool, "
            "MultiplyTool, DivideTool, PowerTool, SqrtTool) to perform ALL "
            "mathematical calculations. Do not calculate in your head. "
            "Show your step-by-step reasoning and use tools for each calculation. "
            "After completing calculations, provide your final answer."
        )

        agent = Agent(llm=llm, tools=tools, system_prompt=system_prompt)

        # Extract user content from messages
        user_content = ""
        for msg in data.get("messages", []):
            if msg.get("role") == "user":
                user_content = msg.get("content", "")
                break

        # Create a temporary workspace for the conversation
        with tempfile.TemporaryDirectory() as workspace:
            conversation = Conversation(agent=agent, workspace=workspace)

            # Send the math problem
            conversation.send_message(user_content)

            # Run the conversation synchronously in a thread pool
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, conversation.run)

            # Extract the final response from the conversation events
            # Look for the last agent message with content
            final_answer = ""
            for event in reversed(list(conversation.state.events)):
                if hasattr(event, "source") and event.source == "agent":
                    if hasattr(event, "llm_message") and event.llm_message:
                        for content in event.llm_message.content:
                            if hasattr(content, "text") and content.text.strip():
                                final_answer = content.text
                                break
                        if final_answer:
                            break

        # Calculate reward
        reward_fn = AsyncRewardWrapper(math_reward_fn)
        return await reward_fn(completions=final_answer, answer=data["answer"])
