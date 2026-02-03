"""OpenHands SDK-based math agent implementation for AReaL training."""

import asyncio
import math
import tempfile
import threading
from collections.abc import Sequence
from typing import Any, Self

from math_verify import parse, verify
from pydantic import Field, SecretStr

from openhands.sdk import (
    LLM,
    Action,
    Agent,
    Conversation,
    Observation,
    TextContent,
    Tool,
    ToolDefinition,
    register_tool,
)
from openhands.sdk.tool import ToolExecutor

from areal.api.reward_api import AsyncRewardWrapper
from areal.api.workflow_api import AgentWorkflow
from areal.utils import logging

logger = logging.getLogger("OpenHandsMathAgent")


def math_reward_fn(completions: str, answer: str) -> float:
    """Calculate reward based on math answer correctness."""
    ans = parse(completions)
    gold = parse(answer)
    return float(verify(ans, gold))


# =============================================================================
# Calculator Tools Definition
# =============================================================================


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


class CalculatorObservation(Observation):
    result: float = Field(description="The calculation result")

    @property
    def to_llm_content(self) -> Sequence[TextContent]:
        return [TextContent(text=f"Result: {self.result}")]


class AddExecutor(ToolExecutor):
    def __call__(self, action: AddAction, conversation=None) -> CalculatorObservation:
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
    def __call__(self, action: PowerAction, conversation=None) -> CalculatorObservation:
        return CalculatorObservation(result=action.base**action.exponent)


class SqrtExecutor(ToolExecutor):
    def __call__(self, action: SqrtAction, conversation=None) -> CalculatorObservation:
        if action.n < 0:
            raise ValueError("Cannot compute square root of a negative number.")
        return CalculatorObservation(result=math.sqrt(action.n))


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


# =============================================================================
# OpenHands Agent Builder
# =============================================================================


class OpenHandsAgentBuilder:
    """Builder for creating OpenHands agents.

    This builder provides a fluent interface for configuring and building
    OpenHands SDK agents. It only handles agent construction - conversation
    execution and reward calculation are left to the caller.

    Example:
        >>> # Build an agent
        >>> agent = (
        ...     OpenHandsAgentBuilder()
        ...     .with_base_url("http://localhost:8000/v1")
        ...     .with_llm_config(temperature=1.0, max_tokens=1024)
        ...     .with_calculator_tools()
        ...     .with_system_prompt("You are a math assistant.")
        ...     .build()
        ... )
        >>>
        >>> # Run conversation yourself
        >>> with tempfile.TemporaryDirectory() as workspace:
        ...     conversation = Conversation(agent=agent, workspace=workspace)
        ...     conversation.send_message("What is 2+2?")
        ...     conversation.run()
        ...     # Process events as needed
    """

    # Calculator tool definitions
    CALCULATOR_TOOLS: dict[str, type[ToolDefinition]] = {
        "AddTool": AddTool,
        "SubtractTool": SubtractTool,
        "MultiplyTool": MultiplyTool,
        "DivideTool": DivideTool,
        "PowerTool": PowerTool,
        "SqrtTool": SqrtTool,
    }

    # Thread-safe tool registration state
    _registered_tools: set[str] = set()
    _registration_lock = threading.Lock()

    def __init__(self):
        self._base_url: str | None = None
        self._model: str = "default"
        self._temperature: float = 1.0
        self._top_p: float = 1.0
        self._max_tokens: int = 1024
        self._tool_names: list[str] = []
        self._system_prompt: str | None = None

    def with_base_url(self, base_url: str | None) -> Self:
        """Set the base URL of the AReaL proxy server."""
        self._base_url = base_url
        return self

    def with_llm_config(
        self,
        model: str = "default",
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int = 1024,
        max_completion_tokens: int | None = None,
    ) -> Self:
        """Configure LLM parameters."""
        self._model = model
        self._temperature = temperature
        self._top_p = top_p
        self._max_tokens = max_completion_tokens or max_tokens
        return self

    def with_tools(self, tool_names: list[str]) -> Self:
        """Set the tools to use for the agent."""
        self._tool_names = tool_names
        self._ensure_tools_registered(tool_names)
        return self

    def with_calculator_tools(self) -> Self:
        """Enable all calculator tools."""
        return self.with_tools(list(self.CALCULATOR_TOOLS.keys()))

    def with_system_prompt(self, prompt: str) -> Self:
        """Set the system prompt for the agent."""
        self._system_prompt = prompt
        return self

    @classmethod
    def _ensure_tools_registered(cls, tool_names: list[str]) -> None:
        """Register tools with OpenHands SDK (thread-safe, idempotent)."""
        with cls._registration_lock:
            for name in tool_names:
                if name not in cls._registered_tools and name in cls.CALCULATOR_TOOLS:
                    register_tool(name, cls.CALCULATOR_TOOLS[name].create)
                    cls._registered_tools.add(name)

    def build(self) -> Agent:
        """Build and return the configured Agent instance.

        Returns:
            Agent: The configured OpenHands Agent ready for use.
        """
        llm = LLM(
            model=self._model,
            base_url=self._base_url,
            api_key=SecretStr("placeholder"),
            temperature=self._temperature,
            top_p=self._top_p,
            max_output_tokens=self._max_tokens,
        )

        tools = [Tool(name=name) for name in self._tool_names]

        if self._system_prompt:
            return Agent(llm=llm, tools=tools, system_prompt=self._system_prompt)
        return Agent(llm=llm, tools=tools)

    @staticmethod
    def extract_user_content(messages: list[dict[str, Any]]) -> str:
        """Extract user content from a list of messages.

        Args:
            messages: List of message dicts with "role" and "content" keys

        Returns:
            The content of the first user message, or empty string if none found
        """
        for msg in messages:
            if msg.get("role") == "user":
                return msg.get("content", "")
        return ""

    @staticmethod
    def extract_all_agent_text(events) -> str:
        """Extract all text content from agent events.

        Args:
            events: Sequence of conversation events

        Returns:
            Concatenated text from all agent messages
        """
        completion_text = ""
        for event in events:
            if hasattr(event, "source") and event.source == "agent":
                if hasattr(event, "llm_message") and event.llm_message:
                    for content in event.llm_message.content:
                        if hasattr(content, "text"):
                            completion_text += content.text
        return completion_text

    @staticmethod
    def extract_final_agent_text(events) -> str:
        """Extract the final text content from agent events.

        Args:
            events: Sequence of conversation events

        Returns:
            Text from the last agent message with non-empty content
        """
        for event in reversed(list(events)):
            if hasattr(event, "source") and event.source == "agent":
                if hasattr(event, "llm_message") and event.llm_message:
                    for content in event.llm_message.content:
                        if hasattr(content, "text") and content.text.strip():
                            return content.text
        return ""


# =============================================================================
# AgentWorkflow Implementations
# =============================================================================


class MathAgent(AgentWorkflow):
    """Simple single-turn math agent using OpenHands SDK.

    This agent uses the AReaL proxy to generate responses and calculates
    rewards based on math answer correctness.

    Example:
        >>> agent = MathAgent(temperature=1.0, max_tokens=1024)
        >>> reward = await agent.run(
        ...     data={"messages": [...], "answer": "42"},
        ...     base_url="http://localhost:8000/v1",
        ... )
    """

    def __init__(self, **kwargs: Any):
        """Initialize the MathAgent.

        Args:
            **kwargs: Configuration options including:
                - model: Model name (default: "default")
                - temperature: Sampling temperature (default: 1.0)
                - top_p: Top-p sampling parameter (default: 1.0)
                - max_tokens: Maximum tokens for completion (default: 1024)
                - max_completion_tokens: Alternative name for max_tokens
        """
        self.kwargs = kwargs.copy()

    async def run(self, data: dict[str, Any], **extra_kwargs: Any) -> float:
        """Run the agent on a single math problem.

        Args:
            data: Input data containing:
                - messages: List of message dicts with "role" and "content"
                - answer: Ground truth answer string
            **extra_kwargs: Contains:
                - base_url: URL of the AReaL proxy server

        Returns:
            Reward value (0.0 or 1.0) based on answer correctness

        Raises:
            ValueError: If "answer" key is missing from data
        """
        answer = data.get("answer")
        if answer is None:
            raise ValueError("Input data must contain 'answer' key")

        user_content = OpenHandsAgentBuilder.extract_user_content(
            data.get("messages", [])
        )
        if not user_content:
            logger.warning("No user message found in input data")
            return 0.0

        # Build agent
        agent = (
            OpenHandsAgentBuilder()
            .with_base_url(extra_kwargs.get("base_url"))
            .with_llm_config(
                model=self.kwargs.get("model", "default"),
                temperature=self.kwargs.get("temperature", 1.0),
                top_p=self.kwargs.get("top_p", 1.0),
                max_tokens=self.kwargs.get("max_tokens", 1024),
                max_completion_tokens=self.kwargs.get("max_completion_tokens"),
            )
            .build()
        )

        # Run conversation
        with tempfile.TemporaryDirectory() as workspace:
            conversation = Conversation(agent=agent, workspace=workspace)
            try:
                conversation.send_message(user_content)
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, conversation.run)
            except Exception as e:
                logger.error(f"OpenHands SDK error: {e}")
                return 0.0

            events = list(conversation.state.events)

        # Extract completion and calculate reward
        completion_text = OpenHandsAgentBuilder.extract_all_agent_text(events)
        logger.debug(f"Extracted completion: {completion_text[:100]}...")

        reward_fn = AsyncRewardWrapper(math_reward_fn)
        return await reward_fn(completions=completion_text, answer=answer)


class MathToolAgent(AgentWorkflow):
    """Math agent with calculator tools using OpenHands SDK.

    This agent extends the basic MathAgent with calculator tools,
    allowing the LLM to perform mathematical operations step by step.

    Example:
        >>> agent = MathToolAgent(temperature=1.0, max_tokens=2048)
        >>> reward = await agent.run(
        ...     data={"messages": [...], "answer": "42"},
        ...     base_url="http://localhost:8000/v1",
        ... )
    """

    SYSTEM_PROMPT = (
        "You are a math assistant with calculator tools. "
        "You MUST use the provided calculator tools (AddTool, SubtractTool, "
        "MultiplyTool, DivideTool, PowerTool, SqrtTool) to perform ALL "
        "mathematical calculations. Do not calculate in your head. "
        "Show your step-by-step reasoning and use tools for each calculation. "
        "After completing calculations, provide your final answer."
    )

    def __init__(self, **kwargs: Any):
        """Initialize the MathToolAgent.

        Args:
            **kwargs: Configuration options including:
                - model: Model name (default: "default")
                - temperature: Sampling temperature (default: 1.0)
                - top_p: Top-p sampling parameter (default: 1.0)
                - max_tokens: Maximum tokens for completion (default: 2048)
                - max_completion_tokens: Alternative name for max_tokens
        """
        self.kwargs = kwargs.copy()

    async def run(self, data: dict[str, Any], **extra_kwargs: Any) -> float:
        """Run the agent with tool-calling capabilities.

        Args:
            data: Input data containing:
                - messages: List of message dicts with "role" and "content"
                - answer: Ground truth answer string
            **extra_kwargs: Contains:
                - base_url: URL of the AReaL proxy server

        Returns:
            Reward value (0.0 or 1.0) based on answer correctness

        Raises:
            ValueError: If "answer" key is missing from data
        """
        answer = data.get("answer")
        if answer is None:
            raise ValueError("Input data must contain 'answer' key")

        user_content = OpenHandsAgentBuilder.extract_user_content(
            data.get("messages", [])
        )
        if not user_content:
            logger.warning("No user message found in input data")
            return 0.0

        # Build agent with tools
        agent = (
            OpenHandsAgentBuilder()
            .with_base_url(extra_kwargs.get("base_url"))
            .with_llm_config(
                model=self.kwargs.get("model", "default"),
                temperature=self.kwargs.get("temperature", 1.0),
                top_p=self.kwargs.get("top_p", 1.0),
                max_tokens=self.kwargs.get("max_tokens", 2048),
                max_completion_tokens=self.kwargs.get("max_completion_tokens"),
            )
            .with_calculator_tools()
            .with_system_prompt(self.SYSTEM_PROMPT)
            .build()
        )

        # Run conversation
        with tempfile.TemporaryDirectory() as workspace:
            conversation = Conversation(agent=agent, workspace=workspace)
            try:
                conversation.send_message(user_content)
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, conversation.run)
            except Exception as e:
                logger.error(f"OpenHands SDK error: {e}")
                return 0.0

            events = list(conversation.state.events)

        # Extract final answer and calculate reward
        final_answer = OpenHandsAgentBuilder.extract_final_agent_text(events)
        logger.debug(f"Extracted final answer: {final_answer[:100]}...")

        reward_fn = AsyncRewardWrapper(math_reward_fn)
        return await reward_fn(completions=final_answer, answer=answer)
