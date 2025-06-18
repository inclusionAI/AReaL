import re
from copy import deepcopy
from typing import Any, List, Optional

from pydantic import BaseModel

from tau2.agent.base import (
    LocalAgent,
    ValidAgentInputMessage,
    is_valid_agent_history_message,
)
from tau2.data_model.message import AssistantMessage, Message, MultiToolMessage
from tau2.environment.tool import Tool


class GymAgentState(BaseModel):
    """The state of the agent."""

    messages: list[Any]


class GymAgent(LocalAgent[GymAgentState]):
    """
    An LLM agent that can be used to solve a task.
    """

    def __init__(
        self,
        tools: List[Tool],
        domain_policy: str,
    ):
        """
        Initialize the LLMAgent.
        """
        super().__init__(tools=tools, domain_policy=domain_policy)

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> GymAgentState:
        """Get the initial state of the agent.

        Args:
            message_history: The message history of the conversation.

        Returns:
            The initial state of the agent.
        """
        if message_history is None:
            message_history = []
        assert all(is_valid_agent_history_message(m) for m in message_history), (
            "Message history must contain only AssistantMessage, UserMessage, or ToolMessage to Agent."
        )
        return GymAgentState(
            messages=message_history,
        )

    def _display_conversation(self, state: GymAgentState):
        """Display the current conversation history."""
        print("\n" + "=" * 50)
        print("CONVERSATION HISTORY:")
        print("=" * 50)

        for i, msg in enumerate(state.messages):
            if hasattr(msg, "role"):
                role = msg.role
                content = getattr(msg, "content", str(msg))
            else:
                role = "unknown"
                content = str(msg)

            print(f"{i + 1}. [{role.upper()}]: {content}")

        print("=" * 50 + "\n")

    def _parse_tool_call(
        self, tool_input: str, message_count: int = 0
    ) -> Optional[dict]:
        """Parse tool call using regex pattern.

        Expected format: TOOL:tool_name(arg1=value1, arg2=value2)
        """
        # Regex pattern to match TOOL:tool_name(arguments)
        tool_pattern = r"^TOOL:\s*(\w+)\s*\((.*)\)$"
        match = re.match(tool_pattern, tool_input.strip())

        if not match:
            return None

        tool_name = match.group(1)
        args_str = match.group(2)

        # Parse arguments (simple key=value format)
        args = {}
        if args_str.strip():
            # Split by comma, but be careful about commas within quotes
            arg_pairs = re.findall(r"(\w+)\s*=\s*([^,]+)", args_str)
            for key, value in arg_pairs:
                # Remove quotes if present
                value = value.strip().strip("\"'")
                args[key] = value

        return {
            "id": f"call_{message_count}",
            "type": "function",
            "function": {"name": tool_name, "arguments": args},
        }

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: GymAgentState
    ) -> tuple[AssistantMessage, GymAgentState]:
        """
        Respond to a user or tool message.
        """
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        # Display current conversation
        self._display_conversation(state)

        # Get user input for next action
        next_message = input("Enter next action (or TOOL:tool_name(args)): ")

        # Parse the input to determine if it's a tool call or user message
        tool_call = self._parse_tool_call(next_message, len(state.messages))

        if tool_call:
            # Create tool call message
            assistant_message = AssistantMessage(
                role="assistant", content="", tool_calls=[tool_call]
            )
        else:
            # Regular user message
            assistant_message = AssistantMessage(
                role="assistant", content=next_message, cost=0.0
            )

        # Add the assistant message to state
        state.messages.append(assistant_message)

        return assistant_message, state
