import re
import threading
import time
from copy import deepcopy
from typing import Any, List, Optional

import gymnasium as gym
from pydantic import BaseModel

from tau2.agent.base import (
    LocalAgent,
    ValidAgentInputMessage,
    is_valid_agent_history_message,
)
from tau2.data_model.message import AssistantMessage, Message, MultiToolMessage
from tau2.data_model.simulation import Task
from tau2.environment.environment import Environment
from tau2.environment.tool import Tool
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.user.user_simulator import UserSimulator


class GymAgentState(BaseModel):
    """The state of the agent."""

    messages: list[Any]


class GymAgent(LocalAgent[GymAgentState]):
    """
    An LLM agent that can be used to solve a task with a gym-like interface.
    """

    def __init__(
        self,
        tools: List[Tool],
        domain_policy: str,
    ):
        """
        Initialize the GymAgent.
        """
        super().__init__(tools=tools, domain_policy=domain_policy)
        self._orchestrator = None
        self._orchestrator_thread = None
        self._waiting_for_input = False
        self._input_received = threading.Event()
        self._next_action = None
        self._conversation_snapshot = None
        self._simulation_done = False
        self._lock = threading.Lock()

    def set_orchestrator(self, orchestrator):
        """Set the orchestrator that this agent will work with."""
        self._orchestrator = orchestrator

    def reset(self):
        """
        Reset the agent and start the orchestrator.
        Returns the initial conversation state.
        """
        with self._lock:
            # Reset state
            self._waiting_for_input = False
            self._input_received.clear()
            self._next_action = None
            self._conversation_snapshot = None
            self._simulation_done = False

            # Start orchestrator in a separate thread
            if self._orchestrator_thread and self._orchestrator_thread.is_alive():
                # Wait for any existing thread to finish
                self._orchestrator_thread.join(timeout=1.0)

            self._orchestrator_thread = threading.Thread(target=self._run_orchestrator)
            self._orchestrator_thread.daemon = True
            self._orchestrator_thread.start()

            # Wait for the first call to generate_next_message
            while not self._waiting_for_input and not self._simulation_done:
                time.sleep(0.01)

            if self._simulation_done:
                return self._conversation_snapshot

            return self._conversation_snapshot

    def step(self, action: str):
        """
        Provide an action and continue the simulation.
        Returns the conversation state when the next input is needed.
        """
        with self._lock:
            if not self._waiting_for_input:
                raise RuntimeError(
                    "Agent is not waiting for input. Call reset() first."
                )

            self._next_action = action
            self._input_received.set()
            self._waiting_for_input = False

            # Wait for the next call to generate_next_message
            while not self._waiting_for_input and not self._simulation_done:
                time.sleep(0.01)

            if self._simulation_done:
                return self._conversation_snapshot

            return self._conversation_snapshot

    def _run_orchestrator(self):
        """Run the orchestrator in a separate thread."""
        try:
            if self._orchestrator:
                self._orchestrator.run()
        except Exception as e:
            print(f"Orchestrator error: {e}")
        finally:
            self._simulation_done = True

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
        This method is called by the orchestrator and waits for input from step().
        """
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        # Take a snapshot of the conversation
        with self._lock:
            self._conversation_snapshot = deepcopy(state.messages)
            self._waiting_for_input = True
            self._input_received.clear()

        # Wait for input from step() method
        self._input_received.wait()

        # Get the action provided by step()
        next_message = self._next_action
        # if next_message is None:
        #     # Fallback to direct input if step() wasn't called
        #     next_message = input("Enter next action (or TOOL:tool_name(args)): ")

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


class TauGymEnv(gym.Env):
    """
    A gym environment for the tau2 simulation.
    """

    def __init__(self, domain: str, task_id: str):
        self.domain = domain
        self.task_id = task_id
        self._lock = threading.Lock()
        self._orchestrator = None
        self._orchestrator_thread = None
        self._waiting_for_input = False
        self._input_received = threading.Event()
        self._next_action = None
        self._conversation_snapshot = None
        self._simulation_done = False
        self.observation_space = gym.spaces.Text()
        self.action_space = gym.spaces.Text()

    def _create_orchestrator(self) -> Orchestrator:
        from tau2.registry import registry

        environment = registry.get_env_constructor(self.domain)()
        task = [
            task
            for task in registry.get_tasks_loader(self.domain)(self.task_id)
            if task.id == self.task_id
        ]
        if len(task) == 0:
            raise ValueError(f"Task {self.task_id} not found in domain {self.domain}")
        task = task[0]
        agent = GymAgent(
            tools=environment.get_tools(), domain_policy=environment.get_policy()
        )
        user = UserSimulator(user_info=environment.get_user_info())
        orchestrator = Orchestrator(
            environment=environment,
            task=task,
            agent=agent,
            user=user,
        )
        return orchestrator

    def _get_obs(self) -> str:
        # Wait for call to generate_next_message
        while not self._waiting_for_input and not self._simulation_done:
            time.sleep(0.01)

        if self._simulation_done:
            return self._conversation_snapshot

        return self._conversation_snapshot

    def _get_info(self) -> dict:
        return {}

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[str, dict]:
        """
        Reset the environment.

        Args:
            seed: The seed for the environment.
            options: The options for the environment.

        Returns:
            The initial observation and info.



        """
        super().reset(seed=seed)
        with self._lock:
            # Reset state
            self._waiting_for_input = False
            self._input_received.clear()
            self._next_action = None
            self._conversation_snapshot = None
            self._simulation_done = False

            # Start orchestrator in a separate thread
            if self._orchestrator_thread and self._orchestrator_thread.is_alive():
                # Wait for any existing thread to finish
                self._orchestrator_thread.join(timeout=1.0)

            self._orchestrator = self._create_orchestrator()
            self._orchestrator_thread = threading.Thread(target=self._run_orchestrator)
            self._orchestrator_thread.daemon = True
            self._orchestrator_thread.start()

            return self._get_obs(), self._get_info()

    def _run_orchestrator(self):
        """Run the orchestrator in a separate thread."""
        try:
            if self._orchestrator:
                self._orchestrator.run()
        except Exception as e:
            print(f"Orchestrator error: {e}")
        finally:
            self._simulation_done = True

    def step(self, action: str):
        """
        Provide an action and continue the simulation.
        Returns the conversation state when the next input is needed.
        """
        if self._orchestrator is None:
            raise RuntimeError("Orchestrator not initialized. Call reset() first.")
        with self._lock:
            if not self._waiting_for_input:
                raise RuntimeError(
                    "TauGymEnv is not waiting for input. Call reset() first."
                )

            # Set the next action. This will trigger the agent to generate a next message.
            self._next_action = action
            self._input_received.set()
            self._waiting_for_input = False

            return self._get_obs(), self._get_info()

    def render(self, mode="human"):
        pass
