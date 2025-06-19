import threading
import time
from copy import deepcopy
from typing import Any, List, Optional

import gymnasium as gym
from loguru import logger
from pydantic import BaseModel

from tau2.agent.base import LocalAgent, ValidAgentInputMessage
from tau2.config import DEFAULT_LLM_ARGS_USER, DEFAULT_LLM_USER
from tau2.data_model.message import AssistantMessage, Message, MultiToolMessage
from tau2.data_model.simulation import Task
from tau2.environment.environment import Environment
from tau2.environment.tool import Tool
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.user.user_simulator import UserSimulator


class GymAgentState(BaseModel):
    """The state of the agent."""

    messages: list[Any]


class GymAgent(LocalAgent):
    """
    A gym agent that can be used to solve a task with a gym-like interface.
    """

    def __init__(self, tools: List[Tool], domain_policy: str):
        super().__init__(tools=tools, domain_policy=domain_policy)
        self._observation: Optional[list[Message]] = None
        self._next_action: Optional[str] = None
        self._action_received = threading.Event()
        self._observation_set = threading.Event()
        self._lock = threading.Lock()

    def step(self, action: str) -> list[Message]:
        """
        Set the next action
        This should allow generate_next_message to continue
        Wait until generate_next_message sets the observation
        return the observation
        """
        with self._lock:
            logger.info(f"Stepping with action: {action}")
            self._next_action = action
            self._action_received.set()
            self._observation_set.clear()

        logger.info(f"Waiting for observation")
        # Wait for generate_next_message to set the observation
        self._observation_set.wait()

        logger.info(f"Got observation: {self._observation}")

        # Return the current observation
        return deepcopy(self._observation) if self._observation else []

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: GymAgentState
    ) -> tuple[AssistantMessage, GymAgentState]:
        """
        Set current observation to messages
        Wait for next_action to be set
        Return the next message
        """
        with self._lock:
            logger.info(f"Got message: {message}")
            if isinstance(message, MultiToolMessage):
                state.messages.extend(message.tool_messages)
            else:
                state.messages.append(message)
            self._observation = deepcopy(state.messages)
            logger.info(f"Setting observation: {self._observation}")
            self._observation_set.set()

        # Wait for step() to provide the next action
        logger.info(f"Waiting for action")
        self._action_received.wait()

        logger.info(f"Continuing with action: {self._next_action}")

        with self._lock:
            action = self._next_action
            # Reset for next iteration
            self._action_received.clear()
            self._observation_set.clear()
            self._next_action = None

        # Create the response message
        response_message = AssistantMessage(role="assistant", content=action) ## FIXME: We need to handle tool calls here
        state.messages.append(response_message)
        return response_message, state

    def get_init_state(
        self,
        message_history: Optional[list[Message]] = None,
    ) -> GymAgentState:
        """Get the initial state of the agent."""
        messages = message_history.copy() if message_history else []
        return GymAgentState(messages=messages)

    @property
    def waiting_for_input(self) -> bool:
        """Check if the agent is waiting for input via step()."""
        return self._observation_set.is_set() and not self._action_received.is_set()

    def reset(self) -> list[Message]:
        """
        Reset the agent state and wait for generate_next_message to be called for the first time.
        Returns the initial observation from generate_next_message.
        """
        with self._lock:
            # Clear any pending synchronization
            self._action_received.clear()
            self._observation_set.clear()
            self._next_action = None
            self._observation = None

        # Wait for generate_next_message to set the observation
        self._observation_set.wait()

        # Return the initial observation
        return deepcopy(self._observation) if self._observation else []


class TauGymEnv(gym.Env):
    """
    A gym environment for the gym agent.
    """

    def __init__(self, domain: str, task_id: str):
        self.domain: str = domain
        self.task_id: str = task_id
        self._lock: threading.Lock = threading.Lock()
        self._agent: Optional[GymAgent] = None
        self._user: Optional[UserSimulator] = None
        self._orchestrator: Optional[Orchestrator] = None
        self._orchestrator_thread: Optional[threading.Thread] = None
        self._simulation_done: bool = False
        self.observation_space: gym.spaces.Space = gym.spaces.Text(max_length=10000)
        self.action_space: gym.spaces.Space = gym.spaces.Text(max_length=1000)

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[str, dict]:
        """Reset the environment.
        This should create a new orchestrator and start it in a separate thread.
        it should call reset on the agent and return the first observation.
        It should return that observation.
        At this point:
            - the Agent should be waiting for input.
            - The orchestrator should be waiting for the agent to get this input so that it can continue the simulation.

        Returns:
        observation, info

        For now only return the observation. The info can be placeholder values.
        """
        super().reset(seed=seed)

        with self._lock:
            # Reset state
            self._simulation_done = False

            # Wait for any existing thread to finish
            if self._orchestrator_thread and self._orchestrator_thread.is_alive():
                self._orchestrator_thread.join(timeout=1.0)

            # Create new orchestrator
            self._orchestrator = self.get_orchestrator()
            self._agent = self._orchestrator.agent
            self._user = self._orchestrator.user

            # Do NOT call self._orchestrator.initialize() here; run() will do it

            # Start orchestrator in a separate thread
            self._orchestrator_thread = threading.Thread(target=self._run_orchestrator)
            self._orchestrator_thread.daemon = True
            self._orchestrator_thread.start()

            # Wait for the agent to be waiting for input
            while not self._agent.waiting_for_input and not self._simulation_done:
                time.sleep(0.01)

            if self._simulation_done:
                # Simulation ended immediately, return empty observation
                return "", {}

            # Get the current observation from the agent (don't call reset())
            current_observation = (
                self._agent._observation.copy() if self._agent._observation else []
            )

            # Convert observation to string format
            observation_str = self._format_observation(current_observation)

            return observation_str, {}

    def step(self, action: str) -> tuple[str, dict, bool, bool, dict]:
        """
        Provide an action and continue the simulation.
        - This should call step on the agent and return the observation along with other required values
        At this point:
            - The agent should be waiting for input.
            - The orchestrator should be waiting for the agent to get this input so that it can continue the simulation.
        We should check if the orchestrator is done.

        Returns:
        observation, reward, terminated, truncated, info

        For now only return the observation and terminated. The rest can be placeholder values.
        """
        if self._orchestrator is None:
            raise RuntimeError("Orchestrator not initialized. Call reset() first.")

        with self._lock:
            if self._simulation_done:
                return "", 0.0, True, False, {}

            # Provide the action to the agent
            observation = self._agent.step(action)

            # Wait for the agent to be waiting for input again or simulation to be done
            while not self._agent.waiting_for_input and not self._simulation_done:
                time.sleep(0.01)

            # Check if simulation is done
            terminated = self._simulation_done

            # Convert observation to string format
            observation_str = self._format_observation(observation)

            return observation_str, 0.0, terminated, False, {}

    def _run_orchestrator(self):
        """Run the orchestrator in a separate thread."""
        try:
            if self._orchestrator:
                self._orchestrator.run()
        except Exception as e:
            print(f"Orchestrator error: {e}")
        finally:
            self._simulation_done = True

    def _format_observation(self, messages: list[Message]) -> str:
        """Format the observation as a string."""
        if not messages:
            return ""
        return "\n".join([f"{m.role}: {m.content}" for m in messages])

    def get_environment(self) -> Environment:
        """Get the environment."""
        from tau2.registry import registry

        return registry.get_env_constructor(self.domain)()

    def get_task(self) -> Task:
        """Get the task."""
        from tau2.registry import registry

        return next(
            task
            for task in registry.get_tasks_loader(self.domain)()
            if task.id == self.task_id
        )

    def get_agent(self) -> GymAgent:
        """Get the agent."""
        environment = self.get_environment()
        return GymAgent(
            tools=environment.get_tools(),
            domain_policy=environment.get_policy(),
        )

    def get_user(self) -> UserSimulator:
        """Get the user."""
        environment = self.get_environment()
        task = self.get_task()
        try:
            user_tools = environment.get_user_tools()
        except ValueError:
            user_tools = None
        return UserSimulator(
            tools=user_tools,
            instructions=task.user_scenario,
            llm=DEFAULT_LLM_USER,
            llm_args=deepcopy(DEFAULT_LLM_ARGS_USER),
        )

    def get_orchestrator(self) -> Orchestrator:
        """Get the orchestrator."""
        return Orchestrator(
            domain=self.domain,
            agent=self.get_agent(),
            user=self.get_user(),
            environment=self.get_environment(),
            task=self.get_task(),
        )
    

# if __name__ == "__main__":
#     env = TauGymEnv(domain="retail", task_id="1")
