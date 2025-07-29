import threading
import time
from copy import deepcopy
from typing import Any, List, Optional

import gymnasium as gym
from loguru import logger
from pydantic import BaseModel

from tau2.agent.base import LocalAgent, ValidAgentInputMessage
from tau2.config import DEFAULT_LLM_ARGS_USER, DEFAULT_LLM_USER
from tau2.data_model.message import (
    APICompatibleMessage,
    AssistantMessage,
    Message,
    MultiToolMessage,
)
from tau2.data_model.simulation import Task
from tau2.environment.environment import Environment
from tau2.environment.tool import Tool
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.user.user_simulator import UserSimulator


class GymAgentState(BaseModel):
    """The state of the agent."""

    messages: list[APICompatibleMessage]


class GymAgent(LocalAgent):
    """
    A gym-style agent that provides a step-based interface for task execution.

    This agent implements a gym-like interface where external code can control
    the agent's actions step-by-step. It uses threading events to synchronize
    between the external step() calls and internal message generation.

    The agent maintains an observation-action cycle:
    1. External code calls step(action) to provide the next action
    2. The agent processes the action and generates a response
    3. The agent waits for the next step() call
    """

    def __init__(self, tools: List[Tool], domain_policy: str):
        """
        Initialize the gym agent with tools and domain policy.

        Args:
            tools: List of tools available to the agent
            domain_policy: Policy string defining the agent's behavior in the domain
        """
        super().__init__(tools=tools, domain_policy=domain_policy)
        self._observation: Optional[list[Message]] = None
        self._next_action: Optional[str] = None
        self._action_received = threading.Event()
        self._observation_set = threading.Event()
        self._lock = threading.Lock()

    def step(self, action: str) -> list[Message]:
        """
        Provide the next action to the agent and get the resulting observation.

        This method implements the gym-style step interface. It provides an action
        to the agent, waits for the agent to process it and generate a response,
        then returns the current observation (message history).

        The method uses threading events to synchronize with generate_next_message():
        - Sets the next action and signals that an action is available
        - Waits for generate_next_message() to process the action and set the observation
        - Returns a copy of the current observation

        Args:
            action: The action string to be executed by the agent

        Returns:
            A deep copy of the current message history (observation)

        Note:
            This method blocks until the agent has sent the action to the orchestrator and received an observation back.
            The returned observation includes all messages up to and including the agent's response to the provided action.
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
        Generate the next message in the conversation.

        This method is called by the orchestrator to process incoming messages
        and generate responses. It implements a two-phase synchronization:

        1. **Observation Phase**: Updates the agent's observation with the current
           message history and signals that an observation is ready
        2. **Action Phase**: Waits for an external action to be provided via step(),
           then generates and returns the response message

        The method handles both regular messages and MultiToolMessages by
        appropriately updating the state's message history.

        Args:
            message: The incoming message to process (can be a regular message
                    or MultiToolMessage containing tool call results)
            state: The current agent state containing message history

        Returns:
            A tuple containing:
            - The generated AssistantMessage response
            - The updated GymAgentState with the new message history

        Note:
            This method blocks during the action phase until step() is called
            to provide the next action. The synchronization ensures that the
            agent's responses are controlled externally through the gym interface.
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
        response_message = AssistantMessage(
            role="assistant", content=action
        )  ## FIXME: We need to handle tool calls here
        state.messages.append(response_message)
        return response_message, state

    def get_init_state(
        self,
        message_history: Optional[list[Message]] = None,
    ) -> GymAgentState:
        """
        Create and return the initial state for the agent.

        Args:
            message_history: Optional list of existing messages to initialize
                           the state with. If None, starts with an empty list.

        Returns:
            A new GymAgentState instance with the provided or empty message history
        """
        messages = message_history.copy() if message_history else []
        return GymAgentState(messages=messages)

    @property
    def waiting_for_input(self) -> bool:
        """
        Check if the agent is currently waiting for input via step().

        This property indicates whether the agent has set an observation
        and is waiting for an external action to be provided through
        the step() method.

        Returns:
            True if the agent is waiting for input, False otherwise
        """
        return self._observation_set.is_set() and not self._action_received.is_set()

    def reset(self) -> list[Message]:
        """
        Reset the agent's internal state and wait for the first observation.

        This method clears all internal synchronization state and waits
        for generate_next_message() to be called for the first time to
        set the initial observation.

        Returns:
            A deep copy of the initial observation (message history) once
            generate_next_message() has been called and set the observation.
            Returns an empty list if no observation is set.

        Note:
            This method blocks until generate_next_message() is called and
            sets the initial observation. It's typically used to initialize
            the agent before starting the step-action cycle.
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


class AgentGymEnv(gym.Env):
    """
    A Gymnasium environment that wraps the Tau2 simulation system.

    This environment provides a standard gym interface for interacting with
    Tau2 simulations. It manages the lifecycle of orchestrators, agents,
    and user simulators in a thread-safe manner.

    The environment coordinates between:
    - The external gym interface (reset/step calls)
    - The internal Tau2 orchestrator running in a separate thread
    - The GymAgent that provides step-by-step control

    Key Features:
    - Thread-safe operation with proper synchronization
    - Automatic orchestrator lifecycle management
    - Standard gym observation/action spaces
    - Graceful handling of simulation termination
    """

    def __init__(self, domain: str, task_id: str):
        """
        Initialize the Tau2 gym environment.

        Args:
            domain: The domain name (e.g., 'retail', 'telecom', 'airline')
            task_id: The specific task ID to run within the domain
        """
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
        """
        Reset the environment and start a new simulation.

        This method creates a fresh simulation by:
        1. Creating a new orchestrator with the specified domain and task
        2. Starting the orchestrator in a separate thread
        3. Waiting for the agent to be ready for input
        4. Returning the initial observation

        The method ensures proper cleanup of any existing simulation
        and thread-safe initialization of the new one.

        Args:
            seed: Optional random seed for reproducibility (passed to gym.Env.reset)
            options: Optional configuration options (passed to gym.Env.reset)

        Returns:
            A tuple containing:
            - observation: String representation of the initial message history
            - info: Dictionary with additional information (currently empty)

        Note:
            This method blocks until the orchestrator has started and the agent
            is waiting for the first action. If the simulation ends immediately
            (e.g., due to an error), an empty observation is returned.
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
        Execute an action and advance the simulation.

        This method provides the standard gym step interface. It:
        1. Passes the action to the GymAgent via its step() method
        2. Waits for the agent to process the action and receive a response
        3. Checks if the simulation has terminated
        4. Returns the updated observation and termination status

        The method handles the coordination between the external gym interface
        and the internal Tau2 simulation running in a separate thread.

        Args:
            action: The action string to be executed by the agent

        Returns:
            A tuple containing:
            - observation: String representation of the current message history
            - reward: Always 0.0 (not used in current implementation)
            - terminated: True if the simulation has ended, False otherwise
            - truncated: Always False (not used in current implementation)
            - info: Dictionary with additional information (currently empty)

        Raises:
            RuntimeError: If reset() has not been called before step()

        Note:
            This method blocks until the agent has processed the action and
            is ready for the next step. The simulation may terminate during
            this process, in which case terminated will be True.
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
        """
        Run the orchestrator in a separate thread.

        This private method is the target for the orchestrator thread.
        It runs the orchestrator's main simulation loop and handles
        any exceptions that occur during execution.

        The method sets the _simulation_done flag when the orchestrator
        finishes (either normally or due to an error), which signals
        to the main thread that the simulation has ended.
        """
        try:
            if self._orchestrator:
                self._orchestrator.run()
        except Exception as e:
            print(f"Orchestrator error: {e}")
        finally:
            self._simulation_done = True

    def _format_observation(self, messages: list[Message]) -> str:
        """
        Convert a list of messages to a string observation.

        This method formats the message history into a readable string
        format for the gym observation space. Each message is formatted
        as "role: content" and messages are separated by newlines.

        Args:
            messages: List of Message objects representing the conversation history

        Returns:
            A string representation of the message history, or empty string
            if no messages are provided
        """
        if not messages:
            return ""
        return "\n".join([f"{m.role}: {m.content}" for m in messages])

    def get_environment(self) -> Environment:
        """
        Create and return the environment for the specified domain.

        This method uses the registry to construct the appropriate
        environment instance based on the domain name.

        Returns:
            An Environment instance configured for the specified domain
        """
        from tau2.registry import registry

        return registry.get_env_constructor(self.domain)()

    def get_task(self) -> Task:
        """
        Retrieve the task configuration for the specified task ID.

        This method loads all tasks for the domain and finds the one
        matching the specified task_id.

        Returns:
            The Task object corresponding to the specified task_id

        Raises:
            StopIteration: If no task is found with the specified task_id
        """
        from tau2.registry import registry

        return next(
            task
            for task in registry.get_tasks_loader(self.domain)()
            if task.id == self.task_id
        )

    def get_agent(self) -> GymAgent:
        """
        Create and return a GymAgent instance for the domain.

        This method creates a GymAgent with the tools and policy
        from the domain's environment.

        Returns:
            A GymAgent instance configured with the domain's tools and policy
        """
        environment = self.get_environment()
        return GymAgent(
            tools=environment.get_tools(),
            domain_policy=environment.get_policy(),
        )

    def get_user(self) -> UserSimulator:
        """
        Create and return a UserSimulator instance for the task.

        This method creates a UserSimulator with the task's user scenario
        and any available user tools from the environment. If user tools
        are not available for the domain, they are set to None.

        Returns:
            A UserSimulator instance configured with the task's user scenario
            and domain-specific user tools (if available)
        """
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
        """
        Create and return an Orchestrator instance for the simulation.

        This method creates a complete Orchestrator with all necessary
        components: agent, user, environment, and task. The orchestrator
        coordinates the interaction between these components during the
        simulation.

        Returns:
            An Orchestrator instance configured with all simulation components
        """
        return Orchestrator(
            domain=self.domain,
            agent=self.get_agent(),
            user=self.get_user(),
            environment=self.get_environment(),
            task=self.get_task(),
        )
