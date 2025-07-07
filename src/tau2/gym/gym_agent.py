import threading
from copy import deepcopy
from typing import Any, List, Optional

import gymnasium as gym
from gymnasium.envs.registration import register
from loguru import logger
from pydantic import BaseModel

from tau2.agent.base import LocalAgent, ValidAgentInputMessage
from tau2.config import DEFAULT_LLM_ARGS_USER, DEFAULT_LLM_USER
from tau2.data_model.message import (
    APICompatibleMessage,
    AssistantMessage,
    Message,
    MultiToolMessage,
    UserMessage,
)
from tau2.data_model.simulation import SimulationRun, Task
from tau2.environment.environment import Environment
from tau2.environment.tool import Tool
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.registry import registry
from tau2.user.user_simulator import DummyUser, UserSimulator
from tau2.utils.tools import parse_action_string, to_functional_format

TAU_BENCH_ENV_NAME = "tau-bench"
TAU_BENCH_ENV_VERSION = "v0"
TAU_BENCH_ENV_ID = f"{TAU_BENCH_ENV_NAME}-{TAU_BENCH_ENV_VERSION}"


class TauSpace(gym.spaces.Space):
    """
    A space for the tau-bench gym environment.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def sample(self, *args, **kwargs) -> str:
        """
        Sample a string from the space.
        """
        logger.warning("Sampling not supported for tau-bench gym environment")
        return ""

    def contains(self, x: Any) -> bool:
        """
        Check if a string is in the space.
        """
        return isinstance(x, str)


def register_gym_agent() -> None:
    """
    Register the tau-bench gym environment with gymnasium.
    """
    register(
        id=TAU_BENCH_ENV_ID,
        entry_point="tau2.gym.gym_agent:AgentGymEnv",
    )


class GymAgentState(BaseModel):
    """The state of the gym agent containing the conversation history."""

    messages: list[APICompatibleMessage]


class GymAgent(LocalAgent):
    """
    A gym-style agent that provides a step-based interface for task execution.

    This agent implements a gym-like interface where external code can control
    the agent's actions step-by-step. It uses threading events to synchronize
    between the external set_action() calls and internal message generation.

    The agent maintains an observation-action cycle:
    1. External code calls set_action(action_msg) to provide the next action
    2. The agent processes the action and generates a response
    3. The agent waits for the next set_action() call
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
        self._next_action: Optional[AssistantMessage] = None
        self._agent_turn_finished = threading.Event()
        self._lock = threading.Lock()
        self._agent_turn_finished.set()

    @property
    def observation(self) -> list[Message]:
        """
        Get the current observation.

        Returns:
            List of messages representing the current conversation state,
            or empty list if no observation has been set yet.
        """
        return self._observation if self._observation else []

    def stop(
        self,
        message: Optional[AssistantMessage] = None,
        state: Optional[GymAgentState] = None,
    ) -> None:
        """
        Stops the agent and finalizes the observation.

        This method is called when the simulation ends. It updates the final
        observation with the complete message history and signals that the
        agent's turn is finished.

        Args:
            message: The last message to the agent (optional)
            state: The current agent state containing message history (optional)
        """
        super().stop(message, state)
        history = deepcopy(state.messages) if state else []
        with self._lock:
            self._observation = history + [message] if message else []
            self._agent_turn_finished.set()

    def set_action(self, action_msg: AssistantMessage) -> None:
        """
        Provide the next action to the agent.

        This method is called by external code to provide the next action
        that the agent should take. It sets the action and signals that
        the agent can continue processing.

        The method uses threading events to synchronize with generate_next_message():
        - Sets the next action and signals that an action is available
        - Raises an error if called when it's not the agent's turn to act

        Args:
            action_msg: The AssistantMessage containing the action to be executed

        Raises:
            RuntimeError: If called when it's not the agent's turn to act
        """
        with self._lock:
            if self._agent_turn_finished.is_set():
                raise RuntimeError("It is not the agent's turn to act.")
            logger.info(f"Stepping with action: {str(action_msg)}")
            self._next_action = action_msg
            self._agent_turn_finished.set()

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: GymAgentState
    ) -> tuple[AssistantMessage, GymAgentState]:
        """
        Generate the next message in the conversation.

        This method is called by the orchestrator to process incoming messages
        and generate responses. It implements a two-phase synchronization:

        1. **Observation Phase**: Updates the agent's observation with the current
           message history and signals that an observation is ready
        2. **Action Phase**: Waits for an external action to be provided via set_action(),
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
            This method blocks during the action phase until set_action() is called
            to provide the next action. The synchronization ensures that the
            agent's responses are controlled externally through the gym interface.
        """
        with self._lock:
            self._agent_turn_finished.clear()
            logger.info(f"Got message: {message}")
            if isinstance(message, MultiToolMessage):
                state.messages.extend(message.tool_messages)
            elif (
                message is not None
            ):  # TODO: Review. Added this to handle solo mode. But there might be a better way to do this.
                state.messages.append(message)
            # If message is None, we don't add it to the messages list
            self._observation = deepcopy(state.messages)
            logger.info(f"Setting observation: {self._observation}")

        # Wait for set_action() to provide the next action
        logger.info(f"Waiting for action")
        self._agent_turn_finished.wait()

        logger.info(f"Continuing with action: {str(self._next_action)}")

        with self._lock:
            response_message = self._next_action
            # Reset for next iteration
            self._next_action = None

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
    def is_agent_turn(self) -> bool:
        """
        Check if the agent is currently waiting for input via set_action().

        This property indicates whether the agent has set an observation
        and is waiting for an external action to be provided through
        the set_action() method.

        Returns:
            True if the agent is waiting for input, False otherwise
        """
        return not self._agent_turn_finished.is_set()


class GymConfig(BaseModel):
    """
    Configuration for the Gym environment.
    """

    domain: str
    task_id: str
    solo_mode: bool = False
    user_llm: str = DEFAULT_LLM_USER
    user_llm_args: dict = deepcopy(DEFAULT_LLM_ARGS_USER)


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

    Action Input Format:
    The step() method accepts action strings in multiple formats:
    1. JSON-formatted tool calls: Valid ToolCall JSON objects
       Example: '{"name": "search", "arguments": {"query": "flights"}}'

    2. Functional tool calls: Function-style syntax with keyword arguments
       Example: "search_flights(origin='NYC', destination='LAX')"
       Example: "book_ticket(flight_id=123, passenger_name='John Doe')"

    3. Plain text content: Regular text messages for communication
       Example: "Hello, how can I help you?"
       Example: "I need to book a flight from New York to Los Angeles"

    The environment automatically detects the format and converts it to the appropriate
    message type (AssistantMessage with tool calls or content). Plain text messages
    are sent to the user simulator, while tool calls are executed against the environment
    to perform actions like searching databases, making bookings, or retrieving information.
    """

    def __init__(
        self,
        domain: str,
        task_id: str,
        solo_mode: bool = False,
        user_llm: Optional[str] = None,
        user_llm_args: Optional[dict] = None,
        all_messages_as_observation: bool = False,
    ):
        """
        Initialize the Tau2 gym environment.

        Args:
            domain: The domain name (e.g., 'retail', 'telecom', 'airline')
            task_id: The specific task ID to run within the domain
        """
        self.domain = domain
        self.task_id = task_id
        self.solo_mode = solo_mode
        self.user_llm = user_llm if user_llm else DEFAULT_LLM_USER
        self.user_llm_args = (
            user_llm_args if user_llm_args else deepcopy(DEFAULT_LLM_ARGS_USER)
        )
        self.all_messages_as_observation = all_messages_as_observation

        self._lock = threading.Lock()
        self._agent: Optional[GymAgent] = None
        self._user: Optional[UserSimulator] = None
        self._orchestrator: Optional[Orchestrator] = None
        self._orchestrator_thread: Optional[threading.Thread] = None
        self._simulation_done = threading.Event()
        self._simulation_run: Optional[SimulationRun] = None
        self.observation_space = TauSpace()
        self.action_space = TauSpace()

    def _get_tools(self) -> List[Tool]:
        """
        Get the tools for the environment.
        """
        if self._orchestrator is None:
            raise ValueError("Orchestrator not initialized. Call reset() first.")
        return self._orchestrator.environment.get_tools()

    def _get_policy(self) -> str:
        """
        Get the policy for the environment.
        """
        if self._orchestrator is None:
            raise ValueError("Orchestrator not initialized. Call reset() first.")
        return self._orchestrator.environment.get_policy()

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
            self._simulation_run = None
            self._simulation_done.clear()

            # Wait for any existing thread to finish
            if self._orchestrator_thread and self._orchestrator_thread.is_alive():
                self._orchestrator_thread.join(timeout=1.0)

            # Create new orchestrator
            self._orchestrator = self._get_orchestrator()
            self._agent = self._orchestrator.agent
            self._user = self._orchestrator.user

            # Start orchestrator in a separate thread
            self._orchestrator_thread = threading.Thread(target=self._run_orchestrator)
            self._orchestrator_thread.daemon = True
            self._orchestrator_thread.start()

            # Wait for orchestrator to send the initial observation
            # Use a timeout to periodically check if simulation is done
            while not self._simulation_done.is_set() and not self._agent.is_agent_turn:
                self._simulation_done.wait(timeout=0.01)

            if self._simulation_done.is_set():
                # Simulation ended immediately, return empty observation
                logger.warning("Simulation ended immediately")
                return "", self._get_info()

            # Get the initial observation from the agent
            initial_observation = self._agent.observation.copy()

            # Convert observation to string format
            observation_str = self._format_observation(initial_observation)

            return observation_str, self._get_info()

    def _get_info(self) -> dict:
        """
        Get the current info dictionary for the gym environment.

        Returns:
            A dictionary containing the current simulation run information
            in JSON format, or an empty dictionary if no simulation has run yet.
        """
        return {
            "task": self._get_task(),
            "simulation_run": self._get_simulation_run(),
            "tools": self._get_tools(),
            "policy": self._get_policy(),
        }

    def _get_simulation_run(self) -> SimulationRun:
        """
        Get the current simulation run as a JSON string.

        Returns:
            A JSON string representation of the current simulation run,
            or an empty dictionary if no simulation has run yet.
        """
        if self._simulation_run is None:
            return {}
        return self._simulation_run.model_dump_json(indent=2)

    def step(self, action: str) -> tuple[str, float, bool, bool, dict]:
        """
        Execute an action and advance the simulation.

        This method provides the standard gym step interface. It:
        1. Passes the action to the GymAgent via its set_action() method
        2. Waits for the agent to process the action and receive a response
        3. Checks if the simulation has terminated
        4. Returns the updated observation and termination status

        The method handles the coordination between the external gym interface
        and the internal Tau2 simulation running in a separate thread.

        Args:
            action: The action string to be executed by the agent. Supports multiple formats:
                - JSON-formatted tool calls: '{"name": "search", "arguments": {"query": "flights"}}'
                - Functional tool calls: "search_flights(origin='NYC', destination='LAX')"
                - Plain text content: "Hello, how can I help you?"
                See the class docstring for detailed format examples.
                Note: Plain text messages are sent to the user simulator, while tool calls
                are executed against the environment to perform actions.

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
            if self._simulation_done.is_set():
                raise ValueError("Simulation already terminated.")

            # Parse the action string into a message
            action_msg = parse_action_string(action)

            # Provide the action to the agent
            self._agent.set_action(action_msg)

            # Wait for the orchestrator to send the next observation
            # Use a timeout to periodically check if simulation is done
            while not self._simulation_done.is_set() and not self._agent.is_agent_turn:
                self._simulation_done.wait(timeout=0.01)

            # Check if simulation is done
            terminated = self._simulation_done.is_set()
            logger.info(f"Simulation done: {terminated}")
            # Convert observation to string format
            observation_str = self._format_observation(self._agent.observation)

            return (
                observation_str,
                self._get_reward(),
                terminated,
                False,
                self._get_info(),
            )

    def _get_reward(self) -> float:
        """
        Compute the reward for the current simulation run.

        This method evaluates the simulation using the Tau2 evaluation
        system and returns the computed reward value. It uses the ALL
        evaluation type and non-solo mode for comprehensive assessment.
        The reward value for the current simulation, or 0.0 if no simulation has been completed.
        Returns:
            The computed reward value based on simulation performance.
        """
        if self._simulation_run is None:
            return 0.0
        evaluation_type = EvaluationType.ALL
        evaluation_result = evaluate_simulation(
            simulation=self._simulation_run,
            task=self._get_task(),
            evaluation_type=evaluation_type,
            solo_mode=False,
            domain=self.domain,
        )
        logger.info(f"Evaluation result: {evaluation_result}")
        return evaluation_result.reward

    def _run_orchestrator(self):
        """
        Run the orchestrator in a separate thread.

        This private method is the target for the orchestrator thread.
        It runs the orchestrator's main simulation loop and handles
        any exceptions that occur during execution.

        The method sets the _simulation_done flag when the orchestrator
        finishes (either normally or due to an error), which signals
        to the main thread that the simulation has ended.
        It also sets the simulation run to the orchestrator's simulation run.

        Thread Safety:
            This method is designed to be run in a separate thread and
            uses the _simulation_done event to communicate with the main thread.
            Any exceptions are logged but do not propagate to avoid thread crashes.

        Error Handling:
            If the orchestrator raises an exception, it is logged as an error
            but the thread continues to set the simulation as done to prevent
            the main thread from hanging indefinitely.
        """
        simulation_run = None
        try:
            if self._orchestrator:
                logger.info("Starting orchestrator")
                simulation_run = self._orchestrator.run()
                logger.info("Orchestrator finished")
        except Exception as e:
            logger.error(f"Orchestrator error: {e}")
        finally:
            self._simulation_run = simulation_run
            self._simulation_done.set()

    def _format_observation(self, messages: list[Message]) -> str:
        """
        Convert a list of messages to a string observation.

        This method formats the message history into a readable string
        format for the gym observation space. Each message is formatted
        as "role: content" and messages are separated by newlines.

        The method handles different message types:
        - UserMessage: Formatted as "user: content" or "user: tool_calls"
        - AssistantMessage: Formatted as "assistant: content" or "assistant: tool_calls"
        - Other messages: Formatted as "role: content"

        Tool calls are converted to functional format for readability.

        Args:
            messages: List of Message objects representing the conversation history

        Returns:
            A string representation of the message history, or empty string
            if no messages are provided. Each message is on a separate line
            in the format "role: content".
        """
        if not messages:
            return ""
        turns = []
        for m in messages:
            if isinstance(m, UserMessage):
                if not m.is_tool_call():
                    turns.append(f"user: {m.content}")
                else:
                    tool_calls = ", ".join(
                        [to_functional_format(t) for t in m.tool_calls]
                    )
                    turns.append(f"user: {tool_calls}")
            elif isinstance(m, AssistantMessage):
                if not m.is_tool_call():
                    turns.append(f"assistant: {m.content}")
                else:
                    tool_calls = ", ".join(
                        [to_functional_format(t) for t in m.tool_calls]
                    )
                    turns.append(f"assistant: {tool_calls}")
                if not self.all_messages_as_observation:
                    # reset the turns contents, only keep the response to the assistant messages.
                    turns = []
            else:
                turns.append(f"{m.role}: {m.content}")
        return "\n".join(turns)

    def _get_environment(self) -> Environment:
        """
        Create and return the environment for the specified domain.

        This method uses the registry to construct the appropriate
        environment instance based on the domain name. The registry
        provides domain-specific environment constructors that are
        configured with the appropriate tools, policies, and data.

        Returns:
            An Environment instance configured for the specified domain.
            The environment contains domain-specific tools, policies,
            and data structures needed for simulation.

        Raises:
            ValueError: If the domain is not registered in the registry
        """
        return registry.get_env_constructor(self.domain)(solo_mode=self.solo_mode)

    def _get_task(self) -> Task:
        """
        Retrieve the task configuration for the specified task ID.

        This method loads all tasks for the domain using the registry's
        task loader and finds the one matching the specified task_id.
        Tasks contain the scenario, user instructions, and evaluation
        criteria for the simulation.

        Returns:
            The Task object corresponding to the specified task_id.
            The task contains the complete scenario definition including
            user instructions, success criteria, and evaluation parameters.

        Raises:
            ValueError: If no task is found with the specified task_id
                       for the given domain
        """
        tasks = registry.get_tasks_loader(self.domain)()
        for task in tasks:
            if task.id == self.task_id:
                return task
        raise ValueError(
            f"No task found with id {self.task_id} for domain {self.domain}"
        )

    def _get_agent(self) -> GymAgent:
        """
        Create and return a GymAgent instance for the domain.

        This method creates a GymAgent with the tools and policy
        from the domain's environment. The GymAgent provides the
        step-by-step interface that allows external control of
        the agent's actions through the gym environment.

        The agent is configured with:
        - Domain-specific tools for performing actions
        - Domain policy that defines the agent's behavior and constraints

        Returns:
            A GymAgent instance configured with the domain's tools and policy.
            The agent is ready to participate in simulations with external
            step-by-step control.
        """
        environment = self._get_environment()
        user_tools = environment.get_user_tools() if environment.user_tools else []
        return GymAgent(
            tools=environment.get_tools() + user_tools,
            domain_policy=environment.get_policy(),
        )

    def _get_user(self) -> UserSimulator:
        """
        Create and return a UserSimulator instance for the task.

        This method creates a UserSimulator with the task's user scenario
        and any available user tools from the environment. If user tools
        are not available for the domain, they are set to None.

        The user simulator is configured with:
        - Task-specific user scenario and instructions
        - Domain-specific user tools (if available)
        - Default LLM configuration for user simulation

        Error Handling:
            If the environment does not support user tools (raises ValueError),
            the user tools are set to None and the simulator continues without them.

        Returns:
            A UserSimulator instance configured with the task's user scenario
            and domain-specific user tools (if available). The simulator is
            ready to participate in the conversation simulation.
        """
        environment = self._get_environment()
        task = self._get_task()
        try:
            user_tools = environment.get_user_tools()
        except ValueError:
            user_tools = None
        if self.solo_mode:
            user_simulator = DummyUser()
        else:
            user_simulator = UserSimulator(
                tools=user_tools,
                instructions=task.user_scenario,
                llm=self.user_llm,
                llm_args=self.user_llm_args,
            )
        return user_simulator

    def _get_orchestrator(self) -> Orchestrator:
        """
        Create and return an Orchestrator instance for the simulation.

        This method creates a complete Orchestrator with all necessary
        components: agent, user, environment, and task. The orchestrator
        coordinates the interaction between these components during the
        simulation.

        The orchestrator manages:
        - Message flow between agent and user
        - Tool execution and environment state
        - Simulation lifecycle and termination conditions
        - Thread-safe coordination between components

        Returns:
            An Orchestrator instance configured with all simulation components.
            The orchestrator is ready to run the complete simulation with
            proper coordination between agent, user, and environment.
        """
        return Orchestrator(
            domain=self.domain,
            agent=self._get_agent(),
            user=self._get_user(),
            environment=self._get_environment(),
            task=self._get_task(),
            solo_mode=self.solo_mode,
        )
