import threading
from copy import deepcopy
from typing import List, Optional

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
    UserMessage,
)
from tau2.data_model.simulation import SimulationRun, Task
from tau2.environment.environment import Environment
from tau2.environment.tool import Tool
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.user.user_simulator import UserSimulator
from tau2.utils.tools import parse_action_string, to_functional_format


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
        self._next_action: Optional[AssistantMessage] = None
        self._action_received = threading.Event()
        self._observation_set = threading.Event()
        self._lock = threading.Lock()
        self._terminated = threading.Event()

    def stop(
        self,
        message: Optional[AssistantMessage] = None,
        state: Optional[GymAgentState] = None,
    ) -> None:
        """
        Stops the agent.
        Args:
            message: The last message to the agent.
        """
        super().stop(message, state)
        history = deepcopy(state.messages) if state else []
        self._observation = history + [message] if message else []
        self._action_received.set()
        self._observation_set.set()
        self._terminated.set()

    def step(self, action_msg: AssistantMessage) -> tuple[list[Message], bool]:
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
            action_msg: The action string to be executed by the agent

        Returns:
            A deep copy of the current message history (observation)

        Note:
            This method blocks until the agent has sent the action to the orchestrator and received an observation back.
            The returned observation includes all messages up to and including the agent's response to the provided action.
        """
        with self._lock:
            logger.info(f"Stepping with action: {str(action_msg)}")
            self._next_action = action_msg
            self._action_received.set()
            self._observation_set.clear()

        logger.info(f"Waiting for observation")
        # Wait for generate_next_message to set the observation
        self._observation_set.wait()

        logger.info(f"Got observation: {self._observation}")

        # Return the current observation
        observation = deepcopy(self._observation) if self._observation else []
        terminated = self._terminated.is_set()
        return observation, terminated

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

        logger.info(f"Continuing with action: {str(self._next_action)}")

        with self._lock:
            response_message = self._next_action
            # Reset for next iteration
            self._action_received.clear()
            self._observation_set.clear()
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
            self._terminated.clear()

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
        self._simulation_done: threading.Event = threading.Event()
        self._simulation_run: Optional[SimulationRun] = None
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
            self._simulation_run = None
            self._simulation_done.clear()

            # Wait for any existing thread to finish
            if self._orchestrator_thread and self._orchestrator_thread.is_alive():
                self._orchestrator_thread.join(timeout=1.0)

            # Create new orchestrator
            self._orchestrator = self.get_orchestrator()
            self._agent = self._orchestrator.agent
            self._user = self._orchestrator.user

            # Start orchestrator in a separate thread
            self._orchestrator_thread = threading.Thread(target=self._run_orchestrator)
            self._orchestrator_thread.daemon = True
            self._orchestrator_thread.start()

            # Wait for the agent to be waiting for input
            # Use a timeout to periodically check if simulation is done
            while (
                not self._simulation_done.is_set() and not self._agent.waiting_for_input
            ):
                self._simulation_done.wait(timeout=0.01)
                if self._simulation_done.is_set():
                    break

            if self._simulation_done.is_set():
                # Simulation ended immediately, return empty observation
                return "", self._get_info()

            # Get the current observation from the agent (don't call reset())
            current_observation = (
                self._agent._observation.copy() if self._agent._observation else []
            )

            # Convert observation to string format
            observation_str = self._format_observation(current_observation)

            return observation_str, self._get_info()

    def _get_info(self) -> dict:
        """
        Get the current info.
        """
        return {"simulation_run": self._get_simulation_run()}

    def _get_simulation_run(self) -> SimulationRun:
        """
        Get the current simulation run.
        """
        if self._simulation_run is None:
            return {}
        return self._simulation_run.model_dump_json(indent=2)

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
                return "", 0.0, True, False, self._get_info()

            # Parse the action string into a message
            action_msg = parse_action_string(action)

            # Provide the action to the agent
            observation, agent_terminated = self._agent.step(action_msg)

            # Wait for the agent to be waiting for input again, but only if simulation is not done
            # Use a timeout to periodically check if simulation is done
            # TODO: Review the logic here!
            while (
                not self._simulation_done.is_set() and not self._agent.waiting_for_input
            ):
                self._simulation_done.wait(timeout=0.01)
                if self._simulation_done.is_set():
                    break
            # Check if simulation is done
            logger.info(f"Agent terminated: {agent_terminated}")
            terminated = self._simulation_done.is_set()
            logger.info(f"Simulation done: {terminated}")
            # Convert observation to string format
            observation_str = self._format_observation(observation)

            return (
                observation_str,
                self.get_reward(),
                terminated,
                False,
                self._get_info(),
            )

    def get_reward(self) -> float:
        """
        Get the reward for the current simulation run.
        """
        if self._simulation_run is None:
            return 0.0
        return self._get_reward()

    def _get_reward(self) -> float:
        """
        Get the reward for the current simulation run.
        """
        from tau2.evaluator.evaluator import (  # TODO: Should not have to import inside func
            EvaluationType,
            evaluate_simulation,
        )

        evaluation_type = EvaluationType.ENV
        evaluation_result = evaluate_simulation(
            simulation=self._simulation_run,
            task=self.get_task(),
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

        Args:
            messages: List of Message objects representing the conversation history

        Returns:
            A string representation of the message history, or empty string
            if no messages are provided
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
            else:
                turns.append(f"{m.role}: {m.content}")
        return "\n".join(turns)

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


if __name__ == "__main__":
    from tau2.utils.display import ConsoleDisplay

    env = AgentGymEnv(domain="mock", task_id="create_task_1")
    observation, info = env.reset()
    print(observation)
    print(info)
    observation, reward, terminated, truncated, info = env.step(
        "create_task(user_id='user_1', title='Important Meeting')"
    )
    print(observation)
    print(reward)
    print(terminated)
    observation, reward, terminated, truncated, info = env.step(
        "ok done! It is called Important Meeting and assigned to user_1"
    )
    print(observation)
    print(reward)
    print(terminated)

    if not terminated:
        observation, reward, terminated, truncated, info = env.step(
            "ok great, have a nice day!"
        )
        print(observation)
        print(reward)
        print(terminated)

    ConsoleDisplay.display_task(env.get_task())
    ConsoleDisplay.display_simulation(env._simulation_run)

    ConsoleDisplay.display_task(env.get_task())
    ConsoleDisplay.display_simulation(env._simulation_run)
