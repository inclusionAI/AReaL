from copy import deepcopy
from typing import Callable

import pytest

from tau2.agent.llm_agent import LLMAgent, LLMSoloAgent
from tau2.data_model.message import AssistantMessage, ToolCall, UserMessage
from tau2.data_model.tasks import EnvAssertion, InitialState, Task
from tau2.environment.environment import Environment
from tau2.orchestrator.orchestrator import (
    DEFAULT_FIRST_AGENT_MESSAGE,
    Orchestrator,
    Role,
)
from tau2.user.user_simulator import DummyUser, UserSimulator


@pytest.fixture
def user_simulator() -> UserSimulator:
    return UserSimulator(
        instructions="You are a user simulator.",
        llm="gpt-3.5-turbo",
        llm_args={"temperature": 0.0},
    )


@pytest.fixture
def dummy_user() -> DummyUser:
    return DummyUser(
        instructions="You are a dummy user.",
        llm="gpt-3.5-turbo",
        llm_args={"temperature": 0.0},
    )


@pytest.fixture
def agent(get_environment: Callable[[], Environment]) -> LLMAgent:
    environment = get_environment()
    return LLMAgent(
        tools=environment.get_tools(),
        domain_policy=environment.get_policy(),
        llm="gpt-3.5-turbo",
        llm_args={"temperature": 0.0},
    )


@pytest.fixture
def solo_agent(
    get_environment: Callable[[], Environment], base_task: Task
) -> LLMSoloAgent:
    environment = get_environment()
    return LLMSoloAgent(
        tools=environment.get_tools(),
        domain_policy=environment.get_policy(),
        task=base_task,
        llm="gpt-3.5-turbo",
        llm_args={"temperature": 0.0},
    )


def test_orchestrator_initialize_base(
    domain_name: str,
    user_simulator: UserSimulator,
    agent: LLMAgent,
    get_environment: Callable[[], Environment],
    base_task: Task,
):
    orchestrator = Orchestrator(
        domain=domain_name,
        user=user_simulator,
        agent=agent,
        environment=get_environment(),
        task=base_task,
    )
    orchestrator.initialize()

    # Check Initialization
    assert orchestrator.from_role == Role.AGENT
    assert orchestrator.to_role == Role.USER
    assert orchestrator.step_count == 0
    assert not orchestrator.done
    assert orchestrator.termination_reason is None
    assert len(orchestrator.trajectory) == 1
    assert isinstance(orchestrator.trajectory[0], AssistantMessage)
    assert orchestrator.trajectory[0].content == DEFAULT_FIRST_AGENT_MESSAGE.content
    assert orchestrator.message.content == DEFAULT_FIRST_AGENT_MESSAGE.content


def test_orchestrator_initialize_with_message_history(
    domain_name: str,
    user_simulator: UserSimulator,
    agent: LLMAgent,
    get_environment: Callable[[], Environment],
    task_with_message_history: Task,
):
    orchestrator = Orchestrator(
        domain=domain_name,
        user=user_simulator,
        agent=agent,
        environment=get_environment(),
        task=task_with_message_history,
    )
    orchestrator.environment.run_env_assertion(
        EnvAssertion(
            env_type="assistant",
            func_name="assert_number_of_tasks",
            arguments={"user_id": "user_1", "expected_number": 1},
        )
    )
    orchestrator.initialize()
    assert orchestrator.from_role == Role.AGENT
    assert orchestrator.to_role == Role.USER
    assert orchestrator.step_count == 0
    assert not orchestrator.done
    assert orchestrator.termination_reason is None
    assert len(orchestrator.get_trajectory()) == len(
        task_with_message_history.initial_state.message_history
    )

    user_state = orchestrator.user_state
    print(user_state.model_dump_json(indent=2))
    assert len(user_state.messages) == 1

    agent_state = orchestrator.agent_state
    print(agent_state.model_dump_json(indent=2))
    assert len(agent_state.messages) == len(
        task_with_message_history.initial_state.message_history
    )
    orchestrator.environment.run_env_assertion(
        EnvAssertion(
            env_type="assistant",
            func_name="assert_task_status",
            arguments={"task_id": "task_2", "expected_status": "pending"},
        )
    )
    orchestrator.environment.run_env_assertion(
        EnvAssertion(
            env_type="assistant",
            func_name="assert_number_of_tasks",
            arguments={"user_id": "user_1", "expected_number": 2},
        )
    )


def test_orchestrator_initialize_with_initialization_data(
    domain_name: str,
    user_simulator: UserSimulator,
    agent: LLMAgent,
    get_environment: Callable[[], Environment],
    task_with_initialization_data: Task,
):
    orchestrator = Orchestrator(
        domain=domain_name,
        user=user_simulator,
        agent=agent,
        environment=get_environment(),
        task=task_with_initialization_data,
    )
    orchestrator.environment.run_env_assertion(
        EnvAssertion(
            env_type="assistant",
            func_name="assert_number_of_tasks",
            arguments={"user_id": "user_1", "expected_number": 1},
        )
    )
    orchestrator.initialize()
    print(orchestrator.environment.tools.db.model_dump_json(indent=2))
    assert orchestrator.from_role == Role.AGENT
    assert orchestrator.to_role == Role.USER
    assert orchestrator.step_count == 0
    assert not orchestrator.done
    assert orchestrator.termination_reason is None
    assert len(orchestrator.get_trajectory()) == 1
    orchestrator.environment.run_env_assertion(
        EnvAssertion(
            env_type="assistant",
            func_name="assert_task_status",
            arguments={"task_id": "task_2", "expected_status": "pending"},
        )
    )
    orchestrator.environment.run_env_assertion(
        EnvAssertion(
            env_type="assistant",
            func_name="assert_number_of_tasks",
            arguments={"user_id": "user_1", "expected_number": 2},
        )
    )


def test_orchestrator_initialize_with_initialization_actions(
    domain_name: str,
    user_simulator: UserSimulator,
    agent: LLMAgent,
    get_environment: Callable[[], Environment],
    task_with_initialization_actions: Task,
):
    orchestrator = Orchestrator(
        domain=domain_name,
        user=user_simulator,
        agent=agent,
        environment=get_environment(),
        task=task_with_initialization_actions,
    )
    orchestrator.environment.run_env_assertion(
        EnvAssertion(
            env_type="assistant",
            func_name="assert_number_of_tasks",
            arguments={"user_id": "user_1", "expected_number": 1},
        )
    )
    orchestrator.initialize()
    print(orchestrator.environment.tools.db.model_dump_json(indent=2))
    assert orchestrator.from_role == Role.AGENT
    assert orchestrator.to_role == Role.USER
    assert orchestrator.step_count == 0
    assert not orchestrator.done
    assert orchestrator.termination_reason is None
    assert len(orchestrator.get_trajectory()) == 1
    orchestrator.environment.run_env_assertion(
        EnvAssertion(
            env_type="assistant",
            func_name="assert_task_status",
            arguments={"task_id": "task_2", "expected_status": "pending"},
        )
    )
    orchestrator.environment.run_env_assertion(
        EnvAssertion(
            env_type="assistant",
            func_name="assert_number_of_tasks",
            arguments={"user_id": "user_1", "expected_number": 2},
        )
    )


def test_orchestrator_step(
    domain_name: str,
    user_simulator: UserSimulator,
    agent: LLMAgent,
    base_task: Task,
    get_environment: Callable[[], Environment],
):
    orchestrator = Orchestrator(
        domain=domain_name,
        user=user_simulator,
        agent=agent,
        environment=get_environment(),
        task=base_task,
    )
    orchestrator.initialize()

    # Check Step 1
    orchestrator.step()
    assert orchestrator.from_role == Role.USER
    assert orchestrator.to_role == Role.AGENT
    assert orchestrator.step_count == 1
    assert not orchestrator.done
    assert orchestrator.termination_reason is None
    assert len(orchestrator.get_trajectory()) == 2
    assert isinstance(orchestrator.get_trajectory()[1], UserMessage)
    assert isinstance(orchestrator.message, UserMessage)

    # Check Step 2
    orchestrator.step()
    assert orchestrator.from_role == Role.AGENT
    assert orchestrator.to_role in [Role.ENV, Role.USER]
    assert orchestrator.step_count == 2
    assert not orchestrator.done
    assert orchestrator.termination_reason is None
    assert len(orchestrator.get_trajectory()) == 3
    assert isinstance(orchestrator.get_trajectory()[2], AssistantMessage)
    assert isinstance(orchestrator.message, AssistantMessage)


def test_orchestrator_restart(
    domain_name: str,
    user_simulator: UserSimulator,
    agent: LLMAgent,
    base_task: Task,
    get_environment: Callable[[], Environment],
):
    orchestrator1 = Orchestrator(
        domain=domain_name,
        user=user_simulator,
        agent=agent,
        environment=get_environment(),
        task=base_task,
        seed=300,
    )
    orchestrator1.initialize()
    # Create a partial message history
    for _ in range(3):
        orchestrator1.step()
    partial_message_history = orchestrator1.get_trajectory()

    # Create a new task with the partial message history
    task2 = deepcopy(base_task)
    initial_state = InitialState(
        message_history=partial_message_history,
        variables={},
        state={},
    )
    task2.initial_state = initial_state
    # Create a new orchestrator with the partial new task
    orchestrator2 = Orchestrator(
        domain=domain_name,
        environment=get_environment(),
        user=user_simulator,
        agent=agent,
        task=task2,
        seed=300,
    )
    orchestrator2.initialize()

    assert orchestrator1.to_role == orchestrator2.to_role
    assert orchestrator1.from_role == orchestrator2.from_role
    assert orchestrator1.message.content == orchestrator2.message.content
    for msg1, msg2 in zip(
        orchestrator1.get_trajectory(), orchestrator2.get_trajectory()
    ):
        assert msg1.content == msg2.content

    ## Step each orchestrator 3 times
    for _ in range(3):
        orchestrator1.step()
        orchestrator2.step()
        print("--------------------------------")
        print("Orchestrator 1")
        print(orchestrator1.message)
        print("--------------------------------")
        print("Orchestrator 2")
        print(orchestrator2.message)
        print("--------------------------------")


def test_orchestrator_run(
    domain_name: str,
    user_simulator: UserSimulator,
    agent: LLMAgent,
    base_task: Task,
    get_environment: Callable[[], Environment],
):
    orchestrator = Orchestrator(
        domain=domain_name,
        environment=get_environment(),
        user=user_simulator,
        agent=agent,
        task=base_task,
        max_steps=10,
    )
    simulation_run = orchestrator.run()
    assert simulation_run is not None


def test_orchestrator_communication_error_empty_messages(
    domain_name: str,
    user_simulator: UserSimulator,
    agent: LLMAgent,
    get_environment: Callable[[], Environment],
    base_task: Task,
):
    """Test that empty messages (no content, no tool calls) raise appropriate errors."""
    orchestrator = Orchestrator(
        domain=domain_name,
        user=user_simulator,
        agent=agent,
        environment=get_environment(),
        task=base_task,
    )
    orchestrator.initialize()

    # Test cases for empty messages
    empty_message_cases = [
        # (message, from_role, expected_error_type, description)
        (
            AssistantMessage(role="assistant", content=None, tool_calls=None),
            Role.AGENT,
            "agent_error",
            "agent empty message",
        ),
        (
            UserMessage(role="user", content=None, tool_calls=None),
            Role.USER,
            "user_error",
            "user empty message",
        ),
        (
            AssistantMessage(role="assistant", content="   \n\t   ", tool_calls=None),
            Role.AGENT,
            "agent_error",
            "agent whitespace-only message",
        ),
        (
            UserMessage(role="user", content="", tool_calls=None),
            Role.USER,
            "user_error",
            "user empty string message",
        ),
    ]

    for message, from_role, expected_error, description in empty_message_cases:
        orchestrator.message = message
        orchestrator.from_role = from_role
        orchestrator.to_role = Role.USER if from_role == Role.AGENT else Role.AGENT
        orchestrator.done = False
        orchestrator.termination_reason = None

        # This should raise appropriate error and set termination reason
        orchestrator.check_communication_error()

        assert orchestrator.done, f"Failed for {description}"
        assert orchestrator.termination_reason.value == expected_error, (
            f"Failed for {description}"
        )

    # Test that ENV role is ignored (should not raise error)
    empty_message = AssistantMessage(role="assistant", content=None, tool_calls=None)
    orchestrator.message = empty_message
    orchestrator.from_role = Role.ENV
    orchestrator.to_role = Role.AGENT
    orchestrator.done = False
    orchestrator.termination_reason = None

    # This should not raise an error because from_role is ENV
    orchestrator.check_communication_error()

    assert not orchestrator.done
    assert orchestrator.termination_reason is None


def test_orchestrator_communication_error_mixed_content_and_tool_calls(
    domain_name: str,
    user_simulator: UserSimulator,
    agent: LLMAgent,
    get_environment: Callable[[], Environment],
    base_task: Task,
):
    """Test that messages with both content and tool calls raise appropriate errors."""
    orchestrator = Orchestrator(
        domain=domain_name,
        user=user_simulator,
        agent=agent,
        environment=get_environment(),
        task=base_task,
    )
    orchestrator.initialize()

    # Create a tool call for testing
    tool_call = ToolCall(id="test", name="test_tool", arguments={})

    # Test cases for mixed content messages
    mixed_message_cases = [
        # (message, from_role, expected_error_type, description)
        (
            AssistantMessage(
                role="assistant", content="Hello there", tool_calls=[tool_call]
            ),
            Role.AGENT,
            "agent_error",
            "agent message with both content and tool calls",
        ),
        (
            UserMessage(role="user", content="Hello there", tool_calls=[tool_call]),
            Role.USER,
            "user_error",
            "user message with both content and tool calls",
        ),
        # Note: Empty string and whitespace-only content are not considered text content
        # by has_text_content(), so these should NOT raise errors
    ]

    for message, from_role, expected_error, description in mixed_message_cases:
        orchestrator.message = message
        orchestrator.from_role = from_role
        orchestrator.to_role = Role.USER if from_role == Role.AGENT else Role.AGENT
        orchestrator.done = False
        orchestrator.termination_reason = None

        # This should raise appropriate error and set termination reason
        orchestrator.check_communication_error()

        assert orchestrator.done, f"Failed for {description}"
        assert orchestrator.termination_reason.value == expected_error, (
            f"Failed for {description}"
        )


def test_orchestrator_communication_error_solo_mode(
    domain_name: str,
    dummy_user: DummyUser,
    solo_agent: LLMSoloAgent,
    base_task: Task,
    get_environment: Callable[[], Environment],
):
    """Test solo mode communication restrictions for agents."""
    orchestrator = Orchestrator(
        domain=domain_name,
        environment=get_environment(solo_mode=True),
        user=dummy_user,
        agent=solo_agent,
        task=base_task,
        solo_mode=True,
    )
    orchestrator.initialize()

    # Create a tool call for testing
    tool_call = ToolCall(id="test", name="test_tool", arguments={})

    # Test cases for solo mode
    solo_mode_cases = [
        # (message, from_role, to_role, should_fail, expected_error, description)
        (
            AssistantMessage(role="assistant", content="Hello there", tool_calls=None),
            Role.AGENT,
            Role.USER,
            True,
            "agent_error",
            "agent text content in solo mode",
        ),
        (
            AssistantMessage(role="assistant", content="###STOP###", tool_calls=None),
            Role.AGENT,
            Role.USER,
            False,
            None,
            "agent stop message in solo mode",
        ),
        (
            AssistantMessage(role="assistant", content=None, tool_calls=[tool_call]),
            Role.AGENT,
            Role.ENV,
            False,
            None,
            "agent tool call in solo mode",
        ),
        # Note: ###TRANSFER### and ###OUT-OF-SCOPE### are user stop messages, not agent stop messages
        # so they should fail in solo mode
        (
            AssistantMessage(
                role="assistant", content="###TRANSFER###", tool_calls=None
            ),
            Role.AGENT,
            Role.USER,
            True,
            "agent_error",
            "agent transfer message in solo mode (not a stop message)",
        ),
        (
            AssistantMessage(
                role="assistant", content="###OUT-OF-SCOPE###", tool_calls=None
            ),
            Role.AGENT,
            Role.USER,
            True,
            "agent_error",
            "agent out-of-scope message in solo mode (not a stop message)",
        ),
    ]

    for (
        message,
        from_role,
        to_role,
        should_fail,
        expected_error,
        description,
    ) in solo_mode_cases:
        orchestrator.message = message
        orchestrator.from_role = from_role
        orchestrator.to_role = to_role
        orchestrator.done = False
        orchestrator.termination_reason = None

        # Check communication error
        orchestrator.check_communication_error()

        if should_fail:
            assert orchestrator.done, f"Failed for {description}"
            assert orchestrator.termination_reason.value == expected_error, (
                f"Failed for {description}"
            )
        else:
            assert not orchestrator.done, f"Failed for {description}"
            assert orchestrator.termination_reason is None, f"Failed for {description}"


def test_orchestrator_communication_error_valid_messages(
    domain_name: str,
    user_simulator: UserSimulator,
    agent: LLMAgent,
    get_environment: Callable[[], Environment],
    base_task: Task,
):
    """Test that valid messages pass communication error checks."""
    orchestrator = Orchestrator(
        domain=domain_name,
        user=user_simulator,
        agent=agent,
        environment=get_environment(),
        task=base_task,
    )
    orchestrator.initialize()

    # Create a tool call for testing
    tool_call = ToolCall(id="test", name="test_tool", arguments={})

    # Test cases for valid messages
    valid_message_cases = [
        # (message, from_role, to_role, description)
        (
            AssistantMessage(role="assistant", content="Hello there", tool_calls=None),
            Role.AGENT,
            Role.USER,
            "agent text message",
        ),
        (
            UserMessage(role="user", content="Hello there", tool_calls=None),
            Role.USER,
            Role.AGENT,
            "user text message",
        ),
        (
            AssistantMessage(role="assistant", content=None, tool_calls=[tool_call]),
            Role.AGENT,
            Role.ENV,
            "agent tool call message",
        ),
        (
            UserMessage(role="user", content=None, tool_calls=[tool_call]),
            Role.USER,
            Role.ENV,
            "user tool call message",
        ),
    ]

    for message, from_role, to_role, description in valid_message_cases:
        orchestrator.message = message
        orchestrator.from_role = from_role
        orchestrator.to_role = to_role
        orchestrator.done = False
        orchestrator.termination_reason = None

        # This should not raise an error
        orchestrator.check_communication_error()

        assert not orchestrator.done, f"Failed for {description}"
        assert orchestrator.termination_reason is None, f"Failed for {description}"


def test_orchestrator_communication_error_invalid_from_role(
    domain_name: str,
    user_simulator: UserSimulator,
    agent: LLMAgent,
    get_environment: Callable[[], Environment],
    base_task: Task,
):
    """Test that an invalid from_role raises ValueError."""
    orchestrator = Orchestrator(
        domain=domain_name,
        user=user_simulator,
        agent=agent,
        environment=get_environment(),
        task=base_task,
    )
    orchestrator.initialize()

    # Set an invalid from_role
    orchestrator.from_role = "invalid_role"  # type: ignore
    orchestrator.message = AssistantMessage(role="assistant", content="test")

    # This should raise ValueError
    with pytest.raises(ValueError, match="Invalid from role: invalid_role"):
        orchestrator._check_communication_error()


def test_orchestrator_run_with_solo_agent(
    domain_name: str,
    dummy_user: DummyUser,
    solo_agent: LLMSoloAgent,
    base_task: Task,
    get_environment: Callable[[], Environment],
):
    orchestrator = Orchestrator(
        domain=domain_name,
        environment=get_environment(solo_mode=True),
        user=dummy_user,
        agent=solo_agent,
        task=base_task,
        max_steps=10,
        solo_mode=True,
    )
    simulation_run = orchestrator.run()
    assert simulation_run is not None

    orchestrator.environment.run_env_assertion(
        EnvAssertion(
            env_type="assistant",
            func_name="assert_task_status",
            arguments={"task_id": "task_2", "expected_status": "pending"},
        )
    )
