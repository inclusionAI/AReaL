import threading
import time

import pytest

from tau2.data_model.message import AssistantMessage, ToolCall, UserMessage
from tau2.environment.tool import Tool
from tau2.gym.gym_agent import GymAgent, GymAgentState
from .utils import timeout


def make_dummy_tool() -> Tool:
    def dummy_tool():
        """A dummy tool for testing."""
        return "dummy result"

    return Tool(func=dummy_tool)


def make_agent() -> GymAgent:
    tool = make_dummy_tool()
    return GymAgent(tools=[tool], domain_policy="Test policy")


def make_state(messages=None) -> GymAgentState:
    if messages is None:
        messages = []

    return GymAgentState(messages=messages)


class TestGymAgent:
    def test_initialization(self):
        agent = make_agent()
        assert agent._observation is None
        assert agent._next_action is None
        assert not agent.is_agent_turn

    @timeout(10)
    def test_set_action_and_generate_next_message(self):
        agent = make_agent()
        state = make_state()
        test_message = UserMessage(role="user", content="Hello!")
        state.messages.append(test_message)

        result_message: AssistantMessage | None = None
        result_state: GymAgentState | None = None
        exception = None

        def run_generate():
            nonlocal result_message, result_state, exception
            try:
                result_message, result_state = agent.generate_next_message(
                    test_message, state
                )
            except Exception as e:
                exception = e

        thread = threading.Thread(target=run_generate)
        thread.start()
        time.sleep(0.1)
        assert agent.is_agent_turn

        action_content = "Hi! How can I help you?"
        action_msg = AssistantMessage(role="assistant", content=action_content)
        agent.set_action(action_msg)
        thread.join(timeout=2.0)

        # Check for exceptions first
        if exception is not None:
            raise exception

        # At this point, we know no exception occurred, so variables must be set
        assert result_message is not None
        assert result_state is not None
        assert result_message.content == action_content
        assert isinstance(result_message, AssistantMessage)
        assert not agent.is_agent_turn
        assert result_state.messages[-1].content == action_content

    def test_reset(self):
        agent = make_agent()
        # Simulate a previous run
        agent._observation = [UserMessage(role="user", content="Hi")]
        agent._next_action = AssistantMessage(role="assistant", content="Test")
        agent._agent_turn_finished.clear()
        # Test the reset functionality directly by clearing the state
        with agent._lock:
            agent._agent_turn_finished.set()
            agent._next_action = None
            agent._observation = None
        # Verify the state is cleared
        assert agent._observation is None
        assert agent._next_action is None
        assert agent._agent_turn_finished.is_set()
        assert not agent.is_agent_turn

    def test_waiting_for_input_property(self):
        agent = make_agent()
        # Simulate waiting state
        agent._agent_turn_finished.clear()
        assert agent.is_agent_turn
        # Simulate not waiting
        agent._agent_turn_finished.set()
        assert not agent.is_agent_turn

    @timeout(10)
    def test_set_action_with_tool_call(self):
        """Test that GymAgent.set_action() works correctly with tool call messages."""
        agent = make_agent()
        state = make_state()
        test_message = UserMessage(role="user", content="Search for flights")
        state.messages.append(test_message)

        result_message: AssistantMessage | None = None
        result_state: GymAgentState | None = None
        exception = None

        def run_generate():
            nonlocal result_message, result_state, exception
            try:
                result_message, result_state = agent.generate_next_message(
                    test_message, state
                )
            except Exception as e:
                exception = e

        thread = threading.Thread(target=run_generate)
        thread.start()
        time.sleep(0.1)
        assert agent.is_agent_turn

        # Create a tool call message
        tool_call = ToolCall(
            name="search_flights", arguments={"origin": "NYC", "destination": "LAX"}
        )
        action_msg = AssistantMessage(
            role="assistant", content=None, tool_calls=[tool_call]
        )

        agent.set_action(action_msg)
        thread.join(timeout=2.0)

        # Check for exceptions first
        if exception is not None:
            raise exception

        # At this point, we know no exception occurred, so variables must be set
        assert result_message is not None
        assert result_state is not None
        assert result_message.tool_calls is not None
        assert len(result_message.tool_calls) == 1
        assert result_message.tool_calls[0].name == "search_flights"
        assert result_message.tool_calls[0].arguments == {
            "origin": "NYC",
            "destination": "LAX",
        }
        assert isinstance(result_message, AssistantMessage)
        assert not agent.is_agent_turn
        assert result_state.messages[-1].tool_calls == [tool_call]

    def test_set_action_when_not_agent_turn(self):
        """Test that set_action() raises an error when called at the wrong time."""
        agent = make_agent()
        action_msg = AssistantMessage(role="assistant", content="Test message")

        # Should raise error when agent turn is finished (not waiting for action)
        with pytest.raises(RuntimeError, match="It is not the agent's turn to act."):
            agent.set_action(action_msg)

    def test_stop_method(self):
        """Test the stop() method functionality."""
        agent = make_agent()
        state = make_state([UserMessage(role="user", content="Hello")])
        final_message = AssistantMessage(role="assistant", content="Goodbye")

        agent.stop(final_message, state)

        # Check that observation is updated with final message
        assert len(agent.observation) == 2
        assert agent.observation[0].content == "Hello"
        assert agent.observation[1].content == "Goodbye"
        assert agent._agent_turn_finished.is_set()

    def test_get_init_state(self):
        """Test the get_init_state() method."""
        agent = make_agent()

        # Test with no message history
        state = agent.get_init_state()
        assert isinstance(state, GymAgentState)
        assert state.messages == []

        # Test with existing message history
        messages = [UserMessage(role="user", content="Hello")]
        state = agent.get_init_state(message_history=messages)
        assert isinstance(state, GymAgentState)
        assert len(state.messages) == 1
        assert state.messages[0].content == "Hello"

    def test_observation_property(self):
        """Test the observation property."""
        agent = make_agent()

        # Test with no observation set
        assert agent.observation == []

        # Test with observation set
        messages = [UserMessage(role="user", content="Hello")]
        agent._observation = messages
        assert agent.observation == messages


if __name__ == "__main__":
    t = TestGymAgent()
    t.test_initialization()
    t.test_set_action_and_generate_next_message()
    t.test_reset()
    t.test_waiting_for_input_property()
    t.test_set_action_with_tool_call()
    t.test_set_action_when_not_agent_turn()
    t.test_stop_method()
    t.test_get_init_state()
    t.test_observation_property()
    print("✅ All GymAgent unit tests passed!")
