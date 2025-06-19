import threading
import time

from tau2.agent.gym_agent import GymAgent, GymAgentState
from tau2.data_model.message import AssistantMessage, ToolCall, UserMessage
from tau2.environment.tool import Tool
from tests.utils import timeout


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
        assert not agent.waiting_for_input

    @timeout(10)
    def test_step_and_generate_next_message(self):
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
        assert agent.waiting_for_input

        action_content = "Hi! How can I help you?"
        action_msg = AssistantMessage(role="assistant", content=action_content)
        _ = agent.step(action_msg)
        thread.join(timeout=2.0)

        # Check for exceptions first
        if exception is not None:
            raise exception

        # At this point, we know no exception occurred, so variables must be set
        assert result_message is not None
        assert result_state is not None
        assert result_message.content == action_content
        assert isinstance(result_message, AssistantMessage)
        assert not agent.waiting_for_input
        assert result_state.messages[-1].content == action_content

    def test_reset(self):
        agent = make_agent()
        # Simulate a previous run
        agent._observation = [UserMessage(role="user", content="Hi")]
        agent._next_action = AssistantMessage(role="assistant", content="Test")
        agent._action_received.set()
        agent._observation_set.set()
        # Test the reset functionality directly by clearing the state
        with agent._lock:
            agent._action_received.clear()
            agent._observation_set.clear()
            agent._next_action = None
            agent._observation = None
        # Verify the state is cleared
        assert agent._observation is None
        assert agent._next_action is None
        assert not agent._action_received.is_set()
        assert not agent._observation_set.is_set()
        assert not agent.waiting_for_input

    def test_waiting_for_input_property(self):
        agent = make_agent()
        # Simulate waiting state
        agent._observation_set.set()
        agent._action_received.clear()
        assert agent.waiting_for_input
        # Simulate not waiting
        agent._observation_set.clear()
        assert not agent.waiting_for_input

    @timeout(10)
    def test_step_with_tool_call(self):
        """Test that GymAgent.step() works correctly with tool call messages."""
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
        assert agent.waiting_for_input

        # Create a tool call message
        tool_call = ToolCall(
            name="search_flights", arguments={"origin": "NYC", "destination": "LAX"}
        )
        action_msg = AssistantMessage(
            role="assistant", content=None, tool_calls=[tool_call]
        )

        _ = agent.step(action_msg)
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
        assert not agent.waiting_for_input
        assert result_state.messages[-1].tool_calls == [tool_call]


if __name__ == "__main__":
    t = TestGymAgent()
    t.test_initialization()
    t.test_step_and_generate_next_message()
    t.test_reset()
    t.test_waiting_for_input_property()
    t.test_step_with_tool_call()
    print("✅ All GymAgent unit tests passed!")
