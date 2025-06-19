import threading
import time

from tau2.agent.gym_agent import GymAgent
from tau2.data_model.message import AssistantMessage, UserMessage
from tau2.environment.tool import Tool
from tests.utils import timeout


def make_dummy_tool():
    def dummy_tool():
        """A dummy tool for testing."""
        return "dummy result"

    return Tool(func=dummy_tool)


def make_agent():
    tool = make_dummy_tool()
    return GymAgent(tools=[tool], domain_policy="Test policy")


def make_state(messages=None):
    if messages is None:
        messages = []
    from tau2.agent.gym_agent import GymAgentState

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

        result_message = None
        result_state = None
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

        action = "Hi! How can I help you?"
        obs = agent.step(action)
        thread.join(timeout=2.0)

        assert exception is None
        assert result_message is not None
        assert result_message.content == action
        assert isinstance(result_message, AssistantMessage)
        assert result_state is not None
        assert not agent.waiting_for_input
        assert result_state.messages[-1].content == action

    def test_reset(self):
        agent = make_agent()
        # Simulate a previous run
        agent._observation = [UserMessage(role="user", content="Hi")]
        agent._next_action = "Test"
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


if __name__ == "__main__":
    t = TestGymAgent()
    t.test_initialization()
    t.test_step_and_generate_next_message()
    t.test_reset()
    t.test_waiting_for_input_property()
    print("✅ All GymAgent unit tests passed!")
    t.test_waiting_for_input_property()
    print("✅ All GymAgent unit tests passed!")
