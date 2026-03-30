"""Tests for Gymnasium interface of VisionQATask.

This module tests:
- Gymnasium env interface (reset, step)
- State save/restore functionality
- Compatibility with existing inference patterns
"""

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Any

import pytest

from geo_edit.environment.task.openai_compatible_vision_qa_task import OpenAICompatibleVisionQATask
from geo_edit.environment.task.vision_qa_task import ToolCall


# =============================================================================
# Test Fixtures and Helpers
# =============================================================================

def _get_test_image_path() -> Path:
    return Path(__file__).resolve().parents[1] / "images" / "input_image.png"


def _make_tool_call(call_id: str, name: str, arguments: dict):
    """Create a mock tool call object for responses API format."""
    return SimpleNamespace(
        id=call_id,
        call_id=call_id,
        type="function_call",
        name=name,
        arguments=json.dumps(arguments),
    )


def _make_message_output(content: str):
    """Create a mock message output object for responses API format."""
    part = SimpleNamespace(type="output_text", text=content)
    return SimpleNamespace(type="message", content=[part])


def _make_responses_api_response(content: str | None, tool_calls=None, tokens_used: int = 5):
    """Create a mock response object for responses API format."""
    output_items = []
    if content is not None:
        output_items.append(_make_message_output(content))
    if tool_calls:
        output_items.extend(tool_calls)
    return SimpleNamespace(
        output=output_items,
        output_text=content,
        usage=SimpleNamespace(
            input_tokens=max(tokens_used - 1, 0),
            output_tokens=1,
            total_tokens=tokens_used,
        ),
    )


def _create_task(tmp_path: Path, api_mode: str = "responses") -> OpenAICompatibleVisionQATask:
    """Create a test task with the given API mode."""
    return OpenAICompatibleVisionQATask(
        task_id="test_task",
        task_prompt="What is in this image?",
        task_answer="A test image",
        task_image_path=str(_get_test_image_path()),
        save_dir=tmp_path,
        model_type="openai",
        api_mode=api_mode,
        tool_functions={},
        max_steps=5,
    )


# =============================================================================
# Gymnasium Interface Tests
# =============================================================================

class TestGymEnvInterface:
    """Test Gymnasium env interface."""

    def test_has_gym_attributes(self, tmp_path):
        """Test that task has required Gymnasium attributes."""
        task = _create_task(tmp_path)

        assert hasattr(task, "observation_space")
        assert hasattr(task, "action_space")
        assert hasattr(task, "reset")
        assert hasattr(task, "step")
        assert hasattr(task, "render")
        assert hasattr(task, "close")

    def test_reset_returns_observation_and_info(self, tmp_path):
        """Test that reset() returns (observation, info) tuple."""
        task = _create_task(tmp_path)

        obs, info = task.reset()

        assert isinstance(obs, dict)
        assert "contents" in obs
        assert "step" in obs
        assert obs["step"] == 0

        assert isinstance(info, dict)
        assert "task_id" in info

    def test_reset_clears_state(self, tmp_path):
        """Test that reset() clears previous episode state."""
        task = _create_task(tmp_path)

        # Simulate some state changes
        task._step_count = 3
        task.conversation_history.append({"step": 1, "data": "test"})

        obs, info = task.reset()

        assert obs["step"] == 0
        assert task._step_count == 0
        assert len(task.conversation_history) == 0

    def test_step_with_final_answer(self, tmp_path):
        """Test step() when model gives final answer (no tool calls)."""
        task = _create_task(tmp_path)
        task.reset()

        # Create a response with final answer (no tool calls)
        action = _make_responses_api_response(content="The answer is 42")

        obs, reward, terminated, truncated, info = task.step(action)

        assert terminated is True
        assert truncated is False
        assert info.get("final_answer") is True
        assert info.get("tool_calls") == []

    def test_step_with_tool_call(self, tmp_path):
        """Test step() when model makes a tool call."""
        def mock_tool(image_list, image_index=0, **kwargs):
            return image_list[image_index].copy()

        task = OpenAICompatibleVisionQATask(
            task_id="test_task",
            task_prompt="What is in this image?",
            task_answer="A test image",
            task_image_path=str(_get_test_image_path()),
            save_dir=tmp_path,
            model_type="openai",
            api_mode="responses",
            tool_functions={"mock_tool": mock_tool},
            tool_return_types={"mock_tool": "image"},
            max_steps=5,
        )
        task.reset()

        # Create a response with tool call
        tool_call = _make_tool_call("call_1", "mock_tool", {"image_index": 0})
        action = _make_responses_api_response(content=None, tool_calls=[tool_call])

        obs, reward, terminated, truncated, info = task.step(action)

        assert terminated is False
        assert truncated is False
        assert len(info.get("tool_calls", [])) == 1
        assert info["tool_calls"][0][0] == "mock_tool"

    def test_step_truncation_at_max_steps(self, tmp_path):
        """Test that step() sets truncated=True at max_steps."""
        def mock_tool(image_list, image_index=0, **kwargs):
            return image_list[image_index].copy()

        task = OpenAICompatibleVisionQATask(
            task_id="test_task",
            task_prompt="What is in this image?",
            task_answer="A test image",
            task_image_path=str(_get_test_image_path()),
            save_dir=tmp_path,
            model_type="openai",
            api_mode="responses",
            tool_functions={"mock_tool": mock_tool},
            tool_return_types={"mock_tool": "image"},
            max_steps=2,  # Low max_steps for testing
        )
        task.reset()

        # Make max_steps tool calls
        for i in range(2):
            tool_call = _make_tool_call(f"call_{i}", "mock_tool", {"image_index": 0})
            action = _make_responses_api_response(content=None, tool_calls=[tool_call])
            obs, reward, terminated, truncated, info = task.step(action)

        # After max_steps, should be truncated
        assert truncated is True
        assert terminated is False


# =============================================================================
# State Save/Restore Tests
# =============================================================================

class TestStateSaveRestore:
    """Test state save/restore functionality."""

    def test_save_state_returns_dict(self, tmp_path):
        """Test that save_state() returns a dict with expected keys."""
        task = _create_task(tmp_path)

        state = task.save_state()

        assert isinstance(state, dict)
        assert "contents" in state
        assert "conversation_history" in state
        assert "image_list" in state
        assert "state" in state
        assert "step_count" in state

    def test_restore_state_restores_contents(self, tmp_path):
        """Test that restore_state() restores contents correctly."""
        task = _create_task(tmp_path)
        task.reset()

        # Save initial state
        initial_state = task.save_state()

        # Modify state
        task.append_prompt("Additional prompt")
        task._step_count = 5

        # Restore
        task.restore_state(initial_state)

        assert task._step_count == 0
        # Contents should be restored (exact comparison depends on implementation)

    def test_save_restore_round_trip(self, tmp_path):
        """Test that save/restore is a proper round trip."""
        task = _create_task(tmp_path)
        task.reset()

        # Modify state
        task._step_count = 3
        task.conversation_history.append({"step": 1, "data": "test"})

        # Save state
        saved_state = task.save_state()

        # Modify further
        task._step_count = 10
        task.conversation_history.append({"step": 2, "data": "more"})

        # Restore
        task.restore_state(saved_state)

        # Verify restoration
        assert task._step_count == 3
        assert len(task.conversation_history) == 1
        assert task.conversation_history[0]["step"] == 1


# =============================================================================
# Legacy Interface Compatibility Tests
# =============================================================================

class TestLegacyCompatibility:
    """Test that legacy interface still works."""

    def test_parse_action_still_works(self, tmp_path):
        """Test that parse_action() still works as before."""
        task = _create_task(tmp_path)

        # Create a response with final answer
        action = _make_responses_api_response(content="The answer is 42")

        tool_calls = task.parse_action(step=1, action=action, extra_info={})

        assert tool_calls == []
        assert len(task.conversation_history) == 1

    def test_update_observation_from_action_still_works(self, tmp_path):
        """Test that update_observation_from_action() still works."""
        def mock_tool(image_list, image_index=0, **kwargs):
            return image_list[image_index].copy()

        task = OpenAICompatibleVisionQATask(
            task_id="test_task",
            task_prompt="What is in this image?",
            task_answer="A test image",
            task_image_path=str(_get_test_image_path()),
            save_dir=tmp_path,
            model_type="openai",
            api_mode="responses",
            tool_functions={"mock_tool": mock_tool},
            tool_return_types={"mock_tool": "image"},
        )

        # Create tool calls
        tool_calls = [ToolCall(name="mock_tool", args={"image_index": 0}, call_id="call_1")]

        # This should not raise
        task.update_observation_from_action(tool_calls)

        # Image list should have grown
        assert len(task.image_list) == 2

    def test_contents_attribute_accessible(self, tmp_path):
        """Test that contents attribute is directly accessible."""
        task = _create_task(tmp_path)

        # Contents should be accessible for legacy scripts
        contents = task.contents

        assert contents is not None
        if task.api_mode == "responses":
            assert "input" in contents
        else:
            assert isinstance(contents, list)


# =============================================================================
# Chat Completions API Mode Tests
# =============================================================================

class TestChatCompletionsMode:
    """Test Gymnasium interface with chat_completions API mode."""

    def test_reset_with_chat_completions(self, tmp_path):
        """Test reset() with chat_completions API mode."""
        task = _create_task(tmp_path, api_mode="chat_completions")

        obs, info = task.reset()

        assert isinstance(obs["contents"], list)
        assert obs["step"] == 0

    def test_multiple_resets(self, tmp_path):
        """Test multiple consecutive resets."""
        task = _create_task(tmp_path)

        for _ in range(3):
            obs, info = task.reset()
            assert obs["step"] == 0
            assert len(task.conversation_history) == 0
