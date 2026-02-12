"""Tests for OpenAICompatibleVisionQATask.

This module tests the unified task class that supports:
- OpenAI (responses API)
- vLLM (responses API)
- SGLang (chat_completions API)
"""

import json
from pathlib import Path
from types import SimpleNamespace

from openai.types.responses.response_create_params import ResponseCreateParamsNonStreaming
from pydantic import TypeAdapter

from geo_edit.config import build_api_agent_configs
from geo_edit.environment.action.image_edition_tool import draw_line_function
from geo_edit.environment.task.openai_compatible_vision_qa_task import OpenAICompatibleVisionQATask
from geo_edit.environment.task.vision_qa_task import ToolCall
from unittest.mock import MagicMock


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
        id="resp_123",
    )


def _make_chat_completions_response(content: str | None, tool_calls=None, tokens_used: int = 5):
    """Create a mock response object for chat_completions API format."""
    message = SimpleNamespace(
        content=content,
        tool_calls=[
            SimpleNamespace(
                id=tc.id,
                type="function",
                function=SimpleNamespace(
                    name=tc.name,
                    arguments=tc.arguments,
                ),
            )
            for tc in (tool_calls or [])
        ] if tool_calls else None,
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(
            prompt_tokens=max(tokens_used - 1, 0),
            completion_tokens=1,
            total_tokens=tokens_used,
        ),
        id="chatcmpl_123",
    )


def _build_task(tmp_path, model_type: str = "openai", api_mode: str = "responses", tool_functions=None):
    """Build a task instance for testing."""
    if tool_functions is None:
        tool_functions = {"draw_line": draw_line_function}
    image_path = _get_test_image_path()
    return OpenAICompatibleVisionQATask(
        task_id=f"{model_type}-task-1",
        task_prompt="Question?",
        task_answer="A",
        task_image_path=str(image_path),
        save_dir=tmp_path / "out",
        tool_functions=tool_functions,
        model_type=model_type,
        api_mode=api_mode,
    )


# =============================================================================
# Tests for Input Image Detail (all model types)
# =============================================================================

class TestInputImageDetail:
    """Tests that input images have the correct 'detail' field."""

    def test_openai_initial_input_image_has_detail_and_matches_schema(self, tmp_path):
        """OpenAI responses API: initial input image should have detail='auto'."""
        task = _build_task(tmp_path, model_type="openai", api_mode="responses")
        user_message = next(item for item in task.contents["input"] if item.get("role") == "user")
        image_part = next(part for part in user_message["content"] if part.get("type") == "input_image")
        assert image_part["detail"] == "auto"
        TypeAdapter(ResponseCreateParamsNonStreaming).validate_python(
            {"model": "dummy-model", "input": task.contents["input"]}
        )

    def test_vllm_initial_input_image_has_detail_and_matches_schema(self, tmp_path):
        """vLLM responses API: initial input image should have detail='auto'."""
        task = _build_task(tmp_path, model_type="vllm", api_mode="responses")
        user_message = next(item for item in task.contents["input"] if item.get("role") == "user")
        image_part = next(part for part in user_message["content"] if part.get("type") == "input_image")
        assert image_part["detail"] == "auto"
        TypeAdapter(ResponseCreateParamsNonStreaming).validate_python(
            {"model": "dummy-model", "input": task.contents["input"]}
        )

    def test_sglang_initial_input_image_has_detail(self, tmp_path):
        """SGLang chat_completions API: initial input image should have detail='auto'."""
        task = _build_task(tmp_path, model_type="sglang", api_mode="chat_completions")
        user_message = next(item for item in task.contents["messages"] if item.get("role") == "user")
        image_part = next(part for part in user_message["content"] if part.get("type") == "image_url")
        assert image_part["image_url"]["detail"] == "auto"

    def test_openai_tool_observation_image_has_detail(self, tmp_path):
        """OpenAI responses API: tool observation images should have detail='auto'."""
        task = _build_task(tmp_path, model_type="openai", api_mode="responses")
        tool_calls = [
            ToolCall(
                name="draw_line",
                args={"image_index": 0, "coordinates": "\\boxed{10,20,30,40}"},
                call_id="call_openai_1",
            )
        ]
        task.update_observation_from_action(tool_calls)
        tool_result_messages = [
            message for message in task.contents["input"]
            if message.get("type") == "function_call_output"
        ]
        assert tool_result_messages
        tool_output = tool_result_messages[-1]["output"]
        image_part = next(part for part in tool_output if part.get("type") == "input_image")
        assert image_part["detail"] == "auto"

    def test_vllm_tool_observation_image_has_detail(self, tmp_path):
        """vLLM responses API: tool observation images should have detail='auto'."""
        task = _build_task(tmp_path, model_type="vllm", api_mode="responses")
        tool_calls = [
            ToolCall(
                name="draw_line",
                args={"image_index": 0, "coordinates": "\\boxed{10,20,30,40}"},
                call_id="call_vllm_1",
            )
        ]
        task.update_observation_from_action(tool_calls)
        tool_result_messages = [
            message for message in task.contents["input"]
            if message.get("type") == "function_call_output"
        ]
        assert tool_result_messages
        tool_output = tool_result_messages[-1]["output"]
        image_part = next(part for part in tool_output if part.get("type") == "input_image")
        assert image_part["detail"] == "auto"

    def test_sglang_tool_observation_image_has_detail(self, tmp_path):
        """SGLang chat_completions API: tool observation images should have detail='auto'."""
        task = _build_task(tmp_path, model_type="sglang", api_mode="chat_completions")
        tool_calls = [
            ToolCall(
                name="draw_line",
                args={"image_index": 0, "coordinates": "\\boxed{10,20,30,40}"},
                call_id="call_sglang_1",
            )
        ]
        task.update_observation_from_action(tool_calls)
        tool_result_messages = [
            message for message in task.contents["messages"]
            if message.get("role") == "tool"
        ]
        assert tool_result_messages
        tool_content = tool_result_messages[-1]["content"]
        image_part = next(part for part in tool_content if part.get("type") == "image_url")
        assert image_part["image_url"]["detail"] == "auto"


# =============================================================================
# Tests for Responses API Parsing (OpenAI, vLLM)
# =============================================================================

class TestResponsesApiParsing:
    """Tests for parsing responses API format (OpenAI and vLLM)."""

    def test_vllm_task_parses_tool_calls(self, tmp_path):
        """vLLM responses API: should correctly parse tool calls with <think> tags."""
        image_path = _get_test_image_path()
        task = OpenAICompatibleVisionQATask(
            task_id="task-1",
            task_prompt="Question?",
            task_answer="A",
            task_image_path=str(image_path),
            save_dir=tmp_path / "out",
            tool_functions={"draw_line": draw_line_function},
            model_type="vllm",
            api_mode="responses",
        )

        arguments = {"image_index": 0, "coordinates": "\\boxed{10,20,30,40}"}
        native_tool_calls = [_make_tool_call("call_abc123", "draw_line", arguments)]
        content = "<think>Need to draw a line.</think>"
        response = _make_responses_api_response(content, tool_calls=native_tool_calls)
        tool_calls = task.parse_action(
            step=1,
            action=response,
            extra_info={"tokens_used": 5},
        )

        assert len(tool_calls) == 1
        assert tool_calls[0].name == "draw_line"
        assert tool_calls[0].args == arguments
        assert tool_calls[0].call_id == "call_abc123"
        assert task.contents["input"][-1]["type"] == "function_call"
        assert task.conversation_history[0]["function_call"][0][0] == "draw_line"
        assert task.conversation_history[0]["thinking_process"] == "Need to draw a line."
        assert task.conversation_history[0]["output_text"] == ""
        assert task.conversation_history[0]["action"]["tool_calls"][0]["args"] == arguments

        observation = task.conversation_history[0]["observation"]
        user_message = next(item for item in observation if item.get("role") == "user")
        image_part = next(part for part in user_message["content"] if part.get("type") == "input_image")
        assert image_part["image_path"] == str(image_path)

        task.update_observation_from_action(tool_calls)
        assert len(task.image_list) == 2
        tool_result_messages = [
            message for message in task.contents["input"]
            if message.get("type") == "function_call_output"
        ]
        assert tool_result_messages
        tool_output = tool_result_messages[-1]["output"]
        tool_image_part = next(part for part in tool_output if part.get("type") == "input_image")
        assert tool_image_part["detail"] == "auto"

        final_content = "<think>Done.</think><answer>Final answer.</answer>"
        final_response = _make_responses_api_response(final_content, tokens_used=7)
        final_tool_calls = task.parse_action(
            step=2,
            action=final_response,
            extra_info={"tokens_used": 7},
        )
        assert final_tool_calls == []
        assert task.conversation_history[1]["output_text"] == "Final answer."
        assert task.conversation_history[1]["function_call"] is None

    def test_openai_task_parses_tool_calls(self, tmp_path):
        """OpenAI responses API: should correctly parse tool calls."""
        task = _build_task(tmp_path, model_type="openai", api_mode="responses")

        arguments = {"image_index": 0, "coordinates": "\\boxed{10,20,30,40}"}
        native_tool_calls = [_make_tool_call("call_openai_1", "draw_line", arguments)]
        response = _make_responses_api_response("Analyzing the image...", tool_calls=native_tool_calls)
        tool_calls = task.parse_action(
            step=1,
            action=response,
            extra_info={"tokens_used": 10},
        )

        assert len(tool_calls) == 1
        assert tool_calls[0].name == "draw_line"
        assert tool_calls[0].args == arguments
        assert tool_calls[0].call_id == "call_openai_1"


# =============================================================================
# Tests for Chat Completions API Parsing (SGLang)
# =============================================================================

class TestChatCompletionsApiParsing:
    """Tests for parsing chat_completions API format (SGLang)."""

    def test_sglang_task_parses_tool_calls(self, tmp_path):
        """SGLang chat_completions API: should correctly parse tool calls with <think> tags."""
        task = _build_task(tmp_path, model_type="sglang", api_mode="chat_completions")

        arguments = {"image_index": 0, "coordinates": "\\boxed{10,20,30,40}"}
        native_tool_calls = [_make_tool_call("call_sglang_1", "draw_line", arguments)]
        content = "<think>Need to draw a line.</think>"
        response = _make_chat_completions_response(content, tool_calls=native_tool_calls)
        tool_calls = task.parse_action(
            step=1,
            action=response,
            extra_info={"tokens_used": 5},
        )

        assert len(tool_calls) == 1
        assert tool_calls[0].name == "draw_line"
        assert tool_calls[0].args == arguments
        assert tool_calls[0].call_id == "call_sglang_1"
        assert task.conversation_history[0]["thinking_process"] == "Need to draw a line."

    def test_sglang_task_parses_final_answer(self, tmp_path):
        """SGLang chat_completions API: should correctly parse final answer with <answer> tags."""
        task = _build_task(tmp_path, model_type="sglang", api_mode="chat_completions")

        content = "<think>Let me think about this.</think><answer>The answer is B.</answer>"
        response = _make_chat_completions_response(content, tool_calls=None)
        tool_calls = task.parse_action(
            step=1,
            action=response,
            extra_info={"tokens_used": 5},
        )

        assert tool_calls == []
        assert task.conversation_history[0]["thinking_process"] == "Let me think about this."
        assert task.conversation_history[0]["output_text"] == "The answer is B."

    def test_sglang_message_format(self, tmp_path):
        """SGLang chat_completions API: should use correct message format."""
        task = _build_task(tmp_path, model_type="sglang", api_mode="chat_completions")

        # Chat completions format uses "messages" key
        assert "messages" in task.contents
        assert "input" not in task.contents

        # Check message structure
        messages = task.contents["messages"]
        user_message = next(m for m in messages if m.get("role") == "user")
        assert "content" in user_message

        # Image should use image_url format
        image_part = next(p for p in user_message["content"] if p.get("type") == "image_url")
        assert "image_url" in image_part
        assert "url" in image_part["image_url"]


# =============================================================================
# Tests for Direct Mode (No Tools)
# =============================================================================

class TestDirectMode:
    """Tests for direct mode (no tool calls)."""

    def test_vllm_direct_initial_input_matches_responses_schema(self, tmp_path):
        """vLLM direct mode: should not include tool-related fields."""
        image_path = _get_test_image_path()
        task = OpenAICompatibleVisionQATask(
            task_id="task-direct-1",
            task_prompt="Question?",
            task_answer="A",
            task_image_path=str(image_path),
            save_dir=tmp_path / "out",
            tool_functions={},
            model_type="vllm",
            api_mode="responses",
        )

        user_message = next(item for item in task.contents["input"] if item.get("role") == "user")
        image_part = next(part for part in user_message["content"] if part.get("type") == "input_image")
        assert image_part["detail"] == "auto"

        assert not any(
            item.get("type") in {"function_call", "function_call_output"}
            for item in task.contents["input"]
            if isinstance(item, dict)
        )

        TypeAdapter(ResponseCreateParamsNonStreaming).validate_python(
            {"model": "dummy-model", "input": task.contents["input"]}
        )

    def test_sglang_direct_initial_input_format(self, tmp_path):
        """SGLang direct mode: should not include tool-related fields."""
        task = _build_task(tmp_path, model_type="sglang", api_mode="chat_completions", tool_functions={})

        assert "messages" in task.contents
        messages = task.contents["messages"]

        # Should not have any tool-related messages
        assert not any(m.get("role") == "tool" for m in messages)
        assert not any(m.get("tool_calls") for m in messages)

    def test_vllm_direct_mode_omits_tool_fields_in_generate_config(self):
        """vLLM direct mode: generate config should not include tool fields."""
        def _create_mock_router(tool_mode: str, declarations: list, tools: dict):
            router = MagicMock()
            router.tool_mode = tool_mode
            router.get_available_declarations.return_value = declarations
            router.get_available_tools.return_value = tools
            return router

        direct_router = _create_mock_router(tool_mode="direct", declarations=[], tools={})
        direct_configs = build_api_agent_configs(
            direct_router,
            api_mode="responses",
            max_output_tokens=None,
            temperature=0.7,
            system_prompt="sys",
        )
        assert "tools" not in direct_configs.generate_config
        assert "tool_choice" not in direct_configs.generate_config
        assert "tools" not in direct_configs.force_final_generate_config
        assert "tool_choice" not in direct_configs.force_final_generate_config

        auto_router = _create_mock_router(
            tool_mode="auto",
            declarations=[{"name": "test_tool", "description": "Test", "parameters": {"type": "object"}}],
            tools={"test_tool": lambda: None},
        )
        auto_configs = build_api_agent_configs(
            auto_router,
            api_mode="responses",
            max_output_tokens=128,
            temperature=0.7,
            system_prompt="sys",
        )
        assert auto_configs.generate_config["tool_choice"] == "auto"
        assert isinstance(auto_configs.generate_config["tools"], list)