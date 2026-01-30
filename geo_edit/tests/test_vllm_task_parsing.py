import json
from pathlib import Path
from types import SimpleNamespace

from geo_edit.environment.action.image_edition_tool import draw_line_function
from geo_edit.environment.task.vllm_vision_qa_task import VLLMVisionQATask


def _make_tool_call(call_id: str, name: str, arguments: dict):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(
            name=name,
            arguments=json.dumps(arguments),
        ),
    )


def _make_response(content: str, tool_calls=None, tokens_used: int = 5):
    message = SimpleNamespace(
        content=content,
        tool_calls=tool_calls,
        role="assistant",
    )
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(
        choices=[choice],
        usage=SimpleNamespace(total_tokens=tokens_used),
    )


def test_vllm_task_parses_tool_calls(tmp_path):
    image_path = Path(__file__).resolve().parents[1] / "images" / "input_image.png"
    task = VLLMVisionQATask(
        task_id="task-1",
        task_prompt="Question?",
        task_answer="A",
        task_image_path=str(image_path),
        save_dir=tmp_path / "out",
        tool_functions={"draw_line": draw_line_function},
        system_prompt="sys",
    )

    arguments = {"image_index": 0, "coordinates": "\\boxed{10,20,30,40}"}
    native_tool_calls = [
        _make_tool_call("call_abc123", "draw_line", arguments)
    ]
    content = "<think>Need to draw a line.</think>"
    response = _make_response(content, tool_calls=native_tool_calls)
    tool_calls = task.parse_action(
        step=1,
        action=response,
        extra_info={"tokens_used": 5},
    )

    assert len(tool_calls) == 1
    assert tool_calls[0].name == "draw_line"
    assert tool_calls[0].args == arguments
    assert tool_calls[0].call_id == "call_abc123"
    assert task.contents[-1]["role"] == "assistant"
    assert task.conversation_history[0]["function_call"][0][0] == "draw_line"
    assert task.conversation_history[0]["thinking_process"] == "Need to draw a line."
    assert task.conversation_history[0]["output_text"] == ""
    assert task.conversation_history[0]["action"]["tool_calls"][0]["args"] == arguments

    observation = task.conversation_history[0]["observation"]
    user_message = next(item for item in observation if item.get("role") == "user")
    image_part = next(
        part for part in user_message["content"] if part.get("type") == "image_url"
    )
    assert image_part["image_path"] == str(image_path)

    task.update_observation_from_action(tool_calls)
    assert len(task.image_list) == 2
    tool_result_messages = [
        message
        for message in task.contents
        if message.get("role") == "tool"
    ]
    assert tool_result_messages

    final_content = "<think>Done.</think><answer>Final answer.</answer>"
    final_response = _make_response(final_content, tokens_used=7)
    final_tool_calls = task.parse_action(
        step=2,
        action=final_response,
        extra_info={"tokens_used": 7},
    )
    assert final_tool_calls == []
    assert task.conversation_history[1]["output_text"] == "Final answer."
    assert task.conversation_history[1]["function_call"] is None
