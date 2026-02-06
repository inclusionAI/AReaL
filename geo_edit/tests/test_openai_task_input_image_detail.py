from pathlib import Path

from openai.types.responses.response_create_params import ResponseCreateParamsNonStreaming
from pydantic import TypeAdapter

from geo_edit.environment.action.image_edition_tool import draw_line_function
from geo_edit.environment.task.openai_vision_qa_task import OpenAIVisionQATask
from geo_edit.environment.task.vision_qa_task import ToolCall


def _build_task(tmp_path):
    image_path = Path(__file__).resolve().parents[1] / "images" / "input_image.png"
    return OpenAIVisionQATask(
        task_id="openai-task-1",
        task_prompt="Question?",
        task_answer="A",
        task_image_path=str(image_path),
        save_dir=tmp_path / "out",
        tool_functions={"draw_line": draw_line_function},
    )


def test_openai_initial_input_image_has_detail_and_matches_schema(tmp_path):
    task = _build_task(tmp_path)
    user_message = next(item for item in task.contents["input"] if item.get("role") == "user")
    image_part = next(part for part in user_message["content"] if part.get("type") == "input_image")
    assert image_part["detail"] == "auto"
    TypeAdapter(ResponseCreateParamsNonStreaming).validate_python(
        {"model": "dummy-model", "input": task.contents["input"]}
    )


def test_openai_tool_observation_image_has_detail(tmp_path):
    task = _build_task(tmp_path)
    tool_calls = [
        ToolCall(
            name="draw_line",
            args={"image_index": 0, "coordinates": "\\boxed{10,20,30,40}"},
            call_id="call_openai_1",
        )
    ]
    task.update_observation_from_action(tool_calls)
    tool_result_messages = [
        message
        for message in task.contents["input"]
        if message.get("type") == "function_call_output"
    ]
    assert tool_result_messages
    tool_output = tool_result_messages[-1]["output"]
    image_part = next(part for part in tool_output if part.get("type") == "input_image")
    assert image_part["detail"] == "auto"
