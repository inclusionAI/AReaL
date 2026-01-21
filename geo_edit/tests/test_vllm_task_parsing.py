import json
from types import SimpleNamespace

from PIL import Image

from ..environment.task.vllm_vision_qa_task import VLLMVisionQATask


def _make_tool_call(name: str, arguments: dict, call_id: str = "call_1"):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


def _make_response(message, tokens_used: int = 5):
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(
        choices=[choice],
        usage=SimpleNamespace(total_tokens=tokens_used),
    )


def test_vllm_task_parses_tool_calls(tmp_path):
    image_path = tmp_path / "input.png"
    Image.new("RGB", (8, 8), color="white").save(image_path)

    task = VLLMVisionQATask(
        task_id="task-1",
        task_prompt="Question?",
        task_answer="A",
        task_image_path=str(image_path),
        save_dir=tmp_path / "out",
        tool_functions={},
        system_prompt="sys",
    )

    tool_call = _make_tool_call(
        "image_label",
        {"image_index": 0, "text": "x", "position": "(1,2)"},
    )
    message = SimpleNamespace(
        content="tool call",
        tool_calls=[tool_call],
        role="assistant",
    )
    response = _make_response(message)

    tool_calls = task.parse_action(
        step=1,
        action=response,
        extra_info={"tokens_used": 5},
    )

    assert len(tool_calls) == 1
    assert tool_calls[0].name == "image_label"
    assert tool_calls[0].args["image_index"] == 0
    assert task.messages[-1]["role"] == "assistant"
    assert task.conversation_history[0]["function_call"][0][0] == "image_label"
