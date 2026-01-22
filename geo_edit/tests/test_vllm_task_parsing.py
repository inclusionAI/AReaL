from types import SimpleNamespace

from PIL import Image

from ..environment.task.vllm_vision_qa_task import VLLMVisionQATask


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

    content = (
        "<think>Need to label.</think>"
        "<action>{\"name\":\"image_label\",\"arguments\":{\"image_index\":0,\"text\":\"x\",\"position\":\"(1,2)\"}}</action>"
    )
    message = SimpleNamespace(
        content=content,
        tool_calls=None,
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
    assert tool_calls[0].args["text"] == "x"
    assert tool_calls[0].call_id == "call_1_1"
    assert task.messages[-1]["role"] == "assistant"
    assert task.conversation_history[0]["function_call"][0][0] == "image_label"
    assert task.conversation_history[0]["thinking_process"] == "Need to label."
    assert "<action>" not in task.conversation_history[0]["output_text"]
