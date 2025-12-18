import traceback
import uuid
from typing import Any

from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
    Function,
)
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall

from areal.utils import logging

logger = logging.getLogger("Tool Call Parser")


def _detect_think_and_return_ori_think(
    text: str, think_start_token: str, think_end_token: str
) -> tuple[str, str]:
    """
    return think text(with <think> and </think>) and normal text
    """
    in_reasoning = think_start_token in text

    if not in_reasoning:
        return "", text

    # The text is considered to be in a reasoning block.
    processed_text = text.replace(think_start_token, "")

    if think_end_token not in processed_text:
        # Assume reasoning was truncated before `</think>` token
        return think_start_token + processed_text, ""

    # Extract reasoning content
    splits = processed_text.split(think_end_token, maxsplit=1)
    reasoning_text = splits[0]
    normal_text = splits[1]

    return think_start_token + reasoning_text + think_end_token, normal_text


# Modified from sglang
def process_tool_calls(
    text: str,
    tools: list[Any],
    tool_call_parser: str,
    reasoning_parser: str,
    finish_reason: str,
    use_responses: bool = False,
) -> tuple[
    list[ChatCompletionMessageFunctionToolCall | ResponseFunctionToolCall] | None,
    str,
    str,
]:
    """Process tool calls in the response"""
    from sglang.srt.entrypoints.openai.protocol import Function as SglFunction
    from sglang.srt.entrypoints.openai.protocol import Tool as SglTool
    from sglang.srt.function_call.function_call_parser import FunctionCallParser
    from sglang.srt.parser.reasoning_parser import ReasoningParser

    if use_responses:
        tools = [
            SglTool(
                type=tool["type"],
                function=SglFunction(
                    name=tool.get("name"),
                    description=tool.get("description"),
                    parameters=tool.get("parameters"),
                ),
            )
            for tool in tools
        ]
    else:
        tools = [
            SglTool(type=tool["type"], function=SglFunction(**tool["function"]))
            for tool in tools
        ]

    parser_p = FunctionCallParser(tools, tool_call_parser)
    reasoning_parser_p = ReasoningParser(reasoning_parser)

    reasoning_text, content_text = _detect_think_and_return_ori_think(
        text,
        reasoning_parser_p.detector.think_start_token,
        reasoning_parser_p.detector.think_end_token,
    )

    if parser_p.has_tool_call(content_text):
        if finish_reason == "stop":
            finish_reason = "tool_calls"
        try:
            content_text, call_info_list = parser_p.parse_non_stream(content_text)

            if use_responses:
                tool_calls = [
                    ResponseFunctionToolCall(
                        type="function_call",
                        id=f"fc-{uuid.uuid4().hex[:24]}",
                        call_id=f"call_{uuid.uuid4().hex[:24]}",
                        name=call_info.name,
                        arguments=call_info.parameters,
                        status="completed",
                    )
                    for call_info in call_info_list
                ]
            else:
                tool_calls = [
                    ChatCompletionMessageFunctionToolCall(
                        type="function",
                        id=f"call_{uuid.uuid4().hex[:24]}",
                        function=Function(
                            name=call_info.name, arguments=call_info.parameters
                        ),
                    )
                    for call_info in call_info_list
                ]

            return tool_calls, reasoning_text + content_text, finish_reason
        except Exception as e:
            logger.error(f"Tool call parsing error: {e}")
            traceback.print_exc()
            # Return error but don't fail the whole request
            return None, text, finish_reason

    return None, text, finish_reason
