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


# Modified from sglang
def process_tool_calls(
    text: str,
    tools: list[Any],
    tool_call_parser: str | None,
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

    parser = FunctionCallParser(tools, tool_call_parser)
    if parser.has_tool_call(text):
        if finish_reason == "stop":
            finish_reason = "tool_calls"
        try:
            text, call_info_list = parser.parse_non_stream(text)

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
            return tool_calls, text, finish_reason
        except Exception as e:
            logger.error(f"Tool call parsing error: {e}")
            traceback.print_exc()
            # Return error but don't fail the whole request
            return None, text, finish_reason

    return None, text, finish_reason
