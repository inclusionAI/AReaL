import uuid
from typing import Any, Dict, List, Optional

from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)

from realhf.base import logging

logger = logging.getLogger("Tool Call Parser")


# Copied from sglang
def process_tool_calls(
    text: str,
    tools: List[Any],
    tool_call_parser: Optional[str],
    finish_reason: Dict[str, Any],
) -> tuple[Optional[List[ChatCompletionMessageToolCall]], str, Dict[str, Any]]:
    """Process tool calls in the response"""
    from sglang.srt.function_call.function_call_parser import FunctionCallParser

    parser = FunctionCallParser(tools, tool_call_parser)
    if parser.has_tool_call(text):
        if finish_reason["type"] == "stop":
            finish_reason["type"] = "tool_calls"
            finish_reason["matched"] = None
        try:
            text, call_info_list = parser.parse_non_stream(text)
            tool_calls = [
                ChatCompletionMessageToolCall(
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
            # Return error but don't fail the whole request
            return None, text, finish_reason

    return None, text, finish_reason
