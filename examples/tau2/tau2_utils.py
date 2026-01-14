import json
from dataclasses import dataclass, field

import yaml
from loguru import logger
from openai.types.chat import ChatCompletion
from pydantic import BaseModel
from tau2.data_model.message import AssistantMessage, Message, ToolCall
from tau2.data_model.simulation import RewardInfo
from tau2.data_model.tasks import Task
import tau2.utils.llm_utils


def _get_message_from_completion(completion: ChatCompletion) -> AssistantMessage:
    """Convert a ChatCompletion (from OpenAI SDK / ArealOpenAI) to AssistantMessage.

    This is a replacement for tau2.utils.llm_utils._get_message_from_response
    that works with ChatCompletion instead of litellm's ModelResponse.

    Args:
        completion: ChatCompletion object from ArealOpenAI client

    Returns:
        AssistantMessage: Tau2 message format
    """
    # Extract usage information
    usage = None
    if completion.usage is not None:
        usage = {
            "completion_tokens": completion.usage.completion_tokens,
            "prompt_tokens": completion.usage.prompt_tokens,
        }

    # Get the first choice
    choice = completion.choices[0]

    # Check finish reason
    try:
        finish_reason = choice.finish_reason
        if finish_reason == "length":
            logger.warning("Output might be incomplete due to token limit!")
    except Exception as e:
        logger.error(e)
        raise e

    assert choice.message.role == "assistant", (
        "The response should be an assistant message"
    )

    content = choice.message.content
    tool_calls_raw = choice.message.tool_calls or []
    tool_calls = [
        ToolCall(
            id=tool_call.id,
            name=tool_call.function.name,
            arguments=json.loads(tool_call.function.arguments),
        )
        for tool_call in tool_calls_raw
    ]
    tool_calls = tool_calls or None

    message = AssistantMessage(
        role="assistant",
        content=content,
        tool_calls=tool_calls,
        cost=0.0,  # No cost tracking for ArealOpenAI
        usage=usage,
        raw_data=completion.model_dump(),
    )
    return message


# Patch tau2.utils.llm_utils._get_message_from_response with our implementation
tau2.utils.llm_utils._get_message_from_response = _get_message_from_completion



class Tau2RunInfo(BaseModel):
    reward: float
    agent_time: list[float]
    user_time: list[float]
    messages: list[Message]
    task: Task
    reward_info: RewardInfo | None = None
    error: str | None = None

    def __str__(self):
        s = f"[REWARD]: {self.reward}\n\n"
        s += "[TASK]\n"
        s += yaml.dump(self.task.model_dump()) + "\n"
        if self.reward_info:
            s += "[REWARD_INFO]\n"
            s += yaml.dump(self.reward_info.model_dump()) + "\n"
        s += f"[TURNS COUNT]: {len(self.messages)}\n"
        s += "[MESSAGES]\n"
        for message in self.messages:
            turn_idx = message.turn_idx
            role = message.role
            content = message.content or ""
            usage = getattr(message, "usage", {})
            tool_calls = getattr(message, "tool_calls", None)
            if tool_calls:
                content += "\n[TOOL_CALLS]\n"
                content += yaml.dump(
                    [tool_call.model_dump() for tool_call in tool_calls]
                )
            s += f"[{turn_idx}][{role}]: {content}\n"
            if usage:
                s += f"[{turn_idx}][{role}][USAGE]: {yaml.dump(usage)}\n"
        if len(self.agent_time):
            s += f"[AGENT_TIME]: total {sum(self.agent_time)}, avg {sum(self.agent_time) / len(self.agent_time)}\n"
        if len(self.user_time):
            s += f"[USER_TIME]: total {sum(self.user_time)}, avg {sum(self.user_time) / len(self.user_time)}\n"
        if self.error:
            s += f"[ERROR]: {self.error}\n"
        return s


# ================================ config ================================
# Customized config for tau2, add env config
@dataclass
class Tau2EnvConfig:
    domain: str = field(
        default="telecom",
        metadata={
            "help": "The tau2 domain name, e.g., 'retail', 'airline', 'telecom'."
        },
    )
    max_steps: int = field(
        default=100, metadata={"help": "Maximum number of steps per episode."}
    )
    add_thinking_tool: bool = field(
        default=True, metadata={"help": "Whether to add a thinking tool."}
    )
    solo_mode: bool = field(
        default=False, metadata={"help": "Whether to use solo mode."}
    )
    user_llm_base_url: str | None = field(
        default=None,
        metadata={"help": "The base URL of the user LLM."},
    )
    user_llm: str | None = field(
        default=None,
        metadata={"help": "The user LLM to use, default to the gpt-4.1 model."},
    )
    user_llm_args: dict | None = field(
        default=None, metadata={"help": "The arguments for the user LLM."}
    )
    turn_discount: float = field(
        default=1.0, metadata={"help": "Discount factor for turn-based learning."}
    )
    invalid_format_penalty: float = field(
        default=0.1, metadata={"help": "Penalty for invalid format in completions."}
    )
