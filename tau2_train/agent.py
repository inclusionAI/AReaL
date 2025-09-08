from copy import deepcopy
from typing import List, Optional
from areal.api.io_struct import ModelRequest
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.managers.io_struct import Tool, Function

from pydantic import BaseModel
import uuid
import json
from dataclasses import dataclass, asdict


from .data_model.message import (
    APICompatibleMessage,
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
    ToolCall,
)
from .utils.llm_utils import to_litellm_messages

@dataclass
class Record:
    text: str
    # for webpage and search results
    # RL data
    input_len: Optional[int] = None
    input_tokens: Optional[List[int]] = None
    output_len: Optional[int] = None
    output_tokens: Optional[List[int]] = None
    output_logprobs: Optional[List[float]] = None
    output_versions: Optional[List[int]] = None

AGENT_INSTRUCTION = """
You are a customer service agent that helps the user according to the <policy> provided below.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.
""".strip()

SYSTEM_PROMPT = """
<instructions>
{agent_instruction}
</instructions>
<policy>
{domain_policy}
</policy>
""".strip()

class LLMAgentState(BaseModel):
    """The state of the agent."""

    system_messages: list[SystemMessage]
    messages: list[APICompatibleMessage]

class LLMAgent:
    """
    An LLM agent that can be used to solve a task.
    """

    def __init__(
        self,
        llm_engine,
        tokenizer,
        gconfig,
        domain_policy,
        tools,
        max_context_length = 32768
    ):
        """
        Initialize the LLMAgent.
        """
        self.llm_engine = llm_engine
        self.traj_rid = uuid.uuid4().hex
        self.tokenizer = tokenizer
        self.gconfig = gconfig
        self.records = []

        self.domain_policy = domain_policy
        self.tools = tools
        self.max_context_length = max_context_length
        self.stop = False

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT.format(
            domain_policy=self.domain_policy, agent_instruction=AGENT_INSTRUCTION
        )

    def is_stop(self, message):
        return self.stop

    def get_init_state(
        self, message_history: Optional[list] = None
    ):
        """Get the initial state of the agent.

        Args:
            message_history: The message history of the conversation.

        Returns:
            The initial state of the agent.
        """
        if message_history is None:
            message_history = []

        return LLMAgentState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=message_history,
        )
    
    def convert_dict_to_tool(self, tool_dict: dict) -> Tool:
        function_dict = tool_dict.get("function", {})
        return Tool(
            type=tool_dict.get("type", "function"),
            function=Function(
                name=function_dict.get("name"),
                description=function_dict.get("description"),
                parameters=function_dict.get("parameters"),
            ),
        )

    async def generate_next_message(
        self, message, state: LLMAgentState
    ) -> tuple[AssistantMessage, LLMAgentState]:
        """
        Respond to a user or tool message.
        """
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)
        messages = state.system_messages + state.messages
        messages = to_litellm_messages(messages)

        tools = [tool.openai_schema for tool in self.tools] if self.tools else None
        input_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tools=tools,
                toeknize=True,
            )
        max_new_tokens = min(self.gconfig.max_new_tokens, self.max_context_length - len(input_ids) - 1)
        if max_new_tokens <= 0:
            self.stop = True
            assistant_message = AssistantMessage(
                role="assistant",
                content="",
                tool_calls=None,
                raw_data=None,
            )
            state.messages.append(assistant_message)
            return assistant_message, state
        
        req = ModelRequest(
                rid=self.traj_rid,
                input_ids=input_ids,
                gconfig=self.gconfig.new(n_samples=1, max_new_tokens=max_new_tokens),
            )

        resp = await self.llm_engine.agenerate(req)
        completion_str = self.tokenizer.decode(resp.output_tokens)
        
        self.records.append(
                Record(
                    text=completion_str,
                    input_len=resp.input_len,
                    input_tokens=resp.input_tokens,
                    output_len=resp.output_len,
                    output_tokens=resp.output_tokens,
                    output_logprobs=resp.output_logprobs,
                    output_versions=resp.output_versions  
                )
            )

        tools = [self.convert_dict_to_tool(raw_tool) for raw_tool in tools] if tools else None
        
        parser = FunctionCallParser(tools=tools, tool_call_parser="qwen25")
        completion_str = completion_str.split("</think>")[-1]
        try:
            normal_text, calls = parser.parse_non_stream(completion_str)
        except Exception as e:
            print("[deubg]: sglang tool parse ", completion_str)
            raise e

        tool_calls = []
        for single_call in calls:
            try:
                parameters = single_call.parameters
                while isinstance(parameters, str):
                    parameters = json.loads(parameters)
            except:
                parameters = {}
            tool_calls.append(
                        ToolCall(
                            id=str(single_call.tool_index),
                            name=single_call.name,
                            arguments=parameters,
                        )   
                    )

        assistant_message = AssistantMessage(
            role="assistant",
            content=normal_text,
            tool_calls=tool_calls if len(tool_calls) > 0 else None,
            raw_data=None,
        )

        state.messages.append(assistant_message)
        return assistant_message, state

