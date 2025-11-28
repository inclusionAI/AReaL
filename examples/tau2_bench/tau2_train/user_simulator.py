from typing import Optional, Tuple
from pydantic import BaseModel
from .data_model.message import (
    APICompatibleMessage,
    AssistantMessage,
    Message,
    MultiToolMessage,
    ToolMessage,
    SystemMessage,
    ToolCall,
    UserMessage,
)

from .data_model.tasks import UserInstructions
from .environment.tool import Tool

from .utils import DATA_DIR
from .utils.llm_utils import to_litellm_messages
from openai import AsyncOpenAI, OpenAI, RateLimitError, APIError, BadRequestError
import time

from loguru import logger

GLOBAL_USER_SIM_GUIDELINES_DIR = DATA_DIR / "tau2" / "user_simulator"


GLOBAL_USER_SIM_GUIDELINES_PATH = (
    GLOBAL_USER_SIM_GUIDELINES_DIR / "simulation_guidelines.md"
)

GLOBAL_USER_SIM_GUIDELINES_PATH_TOOLS = (
    GLOBAL_USER_SIM_GUIDELINES_DIR / "simulation_guidelines_tools.md"
)

STOP = "###STOP###"
TRANSFER = "###TRANSFER###"
OUT_OF_SCOPE = "###OUT-OF-SCOPE###"


def get_global_user_sim_guidelines(use_tools: bool = False) -> str:
    """
    Get the global user simulator guidelines.

    Args:
        use_tools: Whether to use the tools guidelines.

    Returns:
        The global user simulator guidelines.
    """
    if use_tools:
        with open(GLOBAL_USER_SIM_GUIDELINES_PATH_TOOLS, "r") as fp:
            user_sim_guidelines = fp.read()
    else:
        with open(GLOBAL_USER_SIM_GUIDELINES_PATH, "r") as fp:
            user_sim_guidelines = fp.read()
    return user_sim_guidelines

async def create_with_retry(client, max_retries=5, initial_delay=1, is_eval=False, backoff_factor=2, **kwargs):
    retries = 0
    delay = initial_delay
    
    while retries <= max_retries:
        try:
            return await client.chat.completions.create(**kwargs)
        except BadRequestError as e:
            if "is longer than the model's context length" in e.message:
                if is_eval:
                    raise e
                else:
                    return {
                        "error": "reach length limit."
                    }
            else:
                raise e

        except (RateLimitError, APIError) as e:
            retries += 1
            if retries > max_retries:
                raise Exception(f"all {max_retries} trials fialed: {e}")
            
            print(f"try {retries}/{max_retries} fail, wait {delay} s ...")
            time.sleep(delay)
            delay *= backoff_factor
        except Exception as e:
            raise e


SYSTEM_PROMPT = """
{global_user_sim_guidelines}

<scenario>
{instructions}
</scenario>
""".strip()

class UserState(BaseModel):
    """The state of the user simulator."""

    system_messages: list[SystemMessage]
    messages: list[APICompatibleMessage]

    def flip_roles(self) -> list[APICompatibleMessage]:
        """
        Returns a list of messages with the roles flipped.
        """
        # NOTE: also clean the message to a api-compatible format
        flipped_messages = []
        for message in self.messages:
            if isinstance(message, UserMessage):
                flipped_messages.append(
                    AssistantMessage(
                        role="assistant",
                        tool_calls=message.tool_calls,
                        content=message.content,
                    )
                )
            elif isinstance(message, AssistantMessage):
                if not message.is_tool_call():
                    # Only add non tool call messages
                    flipped_messages.append(
                        UserMessage(
                            role="user",
                            content=message.content,
                        )
                    )
                else:
                    raise ValueError(
                        f"Tool calls are not supported in the flipped messages: {message}"
                    )
            elif isinstance(message, ToolMessage):
                if message.requestor == "user":
                    # Only add tool messages for the user
                    flipped_messages.append(
                        ToolMessage(
                            id=message.id,
                            role=message.role,
                            content=message.content,
                        )
                    )
                else:
                    raise ValueError(
                        f"Tool messages should be sent to the user in this message history: {message}"
                    )
            else:
                print(message, type(message))
                raise ValueError(f"Unknown message role: {message.role}")
        return flipped_messages


class UserSimulator:
    """Stateless implementation of a user simulator."""

    def __init__(
        self,
        tools: Optional[list[Tool]] = None,
        instructions: Optional[UserInstructions] = None,
        llm: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        llm_args: Optional[dict] = None,
    ):
        self.instructions = instructions
        
        self.llm_client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url
        )

        self.tools = tools
        self.llm = llm
        self.llm_args = llm_args or {}

    @property
    def global_simulation_guidelines(self) -> str:
        """
        The simulation guidelines for the user simulator.
        """
        use_tools = self.tools is not None
        return get_global_user_sim_guidelines(use_tools=use_tools)

    @property
    def system_prompt(self) -> str:
        """
        The system prompt for the user simulator.
        """
        if self.instructions is None:
            logger.warning("No instructions provided for user simulator")

        system_prompt = SYSTEM_PROMPT.format(
            global_user_sim_guidelines=self.global_simulation_guidelines,
            instructions=self.instructions,
        )
        return system_prompt

    def get_init_state(
        self, message_history: Optional[list] = None
    ) -> UserState:
        """
        Get the initial state of the user simulator.
        """
        if message_history is None:
            message_history = []

        user_state = UserState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=message_history,
        )
        return user_state

    @classmethod
    def is_stop(cls, message: UserMessage) -> bool:
        """
        Check if the message is a stop message.
        """
        if message.is_tool_call():
            return False
        assert message.content is not None
        return (
            STOP in message.content
            or TRANSFER in message.content
            or OUT_OF_SCOPE in message.content
            or "reach length limit" in message.content
        )

    async def generate_next_message(
        self, message, state: UserState
    ) -> Tuple[UserMessage, UserState]:
        return await self._generate_next_message(message, state)

    async def _generate_next_message(
        self, message, state: UserState
    ) -> Tuple[UserMessage, UserState]:
        """Get the response from the user simulator.

        Args:
            message: The assistant or tool message.
            state: The user simulator's state.

        Returns:
            A tuple containing the user message and the updated user state.
        """
        # Updating state with new message
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)
        messages = state.system_messages + state.flip_roles()
        
        messages = to_litellm_messages(messages)
        # Generate response

        tools = [tool.openai_schema for tool in self.tools] if self.tools else None
        # response = self.llm_client.chat.completions.create(
        #         model=self.llm,
        #         messages=messages,
        #         tools=tools,
        #         **self.llm_args,
        #     )
        response = await create_with_retry(
            self.llm_client,
            model=self.llm,
            messages=messages,
            tools=tools,
            **self.llm_args,
        )
        # print("[debug]: ", messages, response)

        if isinstance(response, dict) and "error" in response:
            user_response = response["error"]
        else:
            response = response.choices[0]
            user_response = response.message.content
        
        if user_response is None:
            user_response = OUT_OF_SCOPE

        user_message = UserMessage(
            role="user",
            content=user_response,
            raw_data=response.to_dict() if not isinstance(response, dict) else response,
        )

        # # flip the requestor of the tool calls
        # if assistant_message.tool_calls is not None:
        #     user_message.tool_calls = []
        #     for tool_call in assistant_message.tool_calls:
        #         user_message.tool_calls.append(
        #             ToolCall(
        #                 id=tool_call.id,
        #                 name=tool_call.name,
        #                 arguments=tool_call.arguments,
        #                 requestor="user",
        #             )
        #         )

        # Updating state with response
        state.messages.append(user_message)
        return user_message, state
