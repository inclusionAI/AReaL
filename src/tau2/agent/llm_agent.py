import json
import re
from copy import deepcopy
from typing import List, Optional

from loguru import logger
from pydantic import BaseModel

from tau2.agent.base import (
    AgentError,
    LocalAgent,
    ValidAgentInputMessage,
    is_valid_agent_history_message,
)
from tau2.data_model.message import (
    APICompatibleMessage,
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
    ToolCall,
    UserMessage,
)
from tau2.data_model.tasks import Action, Task
from tau2.environment.tool import Tool, as_tool
from tau2.utils.llm_utils import generate


def parse_message(msg: AssistantMessage) -> None:
    """Parse the text content into reasoning steps, and tool call.
    The message.content should be in the following formats:
    Option 1:
    ```
    Thought:
    A single line of reasoning to process the context and inform the decision making. Do not include extra lines.

    Action:
    {{"name": <The name of tool you want to call>, "arguments": The arguments to pass to the tool in json format>}}
    ```

    Option 2:
    ```
    Thought:
    A single line of reasoning to process the context and inform the decision making. Do not include extra lines.

    To User:
    The message to the user.
    ```
    In both cases the "Thought:" section is optional.

    If the message is in the first format, the tool call will be parsed into the following fields:
    - msg.reasoning: The reasoning steps.
    - msg.content: The message to the user.
    - msg.tool_calls: The json parsed tool call.
    """
    text_content = msg.content
    if text_content is None:
        return

    logger.debug(f"Parsing message content: {text_content}.")
    # Initialize variables
    reasoning = None
    user_message = None
    tool_calls = None

    # Extract reasoning content between "Thought:" and the next section
    thought_match = re.search(
        r"Thought:\s*(.*?)(?=\n\s*(?:Action:|To User:)|$)",
        text_content,
        re.DOTALL | re.MULTILINE,
    )
    if thought_match:
        reasoning = thought_match.group(1).strip()
        # Get the content after the thought section
        thought_end = thought_match.end()
        remaining_content = text_content[thought_end:].strip()
    else:
        # No thought section, use entire content
        remaining_content = text_content.strip()

    # Parse what comes after the thought (or entire content if no thought)
    if remaining_content.startswith("Action:"):
        # Extract Action section (tool call)
        action_match = re.search(r"Action:\s*(\{.*\})", remaining_content, re.DOTALL)
        if action_match:
            try:
                action_json = action_match.group(1).strip()
                tool_calls = [ToolCall(**json.loads(action_json))]
            except (json.JSONDecodeError, TypeError) as e:
                raise AgentError(f"Failed to parse action JSON '{action_json}': {e}")
        else:
            raise AgentError(
                "Action section found but no valid JSON could be extracted"
            )
    elif remaining_content.startswith("To User:"):
        # Extract "To User" section
        to_user_match = re.search(
            r"To User:\s*(.*?)$", remaining_content, re.DOTALL | re.MULTILINE
        )
        if to_user_match:
            user_message = to_user_match.group(1).strip()
        else:
            raise AgentError("To User section found but no content could be extracted")
    else:
        # Neither Action nor To User section found
        raise AgentError(
            f"Invalid message format. Expected 'Action:' or 'To User:' section after thought, but found: {remaining_content[:100]}..."
        )

    # Update the message object
    msg.reasoning = reasoning
    msg.content = user_message
    msg.tool_calls = tool_calls

    logger.debug(
        f"Reasoning: {reasoning}.\n"
        f"User message: {msg.content}.\n"
        f"Tool calls: {msg.tool_calls}."
    )


AGENT_INSTRUCTION = """
You are a customer service agent that helps the user according to the <policy> provided below.
You are also provided with a list of <tools> that you can use to solve the ticket.

CRITICAL: You must format your responses in exactly one of these two formats:

FORMAT 1 - Send a message to the user:
```
Thought:
[Your reasoning in a single line]

To User:
[Your message to the user]
```

FORMAT 2 - Make a tool call:
```
Thought:
[Your reasoning in a single line]

Action:
{"name": "tool_name", "arguments": {"param1": "value1", "param2": "value2"}}
```

IMPORTANT RULES:
1. The "Thought:" section is OPTIONAL but recommended
2. You must choose either "To User:" OR "Action:" - never both
3. The Action JSON must be valid JSON with double quotes around keys and string values
4. Do not use placeholder values - use actual data from the context
5. Each section must be on its own line with proper spacing
6. The Thought section should be a single line of reasoning
7. Tool names and arguments must match exactly what's available in the tools list

VALID EXAMPLES:

Example 1 - Message to user:
```
Thought:
I need the user's username to reset their password.

To User:
Could you please provide me with your username?
```

Example 2 - Tool call:
```
Thought:
I will reset the password for user "john_doe" using the reset_password tool.

Action:
{"name": "reset_password", "arguments": {"username": "john_doe"}}
```

Example 3 - Tool call with multiple arguments:
```
Thought:
I need to update the user's account with their new email and phone number.

Action:
{"name": "update_account", "arguments": {"email": "new@example.com", "phone": "555-1234"}}
```

INVALID FORMATS TO AVOID:
- Don't use single quotes in JSON: {"name": 'tool'} ❌
- Don't mix sections: Thought + Action + To User ❌
- Don't use placeholder values: {"username": "user123"} when you don't know the username ❌
- Don't forget quotes around JSON keys: {name: "tool"} ❌

Be helpful and always follow the policy!
""".strip()

SYSTEM_PROMPT = """
<instructions>
{agent_instruction}
</instructions>
<policy>
{domain_policy}
</policy>
<tools>
{tool_prompt}
</tools>
""".strip()


class LLMAgentState(BaseModel):
    """The state of the agent."""

    system_messages: list[SystemMessage]
    messages: list[APICompatibleMessage]


class LLMAgent(LocalAgent[LLMAgentState]):
    """
    An LLM agent that can be used to solve a task.
    """

    def __init__(
        self,
        tools: List[Tool],
        domain_policy: str,
        llm: Optional[str] = None,
        llm_args: Optional[dict] = None,
    ):
        """
        Initialize the LLMAgent.
        """
        super().__init__(tools=tools, domain_policy=domain_policy)
        self.llm = llm
        self.llm_args = deepcopy(llm_args) if llm_args is not None else {}
        self.tool_prompt = json.dumps(
            [tool.openai_schema for tool in self.tools], indent=2
        )

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT.format(
            domain_policy=self.domain_policy,
            agent_instruction=AGENT_INSTRUCTION,
            tool_prompt=self.tool_prompt,
        )

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> LLMAgentState:
        """Get the initial state of the agent.

        Args:
            message_history: The message history of the conversation.

        Returns:
            The initial state of the agent.
        """
        if message_history is None:
            message_history = []
        assert all(is_valid_agent_history_message(m) for m in message_history), (
            "Message history must contain only AssistantMessage, UserMessage, or ToolMessage to Agent."
        )
        return LLMAgentState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=message_history,
        )

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: LLMAgentState
    ) -> tuple[AssistantMessage, LLMAgentState]:
        """
        Respond to a user or tool message.
        """
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)
        messages = state.system_messages + state.messages
        assistant_message = generate(
            model=self.llm,
            tools=self.tools,
            messages=messages,
            **self.llm_args,
        )
        state.messages.append(assistant_message)
        parse_message(assistant_message)
        return assistant_message, state

    def set_seed(self, seed: int):
        """Set the seed for the LLM."""
        if self.llm is None:
            raise ValueError("LLM is not set")
        cur_seed = self.llm_args.get("seed", None)
        if cur_seed is not None:
            logger.warning(f"Seed is already set to {cur_seed}, resetting it to {seed}")
        self.llm_args["seed"] = seed


AGENT_GT_INSTRUCTION = """
You are testing that our user simulator is working correctly.
User simulator will have an issue for you to solve.
You must behave according to the <policy> provided below.
To make following the policy easier, we give you the list of resolution steps you are expected to take.
These steps involve either taking an action or asking the user to take an action.
You are also provided with a list of <tools> that you can use to solve the ticket.

CRITICAL: You must format your responses in exactly one of these two formats:

FORMAT 1 - Send a message to the user:
```
Thought:
[Your reasoning in a single line]

To User:
[Your message to the user]
```

FORMAT 2 - Make a tool call:
```
Thought:
[Your reasoning in a single line]

Action:
{"name": "tool_name", "arguments": {"param1": "value1", "param2": "value2"}}
```

IMPORTANT RULES:
1. The "Thought:" section is OPTIONAL but recommended
2. You must choose either "To User:" OR "Action:" - never both
3. The Action JSON must be valid JSON with double quotes around keys and string values
4. Do not use placeholder values - use actual data from the context
5. Each section must be on its own line with proper spacing
6. The Thought section should be a single line of reasoning
7. Tool names and arguments must match exactly what's available in the tools list
8. Follow the resolution steps provided to guide your actions

VALID EXAMPLES:

Example 1 - Message to user:
```
Thought:
I need the user's username to reset their password.

To User:
Could you please provide me with your username?
```

Example 2 - Tool call:
```
Thought:
I will reset the password for user "john_doe" using the reset_password tool.

Action:
{"name": "reset_password", "arguments": {"username": "john_doe"}}
```

INVALID FORMATS TO AVOID:
- Don't use single quotes in JSON: {"name": 'tool'} ❌
- Don't mix sections: Thought + Action + To User ❌
- Don't use placeholder values: {"username": "user123"} when you don't know the username ❌
- Don't forget quotes around JSON keys: {name: "tool"} ❌

Be helpful and always follow the policy!
""".strip()

SYSTEM_PROMPT_GT = """
<instructions>
{agent_instruction}
</instructions>
<policy>
{domain_policy}
</policy>
<tools>
{tool_prompt}
</tools>
<resolution_steps>
{resolution_steps}
</resolution_steps>
""".strip()


class LLMGTAgent(LocalAgent[LLMAgentState]):
    """
    An GroundTruth agent that can be used to solve a task.
    This agent will receive the expected actions.
    """

    def __init__(
        self,
        tools: List[Tool],
        domain_policy: str,
        task: Task,
        llm: Optional[str] = None,
        llm_args: Optional[dict] = None,
        provide_function_args: bool = True,
    ):
        """
        Initialize the LLMAgent.
        If provide_function_args is True, the resolution steps will include the function arguments.
        """
        super().__init__(tools=tools, domain_policy=domain_policy)
        assert self.check_valid_task(task), (
            f"Task {task.id} is not valid. Cannot run GT agent."
        )
        self.task = task
        self.llm = llm
        self.llm_args = deepcopy(llm_args) if llm_args is not None else {}
        self.provide_function_args = provide_function_args
        self.tool_prompt = json.dumps(
            [tool.openai_schema for tool in self.tools], indent=2
        )

    @classmethod
    def check_valid_task(cls, task: Task) -> bool:
        """
        Check if the task is valid.
        Only the tasks that require at least one action are valid.
        """
        if task.evaluation_criteria is None:
            return False
        expected_actions = task.evaluation_criteria.actions or []
        if len(expected_actions) == 0:
            return False
        return True

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT_GT.format(
            agent_instruction=AGENT_GT_INSTRUCTION,
            domain_policy=self.domain_policy,
            resolution_steps=self.make_agent_instructions_from_actions(),
            tool_prompt=self.tool_prompt,
        )

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> LLMAgentState:
        """Get the initial state of the agent.

        Args:
            message_history: The message history of the conversation.

        Returns:
            The initial state of the agent.
        """
        if message_history is None:
            message_history = []
        assert all(is_valid_agent_history_message(m) for m in message_history), (
            "Message history must contain only AssistantMessage, UserMessage, or ToolMessage to Agent."
        )
        return LLMAgentState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=message_history,
        )

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: LLMAgentState
    ) -> tuple[AssistantMessage, LLMAgentState]:
        """
        Respond to a user or tool message.
        """
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)
        messages = state.system_messages + state.messages
        assistant_message = generate(
            model=self.llm,
            tools=self.tools,
            messages=messages,
            **self.llm_args,
        )
        state.messages.append(assistant_message)
        parse_message(assistant_message)
        return assistant_message, state

    def set_seed(self, seed: int):
        """Set the seed for the LLM."""
        if self.llm is None:
            raise ValueError("LLM is not set")
        cur_seed = self.llm_args.get("seed", None)
        if cur_seed is not None:
            logger.warning(f"Seed is already set to {cur_seed}, resetting it to {seed}")
        self.llm_args["seed"] = seed

    def make_agent_instructions_from_actions(self) -> str:
        """
        Make agent instructions from a list of actions
        """
        lines = []
        for i, action in enumerate(self.task.evaluation_criteria.actions):
            lines.append(
                f"[Step {i + 1}] {self.make_agent_instructions_from_action(action=action, include_function_args=self.provide_function_args)}"
            )
        return "\n".join(lines)

    @classmethod
    def make_agent_instructions_from_action(
        cls, action: Action, include_function_args: bool = False
    ) -> str:
        """
        Make agent instructions from an action.
        If the action is a user action, returns instructions for the agent to give to the user.
        If the action is an agent action, returns instructions for the agent to perform the action.
        """
        if action.requestor == "user":
            if include_function_args:
                return f"Instruct the user to perform the following action: {action.get_func_format()}."
            else:
                return f"User action: {action.name}."
        elif action.requestor == "assistant":
            if include_function_args:
                return f"Perform the following action: {action.get_func_format()}."
            else:
                return f"Assistant action: {action.name}."
        else:
            raise ValueError(f"Unknown action requestor: {action.requestor}")


AGENT_SOLO_INSTRUCTION = """
You are a customer service agent that helps the user according to the <policy> provided below.
You are provided with a <ticket> that contains the user's request.
You are also provided with a list of <tools> that you can use to solve the ticket.
You will need to plan and call the appropriate tools to solve the ticket.

You cannot communicate with the user, only make tool calls.

CRITICAL: You must format your responses in exactly this format:

```
Thought:
[Your reasoning in a single line]

Action:
{"name": "tool_name", "arguments": {"param1": "value1", "param2": "value2"}}
```

IMPORTANT RULES:
1. The "Thought:" section is OPTIONAL but recommended
2. The Action JSON must be valid JSON with double quotes around keys and string values
3. Do not use placeholder values - use actual data from the ticket context
4. Each section must be on its own line with proper spacing
5. The Thought section should be a single line of reasoning
6. Tool names and arguments must match exactly what's available in the tools list

COMPLETION SIGNAL:
When you have successfully resolved the ticket, you MUST call the `<STOP_FUNCTION_NAME>` tool to signal completion. This is the ONLY way to indicate you are done with the task.

VALID EXAMPLES:

Example 1 - Tool call:
```
Thought:
I will reset the password for user "john_doe" using the reset_password tool.

Action:
{"name": "reset_password", "arguments": {"username": "john_doe"}}
```

Example 2 - Tool call with multiple arguments:
```
Thought:
I need to update the user's account with their new email and phone number.

Action:
{"name": "update_account", "arguments": {"email": "new@example.com", "phone": "555-1234"}}
```

Example 3 - COMPLETION (call this when done):
```
Thought:
I have successfully resolved the user's issue. I will call the <STOP_FUNCTION_NAME> tool to finish.

Action:
{"name": "<STOP_FUNCTION_NAME>", "arguments": {}}
```

INVALID FORMATS TO AVOID:
- Don't use single quotes in JSON: {"name": 'tool'} ❌
- Don't use placeholder values: {"username": "user123"} when you don't know the username ❌
- Don't forget quotes around JSON keys: {name: "tool"} ❌
- Don't include multiple tool calls in one message ❌
- Don't forget to call the `<STOP_FUNCTION_NAME>` tool when you're done ❌

Be helpful and always follow the policy!
""".strip()


SYSTEM_PROMPT_SOLO = """
<instructions>
{agent_instruction}
</instructions>
<policy>
{domain_policy}
</policy>
<ticket>
{ticket}
</ticket>
<tools>
{tool_prompt}
</tools>
""".strip()


class LLMSoloAgent(LocalAgent[LLMAgentState]):
    """
    An LLM agent that can be used to solve a task without any interaction with the customer.
    The task need to specify a ticket format.
    """

    STOP_FUNCTION_NAME = "done"
    TRANSFER_TOOL_NAME = "transfer_to_human_agents"
    STOP_TOKEN = "###STOP###"

    def __init__(
        self,
        tools: List[Tool],
        domain_policy: str,
        task: Task,
        llm: Optional[str] = None,
        llm_args: Optional[dict] = None,
    ):
        """
        Initialize the LLMAgent.
        """
        super().__init__(tools=tools, domain_policy=domain_policy)
        assert self.check_valid_task(task), (
            f"Task {task.id} is not valid. Cannot run GT agent."
        )
        self.task = task
        self.llm = llm
        self.llm_args = llm_args if llm_args is not None else {}
        self.tool_prompt = json.dumps(
            [tool.openai_schema for tool in self.tools], indent=2
        )
        self.add_stop_tool()
        self.validate_tools()

    def add_stop_tool(self) -> None:
        """Add the stop tool to the tools."""

        def done() -> str:
            """Call this function when you are done with the task."""
            return self.STOP_TOKEN

        self.tools.append(as_tool(done))

    def validate_tools(self) -> None:
        """Check if the tools are valid."""
        tool_names = {tool.name for tool in self.tools}
        if self.TRANSFER_TOOL_NAME not in tool_names:
            logger.warning(
                f"Tool {self.TRANSFER_TOOL_NAME} not found in tools. This tool is required for the agent to transfer the user to a human agent."
            )
        if self.STOP_FUNCTION_NAME not in tool_names:
            raise ValueError(f"Tool {self.STOP_FUNCTION_NAME} not found in tools.")

    @classmethod
    def check_valid_task(cls, task: Task) -> bool:
        """
        Check if the task is valid.
        Task should contain a ticket and evaluation criteria.
        If the task contains an initial state, the message history should only contain tool calls and responses.
        """
        if task.initial_state is not None:
            message_history = task.initial_state.message_history or []
            for message in message_history:
                if isinstance(message, UserMessage):
                    return False
                if isinstance(message, AssistantMessage) and not message.is_tool_call():
                    return False
            return True
        if task.ticket is None:
            return False
        if task.evaluation_criteria is None:
            return False
        expected_actions = task.evaluation_criteria.actions or []
        if len(expected_actions) == 0:
            return False
        return True

    @property
    def system_prompt(self) -> str:
        agent_instruction = AGENT_SOLO_INSTRUCTION.replace(
            "<STOP_FUNCTION_NAME>", self.STOP_FUNCTION_NAME
        )
        return SYSTEM_PROMPT_SOLO.format(
            agent_instruction=agent_instruction,
            domain_policy=self.domain_policy,
            ticket=self.task.ticket,
            tool_prompt=self.tool_prompt,
        )

    def _check_if_stop_toolcall(self, message: AssistantMessage) -> None:
        """Check if the message is a stop message.
        If the message contains a tool call with the name STOP_FUNCTION_NAME, then the message is a stop message.
        """
        if message.tool_calls is None:
            return message
        is_stop = False
        for tool_call in message.tool_calls:
            if tool_call.name == self.STOP_FUNCTION_NAME:
                is_stop = True
                break
        if is_stop:
            message.content = self.STOP_TOKEN
            message.tool_calls = None

    @classmethod
    def is_stop(cls, message: AssistantMessage) -> bool:
        """Check if the message is a stop message."""
        if message.content is None:
            return False
        return cls.STOP_TOKEN in message.content

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> LLMAgentState:
        """Get the initial state of the agent.

        Args:
            message_history: The message history of the conversation.

        Returns:
            The initial state of the agent.
        """
        if message_history is None:
            message_history = []
        assert all(is_valid_agent_history_message(m) for m in message_history), (
            "Message history must contain only AssistantMessage, UserMessage, or ToolMessage to Agent."
        )
        return LLMAgentState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=message_history,
        )

    def generate_next_message(
        self, message: Optional[ValidAgentInputMessage], state: LLMAgentState
    ) -> tuple[AssistantMessage, LLMAgentState]:
        """
        Respond to a user or tool message.
        """
        if isinstance(message, UserMessage):
            raise ValueError("LLMSoloAgent does not support user messages.")
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        elif message is None:
            assert len(state.messages) == 0, "Message history should be empty"
        else:
            state.messages.append(message)
        messages = state.system_messages + state.messages
        assistant_message = generate(
            model=self.llm,
            messages=messages,
            **self.llm_args,
        )
        state.messages.append(assistant_message)
        parse_message(assistant_message)
        self._check_if_stop_toolcall(assistant_message)
        return assistant_message, state

    def set_seed(self, seed: int):
        """Set the seed for the LLM."""
        if self.llm is None:
            raise ValueError("LLM is not set")
        cur_seed = self.llm_args.get("seed", None)
        if cur_seed is not None:
            logger.warning(f"Seed is already set to {cur_seed}, resetting it to {seed}")
        self.llm_args["seed"] = seed
