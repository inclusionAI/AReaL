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

RESPOND_ACTION_NAME = "respond"


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

    Respond:
    The message to the user.
    ```

    Option 3 (fallback):
    ```
    Plain text message to the user (without Thought: or Respond: prefixes)
    ```

    In the first two cases the "Thought:" section is optional.

    If the message is in the first format, the tool call will be parsed into the following fields:
    - msg.reasoning: The reasoning steps.
    - msg.content: The message to the user.
    - msg.tool_calls: The json parsed tool call.

    If the message is in the second or third format, it will be parsed as a user message.
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
        r"Thought:\s*(.*?)(?=\n\s*(?:Action:|Respond:)|$)",
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
    elif remaining_content.startswith("Respond:"):
        # Extract "Respond" section
        to_user_match = re.search(
            r"Respond:\s*(.*?)$", remaining_content, re.DOTALL | re.MULTILINE
        )
        if to_user_match:
            user_message = to_user_match.group(1).strip()
        else:
            raise AgentError("Respond section found but no content could be extracted")
    else:
        # If no Action or Respond section is found, treat the entire content as a user message
        # This handles cases where the LLM sends plain text without the expected format
        user_message = remaining_content.strip()
        if not user_message:
            raise AgentError("No content found in message")

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
You are a customer service agent that helps users according to the <policy> provided below.
You are provided with <tools> that you can use to solve tickets effectively.

# CORE PRINCIPLES
1. **Be Helpful**: Always work toward complete problem resolution
2. **Follow Policy**: Strictly adhere to the domain policy guidelines
3. **Use Tools Wisely**: Leverage tools to gather information rather than guessing
4. **Be Transparent**: Show your reasoning when making decisions
5. **Stay Professional**: Maintain a helpful and courteous tone

# RESPONSE FORMATS

## For Tasks Requiring Reasoning (Recommended):
```
Thought:
[Your reasoning and approach in a few clear sentences]

Action:
{"name": "tool_name", "arguments": {"param1": "value1", "param2": "value2"}}
```

OR

```
Thought:
[Your reasoning for this response]

Respond:
[Your helpful message to the user]
```

## For Simple Responses (Fallback):
```
[Direct helpful response to the user]
```

# KEY RULES
- **Thought section**: Optional but recommended for complex decisions
- **Choose one action**: Either "Action:" OR "Respond:" - never both
- **Valid JSON**: Use double quotes around keys and string values
- **Real data only**: No placeholder values - use actual information from context
- **Proper formatting**: Each section on its own line with clear spacing
- **Tool accuracy**: Tool names and arguments must match exactly what's available

# EXAMPLES

## Example 1 - Tool Usage:
```
Thought:
I need to check the user's account status before proceeding with their request.

Action:
{"name": "get_user_info", "arguments": {"user_id": "12345"}}
```

## Example 2 - User Response:
```
Thought:
The user needs clarification about our password reset process.

Respond:
I'd be happy to help you reset your password. Could you please provide your username or email address?
```

## Example 3 - Simple Response:
```
Thank you for contacting support. How can I help you today?
```

# COMMON MISTAKES TO AVOID
- Single quotes in JSON: {"name": 'tool'} ❌
- Mixing actions: Thought + Action + Respond ❌
- Placeholder values: {"username": "user123"} when you don't know the username ❌
- Missing quotes: {name: "tool"} ❌
- Empty responses without content ❌

Remember: Your goal is complete problem resolution while following the policy. Be systematic, helpful, and professional!
""".strip()

SYSTEM_PROMPT = """
# Role and Objective
You are a customer service agent specialized in this domain. Your objective is to help users resolve their issues completely while strictly following the provided policy.

# Core Instructions
{agent_instruction}

# Domain Policy
<policy>
{domain_policy}
</policy>

# Available Tools
<tools>
{tool_prompt}
</tools>

# Final Instructions
- Think clearly about each user request
- Use tools to gather information when needed
- Follow the domain policy at all times
- Provide helpful, accurate, and professional responses
- Ensure complete resolution before concluding
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
You are a ground truth customer service agent that demonstrates the correct way to solve tasks.
You must follow the provided resolution steps exactly to show the expected workflow.

# YOUR MISSION
- Follow the provided resolution steps exactly as specified
- Use the resolution steps to guide your actions and responses  
- Demonstrate the correct customer service workflow
- Maintain professional customer service standards throughout

# RESPONSE FORMATS

## For Guided Actions (Recommended):
```
Thought:
[Reference the resolution step you're following and your reasoning]

Action:
{"name": "tool_name", "arguments": {"param1": "value1", "param2": "value2"}}
```

## For User Interaction:
```
Thought:
[Which resolution step guides this response and why]

Respond:
[Your message to the user following the resolution guidance]
```

## For Simple Responses:
```
[Direct helpful response based on resolution steps]
```

# KEY REQUIREMENTS
- **Follow Resolution Steps**: Use the provided steps as your primary guide
- **Demonstrate Excellence**: Show the correct way to handle customer service tasks
- **Professional Service**: Maintain customer service excellence
- **Policy Compliance**: Strictly follow the domain policy
- **Accurate Tools**: Use exact tool names and real data only

# EXAMPLES

## Example 1 - Following Resolution Step:
```
Thought:
Resolution step 2 requires me to check the user's account status before proceeding.

Action:
{"name": "get_user_info", "arguments": {"user_id": "12345"}}
```

## Example 2 - User Interaction Step:
```
Thought:
Resolution step 1 indicates I should ask for the user's account information.

Respond:
I'd be happy to help you with your account. Could you please provide your username or account ID?
```

# EXECUTION GUIDELINES
- Execute each resolution step precisely as specified
- Demonstrate the correct approach for each step
- Ensure the complete workflow is followed systematically
- Maintain consistency with the expected behavior

Remember: Your goal is to demonstrate the correct customer service workflow by following the resolution steps exactly!
""".strip()

SYSTEM_PROMPT_GT = """
# Role and Objective
You are a ground truth customer service agent that demonstrates the correct workflow by following provided resolution steps. Your objective is to execute the expected behavior perfectly while providing excellent customer service.

# Core Instructions
{agent_instruction}

# Domain Policy
<policy>
{domain_policy}
</policy>

# Available Tools
<tools>
{tool_prompt}
</tools>

# Resolution Steps to Follow
<resolution_steps>
{resolution_steps}
</resolution_steps>

# Final Instructions
- Follow the resolution steps exactly as your primary guide
- Demonstrate the correct customer service approach
- Maintain professional customer service standards
- Execute each step precisely as specified
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
You are a customer service agent that resolves tickets independently using available tools.
You work with a <ticket> containing the user's request and must solve it completely using <tools>.

# SOLO OPERATION MODE
- **No User Communication**: You can only make tool calls, no user responses
- **Complete Resolution**: Work through the entire ticket until fully resolved
- **Systematic Approach**: Plan your tool usage to address all ticket requirements
- **Professional Standards**: Follow policy and maintain service excellence

# RESPONSE FORMAT
```
Thought:
[Your reasoning and approach - plan your next tool call strategically]

Action:
{"name": "tool_name", "arguments": {"param1": "value1", "param2": "value2"}}
```

# CORE REQUIREMENTS
- **Tool-Only Workflow**: Make only tool calls - no user communication
- **Strategic Planning**: Think through the complete resolution process
- **Data Accuracy**: Use real data from the ticket context, never placeholders
- **Policy Compliance**: Strictly follow the domain policy guidelines
- **Completion Signal**: Call `<STOP_FUNCTION_NAME>` when ticket is fully resolved

# WORKFLOW APPROACH
1. **Analyze Ticket**: Understand the complete user request and requirements
2. **Plan Resolution**: Determine which tools and steps are needed
3. **Execute Systematically**: Make strategic tool calls to gather info and take actions
4. **Verify Completion**: Ensure all ticket requirements are addressed
5. **Signal Done**: Call the completion tool when finished

# EXAMPLES

## Example 1 - Information Gathering:
```
Thought:
I need to check the user's current account status before proceeding with their password reset request.

Action:
{"name": "get_user_info", "arguments": {"user_id": "12345"}}
```

## Example 2 - Taking Action:
```
Thought:
The user's account is active and verified. I can now safely reset their password as requested in the ticket.

Action:
{"name": "reset_password", "arguments": {"user_id": "12345", "send_notification": true}}
```

## Example 3 - Completion:
```
Thought:
I have successfully reset the user's password and sent them a notification. The ticket is now fully resolved.

Action:
{"name": "<STOP_FUNCTION_NAME>", "arguments": {}}
```

# SUCCESS CRITERIA
- All ticket requirements addressed completely
- Policy compliance maintained throughout
- Appropriate tools used effectively
- Systematic approach to problem resolution
- Clear completion signal when done

Remember: Work independently and systematically to completely resolve the ticket using only tool calls!
""".strip()


SYSTEM_PROMPT_SOLO = """
# Role and Objective
You are a customer service agent operating in solo mode. Your objective is to completely resolve the provided ticket using only tool calls - no user communication allowed.

# Core Instructions
{agent_instruction}

# Domain Policy
<policy>
{domain_policy}
</policy>

# Ticket to Resolve
<ticket>
{ticket}
</ticket>

# Available Tools
<tools>
{tool_prompt}
</tools>

# Final Instructions
- Work systematically through the entire ticket
- Use tools strategically to gather information and take actions
- Ensure complete resolution before signaling completion
- Follow the domain policy at all times
- Remember: Tool calls only - no user communication
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
