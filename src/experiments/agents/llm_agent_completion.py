"""
This modules implement LLMAgent using the base completion API (instead of the chat API with tools passed as a special parameter).
This allows for full control over the prompt sent to the LLM.
"""

import json
import re
from copy import deepcopy
from typing import List, Optional

from loguru import logger
from pydantic import BaseModel

from tau2.agent.base import AgentError, ValidAgentInputMessage, validate_message_format
from tau2.agent.llm_agent import LLMAgent, LLMGTAgent, LLMSoloAgent
from tau2.data_model.message import (
    APICompatibleMessage,
    AssistantMessage,
    MultiToolMessage,
    SystemMessage,
    ToolCall,
    UserMessage,
)
from tau2.data_model.tasks import Task
from tau2.environment.tool import Tool
from tau2.utils.llm_utils import generate

RESPOND_ACTION_NAME = "respond"  # Not used.


def parse_message(msg: AssistantMessage, solo: bool = False) -> AssistantMessage:
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
    - msg.tool_calls: The json parsed tool call.

    If the message is in the second or third format, it will be parsed into the following fields:
    - msg.reasoning: The reasoning steps.
    - msg.content: The message to the user.
    """
    msg = deepcopy(msg)
    text_content = msg.content
    if text_content is None:
        return msg

    logger.debug(f"Parsing message content: {text_content}.")
    # Initialize variables
    reasoning = None
    message_to_user = None
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
            message_to_user = to_user_match.group(1).strip()
        else:
            raise AgentError("Respond section found but no content could be extracted")
    else:
        # If no Action or Respond section is found, treat the entire content as a user message
        # This handles cases where the LLM sends plain text without the expected format
        message_to_user = remaining_content.strip()
        if not message_to_user:
            raise AgentError("No content found in message")

    # Update the message object
    msg.reasoning = reasoning
    msg.content = message_to_user
    msg.tool_calls = tool_calls

    logger.debug(
        f"Reasoning: {msg.reasoning}.\n"
        f"User message: {msg.content}.\n"
        f"Tool calls: {msg.tool_calls}."
    )
    valid, error_msg = validate_message_format(msg, solo=solo)
    if not valid:
        raise AgentError(f"Invalid message format: {error_msg}")

    return msg


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


class LLMAgentCompletion(LLMAgent):
    """
    An LLM agent that can be used to solve a task using the completion API.
    Inherits from LLMAgent but overrides the message generation to use completion API.
    """

    def __init__(
        self,
        tools: List[Tool],
        domain_policy: str,
        llm: Optional[str] = None,
        llm_args: Optional[dict] = None,
        allow_format_retry: bool = True,
    ):
        """
        Initialize the LLMAgentCompletion.
        """
        super().__init__(
            tools=tools,
            domain_policy=domain_policy,
            llm=llm,
            llm_args=llm_args,
            allow_format_retry=allow_format_retry,
        )
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

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: LLMAgentState
    ) -> tuple[AssistantMessage, LLMAgentState]:
        """
        Respond to a user or tool message using the completion API.
        """
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)
        messages = state.system_messages + state.messages
        completion_assistant_message = generate(
            model=self.llm,
            messages=messages,
            **self.llm_args,
        )
        valid, error_msg = True, None
        try:
            assistant_message = parse_message(completion_assistant_message, solo=False)
        except AgentError as e:
            logger.warning(f"Error: {e}. Retrying...")
            valid, error_msg = False, str(e)
        if not valid and self.allow_format_retry:
            logger.warning(f"Format error: {error_msg}. Retrying...")
            retry_messages = messages[:]
            retry_messages.append(completion_assistant_message)
            retry_messages.append(
                SystemMessage(role="system", content=f"Error: {error_msg}. Try again.")
            )
            completion_assistant_message = generate(
                model=self.llm,
                messages=retry_messages,
                **self.llm_args,
            )
            assistant_message = parse_message(completion_assistant_message, solo=False)
        elif not valid and not self.allow_format_retry:
            logger.warning(f"Format error: {error_msg}. Format retry is disabled.")
            assistant_message.errors = [error_msg]
        state.messages.append(completion_assistant_message)
        return assistant_message, state


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


class LLMGTAgentCompletion(LLMGTAgent):
    """
    A GroundTruth agent that can be used to solve a task using the completion API.
    Inherits from LLMGTAgent but overrides the message generation to use completion API.
    """

    def __init__(
        self,
        tools: List[Tool],
        domain_policy: str,
        task: Task,
        llm: Optional[str] = None,
        llm_args: Optional[dict] = None,
        provide_function_args: bool = True,
        allow_format_retry: bool = True,
    ):
        """
        Initialize the LLMGTAgentCompletion.
        If provide_function_args is True, the resolution steps will include the function arguments.
        """
        super().__init__(
            tools=tools,
            domain_policy=domain_policy,
            task=task,
            llm=llm,
            llm_args=llm_args,
            provide_function_args=provide_function_args,
            allow_format_retry=allow_format_retry,
        )
        self.tool_prompt = json.dumps(
            [tool.openai_schema for tool in self.tools], indent=2
        )

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT_GT.format(
            agent_instruction=AGENT_GT_INSTRUCTION,
            domain_policy=self.domain_policy,
            resolution_steps=self.make_agent_instructions_from_actions(),
            tool_prompt=self.tool_prompt,
        )

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: LLMAgentState
    ) -> tuple[AssistantMessage, LLMAgentState]:
        """
        Respond to a user or tool message using the completion API.
        """
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)
        messages = state.system_messages + state.messages
        completion_assistant_message = generate(
            model=self.llm,
            messages=messages,
            **self.llm_args,
        )
        valid, error_msg = True, None
        try:
            assistant_message = parse_message(completion_assistant_message, solo=False)
        except AgentError as e:
            logger.warning(f"Error: {e}. Retrying...")
            valid, error_msg = False, str(e)
        if not valid and self.allow_format_retry:
            logger.warning(f"Format error: {error_msg}. Retrying...")
            retry_messages = messages[:]
            retry_messages.append(completion_assistant_message)
            retry_messages.append(
                SystemMessage(role="system", content=f"Error: {error_msg}. Try again.")
            )
            completion_assistant_message = generate(
                model=self.llm,
                messages=retry_messages,
                **self.llm_args,
            )
            assistant_message = parse_message(completion_assistant_message)
        elif not valid and not self.allow_format_retry:
            logger.warning(f"Format error: {error_msg}. Format retry is disabled.")
            assistant_message.errors = [error_msg]
        state.messages.append(completion_assistant_message)
        return assistant_message, state


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


class LLMSoloAgentCompletion(LLMSoloAgent):
    """
    An LLM agent that can be used to solve a task without any interaction with the customer using the completion API.
    Inherits from LLMSoloAgent but overrides the message generation to use completion API.
    """

    def __init__(
        self,
        tools: List[Tool],
        domain_policy: str,
        task: Task,
        llm: Optional[str] = None,
        llm_args: Optional[dict] = None,
        allow_format_retry: bool = True,
    ):
        """
        Initialize the LLMSoloAgentCompletion.
        """
        super().__init__(
            tools=tools,
            domain_policy=domain_policy,
            task=task,
            llm=llm,
            llm_args=llm_args,
            allow_format_retry=allow_format_retry,
        )
        self.tool_prompt = json.dumps(
            [tool.openai_schema for tool in self.tools], indent=2
        )

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

    def generate_next_message(
        self, message: Optional[ValidAgentInputMessage], state: LLMAgentState
    ) -> tuple[AssistantMessage, LLMAgentState]:
        """
        Respond to a user or tool message using the completion API.
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
