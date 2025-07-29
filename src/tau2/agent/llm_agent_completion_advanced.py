import json
import re
from copy import deepcopy
from typing import Any, Dict, List, Optional

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
    ToolMessage,
)
from tau2.environment.tool import Tool
from tau2.utils.llm_utils import generate

RESPOND_ACTION_NAME = "respond"


def parse_message_enhanced(msg: AssistantMessage) -> None:
    """Enhanced parsing with multi-step reasoning and self-consistency.

    The message.content should support these formats:

    Option 1 - Enhanced Reasoning (for complex tasks):
    ```
    Analysis:
    [Break down the problem into components]

    Planning:
    [Outline your approach step-by-step]

    Execution Reasoning:
    [Explain your chosen action/response]

    Verification:
    [Check your reasoning for errors or gaps]

    Action:
    {"name": "tool_name", "arguments": {"param1": "value1"}}
    ```

    Option 2 - Standard Reasoning:
    ```
    Thought:
    [Your reasoning]

    Action/Respond:
    [Tool call or user response]
    ```

    Option 3 - Simple format (fallback):
    ```
    [Direct response to user]
    ```
    """
    text_content = msg.content
    if text_content is None:
        return

    logger.debug(f"Parsing enhanced message content: {text_content}")

    # Initialize variables
    reasoning = None
    enhanced_reasoning = None
    user_message = None
    tool_calls = None

    # Try to extract enhanced reasoning block first
    enhanced_pattern = r"Analysis:\s*(.*?)(?=\n\s*Planning:|$).*?Planning:\s*(.*?)(?=\n\s*Execution Reasoning:|$).*?Execution Reasoning:\s*(.*?)(?=\n\s*Verification:|$).*?Verification:\s*(.*?)(?=\n\s*(?:Action:|Respond:)|$)"
    enhanced_match = re.search(enhanced_pattern, text_content, re.DOTALL | re.MULTILINE)

    if enhanced_match:
        enhanced_reasoning = {
            "analysis": enhanced_match.group(1).strip(),
            "planning": enhanced_match.group(2).strip(),
            "execution": enhanced_match.group(3).strip(),
            "verification": enhanced_match.group(4).strip(),
        }
        # Get content after the enhanced reasoning block
        verification_end = enhanced_match.end()
        remaining_content = text_content[verification_end:].strip()
    else:
        # Try standard thought pattern
        thought_match = re.search(
            r"Thought:\s*(.*?)(?=\n\s*(?:Action:|Respond:)|$)",
            text_content,
            re.DOTALL | re.MULTILINE,
        )
        if thought_match:
            reasoning = thought_match.group(1).strip()
            thought_end = thought_match.end()
            remaining_content = text_content[thought_end:].strip()
        else:
            remaining_content = text_content.strip()

    # Parse the action/response section
    if remaining_content.startswith("Action:"):
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
        respond_match = re.search(
            r"Respond:\s*(.*?)$", remaining_content, re.DOTALL | re.MULTILINE
        )
        if respond_match:
            user_message = respond_match.group(1).strip()
        else:
            raise AgentError("Respond section found but no content could be extracted")
    else:
        # Fallback: treat remaining content as user message
        user_message = remaining_content.strip()
        if not user_message:
            raise AgentError("No content found in message")

    # Update the message object
    if enhanced_reasoning:
        msg.reasoning = enhanced_reasoning
    else:
        msg.reasoning = reasoning
    msg.content = user_message
    msg.tool_calls = tool_calls

    logger.debug(
        f"Enhanced reasoning: {enhanced_reasoning}.\n"
        f"Standard reasoning: {reasoning}.\n"
        f"User message: {msg.content}.\n"
        f"Tool calls: {msg.tool_calls}."
    )


ENHANCED_AGENT_INSTRUCTION = """
You are a customer service agent that helps users according to the <policy> provided below.
You are provided with <tools> that you can use to solve tickets effectively.

# CORE AGENTIC PRINCIPLES
1. **Persistence**: Keep working until the user's request is completely resolved
2. **Planning**: Before each action, plan your next 2-3 steps  
3. **Tool Mastery**: Always use tools to gather information rather than guessing
4. **Reflection**: After tool results, reflect on what you learned and adjust your plan
5. **Verification**: Check your reasoning and actions for accuracy and completeness

# RESPONSE FORMATS

## For Complex Multi-Step Tasks (Recommended):
```
Analysis:
[Break down the problem into key components and identify what information you need]

Planning: 
[Outline your step-by-step approach - what will you do next and why]

Execution Reasoning:
[Explain your chosen action/response and how it advances toward resolution]

Verification:
[Check: Is this action appropriate? Will it help resolve the user's issue? Any risks?]

Action:
{"name": "tool_name", "arguments": {"param1": "value1", "param2": "value2"}}
```

OR

```
Analysis:
[Break down what the user needs and your understanding of their situation]

Planning:
[Your approach to help them effectively]

Execution Reasoning: 
[Why this response will be helpful and appropriate]

Verification:
[Check: Is this response complete, accurate, and helpful?]

Respond:
[Your helpful message to the user]
```

## For Simple Tasks:
```
Thought:
[Brief reasoning for your action/response]

Action:
{"name": "tool_name", "arguments": {"param1": "value1"}}
```

OR

```
Thought:
[Brief reasoning for your response]

Respond:
[Your message to the user]
```

## For Very Simple Responses (Fallback):
```
[Direct helpful response to the user]
```

# ENHANCED WORKFLOW
1. **Understand**: Fully comprehend the user's complete request and context
2. **Analyze**: Break down the problem into components  
3. **Plan**: Outline your systematic approach
4. **Execute**: Take one thoughtful action at a time
5. **Reflect**: Learn from results and adjust your plan
6. **Persist**: Continue until the issue is completely resolved
7. **Verify**: Ensure your final resolution is complete and satisfactory

# ERROR RECOVERY PROTOCOL
When you encounter errors or unexpected results:
1. **Acknowledge**: "I encountered an issue with [specific problem]. Let me try a different approach."
2. **Alternative Strategy**: Try different tools, break into smaller steps, or ask for clarification
3. **Escalation**: If multiple attempts fail, consider transferring to human support

# SELF-CONSISTENCY CHECK
For critical decisions, briefly verify:
- **Accuracy**: Are my facts and actions correct?
- **Completeness**: Did I address all parts of the request?
- **Policy Compliance**: Does this follow the domain policy?
- **Tool Usage**: Am I using the most appropriate tools?

# IMPORTANT RULES
- Use the enhanced reasoning format for any multi-step or complex tasks
- Always plan your approach before acting
- Use tools to gather information rather than making assumptions
- Keep working until the user's issue is completely resolved
- Be transparent about your reasoning process
- Follow the domain policy strictly
- Provide clear, helpful responses

Remember: Your goal is complete problem resolution, not just providing information. Be persistent, systematic, and helpful!
""".strip()

ENHANCED_SYSTEM_PROMPT = """
# Role and Objective
You are a customer service agent specialized in this domain. Your objective is to completely resolve user issues while strictly following the provided policy.

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
- Think systematically and plan your approach
- Use enhanced reasoning for complex tasks
- Ensure complete resolution before concluding
- Be helpful, accurate, and policy-compliant at all times
""".strip()


class LLMAgentV2State(BaseModel):
    """Enhanced state for the LLM agent with context tracking."""

    system_messages: list[SystemMessage]
    messages: list[APICompatibleMessage]
    context_metadata: Optional[Dict[str, Any]] = None
    reasoning_history: List[Dict[str, Any]] = []


class LLMAgentV2(LocalAgent[LLMAgentV2State]):
    """
    Enhanced LLM agent with state-of-the-art prompting techniques.

    Key improvements:
    - Multi-step reasoning (Analysis/Planning/Execution/Verification)
    - Self-consistency checking
    - Agentic workflow with persistence and planning
    - Enhanced error handling and recovery
    - Better context management
    """

    def __init__(
        self,
        tools: List[Tool],
        domain_policy: str,
        llm: Optional[str] = None,
        llm_args: Optional[dict] = None,
        enable_self_consistency: bool = True,
        max_context_tokens: int = 100000,
    ):
        """
        Initialize the enhanced LLM Agent.

        Args:
            tools: Available tools for the agent
            domain_policy: Domain-specific policy text
            llm: LLM model identifier
            llm_args: Additional arguments for LLM calls
            enable_self_consistency: Whether to enable self-consistency checking
            max_context_tokens: Maximum context window size
        """
        super().__init__(tools=tools, domain_policy=domain_policy)
        self.llm = llm
        self.llm_args = deepcopy(llm_args) if llm_args is not None else {}
        self.enable_self_consistency = enable_self_consistency
        self.max_context_tokens = max_context_tokens
        self.tool_prompt = json.dumps(
            [tool.openai_schema for tool in self.tools], indent=2
        )

    @property
    def system_prompt(self) -> str:
        return ENHANCED_SYSTEM_PROMPT.format(
            agent_instruction=ENHANCED_AGENT_INSTRUCTION,
            domain_policy=self.domain_policy,
            tool_prompt=self.tool_prompt,
        )

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> LLMAgentV2State:
        """Get the initial enhanced state of the agent."""
        if message_history is None:
            message_history = []

        assert all(is_valid_agent_history_message(m) for m in message_history), (
            "Message history must contain only AssistantMessage, UserMessage, or ToolMessage to Agent."
        )

        return LLMAgentV2State(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=message_history,
            context_metadata={"total_tokens": 0, "optimization_count": 0},
            reasoning_history=[],
        )

    def optimize_context(
        self, messages: list[APICompatibleMessage], max_tokens: int
    ) -> list[APICompatibleMessage]:
        """
        Optimize context by prioritizing important messages while staying under token limit.

        This is a simplified version - in production you'd want proper token counting.
        """
        if len(messages) <= 20:  # If context is small enough, don't optimize
            return messages

        # Always keep system messages
        system_msgs = [m for m in messages if isinstance(m, SystemMessage)]

        # Keep recent messages (last 10)
        recent_msgs = messages[-10:]

        # Keep important tool results and reasoning from middle
        important_msgs = []
        middle_msgs = messages[:-10]

        for msg in middle_msgs:
            # Keep successful tool results
            if isinstance(msg, ToolMessage) and not msg.error:
                important_msgs.append(msg)
            # Keep messages with enhanced reasoning
            elif (
                isinstance(msg, AssistantMessage)
                and msg.reasoning
                and isinstance(msg.reasoning, dict)
            ):
                important_msgs.append(msg)

        # Combine, avoiding duplicates
        optimized = system_msgs + important_msgs + recent_msgs

        # Remove duplicates while preserving order
        seen = set()
        final_messages = []
        for msg in optimized:
            msg_id = id(msg)
            if msg_id not in seen:
                seen.add(msg_id)
                final_messages.append(msg)

        return final_messages

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: LLMAgentV2State
    ) -> tuple[AssistantMessage, LLMAgentV2State]:
        """
        Generate the next message with enhanced reasoning and self-consistency.
        """
        # Update state with incoming message
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        # Optimize context if needed
        optimized_messages = self.optimize_context(
            state.messages, self.max_context_tokens
        )

        if len(optimized_messages) != len(state.messages):
            state.context_metadata["optimization_count"] += 1
            logger.info(
                f"Context optimized: {len(state.messages)} -> {len(optimized_messages)} messages"
            )

        # Prepare messages for generation
        messages = state.system_messages + optimized_messages

        # Generate response
        try:
            assistant_message = generate(
                model=self.llm,
                tools=self.tools,
                messages=messages,
                **self.llm_args,
            )
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            # Create fallback error response
            assistant_message = AssistantMessage(
                role="assistant",
                content="I encountered an error while processing your request. Please try again or contact support if the issue persists.",
            )

        # Parse the message with enhanced parsing
        try:
            parse_message_enhanced(assistant_message)
        except AgentError as e:
            logger.error(f"Message parsing failed: {e}")
            # Try to recover with basic parsing
            assistant_message.content = (
                assistant_message.content
                or "I apologize, but I had trouble formulating a proper response. Could you please rephrase your request?"
            )
            assistant_message.reasoning = None
            assistant_message.tool_calls = None

        # Store reasoning in history for analysis
        if assistant_message.reasoning:
            state.reasoning_history.append(
                {
                    "turn": len(state.messages),
                    "reasoning": assistant_message.reasoning,
                    "message_type": "tool_call"
                    if assistant_message.tool_calls
                    else "response",
                }
            )

        # Add to state
        state.messages.append(assistant_message)

        return assistant_message, state

    def set_seed(self, seed: int):
        """Set the seed for the LLM."""
        if self.llm is None:
            raise ValueError("LLM is not set")
        cur_seed = self.llm_args.get("seed", None)
        if cur_seed is not None:
            logger.warning(f"Seed is already set to {cur_seed}, resetting it to {seed}")
        self.llm_args["seed"] = seed

    def get_reasoning_summary(self, state: LLMAgentV2State) -> Dict[str, Any]:
        """Get a summary of the reasoning process for analysis."""
        return {
            "total_reasoning_steps": len(state.reasoning_history),
            "enhanced_reasoning_count": sum(
                1
                for r in state.reasoning_history
                if isinstance(r.get("reasoning"), dict)
            ),
            "context_optimizations": state.context_metadata.get(
                "optimization_count", 0
            ),
            "reasoning_history": state.reasoning_history[-5:],  # Last 5 reasoning steps
        }
