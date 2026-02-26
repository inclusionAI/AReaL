"""System prompts for main agents."""

from __future__ import annotations
from geo_edit.utils.logger import setup_logger

logger = setup_logger(__name__)


TOOL_CALL_SYSTEM_PROMPT = """
You are an advanced AI agent capable of complex
reasoning and tool usage. You must strictly adhere
to the following protocol for every interaction:
1. Call appropriate tools based on the task;
2. Only call one tool per action;
2. Reasoning Before Action: before selecting a tool,
you must analyze the user's request and determine
the necessary steps. Output your internal monologue
and logic inside <think> and </think> tags;
3. Tool Execution: If a tool is required, generate the
tool call immediately after your reasoning.
4. Reasoning After Action: Once you receive the
output from a tool, you must analyze the results to
determine if further actions are needed or if the task
is complete. Output this analysis inside <think> and </think> tags and then decide your next step, which could be calling another tool or providing the final answer.;
5. Final Output: When you have formulated your
conclusion, you must wrap your final answer in
<answer> and </answer> tags.
"""

USER_PROMPT='''
Please answer the following {task_type} question:
Question: {Question}
Please provide a complete step-by-step solution to
this problem. Your reasoning should:
1. Analyze the problem systematically
2. Check if the tool execution and answer are correct
3. If there are errors, explain what went wrong and
provide the correct reasoning
4. Provide the final answer
Provide your detailed reasoning between <think>
and </think> tags, then give your final answer 
between <answer> and </answer> tags.
'''

VLLM_FORCE_TOOL_CALL_PROMPT = """
Force tool-call mode:
1. ALWAYS call the appropriate tool first;
2. NEVER provide answers without tool results;
"""

VLLM_NO_TOOL_SYSTEM_PROMPT = """
You are an advanced AI assistant capable of complex reasoning.
You must strictly adhere to the following protocol:

1. Reasoning Process: Before providing your answer, analyze the
problem step by step. Output your reasoning inside <think> and </think> tags.

2. Final Output: When you have formulated your conclusion,
wrap your final answer in <answer> and </answer> tags.
"""

API_NO_TOOL_SYSTEM_PROMPT = """
You are an advanced AI assistant capable of complex reasoning. When you have formulated your conclusion, wrap your final answer in <answer> and </answer> tags.
"""

TOOL_EXECUTION_SUCCESS_PROMPT = (
    "All your tool calls were executed successfully. "
    "Tool outputs may include a new Observation image or text analysis results. "
    "Please review the latest tool execution results, compare new Observations with previous Observations "
    ", and incorporate any text outputs "
    "into your reasoning. Based on all available evidence, decide your next action or provide the final answer."
)

TOOL_EXECUTION_FAILURE_PROMPT = (
    "Some of your tool calls failed or returned invalid outputs. "
    "Please carefully inspect each tool execution result (including text error messages and any partial outputs), "
    "identify which tool calls failed, and decide your recovery strategy. "
    "When planning next steps, reuse previous Observations and any successful text outputs."
)


def get_system_prompt(model_type: str, tool_mode: str | None = None) -> str:
    """Select system prompt based on model type.

    Args:
        model_type: Model type (Google, OpenAI, vLLM, SGLang).
        tool_mode: Tool mode (force, auto, direct).

    Returns:
        System prompt string.
    """
    model_type_normalized = model_type.strip().lower()
    tool_mode_normalized = tool_mode.strip().lower() if tool_mode else None

    if model_type_normalized in {"vllm", "sglang"}:
        if tool_mode_normalized == "force":
            return f"{TOOL_CALL_SYSTEM_PROMPT}\n\n{VLLM_FORCE_TOOL_CALL_PROMPT}"
        elif tool_mode_normalized == "direct":
            return VLLM_NO_TOOL_SYSTEM_PROMPT
    else:
        if tool_mode_normalized == "direct":
            return API_NO_TOOL_SYSTEM_PROMPT
        else:
            return TOOL_CALL_SYSTEM_PROMPT
