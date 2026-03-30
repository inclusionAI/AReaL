"""Prompt templates for GeoEdit environment."""

from geo_edit.prompts.system_prompts import TOOL_CALL_SYSTEM_PROMPT

# Initial observation: system prompt + tools + question + image placeholder
GEO_EDIT_TEMPLATE_NO_HIS = """{system_prompt}

Available tools:
{tool_definitions}

Use this format for tool calls:
<action>{{"name": "tool_name", "arguments": {{"param1": "value1"}}}}</action>

When you have the final answer:
<answer>your answer here</answer>

Task: {task_prompt}
<image>"""

# Subsequent turns: include history and latest tool result
GEO_EDIT_TEMPLATE = """{system_prompt}

Available tools:
{tool_definitions}

Use this format for tool calls:
<action>{{"name": "tool_name", "arguments": {{"param1": "value1"}}}}</action>

When you have the final answer:
<answer>your answer here</answer>

Task: {task_prompt}

Previous interactions (last {history_length} steps):
{history}

Latest tool result:
{tool_result}
<image>"""
