import re
from typing import Any, Dict, Tuple

from areal.utils import logging

from .base import BaseTool, ToolCallStatus, ToolDescription, ToolMarkers, ToolType

logger = logging.getLogger("Python Tool")


def extract_python_code(text: str) -> str:
    """Extract Python code from text, supporting two formats:
    1. ```python\n...\n```
    2. <python>...</python>

    Args:
        text: Text containing Python code

    Returns:
        Extracted Python code, returns empty string if not found
    """
    # Try to match ```python``` format, match only the last occurrence from back to front
    pattern1 = r"```python\n(.*?)\n```"
    matches1 = list(re.finditer(pattern1, text, re.DOTALL | re.IGNORECASE))
    if matches1:
        last_match = matches1[-1]
        code = last_match.group(1).strip()
        logger.debug(
            f"Extracted Python code from ```python``` format (last occurrence): {code[:100]}..."
        )
        return code

    # Try to match <python></python> format, match only the last occurrence from back to front
    pattern2 = r"<python>(.*?)</python>"
    matches2 = list(re.finditer(pattern2, text, re.DOTALL | re.IGNORECASE))
    if matches2:
        last_match = matches2[-1]
        code = last_match.group(1).strip()
        logger.debug(
            f"Extracted Python code from <python> format (last occurrence): {code[:100]}..."
        )
        return code

    logger.warning("No Python code block found in either format")
    return ""


class PythonTool(BaseTool):
    """Qwen Python code execution tool"""

    def __init__(self, timeout: int = 30, debug_mode: bool = False):
        super().__init__(timeout, debug_mode)
        from qwen_agent.tools.python_executor import PythonExecutor

        self.python_executor = PythonExecutor()

    @property
    def tool_type(self) -> ToolType:
        return ToolType.PYTHON

    @property
    def description(self) -> ToolDescription:
        return ToolDescription(
            name="python_executor",
            description="Execute Python code. Supports variable calculation, data processing, algorithm implementation, etc.",
            parameters={"code": "The Python code string to execute"},
            parameter_prompt="Please provide the Python code to execute. Supports variable calculation, data processing, algorithm implementation, etc.",
            example="```python\na=1\nb=1\nprint(f'The a+b result is {a+b}')\n```\n or \n<python>\na=1\nb=1\nprint(f'The a+b result is {a+b}')\n</python>",
        )

    @property
    def markers(self) -> ToolMarkers:
        return ToolMarkers(
            start_markers=["```python", "<python>"], end_markers=["```", "</python>"]
        )

    def parse_parameters(self, text: str) -> Dict[str, Any]:
        """Extract Python code from text, supporting two formats: ```python``` and <python>"""
        code = extract_python_code(text)
        return {"code": code}

    def execute(self, parameters: Dict[str, Any]) -> Tuple[str, ToolCallStatus]:
        """Execute Python code"""
        code = parameters.get("code", "")
        if not code:
            return "Error: No code provided", ToolCallStatus.ERROR

        if self.debug_mode:
            logger.debug(f"[FAKE] Executing Python code: {code[:100]}...")
            return "dummy python output", ToolCallStatus.SUCCESS

        try:
            # Directly call apply to avoid using ProcessPool in async environment
            result = self.python_executor.apply(code)
            logger.debug(f"Python execution completed: {str(result)[:100]}...")
            return str(result), ToolCallStatus.SUCCESS
        except Exception as e:
            logger.error(f"Python execution error: {e}")
            return f"Error: {str(e)}", ToolCallStatus.ERROR
