import asyncio
import re
import sys
import tempfile
import os
from typing import Dict, Any, Tuple

from areal.utils import logging
from .base import BaseTool, ToolType, ToolDescription, ToolMarkers, ToolCallStatus

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
        logger.debug(f"Extracted Python code from ```python``` format (last occurrence): {code[:100]}...")
        return code
    
    # Try to match <python></python> format, match only the last occurrence from back to front
    pattern2 = r"<python>(.*?)</python>"
    matches2 = list(re.finditer(pattern2, text, re.DOTALL | re.IGNORECASE))
    if matches2:
        last_match = matches2[-1]
        code = last_match.group(1).strip()
        logger.debug(f"Extracted Python code from <python> format (last occurrence): {code[:100]}...")
        return code
    
    logger.warning("No Python code block found in either format")
    return ""


class QwenPythonTool(BaseTool):
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
            parameters={
                "code": "The Python code string to execute"
            },
            parameter_prompt="Please provide the Python code to execute. Supports variable calculation, data processing, algorithm implementation, etc.",
            example="```python\na=1\nb=1\nprint(f'The a+b result is {a+b}')\n```\n or \n<python>\na=1\nb=1\nprint(f'The a+b result is {a+b}')\n</python>"
        )
    
    @property
    def markers(self) -> ToolMarkers:
        return ToolMarkers(
            start_markers=["```python", "<python>"],
            end_markers=["```", "</python>"]
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


class PythonTool(BaseTool):
    """Python code execution tool (sandbox version)"""
    
    @property
    def tool_type(self) -> ToolType:
        return ToolType.PYTHON
    
    @property
    def description(self) -> ToolDescription:
        return ToolDescription(
            name="python_executor",
            description="Execute Python code, supporting variable calculation, data processing, algorithm implementation, etc.",
            parameters={
                "code": "Python code string to execute"
            },
            parameter_prompt="Please provide Python code to execute, supporting variable calculation, data processing, algorithm implementation, etc.",
            example="```python\na=1\nb=1\nprint(f'The a+b result is {a+b}')\n```\nor\n<python>\na=1\nb=1\nprint(f'The a+b result is {a+b}')\n</python>"
        )
    
    @property
    def markers(self) -> ToolMarkers:
        return ToolMarkers(
            start_markers=["```python", "<python>"],
            end_markers=["```", "</python>"]
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
        
        logger.debug(f"Executing Python code: {code[:100]}...")
        
        try:
            # Security check
            if not self._is_safe_code(code):
                logger.warning("Unsafe code detected, blocking execution")
                return "Error: Unsafe code detected", ToolCallStatus.ERROR
            
            # Execute in sandbox
            result = asyncio.run(self._execute_in_sandbox(code))
            logger.debug(f"Python execution completed: {result[:100]}...")
            return result, ToolCallStatus.SUCCESS
            
        except Exception as e:
            logger.error(f"Python execution error: {e}")
            return f"Error: {str(e)}", ToolCallStatus.ERROR
    
    def _is_safe_code(self, code: str) -> bool:
        """Check if code is safe"""
        dangerous_patterns = [
            r"import\s+os",
            r"import\s+subprocess",
            r"import\s+sys",
            r"__import__",
            r"exec\s*\(",
            r"eval\s*\(",
            r"open\s*\(",
            r"file\s*\(",
            r"input\s*\(",
            r"raw_input\s*\(",
            r"exit\s*\(",
            r"quit\s*\(",
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                return False
        
        return True
    
    async def _execute_in_sandbox(self, code: str) -> str:
        """Execute Python code in sandbox"""
        sandbox_dir = tempfile.mkdtemp(prefix="python_sandbox_")
        
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(
                mode='w', 
                suffix='.py', 
                dir=sandbox_dir, 
                delete=False
            ) as f:
                f.write(code)
                temp_file = f.name
            
            # Execute Python code
            process = await asyncio.create_subprocess_exec(
                sys.executable, temp_file,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=sandbox_dir
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), 
                timeout=self.timeout
            )
            
            # Clean up temporary file
            os.unlink(temp_file)
            
            if process.returncode == 0:
                result = stdout.decode('utf-8').strip()
                return result
            else:
                error_msg = stderr.decode('utf-8').strip()
                return f"Error: {error_msg}"
                
        except asyncio.TimeoutError:
            return "Error: Execution timeout"
        except Exception as e:
            return f"Error: {str(e)}"
        finally:
            # Clean up sandbox directory
            try:
                import shutil
                if os.path.exists(sandbox_dir):
                    shutil.rmtree(sandbox_dir)
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup sandbox: {cleanup_error}")
