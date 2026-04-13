from .base import BaseTool, register_tool
import regex as re
import requests
import json
from typing import Dict, List, Any, Optional, Tuple
import os


def is_code_safe(code: str, language: str) -> bool:
    """
    Basic safety check for code before execution.
    
    Args:
        code: Code string to analyze
        language: Programming language of the code
        
    Returns:
        Boolean indicating if the code appears safe
    """
    # Dictionary of dangerous patterns by language
    dangerous_patterns = {
        "python": [
            "import subprocess", "from subprocess", 
            "import multiprocessing", "from multiprocessing",
            "import threading", "from threading",
            "import socket", "from socket",
            "os.system", "os.popen", "os.spawn", "os.fork", 
            "os.exec", "sys.exit", "os._exit", "os.kill"
        ],
        "javascript": [
            "process.exit", "child_process", "require('child_process')",
            "fs.writeFile", "fs.write", "fs.unlink", "fs.rmdir"
        ],
        "cpp": [
            "system(", "exec(", "fork(", "popen("
        ],
        # Add patterns for other languages as needed
    }
    
    # Get patterns for the specific language or use an empty list if not defined
    patterns = dangerous_patterns.get(language.lower(), [])
    
    # Check for dangerous patterns
    for pattern in patterns:
        if pattern in code:
            return False
    
    return True


@register_tool
class SandboxFusionTool(BaseTool):
    tool_type = "sandbox_fusion"
    timeout = 10  # Default timeout in seconds
    sandbox_url = os.getenv("SANDBOX_FUSION_URL", "http://localhost:8080")
    
    def get_usage_inst(self):
        return "This tool allows execution of code in various programming languages using SandboxFusion."
    
    def parse_action(self, action: str) -> Tuple[Dict[str, Any], bool]:
        """
        Parse the raw action string to extract code and language.
        
        Args:
            action: The raw action string (LLM response)
            
        Returns:
            Tuple containing:
            - Dictionary with 'code' and 'language' keys
            - Boolean indicating if the parsing was successful
        """
        # Try to extract code from different formats
        code_block = None
        language = "python"  # Default language
        
        # Try explicit XML tags with language
        lang_tag_match = re.search(r"<([a-zA-Z0-9_]+)>(.*?)</\1>", action, re.DOTALL)
        if lang_tag_match:
            language = lang_tag_match.group(1).lower()
            code_block = lang_tag_match.group(2).strip()
        
        # Try markdown code blocks with language
        if not code_block:
            md_match = re.search(r"```([a-zA-Z0-9_]+)(.*?)```", action, re.DOTALL)
            if md_match:
                language = md_match.group(1).lower()
                code_block = md_match.group(2).strip()
        
        # Try plain markdown code blocks
        if not code_block:
            md_match = re.search(r"```(.*?)```", action, re.DOTALL)
            if md_match:
                code_block = md_match.group(1).strip()
        
        if not code_block:
            return {}, False
        
        # Map some common language aliases
        language_map = {
            "js": "javascript",
            "py": "python",
            "ts": "typescript",
            "rb": "ruby",
            "sh": "bash",
            # Add more mappings as needed
        }
        
        # Normalize language name
        language = language_map.get(language, language)
        
        return {"code": code_block, "language": language}, True
    
    def conduct_action(self, trajectory_id, action, extra_field):
        parsed_action, is_valid = self.parse_action(action)
        
        if not is_valid:
            observation = "No valid code block found. Please provide code in markdown format ```language\ncode\n``` or <language>code</language>."
            return observation, True, False
        
        code = parsed_action["code"]
        language = parsed_action["language"]
        
        # Check if code seems safe
        if not is_code_safe(code, language):
            observation = f"Execution blocked: Code contains potentially dangerous operations that are not allowed."
            return observation, True, False
        
        # Execute code using SandboxFusion
        try:
            result = self._execute_in_sandbox(code, language)
            observation = self._format_result(result)
            return observation, False, True
        except Exception as e:
            observation = f"Error executing code in SandboxFusion: {str(e)}"
            return observation, True, False
    
    def _execute_in_sandbox(self, code: str, language: str) -> Dict[str, Any]:
        """
        Execute code using the SandboxFusion API.
        
        Args:
            code: The code to execute
            language: The programming language
            
        Returns:
            Dictionary containing the execution results
        """
        endpoint = f"{self.sandbox_url}/run_code"
        
        payload = {
            "code": code,
            "language": language
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        response = requests.post(endpoint, json=payload, headers=headers, timeout=self.timeout)
        
        if response.status_code != 200:
            raise Exception(f"SandboxFusion API returned status code {response.status_code}: {response.text}")
        
        return response.json()
    
    def _format_result(self, result: Dict[str, Any]) -> str:
        """
        Format the execution result into a readable string.
        
        Args:
            result: The execution result from SandboxFusion
            
        Returns:
            Formatted string for display
        """
        formatted = "Execution results:\n\n"
        
        # Handle compile result if present
        if result.get("compile_result"):
            compile_status = result["compile_result"]["status"]
            formatted += f"Compilation: {compile_status}\n"
            
            if compile_status != "Finished":
                if result["compile_result"].get("stderr"):
                    formatted += f"Compilation errors:\n{result['compile_result']['stderr']}\n\n"
                return formatted
        
        # Handle run result
        if result.get("run_result"):
            run_status = result["run_result"]["status"]
            execution_time = result["run_result"].get("execution_time", 0)
            formatted += f"Execution: {run_status} (took {execution_time:.4f}s)\n"
            
            # Add stdout if available
            if result["run_result"].get("stdout"):
                formatted += f"\nOutput:\n{result['run_result']['stdout']}"
            
            # Add stderr if available
            if result["run_result"].get("stderr"):
                formatted += f"\nErrors:\n{result['run_result']['stderr']}"
        
        # Handle overall status
        if result.get("status") != "Success":
            formatted += f"\nStatus: {result.get('status')}"
            if result.get("message"):
                formatted += f" - {result.get('message')}"
        
        return formatted
    
"""
To start the docker image locally, see https://bytedance.github.io/SandboxFusion/docs/docs/get-started:
```bash
docker run -it -p 8080:8080 volcengine/sandbox-fusion:server-20241204
```
"""