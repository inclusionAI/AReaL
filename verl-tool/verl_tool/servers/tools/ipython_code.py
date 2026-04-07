from .base import BaseTool, register_tool
import regex as re
import json
from typing import Tuple
import subprocess
import time
import requests
import atexit
import socket
import logging
from .utils.ipython_tool import call_python_script_with_ipython, remove_kernel
logger = logging.getLogger(__name__)

# Timeout for code execution in seconds
TIMEOUT = 10
PRE_IMPORT_LIBS = "from string import *\nfrom re import *\nfrom datetime import *\nfrom collections import *\nfrom heapq import *\nfrom bisect import *\nfrom copy import *\nfrom math import *\nfrom random import *\nfrom statistics import *\nfrom itertools import *\nfrom functools import *\nfrom operator import *\nfrom io import *\nfrom sys import *\nfrom json import *\nfrom builtins import *\nfrom typing import *\nimport string\nimport re\nimport datetime\nimport collections\nimport heapq\nimport bisect\nimport copy\nimport math\nimport random\nimport statistics\nimport itertools\nimport functools\nimport operator\nimport io\nimport sys\nimport json\nsys.setrecursionlimit(6*10**5)\n\n"

def check_forbidden_imports(code: str) -> bool:
    """
    Checks if the code contains imports of potentially dangerous packages.
    
    Args:
        code: Python code string to analyze
        
    Returns:
        Boolean indicating if the code contains forbidden imports
    """
    # List of potentially dangerous modules that could affect the host system
    forbidden_modules = [
        'subprocess', 'multiprocessing', 'threading',
        'socket', 'psutil', 'resource', 'ctypes'
    ]
    
    # Simple string-based check for import statements
    for module in forbidden_modules:
        if f"import {module}" in code or f"from {module}" in code:
            return True
    
    # Check for os.system, os.popen, and similar dangerous calls
    dangerous_patterns = [
        "os.system", "os.popen", "os.spawn", "os.fork", 
        "os.exec", "sys.exit", "os._exit", "os.kill"
    ]
    
    for pattern in dangerous_patterns:
        if pattern in code:
            return True
    
    return False


def find_free_port(start_port: int = 8000, max_attempts: int = 100) -> int:
    """
    Find a free port starting from start_port.
    
    Args:
        start_port: Port to start searching from
        max_attempts: Maximum number of ports to try
        
    Returns:
        Free port number
    """
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"Could not find free port after {max_attempts} attempts")

@register_tool
class IPythonTool(BaseTool):
    tool_type = "ipython_code"
    timeout = TIMEOUT
    stop_tokens = ["```output", "<output>", "<tool_call>"]
    enable_history_code_execution = False
    done_without_error = False
    pre_import_lib = False
    
    def __init__(self, server_host: str = "127.0.0.1", server_port: int = None, **kwargs):
        """
        
        Args:
            server_host: Host for the IPython server
            server_port: Port for the IPython server (None = auto-select)
            **kwargs: Additional arguments passed to BaseTool
        """
        super().__init__(**kwargs)
    
    def get_usage_inst(self):
        return "You are able to write and execute Python code using IPython with persistent state across executions."
    
    def has_env(self, trajectory_id):
        """
        Check if the environment for the given trajectory_id exists
        """
        return trajectory_id in self.env_cache
    
    def load_env(self, trajectory_id):
        """
        Load the environment for the given trajectory_id
        """
        env = self.env_cache.get(trajectory_id)
        if env is None:
            env = {
                "trajectory_id": trajectory_id,
                "metadata": {
                    "turns": 0,
                },
                "previous_obs": [],
            }
        
        return env
    
    def save_env(self, trajectory_id, env):
        """
        Save the environment for the given trajectory_id
        """
        self.env_cache[trajectory_id] = env
    
    def update_env(self, trajectory_id, env, action, is_valid, extra_field, observation, **kwargs):
        """
        Update the environment for the given trajectory_id
        """
        env["metadata"]["turns"] += 1
        env["previous_obs"].append({
            "action": action,
            "is_valid": is_valid,
            "observation": observation,
            "extra_field": extra_field,
            **kwargs
        })
    
    def delete_env(self, trajectory_id):
        """
        Delete the environment for the given trajectory_id
        """
        if trajectory_id in self.env_cache:
            del self.env_cache[trajectory_id]
        
        remove_kernel(trajectory_id)
    
    def parse_action(self, action: str) -> Tuple[str, bool]:
        """
        Parse the raw action string (which is the llm response) into an actual action and its contents.
        Ensures that the parsed code is valid and safe for execution.
        
        Args:
            action: Raw action string containing Python code
            
        Returns:
            Tuple containing the extracted code and a validity flag
        """
        # Try to find Python code in various formats
        all_valid_python_code = re.findall(r"<python>(.*?)</python>", action, re.DOTALL)
        
        if not all_valid_python_code:
            all_valid_python_code = re.findall(r"```\n?python(.*?)```", action, re.DOTALL)
            
            if not all_valid_python_code:
                tool_call_match = re.search(r"<tool_call>(.*?)</tool_call>", action, re.DOTALL)
                if not tool_call_match:
                    all_valid_python_code = []
                else:
                    tool_call_content = tool_call_match.group(1).strip()
                    call_json = json.loads(tool_call_content)
                    tool_name = call_json.get("name", "").strip()
                    args = call_json.get("arguments", {})
                    if tool_name == "python":
                        code = args.get("code", "").strip()
                        if code:
                            all_valid_python_code = [code]
                        else:
                            all_valid_python_code = []
        
        if len(all_valid_python_code) == 0:
            return "", False
        
        # Use all the code blocks
        parsed_code = "\n".join([code.strip() for code in all_valid_python_code])
        
        return parsed_code, True
    
    def conduct_action(self, trajectory_id, action, extra_field):
        """
        Execute the parsed action using IPython
        
        Args:
            trajectory_id: ID for tracking the action
            action: Raw action string
            extra_field: Additional parameters
            
        Returns:
            Tuple containing observation, done flag, and validity flag
        """
        parsed_action, is_valid = self.parse_action(action)
        env = self.load_env(trajectory_id)
        
        if not is_valid:
            observation = ""
            execution_result = ""
            done = False
            valid = False
        else:
            # Extract stdin if provided in extra_field
            stdin = extra_field.get("stdin", "") if extra_field else None
            
            test_input = re.findall(r"```input\n(.*?)\n```", action, re.DOTALL)
            if len(test_input) > 0:
                stdin = test_input[0].strip()
            
            # Determine what code to execute
            if self.enable_history_code_execution:
                previous_parsed_code = [obs["action"] for obs in env["previous_obs"] if obs["is_valid"]]
                code_to_execute = previous_parsed_code + [parsed_action]
            else:
                code_to_execute = parsed_action
            
            stdout, success = call_python_script_with_ipython(
                request_id=trajectory_id,
                script=code_to_execute,
                timeout=self.timeout,
            )
            has_error = not success
            
            execution_result = stdout
            execution_result = execution_result.lstrip(' \n')
            observation = execution_result
            
            # Format the observation based on the action type
            if action.endswith("```output"):
                observation = "\n" + observation + "\n```\n"
            elif action.endswith("</tool_call>"):
                observation = observation
                # observation = "\n```output\n" + observation + "\n```\n"
            elif action.endswith("<output>"):
                observation = "\n" + observation + "\n</output>\n"
            elif action.endswith("</python>") or "</python>" in action:
                observation = "\n<output>\n" + observation + "\n</output>\n"
            elif "<|calling system for feedback|>" in action:
                if "```python" in action:
                    observation = "\n```output\n" + observation + "\n```\n"
                elif "<python>" in action:
                    observation = "\n<output>\n" + observation + "\n</output>\n"
                else:
                    observation = "\n" + observation + "\n"
            elif action.strip(' \n').endswith("```") or "```python" in action:
                if action.count("```") % 2 == 0:
                    observation = "\n```output\n" + observation + "\n```\n"
                else:
                    observation = "output\n" + observation + "\n```\n"
            else:
                observation = "\n" + observation + "\n"

            if self.done_without_error:
                if has_error:
                    done = False
                else:
                    done = True
            else: 
                done = False
            valid = True
            
            observation = {"obs": observation, "metrics": {"code_success": success, "code_lines": parsed_action.count('\n') + 1}, "timeout": "execution time out" in execution_result.lower()}
        
        self.update_env(trajectory_id, env, parsed_action, is_valid, extra_field, execution_result)
        self.save_env(trajectory_id, env)
        
        return observation, done, valid
    
    @classmethod
    def shutdown_server(cls):
        """
        Class method to shutdown the shared server.
        Call this when shutting down the application.
        """
        if cls._server_manager:
            cls._server_manager.shutdown()
            cls._server_manager = None