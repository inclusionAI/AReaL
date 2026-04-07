"""
Bash Terminal Tool for secure command execution with persistent shell sessions
"""
from .base import BaseTool, register_tool
import regex as re
import os
import uuid
import shutil
from typing import Tuple, Dict, Any, Optional, Union, List
import pty
from .utils.bash_session import BashSession, check_forbidden_commands
# Timeout for command execution in seconds
TIMEOUT = 10


@register_tool
class BashTerminalTool(BaseTool):
    tool_type = "bash_terminal"
    timeout = TIMEOUT
    stop_tokens = ["```output", "<o>", "<tool_call>"]
    use_firejail = True  # Default to False to avoid resource issues
    
    def __init__(self, num_workers=1):
        super().__init__(num_workers)
        self.sessions = {}  # trajectory_id -> BashSession
    
    def get_usage_inst(self):
        return "You are able to execute bash commands in a persistent shell session with file operations restricted to temporary directories."
    
    def has_env(self, trajectory_id):
        """Check if the environment for the given trajectory_id exists"""
        return trajectory_id in self.env_cache
    
    def load_env(self, trajectory_id):
        """Load the environment for the given trajectory_id"""
        env = self.env_cache.get(trajectory_id)
        if env is None:
            env = {
                "trajectory_id": trajectory_id,
                "metadata": {
                    "turns": 0,
                },
                "previous_obs": [],
                "temp_dir": None,
                "session_active": False,
            }
        return env
    
    def save_env(self, trajectory_id, env):
        """Save the environment for the given trajectory_id"""
        self.env_cache[trajectory_id] = env
    
    def update_env(self, trajectory_id, env, action, is_valid, extra_field, observation, **kwargs):
        """Update the environment for the given trajectory_id"""
        env["metadata"]["turns"] += 1
        env["previous_obs"].append({
            "action": action,
            "is_valid": is_valid,
            "observation": observation,
            "extra_field": extra_field,
            **kwargs
        })
    
    def delete_env(self, trajectory_id):
        """Delete the environment for the given trajectory_id"""
        # Clean up session if it exists
        if trajectory_id in self.sessions:
            self.sessions[trajectory_id].cleanup()
            del self.sessions[trajectory_id]
        
        # Clean up temp directory
        if trajectory_id in self.env_cache:
            env = self.env_cache[trajectory_id]
            if env.get("temp_dir") and os.path.exists(env["temp_dir"]):
                try:
                    shutil.rmtree(env["temp_dir"])
                except Exception:
                    pass
            del self.env_cache[trajectory_id]
    
    def _get_or_create_session(self, trajectory_id, env):
        """Get existing session or create a new one"""
        if trajectory_id not in self.sessions:
            # Create temp directory if it doesn't exist
            if not env.get("temp_dir"):
                temp_dir = os.path.join(os.getcwd(), "tmp/bash", str(uuid.uuid4().hex))
                os.makedirs(temp_dir, exist_ok=True)
                env["temp_dir"] = temp_dir
            
            # Create new session
            try:
                session = BashSession(env["temp_dir"], self.use_firejail)
                self.sessions[trajectory_id] = session
                env["session_active"] = True
            except Exception as e:
                env["session_active"] = False
                raise e
        
        return self.sessions[trajectory_id]
    
    def parse_action(self, action: str) -> Tuple[str, bool]:
        """
        Parse the raw action string into bash commands.
        
        Args:
            action: Raw action string containing bash commands
            
        Returns:
            Tuple containing the extracted commands and a validity flag
        """
        # Try to find bash commands in various formats
        all_valid_bash_code = re.findall(r"<bash>(.*?)</bash>", action, re.DOTALL)
        
        if not all_valid_bash_code:
            all_valid_bash_code = re.findall(r"```\s*(?:bash|sh|shell)(.*?)```", action, re.DOTALL)
        
        if not all_valid_bash_code:
            all_valid_bash_code = re.findall(r"```\s*terminal(.*?)```", action, re.DOTALL)
        
        if len(all_valid_bash_code) == 0:
            return "", False
        
        # Combine all command blocks
        parsed_commands = "\n".join([cmd.strip() for cmd in all_valid_bash_code])
        
        return parsed_commands, True
    
    def conduct_action(self, trajectory_id, action, extra_field):
        """
        Execute the parsed bash commands in a persistent shell session.
        
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
            valid = False
        else:
            # Check for forbidden commands first
            detected_forbidden_commands = check_forbidden_commands(parsed_action)
            if detected_forbidden_commands:
                execution_result = f"Execution blocked: Command contains potentially dangerous operations. Forbidden commands detected: {', '.join(detected_forbidden_commands)}"
                observation = execution_result
                valid = True
            else:
                try:
                    # Get or create persistent session
                    session = self._get_or_create_session(trajectory_id, env)
                    # Execute command in persistent session
                    execution_result = session.execute_command_like_shell(parsed_action.splitlines(), self.timeout)
                    observation = execution_result.strip(' \n')
                    
                except Exception as e:
                    raise e
                    execution_result = f"Session error: {str(e)}"
                    observation = execution_result
                    env["session_active"] = False
            
            # Format the observation based on the action type
            if action.endswith("```output"):
                observation = "\n" + observation + "\n```\n"
            elif action.endswith("</tool_call>"):
                observation = "\n```output\n" + observation + "\n```\n"
            elif action.endswith("<o>"):
                observation = "\n" + observation + "\n</o>\n"
            elif action.endswith("</bash>") or "</bash>" in action:
                observation = "\n<o>\n" + observation + "\n</o>\n"
            elif "<|calling system for feedback|>" in action:
                if "```bash" in action or "```sh" in action:
                    observation = "\n```output\n" + observation + "\n```\n"
                elif "<bash>" in action:
                    observation = "\n<o>\n" + observation + "\n</o>\n"
                else:
                    observation = "\n" + observation + "\n"
            elif action.strip(' \n').endswith("```") or ("```bash" in action or "```sh" in action):
                if action.count("```") % 2 == 0:
                    observation = "\n```output\n" + observation + "\n```\n"
                else:
                    observation = "output\n" + observation + "\n```\n"
            else:
                observation = "\n" + observation + "\n"

            valid = True
            done = False
        
        self.update_env(trajectory_id, env, parsed_action, is_valid, extra_field, execution_result)
        self.save_env(trajectory_id, env)
        
        return observation, done, valid