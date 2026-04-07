from .base import BaseTool, register_tool
import regex as re
import ray
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@register_tool
class FinishTool(BaseTool):
    tool_type = "finish"
    timeout = 10
    
    def __init__(self, num_workers=1, other_tools:dict = {}):
        super().__init__(num_workers)
        self.other_tools = other_tools
    
    def get_usage_inst(self):
        return ""
    
    def parse_action(self, action:str):
        """
        Parse the raw action string to check for answer tags or finish conditions.
        Implements the finish condition logic that was originally in serve.py lines 107-109.
        """
        # Default behavior - trajectory ends without explicit answer. This finish tool will not be in matched until there is an extra field finish=True in the request
        return "", True
    
    def conduct_action(self, trajectory_id, action, extra_data):
        action, is_valid = self.parse_action(action)
        
        observation = ""
        done = True
        
        # Clean up environments for all tools
        for tool_type, tool in self.other_tools.items():
            if isinstance(tool, ray.actor.ActorHandle):
                logger.debug(f"FinishTool: Deleting env for tool {tool_type} and trajectory_id={trajectory_id}, has_env={ray.get(tool.has_env.remote(trajectory_id))}, env_cache_keys={ray.get(tool.get_env_cache_keys.remote())}")
                has_env = ray.get(tool.has_env.remote(trajectory_id))
                if has_env:
                    ray.get(tool.delete_env.remote(trajectory_id))
            else:
                logger.debug(f"FinishTool: Deleting env for tool {tool_type} and trajectory_id={trajectory_id}, has_env={tool.has_env(trajectory_id)}, env_cache_keys={list(tool.env_cache.keys())}")
                if tool.has_env(trajectory_id):
                    tool.delete_env(trajectory_id)
        
        return observation, done, is_valid
    
