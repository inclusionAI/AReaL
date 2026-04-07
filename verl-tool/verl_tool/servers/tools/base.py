from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tqdm import tqdm
registered_tools = {}
ALL_TOOLS = []
use_tqdm = False
def set_use_tqdm(value: bool):
    """
    Set whether to use tqdm for progress bars.
    """
    global use_tqdm
    use_tqdm = value

def get_tool_cls(tool_type):
    if tool_type in ALL_TOOLS:
        if tool_type == "base":
            return BaseTool

        import importlib
        module_path = f"verl_tool.servers.tools.{tool_type}"
        importlib.import_module(module_path)
        
        tool_class = registered_tools.get(tool_type)
        if tool_class is None:
            raise ValueError(f"Tool class for {tool_type} was not registered properly")
        return tool_class
    else:
        raise ValueError(f"Tool type {tool_type} not found. Available tools: {ALL_TOOLS}")


def register_tool(cls):
    """
    Decorator to register a tool class in the registered_tools dictionary.
    The class is registered using its tool_type attribute.
    """
    tool_type = getattr(cls, 'tool_type', cls.__name__)
    registered_tools[tool_type] = cls
    return cls


class BaseTool:
    tool_type = __name__
    
    def __init__(self, num_workers=1):
        self.num_workers = num_workers
        registered_tools[self.tool_type] = self.__class__
        self.env_cache = {}
        # self.executor = ThreadPoolExecutor(max_workers=num_workers)
    
    def get_usage_inst(self):
        """
        Get the usage instructions for the tool
        """
        return "Base usage instructions"
    
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
        if env == None:
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
        self.env_cache.pop(trajectory_id, None)
    
    def get_env_cache_keys(self):
        return list(self.env_cache.keys())
    
    def parse_action(self, action:str):
        """
        Parse the raw action string (which is the llm response) into a actual action and it's contents
        Args:
            action: The raw action string
        Returns:
            action: The parsed action
            valid: Whether the action is valid
        """
        action = action[:10]
        valid = True
        return action, valid
    
    def get_action_priority(self, action: str, extra_field: dict) -> int:
        """
        Get the priority for handling this action. Higher numbers = higher priority.
        Return -1 if this tool cannot handle the action.
        
        Args:
            action: The raw action string
            extra_field: Extra fields associated with the action
        Returns:
            priority: Integer priority (-1 means cannot handle, 0 = default, positive = higher priority)
        """
        _, valid = self.parse_action(action)
        return 0 if valid else -1
    
    def conduct_action(self, trajectory_id, action, extra_field):
        """
        Conduct the action on the environment and return the observation
        Args:
            trajectory_id: The trajectory ID
            action: The action to conduct
            extra_field: Extra data to include in the request
        Returns:
            observation: The observation after conducting the
            done: Whether the trajectory is done
            valid: Whether the action is valid
        """
        parsed_action, is_valid = self.parse_action(action)
        env = self.load_env(trajectory_id)
        
        # any other processing that gets the observation, whether the trajectory is done, and whether the action is valid
        observation = f"Base observation for {trajectory_id} in turn {env['metadata']['turns']}"
        done = True
        valid = True
        
        self.update_env(trajectory_id, env, parsed_action, is_valid, extra_field, observation)
        self.save_env(trajectory_id, env)
        
        return observation, done, valid
    
    def maybe_cleanup_env(self, trajectory_ids, actions, extra_fields):
        """
        Maybe clean up the environments for the given trajectory IDs and actions
        Args:
            trajectory_ids: The list of trajectory IDs
            actions: The list of actions
            extra_fields: Extra data to include in the request
        """
        for i in range(len(trajectory_ids)):
            if extra_fields[i].get('is_last_step', False):
                # delete the environment if it's the last step
                if self.has_env(trajectory_ids[i]):
                    self.delete_env(trajectory_ids[i])
        
    def get_observations(self, trajectory_ids, actions, extra_fields):
        """
        Get the observations for the given trajectory IDs and actions
        Args:
            trajectory_ids: The list of trajectory IDs
            actions: The list of actions
            extra_fields: Extra data to include in the request
        Returns:
            observations: The list of observations
            dones: The list of done flags
            valids: The list of valid flags
        """
        if len(trajectory_ids) <= 4: # heuristic to use single-threaded execution for small number of trajectories
            results = []
            for i in tqdm(range(len(trajectory_ids)), desc=f"Getting observations using tool {self.tool_type}", disable=not use_tqdm):
                trajectory_id = trajectory_ids[i]
                action = actions[i]
                extra_field = extra_fields[i]
                results.append(self.conduct_action(trajectory_id, action, extra_field))
        else:
            with ThreadPoolExecutor(max_workers=min(self.num_workers, len(trajectory_ids))) as executor:
                results = list(tqdm(executor.map(self.conduct_action, trajectory_ids, actions, extra_fields),
                                                total=len(trajectory_ids), desc=f"Getting observations using tool {self.tool_type}", 
                                                disable=not use_tqdm))
        
        observations, dones, valids = zip(*results)
        self.maybe_cleanup_env(trajectory_ids, actions, extra_fields)
        return observations, dones, valids

# go through all files in the tools directory and register them
cur_dir = Path(__file__).parent
excluding_files = ["__init__.py", "base.py"]
ALL_TOOLS.append("base")
for file in cur_dir.iterdir():
    if file.is_file() and file.name not in excluding_files:
        ALL_TOOLS.append(file.stem)
