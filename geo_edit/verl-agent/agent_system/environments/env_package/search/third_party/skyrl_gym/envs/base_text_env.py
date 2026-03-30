from typing import Any, Dict, List, Optional, TypedDict
from agent_system.environments.env_package.search.third_party.skyrl_gym import Env
from typing import Tuple

MessageType = Dict[str, str]
ConversationType = List[MessageType]


class BaseTextEnvStepOutput(TypedDict):
    observations: List[Dict[str, str]]  # OpenAI API Messages Format
    reward: float
    done: bool
    metadata: Dict[str, Any]
    postprocessed_action: Optional[str] = None


class BaseTextEnv(Env[str, str]):
    """
    Base environment class for all text-in / text-out environments.
    Supports tool-calling and multi-turn trajectories.

    Exposes only `step`, `init` and `close`.

    Input Types:
        - ObsType: str (tool output, LLM input)
        - ActType: str (LLM output)
    """

    def __init__(self):
        super().__init__()

        # Metadata
        self.turns = 0
        self.max_turns = 1

        # Tool groups
        self.tool_groups = []
        self.tool_to_toolgroup = {}

    def init_tool_groups(self, tool_groups: List = []) -> None:
        """
        Initialize the tool groups for the environment.
        """
        # Find ToolGroup for a given tool
        self.tool_groups = tool_groups
        self.tool_to_toolgroup = {}
        for tool_group in self.tool_groups:
            self.tool_to_toolgroup.update(tool_group.get_tool_to_group_mapping())

    def _execute_tool(self, tool_group_name: str, tool_name: str, tool_input: Any) -> str:
        """
        Find the right ToolGroup and Tool and execute it.
        """
        for group in self.tool_groups:
            if group.name == tool_group_name:
                return group.execute_tool(tool_name, *tool_input)  # tool_input must be tuple or list

        raise ValueError(f"ToolGroup '{tool_group_name}' not found.")

    def step(self, action: str) -> BaseTextEnvStepOutput:
        """
        Runs one environment step.

        Return:
        - new_obs: [{"role": "user", "content": observation}]
        - reward: float
        - done: bool
        - postprocessed_action: Optional[str]
        - Dict[str, Any]: any metadata
        """
        pass

    def init(self, prompt: ConversationType) -> Tuple[ConversationType, Dict[str, Any]]:
        """
        Return the first prompt to be given to the model and optional metadata.
        """
        return prompt, {}

    def close(self):
        """
        Closes the environment, override if needed by subclasses.
        """
        pass
