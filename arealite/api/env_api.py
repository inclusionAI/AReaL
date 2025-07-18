import abc
from typing import Any, Dict, List


class Environment(abc.ABC):

    async def ainitialize(self):
        """
        Performs the initialization logic for the environment asynchronously.

        For stateful environments, this is where resources are created and
        prepared (e.g., launching a browser).
        """

    def list_tools(self) -> List[Dict[str, Any]]:
        """Lists all available tools in the environment."""
        return []

    async def aexecute(self, tool_name: str, tool_args: Dict[str, Any]) -> Any:
        """Executes a tool in the environment asynchronously."""
        raise NotImplementedError()

    async def aclose(self):
        """
        Destroys the environment asynchronously, releasing all held resources.

        This method is critical for stateful environments (e.g., a browser session).
        """
