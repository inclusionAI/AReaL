from typing import Dict, Callable, Any, Optional, List


class tool:
    """
    A tool that can be used to execute a function.
    """

    def __init__(self, func: Callable):
        self.func = func
        self.name = func.__name__

    def __get__(self, instance, owner):
        if instance is None:
            return self  # Return the descriptor itself when accessed from the class
        return lambda *args, **kwargs: self.func(instance, *args, **kwargs)


class ToolGroup:
    """
    A group of tools that can be used together.
    """

    def __init__(self, name: str):
        self.name = name
        self._tool_registry: Dict[str, Callable] = {}
        self._register_tools()

    def get_name(self):
        return self.name

    def _register_tools(self):
        # Register all methods decorated with @tool

        # Tool names must be unique across tool groups.
        # TODO: Support duplicate tool names across tool groups via namespacing
        for attr_name in dir(self):
            # Look for the descriptor on the class, not the instance
            raw = getattr(type(self), attr_name, None)
            if isinstance(raw, tool):
                self._tool_registry[raw.name] = getattr(self, attr_name)

    def get_tool(self, name: str) -> Optional[Callable]:
        # Get a tool by name, returns None if not found
        return self._tool_registry.get(name)

    def get_tool_names(self) -> List[str]:
        # Get all available tool names
        return list(self._tool_registry.keys())

    def execute_tool(self, name: str, *args, **kwargs) -> Any:
        # Execute a tool by name with given arguments
        tool_func = self.get_tool(name)
        if tool_func:
            return tool_func(*args, **kwargs)
        raise ValueError(f"Tool '{name}' not found in group '{self.name}'.")

    def get_tool_to_group_mapping(self) -> Dict[str, str]:
        # Get mapping of tool names to group name
        return {name: self.name for name in self._tool_registry}
