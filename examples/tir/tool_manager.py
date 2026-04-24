import re
from typing import Any

from areal.api.sandbox_api import SandboxConfig
from areal.utils import logging

from tools import (  # isort: skip
    BaseTool,
    CalculatorTool,
    E2BSandboxPythonTool,
    PythonTool,
    ToolCallStatus,
    ToolType,
)


logger = logging.getLogger("Tool Manager")


class ToolRegistry:
    """Tool registry that manages all available tools.

    When a ``SandboxConfig`` is provided, ``ToolType.PYTHON`` is
    automatically backed by :class:`E2BSandboxPythonTool` (KVM-isolated)
    instead of the local in-process :class:`PythonTool`.

    Parameters
    ----------
    timeout : int
        Per-execution timeout.
    enabled_tools : str
        Semicolon-separated list of tool names to enable.
    debug_mode : bool
        If True, tools return dummy output.
    sandbox_config : SandboxConfig | None
        When provided and ``enabled=True``, Python execution uses the
        sandbox backend.  Calculator is always local (pure compute).
    """

    TOOL_NAMES: dict[str, ToolType] = {
        "python": ToolType.PYTHON,
        "calculator": ToolType.CALCULATOR,
        "bash": ToolType.BASH,
    }

    def __init__(
        self,
        timeout: int = 30,
        enabled_tools: str = "python;calculator",
        debug_mode: bool = False,
        sandbox_config: SandboxConfig | None = None,
    ):
        use_sandbox = sandbox_config is not None and sandbox_config.enabled

        # Build tool instances — backend selection is automatic
        self.all_tools: dict[ToolType, BaseTool] = {
            ToolType.CALCULATOR: CalculatorTool(timeout, debug_mode),
        }

        if use_sandbox:
            self.all_tools[ToolType.PYTHON] = E2BSandboxPythonTool(
                timeout=timeout,
                debug_mode=debug_mode,
                sandbox_config=sandbox_config,
            )
            logger.info(
                "Python tool backend: E2B sandbox (api_url=%s)",
                sandbox_config.api_url or "<env>",
            )
        else:
            self.all_tools[ToolType.PYTHON] = PythonTool(timeout, debug_mode)
            logger.info("Python tool backend: local (in-process, NOT sandboxed)")

        # NOTE: ToolType.BASH is registered but has no local implementation
        # (too dangerous).  It is only available when sandbox is enabled.
        # Future: add E2BSandboxBashTool here.

        # Set enabled tools
        if enabled_tools is None:
            # Default: enable all tools
            self.enabled_tools = list(self.TOOL_NAMES.values())
        else:
            # Validate enabled tools
            self.enabled_tools = []
            for tool_type in enabled_tools.split(";"):
                tool_type = tool_type.strip()
                if not tool_type:
                    continue
                if tool_type in self.TOOL_NAMES:
                    tt = self.TOOL_NAMES[tool_type]
                    if tt in self.all_tools:
                        self.enabled_tools.append(tt)
                    else:
                        logger.warning(
                            "Tool type %r requires sandbox backend but sandbox is "
                            "not configured, skipping",
                            tool_type,
                        )
                else:
                    logger.warning(f"Unknown tool type: {tool_type}, skipping")

        # Only keep enabled tools
        self.tools = {
            tool_type: self.all_tools[tool_type] for tool_type in self.enabled_tools
        }

        logger.info(
            "ToolRegistry initialized with enabled tools: %s (sandbox=%s)",
            [t.value for t in self.enabled_tools],
            use_sandbox,
        )

    def get_tool(self, tool_type: ToolType) -> BaseTool | None:
        """Get tool instance"""
        return self.tools.get(tool_type)

    def get_all_tools(self) -> dict[ToolType, BaseTool]:
        """Get all tool instances"""
        return self.tools

    def get_tool_markers(self) -> dict[ToolType, tuple[list[str], list[str]]]:
        """Get marker information for enabled tools only

        Returns:
            Dict[ToolType, Tuple[List[str], List[str]]]: Tool type -> (start markers list, end markers list)
        """
        return {
            tool_type: (tool.markers.start_markers, tool.markers.end_markers)
            for tool_type, tool in self.tools.items()
        }

    def get_all_start_markers(self) -> list[str]:
        """Get all start markers for enabled tools only

        Returns:
            List[str]: List of all start markers
        """
        start_markers = []
        for tool in self.tools.values():
            start_markers.extend(tool.markers.start_markers)
        return start_markers

    def get_all_end_markers(self) -> list[str]:
        """Get all end markers for enabled tools only

        Returns:
            List[str]: List of all end markers
        """
        end_markers = []
        for tool in self.tools.values():
            end_markers.extend(tool.markers.end_markers)
        return end_markers

    def get_all_markers(self) -> list[str]:
        """Get all markers (start and end) for enabled tools only

        Returns:
            List[str]: List of all markers
        """
        all_markers = []
        all_markers.extend(self.get_all_start_markers())
        all_markers.extend(self.get_all_end_markers())
        return all_markers

    def get_tool_descriptions_prompt(self) -> str:
        """Generate tool description prompt text for external calls (enabled tools only)"""
        prompt_parts = ["Tools List:\n"]

        for tool_type, tool in self.tools.items():
            desc = tool.description
            prompt_parts.append(f"Tool Name: {desc.name}")
            prompt_parts.append(f"Description: {desc.description}")
            prompt_parts.append(f"Parameter Description: {desc.parameter_prompt}")
            prompt_parts.append(f"Usage Example: {desc.example}")
            prompt_parts.append("---")

        return "\n".join(prompt_parts)

    def get_enabled_tools(self) -> list[ToolType]:
        """Get list of enabled tools

        Returns:
            List[ToolType]: List of enabled tool types
        """
        return self.enabled_tools.copy()


class ToolRouter:
    """Tool router that determines which tool to call based on markers"""

    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        # Build tool markers dynamically based on enabled tools
        self.tool_markers = self._build_tool_markers()

    def _build_tool_markers(self) -> list[tuple[ToolType, str]]:
        """Build tool markers based on enabled tools"""
        markers = []

        for tool_type, tool in self.registry.tools.items():
            # Build regex patterns for each tool's markers
            for start_marker in tool.markers.start_markers:
                for end_marker in tool.markers.end_markers:
                    # Escape special regex characters in markers
                    escaped_start = re.escape(start_marker)
                    escaped_end = re.escape(end_marker)
                    # Create pattern that matches content between markers
                    pattern = f"{escaped_start}(.*?){escaped_end}"
                    markers.append((tool_type, pattern))

        return markers

    def route(self, text: str) -> ToolType | None:
        """Determine tool type to call based on markers (enabled tools only)"""
        text = text.strip()

        # Check markers for each enabled tool
        for tool_type, pattern in self.tool_markers:
            if re.search(pattern, text, re.DOTALL | re.IGNORECASE):
                return tool_type

        return None


class ToolManager:
    """General tool manager responsible for coordinating tool calls.

    Parameters
    ----------
    timeout : int
        Per-tool-execution timeout.
    enabled_tools : str
        Semicolon-separated tool names (e.g. ``"python;calculator"``).
    debug_mode : bool
        If True, tools return dummy output.
    sandbox_config : SandboxConfig | None
        When provided, Python execution uses an isolated sandbox backend.
    """

    def __init__(
        self,
        timeout: int = 30,
        enabled_tools: str = "python;calculator",
        debug_mode: bool = False,
        sandbox_config: SandboxConfig | None = None,
    ):
        self.timeout = timeout
        self.debug_mode = debug_mode
        self.registry = ToolRegistry(
            timeout, enabled_tools, debug_mode, sandbox_config=sandbox_config
        )
        self.router = ToolRouter(self.registry)

        logger.info(
            "Initialized ToolManager (debug_mode=%s, enabled_tools=%s)",
            debug_mode,
            [t.value for t in self.registry.get_enabled_tools()],
        )

    def get_tool_descriptions_prompt(self) -> str:
        """Get tool description prompt text for external calls"""
        return self.registry.get_tool_descriptions_prompt()

    def get_tool_markers(self) -> dict[ToolType, tuple[list[str], list[str]]]:
        """Get marker information for all tools

        Returns:
            Dict[ToolType, Tuple[List[str], List[str]]]: Tool type -> (start markers list, end markers list)
        """
        return self.registry.get_tool_markers()

    def get_all_start_markers(self) -> list[str]:
        """Get all start markers for setting stop tokens

        Returns:
            List[str]: List of all start markers, e.g. ['```python\\n', '<calculator>']
        """
        return self.registry.get_all_start_markers()

    def get_all_end_markers(self) -> list[str]:
        """Get all end markers for setting stop tokens

        Returns:
            List[str]: List of all end markers, e.g. ['\\n```', '</calculator>']
        """
        return self.registry.get_all_end_markers()

    def get_all_markers(self) -> list[str]:
        """Get all markers (start and end) for setting stop tokens

        Returns:
            List[str]: List of all markers, e.g. ['```python\\n', '\\n```', '<calculator>', '</calculator>']
        """
        return self.registry.get_all_markers()

    def execute_tool_call(self, text: str) -> tuple[str, ToolCallStatus]:
        """Unified tool call interface

        Returns:
            Tuple[str, ToolCallStatus]: (result, status)
        """

        # 1. Routing: determine which tool to call
        tool_type = self.router.route(text)
        if not tool_type:
            return (
                "Error: No suitable tool found for the given text",
                ToolCallStatus.NOT_FOUND,
            )

        # 2. Get tool instance
        tool = self.registry.get_tool(tool_type)
        if not tool:
            return f"Error: Tool {tool_type.value} not found", ToolCallStatus.NOT_FOUND

        # 3. Parse parameters
        try:
            parameters = tool.parse_parameters(text)
            logger.debug(f"Parsed parameters: {parameters}")
        except Exception as e:
            logger.error(f"Parameter parsing error: {e}")
            return f"Error: Failed to parse parameters - {str(e)}", ToolCallStatus.ERROR

        # 4. Execute tool
        result, status = tool.execute(parameters)
        if status == ToolCallStatus.SUCCESS:
            logger.debug(f"Tool execution completed: {result}")
            return result, status
        else:
            logger.error(f"Tool execution error: {result}")
            return f"Error: Tool execution failed - {result}", status

    async def async_execute_tool_call(
        self, text: str
    ) -> tuple[str, ToolCallStatus]:
        """Async version of :meth:`execute_tool_call`.

        Prefers ``BaseTool.async_execute`` for tools that have native
        async backends (e.g. sandbox tools), avoiding thread-pool
        overhead.
        """
        tool_type = self.router.route(text)
        if not tool_type:
            return (
                "Error: No suitable tool found for the given text",
                ToolCallStatus.NOT_FOUND,
            )

        tool = self.registry.get_tool(tool_type)
        if not tool:
            return f"Error: Tool {tool_type.value} not found", ToolCallStatus.NOT_FOUND

        try:
            parameters = tool.parse_parameters(text)
        except Exception as e:
            logger.error(f"Parameter parsing error: {e}")
            return f"Error: Failed to parse parameters - {str(e)}", ToolCallStatus.ERROR

        result, status = await tool.async_execute(parameters)
        if status == ToolCallStatus.SUCCESS:
            logger.debug(f"Async tool execution completed: {result}")
        else:
            logger.error(f"Async tool execution error: {result}")
            result = f"Error: Tool execution failed - {result}"
        return result, status
