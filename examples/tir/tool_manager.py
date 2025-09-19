import asyncio
import re
from typing import Dict, Optional, List, Tuple

from areal.utils import logging
from .tools import (
    ToolCallStatus,
    ToolType,
    BaseTool,
    QwenPythonTool,
    CalculatorTool,
)


logger = logging.getLogger("Tool Manager")


class ToolRegistry:
    """Tool registry that manages all available tools"""
    
    def __init__(self, timeout: int = 30, fake_mode: bool = False):
        self.tools = {
            ToolType.PYTHON: QwenPythonTool(timeout, fake_mode),
            ToolType.CALCULATOR: CalculatorTool(timeout, fake_mode),
        }
        # Tool marker mapping - defines start and end markers separately
        self.tool_start_markers = {
            ToolType.PYTHON: ["```python", "<python>"],
            ToolType.CALCULATOR: ["<calculator>"],
        }
        self.tool_end_markers = {
            ToolType.PYTHON: ["```", "</python>"],
            ToolType.CALCULATOR: ["</calculator>"],
        }
    
    def get_tool(self, tool_type: ToolType) -> Optional[BaseTool]:
        """Get tool instance"""
        return self.tools.get(tool_type)
    
    def get_all_tools(self) -> Dict[ToolType, BaseTool]:
        """Get all tool instances"""
        return self.tools
    
    def get_tool_markers(self) -> Dict[ToolType, Tuple[List[str], List[str]]]:
        """Get marker information for all tools
        
        Returns:
            Dict[ToolType, Tuple[List[str], List[str]]]: Tool type -> (start markers list, end markers list)
        """
        return {
            tool_type: (self.tool_start_markers[tool_type], self.tool_end_markers[tool_type])
            for tool_type in self.tool_start_markers.keys()
        }
    
    def get_all_start_markers(self) -> List[str]:
        """Get all start markers
        
        Returns:
            List[str]: List of all start markers
        """
        start_markers = []
        for markers in self.tool_start_markers.values():
            start_markers.extend(markers)
        return start_markers
    
    def get_all_end_markers(self) -> List[str]:
        """Get all end markers
        
        Returns:
            List[str]: List of all end markers
        """
        end_markers = []
        for markers in self.tool_end_markers.values():
            end_markers.extend(markers)
        return end_markers
    
    def get_all_markers(self) -> List[str]:
        """Get all markers (start and end)
        
        Returns:
            List[str]: List of all markers
        """
        all_markers = []
        all_markers.extend(self.get_all_start_markers())
        all_markers.extend(self.get_all_end_markers())
        return all_markers
    
    def get_tool_descriptions_prompt(self) -> str:
        """Generate tool description prompt text for external calls"""
        prompt_parts = ["Tools List:\n"]
        
        for tool_type, tool in self.tools.items():
            desc = tool.description
            prompt_parts.append(f"Tool Name: {desc.name}")
            prompt_parts.append(f"Description: {desc.description}")
            prompt_parts.append(f"Parameter Description: {desc.parameter_prompt}")
            prompt_parts.append(f"Usage Example: {desc.example}")
            prompt_parts.append("---")
        
        return "\n".join(prompt_parts)


class ToolRouter:
    """Tool router that determines which tool to call based on markers"""
    
    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self.tool_markers = [
            (ToolType.PYTHON, r"```python(.*?)```"),
            (ToolType.PYTHON, r"<python>(.*?)</python>"),
            (ToolType.CALCULATOR, r"<calculator>(.*?)</calculator>"),
        ]
    
    def route(self, text: str) -> Optional[ToolType]:
        """Determine tool type to call based on markers"""
        text = text.strip()
        
        # Check markers for each tool
        for tool_type, pattern in self.tool_markers:
            if re.search(pattern, text, re.DOTALL | re.IGNORECASE):
                return tool_type
        
        return None


class FakeToolManager:
    """Fake tool manager for debugging, returns dummy results directly"""
    
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        logger.info("Initializing FakeToolManager for debugging")
        
    async def execute_python(self, code: str) -> str:
        """Execute Python code - fake version"""
        logger.info(f"[FAKE] Executing Python code: {code[:100]}...")
        return "dummy code output"
    
    async def execute_calculator(self, expression: str) -> str:
        """Execute basic math calculation - fake version"""
        logger.info(f"[FAKE] Executing calculator: {expression}")
        return "dummy calculator output"
    
    def cleanup(self):
        """Cleanup - fake version"""
        logger.info("[FAKE] Cleanup completed")


class ToolManager:
    """General tool manager responsible for coordinating tool calls"""
    
    def __init__(self, timeout: int = 30, fake_mode: bool = False):
        self.timeout = timeout
        self.fake_mode = fake_mode
        self.registry = ToolRegistry(timeout, fake_mode)
        self.router = ToolRouter(self.registry)
        
        logger.info(f"Initialized ToolManager (fake_mode={fake_mode})")
    
    def get_tool_descriptions_prompt(self) -> str:
        """Get tool description prompt text for external calls"""
        return self.registry.get_tool_descriptions_prompt()
    
    def get_tool_markers(self) -> Dict[ToolType, Tuple[List[str], List[str]]]:
        """Get marker information for all tools
        
        Returns:
            Dict[ToolType, Tuple[List[str], List[str]]]: Tool type -> (start markers list, end markers list)
        """
        return self.registry.get_tool_markers()
    
    def get_all_start_markers(self) -> List[str]:
        """Get all start markers for setting stop tokens
        
        Returns:
            List[str]: List of all start markers, e.g. ['```python\n', '<calculator>']
        """
        return self.registry.get_all_start_markers()
    
    def get_all_end_markers(self) -> List[str]:
        """Get all end markers for setting stop tokens
        
        Returns:
            List[str]: List of all end markers, e.g. ['\n```', '</calculator>']
        """
        return self.registry.get_all_end_markers()
    
    def get_all_markers(self) -> List[str]:
        """Get all markers (start and end) for setting stop tokens
        
        Returns:
            List[str]: List of all markers, e.g. ['```python\n', '\n```', '<calculator>', '</calculator>']
        """
        return self.registry.get_all_markers()
    
    def execute_tool_call(self, text: str) -> Tuple[str, ToolCallStatus]:
        """Unified tool call interface
        
        Returns:
            Tuple[str, ToolCallStatus]: (result, status)
        """
        
        # 1. Routing: determine which tool to call
        tool_type = self.router.route(text)
        if not tool_type:
            return "Error: No suitable tool found for the given text", ToolCallStatus.NOT_FOUND
        
        # 2. Get tool instance
        tool = self.registry.get_tool(tool_type)
        if not tool:
            return f"Error: Tool {tool_type.value} not found", ToolCallStatus.NOT_FOUND
        
        # 3. Parse parameters
        try:
            parameters = tool.parse_parameters(text)
            logger.info(f"Parsed parameters: {parameters}")
        except Exception as e:
            logger.error(f"Parameter parsing error: {e}")
            return f"Error: Failed to parse parameters - {str(e)}", ToolCallStatus.ERROR
        
        # 4. Execute tool
        result, status = tool.execute(parameters)
        if status == ToolCallStatus.SUCCESS:
            logger.info(f"Tool execution completed: {result}")
            return result, status
        else:
            logger.error(f"Tool execution error: {result}")
            return f"Error: Tool execution failed - {result}", status
    
    def cleanup(self):
        """Clean up resources"""
        logger.info("ToolManager cleanup completed")


# Usage example
async def main():
    """Usage example"""
    manager = ToolManager(fake_mode=False)  # Use fake mode for testing
    
    # Get tool descriptions
    print("=== Tool Descriptions ===")
    print(manager.get_tool_descriptions_prompt())
    print()
    
    # Get marker information (for setting stop tokens)
    print("=== Tool Marker Information ===")
    print("All tool markers:", manager.get_tool_markers())
    print("Start markers:", manager.get_all_start_markers())
    print("End markers:", manager.get_all_end_markers())
    print("All markers:", manager.get_all_markers())
    print()
    
    # Test tool calls
    test_cases = [
        "<calculator>1 + 2 * 3</calculator>",
        "```python\nprint('Hello World')\n```",
        "```python\nfor i in range(3):\n    print(i)\n```",
        "<python>print('Hello from <python> tag')\nfor i in range(2):\n    print(f'Count: {i}')\n</python>",
        "<calculator>(10 + 5) / 3</calculator>",
    ]
    
    print("=== Tool Call Tests ===")
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case}")
        result, status = manager.execute_tool_call(test_case)
        print(f"Result: {result}")
        print(f"Status: {status}")
    
    manager.cleanup()


if __name__ == "__main__":
    asyncio.run(main())