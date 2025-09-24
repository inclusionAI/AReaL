from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Tuple

from areal.utils import logging

logger = logging.getLogger("Tool Base")


class ToolCallStatus(Enum):
    """Tool call status enumeration"""

    SUCCESS = "success"
    ERROR = "error"
    NOT_FOUND = "not_found"


class ToolType(Enum):
    """Tool type enumeration"""

    PYTHON = "python"
    CALCULATOR = "calculator"


@dataclass
class ToolCall:
    """Tool call data structure"""

    tool_type: ToolType
    parameters: Dict[str, Any]
    raw_text: str


@dataclass
class ToolDescription:
    """Tool description data structure"""

    name: str
    description: str
    parameters: Dict[str, str]  # Parameter name -> parameter description
    parameter_prompt: str  # Prompt for parameter parsing
    example: str


@dataclass
class ToolMarkers:
    """Tool markers data structure"""

    start_markers: List[str]  # Start markers for the tool
    end_markers: List[str]  # End markers for the tool


class BaseTool(ABC):
    """Base tool abstract class"""

    def __init__(self, timeout: int = 30, debug_mode: bool = False):
        self.timeout = timeout
        self.debug_mode = debug_mode

    @property
    @abstractmethod
    def tool_type(self) -> ToolType:
        """Tool type"""

    @property
    @abstractmethod
    def description(self) -> ToolDescription:
        """Tool description"""

    @property
    @abstractmethod
    def markers(self) -> ToolMarkers:
        """Tool markers for parsing"""

    @abstractmethod
    def parse_parameters(self, text: str) -> Dict[str, Any]:
        """Parse parameters"""

    @abstractmethod
    def execute(self, parameters: Dict[str, Any]) -> Tuple[str, ToolCallStatus]:
        """Execute tool

        Returns:
            Tuple[str, ToolCallStatus]: (result, status)
        """
