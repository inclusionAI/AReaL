"""Core components for AREAL."""

from areal.core.staleness_manager import StalenessManager
from areal.core.workflow_executor import (
    WorkflowExecutor,
    check_trajectory_format,
)

__all__ = [
    "StalenessManager",
    "WorkflowExecutor",
    "check_trajectory_format",
]
