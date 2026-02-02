"""Core components for AREAL."""

from . import workflow_context
from .controller import RolloutController, TrainController
from .platforms import Platform, current_platform, is_npu_available
from .remote_inf_engine import (
    RemoteInfBackendProtocol,
    RemoteInfEngine,
)
from .staleness_manager import StalenessManager
from .workflow_executor import (
    WorkflowExecutor,
    check_trajectory_format,
)

__all__ = [
    "RemoteInfBackendProtocol",
    "RemoteInfEngine",
    "StalenessManager",
    "WorkflowExecutor",
    "check_trajectory_format",
    "RolloutController",
    "TrainController",
    "workflow_context",
    "Platform",
    "current_platform",
    "is_npu_available",
]
