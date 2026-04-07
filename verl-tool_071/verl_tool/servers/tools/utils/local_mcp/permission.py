"""Tool permission management for MCP operations."""
# pylint: disable=unused-argument,too-few-public-methods
import re
from typing import Dict, Literal, Any, Optional, List
from pydantic import BaseModel, Field


class PermissionStatus(BaseModel):
    """
    Status result of permission evaluation.
    
    Attributes:
        approved: Whether the permission was granted.
        reason: Explanation for the permission decision.
    """
    approved: bool
    reason: str


class ToolPermission(BaseModel):
    """
    Defines permission rules for MCP tool execution.
    
    Attributes:
        tool: Tool name or regex pattern with wildcards.
        arguments: Tool argument constraints as key-value pairs.
        action: Permission action (allow, reject, or human_review).
    """
    tool: str
    arguments: Dict[str, str] = Field(default_factory=dict)
    action: Literal["allow", "reject", "human_review"] = "allow"

    @staticmethod
    def _match(pattern: str, text: str):
        """
        Matches text against pattern with regex support.
        
        Args:
            pattern: Pattern string with optional wildcards.
            text: Text to match against pattern.
            
        Returns:
            Match object or boolean indicating match result.
        """
        if "*" in pattern or "+" in pattern or "?" in pattern:
            return re.match(pattern, text)
        return pattern == text

    def match(self, tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Checks if tool execution matches this permission rule.
        
        Args:
            tool_name: Name of the tool to check.
            arguments: Tool arguments to validate.
            
        Returns:
            Permission action if matched, None otherwise.
        """
        if not ToolPermission._match(self.tool, tool_name):
            return None
        if arguments is None:
            arguments = {}
        match = True
        for key, value in self.arguments.items():
            if key not in arguments or not ToolPermission._match(value, arguments[key]):
                match = False
                break
        if not match:
            return None
        return self.action


class PermissionAction:
    """Executes permission actions for tool requests."""

    def __init__(self):
        self._actions = {
            "allow": PermissionAction._allow,
            "reject": PermissionAction._reject,
            "human_review": PermissionAction._human_review
        }

    def execute(
            self,
            action: str,
            tool_name: str,
            arguments: Optional[Dict[str, Any]] = None,
            **kwargs
    ) -> PermissionStatus:
        """
        Executes the specified permission action.
        
        Args:
            action: Permission action to execute (allow, reject, human_review).
            tool_name: Name of the tool requesting permission.
            arguments: Tool arguments for review.
            
        Returns:
            PermissionStatus indicating approval or rejection.
            
        Raises:
            RuntimeError: If action is not recognized.
        """
        if action.lower() not in self._actions:
            raise RuntimeError(f"Unknown permission action: {action}. "
                               f"Please choose from {list(self._actions.keys())}")
        return self._actions[action.lower()](tool_name, arguments, **kwargs)

    @staticmethod
    def _allow(*args, **kwargs) -> PermissionStatus:
        """
        Automatically approves tool execution.

        Returns:
            PermissionStatus with approved=True.
        """
        return PermissionStatus(approved=True, reason="")

    @staticmethod
    def _reject(*args, **kwargs) -> PermissionStatus:
        """
        Automatically rejects tool execution.

        Returns:
            PermissionStatus with approved=False.
        """
        return PermissionStatus(approved=False, reason="This tool is not permitted to perform the requested operation.")

    @staticmethod
    def _human_review(tool_name: str, arguments: Optional[Dict[str, Any]] = None, **kwargs) -> PermissionStatus:
        """
        Prompts user for approval of tool execution.

        Args:
            tool_name: Name of the tool requesting execution.
            arguments: Tool arguments to be reviewed.

        Returns:
            PermissionStatus based on user input.
        """
        if arguments is None:
            arguments = {}

        print(f"Tool '{tool_name}' is requesting permission to execute.")
        if arguments:
            print(f"Arguments: {arguments}")

        while True:
            response = input("Do you want to approve this tool execution? (y/n): ").strip().lower()
            if response in ['y', 'yes']:
                return PermissionStatus(approved=True, reason="Tool execution was approved by user")
            if response in ['n', 'no']:
                return PermissionStatus(approved=False, reason="Tool execution was rejected by user")
            print("Please enter 'y' for yes or 'n' for no.")


def check_permissions(
        permissions: List[ToolPermission],
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None
) -> PermissionStatus:
    """
    Checks tool execution against permission rules.
    
    Args:
        permissions: List of permission rules to evaluate.
        tool_name: Name of the tool requesting execution.
        arguments: Tool arguments to validate.
        
    Returns:
        PermissionStatus indicating final approval decision.
    """
    if not permissions:
        return PermissionStatus(approved=True, reason="Approved")
    actor = PermissionAction()
    for permission in permissions:
        action = permission.match(tool_name=tool_name, arguments=arguments)
        if not action:
            continue
        status = actor.execute(action=action, tool_name=tool_name, arguments=arguments)
        if not status.approved:
            return status
    return PermissionStatus(approved=True, reason="Approved")
