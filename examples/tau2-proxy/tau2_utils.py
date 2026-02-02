from dataclasses import dataclass, field
import json
import os
from datetime import datetime
from typing import Any

import yaml
from pydantic import BaseModel
from tau2.data_model.message import Message
from tau2.data_model.simulation import RewardInfo
from tau2.data_model.tasks import Task


class Tau2RunInfo(BaseModel):
    reward: float
    agent_time: list[float]
    user_time: list[float]
    messages: list[Message]
    task: Task
    reward_info: RewardInfo | None = None
    error: str | None = None

    def __str__(self):
        s = f"[REWARD]: {self.reward}\n\n"
        s += "[TASK]\n"
        s += yaml.dump(self.task.model_dump()) + "\n"
        if self.reward_info:
            s += "[REWARD_INFO]\n"
            s += yaml.dump(self.reward_info.model_dump()) + "\n"
        s += f"[TURNS COUNT]: {len(self.messages)}\n"
        s += "[MESSAGES]\n"
        for message in self.messages:
            turn_idx = message.turn_idx
            role = message.role
            content = message.content or ""
            usage = getattr(message, "usage", {})
            tool_calls = getattr(message, "tool_calls", None)
            if tool_calls:
                content += "\n[TOOL_CALLS]\n"
                content += yaml.dump(
                    [tool_call.model_dump() for tool_call in tool_calls]
                )
            s += f"[{turn_idx}][{role}]: {content}\n"
            if usage:
                s += f"[{turn_idx}][{role}][USAGE]: {yaml.dump(usage)}\n"
        if len(self.agent_time):
            s += f"[AGENT_TIME]: total {sum(self.agent_time)}, avg {sum(self.agent_time) / len(self.agent_time)}\n"
        if len(self.user_time):
            s += f"[USER_TIME]: total {sum(self.user_time)}, avg {sum(self.user_time) / len(self.user_time)}\n"
        if self.error:
            s += f"[ERROR]: {self.error}\n"
        return s

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return {
            "task": {
                "id": self.task.id,
                "domain": getattr(self.task, "domain", None),
                "user_scenario": str(self.task.user_scenario) if self.task.user_scenario else None,
                "expected_actions": [a.model_dump() if hasattr(a, "model_dump") else str(a)
                                    for a in (self.task.expected_actions or [])],
            },
            "reward": self.reward,
            "reward_info": self.reward_info.model_dump() if self.reward_info else None,
            "messages": [
                {
                    "turn_idx": m.turn_idx,
                    "role": m.role,
                    "content": m.content,
                    "tool_calls": [tc.model_dump() for tc in m.tool_calls] if m.tool_calls else None,
                }
                for m in self.messages
            ],
            "agent_time": self.agent_time,
            "user_time": self.user_time,
            "error": self.error,
        }


class TrajectoryLogger:
    """Logger for saving tau2 trajectories to JSONL files."""

    def __init__(self, save_dir: str | None = None, enabled: bool = True):
        """Initialize the trajectory logger.

        Args:
            save_dir: Directory to save trajectories. If None, uses current directory.
            enabled: Whether logging is enabled.
        """
        self.enabled = enabled
        if not enabled:
            return

        if save_dir is None:
            save_dir = os.getcwd()
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # Create a unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filepath = os.path.join(save_dir, f"trajectories_{timestamp}.jsonl")
        self._file = None
        self._count = 0

    def log(self, run_info: "Tau2RunInfo", extra_info: dict[str, Any] | None = None):
        """Log a single trajectory.

        Args:
            run_info: The Tau2RunInfo object containing trajectory data.
            extra_info: Optional extra information to include.
        """
        if not self.enabled:
            return

        record = {
            "timestamp": datetime.now().isoformat(),
            "trajectory_id": self._count,
            **run_info.to_dict(),
        }
        if extra_info:
            record.update(extra_info)

        # Append to file
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        self._count += 1

    def get_count(self) -> int:
        """Get the number of trajectories logged."""
        return self._count

    def get_filepath(self) -> str | None:
        """Get the filepath of the log file."""
        return self.filepath if self.enabled else None


# ================================ config ================================
# Customized config for tau2, add env config
@dataclass
class Tau2EnvConfig:
    domain: str = field(
        default="telecom",
        metadata={
            "help": "The tau2 domain name, e.g., 'retail', 'airline', 'telecom'."
        },
    )
    max_steps: int = field(
        default=100, metadata={"help": "Maximum number of steps per episode."}
    )
    add_thinking_tool: bool = field(
        default=True, metadata={"help": "Whether to add a thinking tool."}
    )
    solo_mode: bool = field(
        default=False, metadata={"help": "Whether to use solo mode."}
    )
    user_llm_base_url: str | None = field(
        default=None,
        metadata={"help": "The base URL of the user LLM."},
    )
    user_llm: str | None = field(
        default=None,
        metadata={"help": "The user LLM to use, default to the gpt-4.1 model."},
    )
    user_llm_args: dict | None = field(
        default=None, metadata={"help": "The arguments for the user LLM."}
    )
    turn_discount: float = field(
        default=1.0, metadata={"help": "Discount factor for turn-based learning."}
    )
    invalid_format_penalty: float = field(
        default=0.1, metadata={"help": "Penalty for invalid format in completions."}
    )
    save_trajectories: bool = field(
        default=True,
        metadata={"help": "Whether to save trajectories to JSONL files."},
    )
    trajectory_save_dir: str | None = field(
        default=None,
        metadata={
            "help": "Directory to save trajectories. "
            "If None, saves to the experiment log directory."
        },
    )
