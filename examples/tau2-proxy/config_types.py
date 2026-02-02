"""Tau2 configuration types.

This module contains only dataclass definitions with minimal dependencies,
so it can be imported by worker processes for deserialization without
requiring heavy dependencies like tau2-bench.
"""

from dataclasses import dataclass, field

from areal.api.cli_args import PPOConfig


@dataclass
class Tau2EnvConfig:
    """Environment configuration for Tau2 benchmark."""

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


@dataclass
class Tau2PPOConfig(PPOConfig):
    """PPO configuration with Tau2-specific settings."""

    econfig: Tau2EnvConfig = field(default_factory=Tau2EnvConfig)
    do_eval: bool = field(
        default=False,
        metadata={"help": "Whether to do evaluation."},
    )
