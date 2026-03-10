"""Utilities for SWE-bench training with AReaL."""

import sys
from dataclasses import dataclass, field

from loguru import logger

from areal.api.cli_args import PPOConfig


@dataclass
class SWEEnvConfig:
    """Environment configuration for SWE-bench agent.

    Attributes:
        dataset_path: Path to the SWE-bench JSONL dataset file.
        swe_agent_config: Name of the SWE agent YAML config in
            SWEAgent/src/swe/configs/ (e.g., 'train', 'eval').
        swe_agent_root: Root directory of the SWEAgent project.
        step_limit: Maximum number of agent interaction steps per episode.
        max_completion_tokens: Maximum completion tokens for the agent LLM.
        timeout: Maximum time allowed for a single episode in seconds.
    """

    dataset_path: str = field(
        default="",
        metadata={"help": "Path to the SWE-bench JSONL dataset file."},
    )
    swe_agent_config: str = field(
        default="train",
        metadata={
            "help": (
                "Name of the SWE agent YAML config in SWEAgent/src/swe/configs/. "
                "E.g., 'train', 'distill_v2', 'eval'."
            )
        },
    )
    swe_agent_root: str = field(
        default="",
        metadata={
            "help": (
                "Root directory of the SWEAgent project. "
                "Used to resolve config files and tool scripts."
            )
        },
    )
    step_limit: int = field(
        default=100,
        metadata={"help": "Maximum number of agent interaction steps per episode."},
    )
    max_completion_tokens: int = field(
        default=16384,
        metadata={"help": "Maximum completion tokens for the agent LLM."},
    )
    timeout: float = field(
        default=1800.0,
        metadata={"help": "Maximum time allowed for a single episode in seconds."},
    )


@dataclass
class SWEPPOConfig(PPOConfig):
    """PPO configuration with SWE-bench-specific settings."""

    econfig: SWEEnvConfig = field(default_factory=SWEEnvConfig)


# Configure loguru logger for SWEAgent package.
# This runs at import time so workers also have the configuration.
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time} {level} {message}")
