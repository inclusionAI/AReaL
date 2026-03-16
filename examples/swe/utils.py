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


@dataclass
class CCEnvConfig:
    """Environment configuration for CC (Claude Code) agent.

    Attributes:
        dataset_path: Path to the SWE-bench JSONL dataset file.
        cc_agent_config: Name of the CC agent YAML config in
            SWEAgent/src/swe/configs/ (e.g., 'train_cc').
        swe_agent_root: Root directory of the SWEAgent project.
        cc_timeout: Maximum time for a single Claude Code invocation in seconds.
        timeout: Maximum time allowed for a single episode in seconds
            (includes setup, CC execution, and reward evaluation).
    """

    dataset_path: str = field(
        default="",
        metadata={"help": "Path to the SWE-bench JSONL dataset file."},
    )
    cc_agent_config: str = field(
        default="train_cc",
        metadata={
            "help": (
                "Name of the CC agent YAML config in SWEAgent/src/swe/configs/. "
                "E.g., 'train_cc', 'eval_cc'."
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
    cc_timeout: int = field(
        default=1800,
        metadata={
            "help": "Maximum time for a single Claude Code invocation in seconds."
        },
    )
    timeout: float = field(
        default=3600.0,
        metadata={
            "help": (
                "Maximum time allowed for a single episode in seconds "
                "(includes setup, CC execution, and reward evaluation)."
            )
        },
    )


@dataclass
class CCPPOConfig(PPOConfig):
    """PPO configuration with CC (Claude Code) agent settings."""

    econfig: CCEnvConfig = field(default_factory=CCEnvConfig)


# Configure loguru logger for SWEAgent package.
# This runs at import time so workers also have the configuration.
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time} {level} {message}")
