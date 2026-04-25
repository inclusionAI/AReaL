# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

from areal.api.cli_args import PPOConfig
from areal.api.sandbox_api import SandboxConfig


@dataclass
class SandboxAgentConfig(PPOConfig):
    """Configuration for sandbox agent RL training.

    Extends ``PPOConfig`` with sandbox-specific settings.
    """

    workflow: str = field(
        default="examples.sandbox.sandbox_math_agent.SandboxMathAgent",
        metadata={"help": "Path to the workflow class for training."},
    )
    eval_workflow: str = field(
        default="examples.sandbox.sandbox_math_agent.SandboxMathAgent",
        metadata={"help": "Path to the workflow class for evaluation."},
    )
    sandbox: SandboxConfig = field(
        default_factory=lambda: SandboxConfig(enabled=True),
        metadata={"help": "Sandbox execution configuration."},
    )
