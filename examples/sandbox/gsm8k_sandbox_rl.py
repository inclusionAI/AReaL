# SPDX-License-Identifier: Apache-2.0

"""Example: GSM8K RL training with CubeSandbox code execution.

This example demonstrates how to use SandboxToolWorkflow for training
a model to solve math problems using code execution. Generated Python
code is executed in isolated CubeSandbox instances for safety.

Usage
-----
1. Set up CubeSandbox:
   ```bash
   export SANDBOX_API_URL="http://localhost:3000"
   export SANDBOX_API_KEY="your-key"
   ```

2. Run training:
   ```bash
   uv run python gsm8k_sandbox_rl.py --config gsm8k_sandbox.yaml
   ```

See Also
--------
- examples/math/ — Standard GSM8K training without sandbox
- examples/tir/ — TIR workflow with in-process execution
"""

from __future__ import annotations

from dataclasses import dataclass, field

from areal.api.cli_args import GRPOConfig
from areal.api.sandbox_api import SandboxConfig


@dataclass
class GSM8KSandboxConfig(GRPOConfig):
    """Config for GSM8K training with sandbox code execution."""

    sandbox: SandboxConfig = field(
        default_factory=lambda: SandboxConfig(
            enabled=True,
            backend="cube",
            timeout=30.0,
            max_tool_turns=3,
            pool_size=0,
        )
    )


def make_sandbox_workflow(config: GSM8KSandboxConfig):
    """Create a SandboxToolWorkflow for GSM8K training.

    Parameters
    ----------
    config : GSM8KSandboxConfig
        Training configuration with sandbox settings.

    Returns
    -------
    SandboxToolWorkflow
        Configured workflow instance.
    """
    from areal.workflow.sandbox_tool import SandboxToolWorkflow

    return SandboxToolWorkflow(
        reward_fn="areal.reward.gsm8k.gsm8k_reward_fn",
        gconfig=config.gen,
        tokenizer=config.model.model_path,
        sandbox_config=config.sandbox,
        enable_thinking=config.gen.enable_thinking,
        system_prompt=(
            "You are a helpful math assistant. You can write and execute "
            "Python code to solve math problems. Put your code in "
            "```python\\n...``` blocks. The code will be executed and the "
            "output will be shown to you. After computing, provide your "
            "final answer in the format: #### <answer>"
        ),
    )


if __name__ == "__main__":
    print("GSM8K Sandbox RL Training Example")
    print("=" * 50)
    print()
    print("This script demonstrates SandboxToolWorkflow configuration.")
    print("For actual training, integrate with AReaL's training pipeline:")
    print()
    print("  1. Create config: GSM8KSandboxConfig")
    print("  2. Create workflow: make_sandbox_workflow(config)")
    print("  3. Pass to trainer: PPOTrainer(workflow=workflow, ...)")
    print()
    print("See README.md for detailed usage instructions.")
