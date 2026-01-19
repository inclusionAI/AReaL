"""Anthropic Agent training script using claude_agent_sdk.

This script uses claude_agent_sdk agents that communicate with Claude Code,
which can be configured to use a proxy by setting ANTHROPIC_BASE_URL.

Following the same pattern as examples/openai_agents/train_agents.py.

Usage:
    python examples/anthropic_agents/train_agents.py --config examples/anthropic_agents/config.yaml
"""

import sys
from dataclasses import dataclass
from pathlib import Path

from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.dataset import get_custom_dataset
from areal.experimental.trainer import PPOTrainer
from areal.utils.hf_utils import load_hf_tokenizer

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


@dataclass
class AnthropicAgentRLConfig(GRPOConfig):
    """Configuration for Anthropic agent RL training."""

    pass


def main(args):
    """Main training function."""
    config, _ = load_expr_config(args, GRPOConfig)

    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    train_dataset = get_custom_dataset(
        split="train",
        dataset_config=config.train_dataset,
        tokenizer=tokenizer,
    )

    valid_dataset = get_custom_dataset(
        split="test",
        dataset_config=config.valid_dataset,
        tokenizer=tokenizer,
    )

    workflow_kwargs = dict(
        max_tokens=config.gconfig.max_new_tokens,
        temperature=config.gconfig.temperature,
        top_p=config.gconfig.top_p,
    )

    with PPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        trainer.train(
            workflow="examples.anthropic_agents.math_agent.MathAgent",
            workflow_kwargs=workflow_kwargs,
            eval_workflow="examples.anthropic_agents.math_agent.MathAgent",
            eval_workflow_kwargs=workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
