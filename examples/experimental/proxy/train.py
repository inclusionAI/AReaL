"""Unified training script for proxy-based agent workflows.

Supports both Anthropic and OpenAI agent workflows:
- Anthropic: MathAgent (simple), MathToolAgent (claude_agent_sdk with MCP tools)
- OpenAI: MathAgent, MultiTurnMathAgent, MathToolAgent (openai-agents SDK)
"""

import sys
from dataclasses import dataclass, field

from areal.api.cli_args import PPOConfig, load_expr_config
from areal.dataset import get_custom_dataset
from areal.experimental.trainer.rl import PPOTrainer
from areal.utils.hf_utils import load_hf_tokenizer


@dataclass
class WorkflowConfig:
    """Configuration for workflow selection and parameters."""

    path: str = field(
        default="areal.workflow.anthropic.math_agent.MathAgent",
        metadata={"help": "Dotted path to the workflow class."},
    )
    eval_path: str | None = field(
        default=None,
        metadata={
            "help": "Dotted path to the eval workflow class. If None, uses the same as path."
        },
    )
    # MathToolAgent specific (Anthropic claude_agent_sdk)
    use_mcp_tools: bool = field(
        default=False,
        metadata={"help": "Whether to use MCP tools (for Anthropic MathToolAgent)."},
    )
    # Multi-turn agent specific
    max_turns: int = field(
        default=10,
        metadata={"help": "Maximum turns for multi-turn agents."},
    )


@dataclass
class ProxyAgentConfig(PPOConfig):
    """Configuration for proxy-based agent training."""

    workflow: WorkflowConfig = field(default_factory=WorkflowConfig)


def main(args):
    """Main training function."""
    config, _ = load_expr_config(args, ProxyAgentConfig)

    # Anthropic agent SDK invokes Claude Code, which injects long system prompts;
    # raise max_tokens_per_mb so actor/ref micro-batches can fit those sequences.
    workflow_path_lower = config.workflow.path.lower()
    if "anthropic" in workflow_path_lower and "math_tool_agent" in workflow_path_lower:
        config.actor.mb_spec.max_tokens_per_mb = 24576
        config.ref.mb_spec.max_tokens_per_mb = 24576

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

    # Build workflow kwargs from config
    workflow_kwargs = dict(
        temperature=config.gconfig.temperature,
        top_p=config.gconfig.top_p,
        # For anthropic
        max_tokens=config.gconfig.max_tokens,
        # For openai
        max_completion_tokens=config.gconfig.max_new_tokens,
        # For agent-specific kwargs
        use_mcp_tools=config.workflow.use_mcp_tools,
        max_turns=config.workflow.max_turns,
    )

    eval_workflow_kwargs = workflow_kwargs.copy()
    eval_workflow_kwargs["temperature"] = 0.6

    # Determine eval workflow path
    eval_workflow_path = config.workflow.eval_path or config.workflow.path

    with PPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        trainer.train(
            workflow=config.workflow.path,
            workflow_kwargs=workflow_kwargs,
            eval_workflow=eval_workflow_path,
            eval_workflow_kwargs=eval_workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
