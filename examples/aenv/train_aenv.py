"""Training entrypoint for AEnvironment-integrated workflow."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

from areal import PPOTrainer
from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.dataset import get_custom_dataset
from areal.infra.aenv import AenvConfig
from areal.utils.hf_utils import load_hf_tokenizer


@dataclass
class AenvGRPOConfig(GRPOConfig):
    """GRPO config for the AEnvironment workflow example."""

    reward_fn_path: str = field(
        default="areal.reward.gsm8k.gsm8k_reward_fn",
        metadata={"help": "Import path of reward function used by AenvWorkflow"},
    )
    max_turns: int = field(
        default=8,
        metadata={"help": "Maximum turns for each rollout episode"},
    )
    export_style: str = field(
        default="individual",
        metadata={"help": "ArealOpenAI export style: individual or concat"},
    )
    tool_call_parser: str = field(
        default="qwen25",
        metadata={"help": "Tool call parser name for ArealOpenAI"},
    )
    system_prompt: str | None = field(
        default=None,
        metadata={"help": "Optional system prompt prepended to each episode"},
    )
    aenv: AenvConfig = field(
        default_factory=AenvConfig,
        metadata={"help": "AEnvironment adapter configuration"},
    )


def main(args: list[str]) -> None:
    """Launch training with AEnvironment-integrated workflow."""
    config, _ = load_expr_config(args, AenvGRPOConfig)
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

    workflow_kwargs = {
        "gconfig": config.gconfig,
        "tokenizer": config.tokenizer_path,
        "aenv_config": config.aenv,
        "reward_fn": config.reward_fn_path,
        "max_turns": config.max_turns,
        "export_style": config.export_style,
        "tool_call_parser": config.tool_call_parser,
        "system_prompt": config.system_prompt,
    }
    eval_workflow_kwargs = workflow_kwargs.copy()
    eval_workflow_kwargs["gconfig"] = config.gconfig.new(temperature=0.6, n_samples=1)

    with PPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        trainer.train(
            workflow="areal.workflow.aenv.AenvWorkflow",
            workflow_kwargs=workflow_kwargs,
            eval_workflow="areal.workflow.aenv.AenvWorkflow",
            eval_workflow_kwargs=eval_workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
