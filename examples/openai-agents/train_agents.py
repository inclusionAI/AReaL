import os
from dataclasses import dataclass, field

from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.dataset import get_custom_dataset
from areal.experimental.trainer import PPOTrainer
from areal.utils.stats_logger import StatsLogger


@dataclass
class AgentRLConfig(GRPOConfig):
    agent_type: str = field(
        default="math",
        metadata={
            "help": "Type of agent workflow to use.",
            "choices": ["math", "multi_agent_math"],
        },
    )
    n_trajs: int = field(
        default=1,
        metadata={
            "help": "We could collect multiple trajectories for a single query. By default n_trajs=1."
        },
    )
    max_turns: int = field(
        default=8,
        metadata={
            "help": "Maximum number of turns per trajectory. By default max_turns=8."
        },
    )
    max_tokens_per_trajectory: int = field(
        default=32768,
        metadata={
            "help": "Maximum number of tokens per trajectory. By default max_tokens_per_trajectory=32768."
        },
    )


def main(args):
    config, _ = load_expr_config(args, AgentRLConfig)

    # Load dataset
    train_dataset = get_custom_dataset(
        split="train", dataset_config=config.train_dataset, tokenizer=None
    )

    # Create trainer (no valid_dataset for this example)
    with PPOTrainer(config, train_dataset, valid_dataset=None) as trainer:
        # Create rollout workflow based on agent type
        if config.agent_type == "math":
            from math_workflow import RLVRAgentWorkflow

            workflow = RLVRAgentWorkflow(
                gconfig=config.gconfig,
                tokenizer=trainer.tokenizer,
                n_trajs=config.n_trajs,
                dump_dir=os.path.join(
                    StatsLogger.get_log_path(config.stats_logger), "generated"
                ),
            )
        elif config.agent_type == "multi_agent_math":
            from multi_agent_math_workflow import MultiAgentRLVRAgentWorkflow

            workflow = MultiAgentRLVRAgentWorkflow(
                gconfig=config.gconfig,
                tokenizer=trainer.tokenizer,
                n_trajs=config.n_trajs,
                max_tokens=config.max_tokens_per_trajectory,
                max_turns=config.max_turns,
                dump_dir=os.path.join(
                    StatsLogger.get_log_path(config.stats_logger), "generated"
                ),
            )
        else:
            raise ValueError(f"Unknown agent_type: {config.agent_type}.")

        # Dummy eval workflow (not used since no valid_dataset)
        eval_workflow = workflow

        # Run training
        trainer.train(workflow, eval_workflow)


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
