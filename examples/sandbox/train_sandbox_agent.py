# SPDX-License-Identifier: Apache-2.0

"""Train a math agent with E2B sandbox code execution (AgentWorkflow).

Usage::

    python examples/sandbox/train_sandbox_agent.py \\
        --config examples/sandbox/gsm8k_agent_sandbox.yaml
"""

import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).parent))

from areal import PPOTrainer
from areal.api.cli_args import load_expr_config
from areal.dataset import get_custom_dataset
from areal.utils.hf_utils import load_hf_tokenizer

from configs import SandboxAgentConfig  # isort: skip


def main(args):
    config, _ = load_expr_config(args, SandboxAgentConfig)
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
        temperature=config.gconfig.temperature,
        top_p=config.gconfig.top_p,
        max_completion_tokens=config.gconfig.max_new_tokens,
        sandbox_config=dict(
            enabled=config.sandbox.enabled,
            backend=config.sandbox.backend,
            api_url=config.sandbox.api_url,
            api_key=config.sandbox.api_key,
            template_id=config.sandbox.template_id,
            ssl_cert_file=config.sandbox.ssl_cert_file,
            timeout=config.sandbox.timeout,
        ),
    )
    eval_workflow_kwargs = workflow_kwargs.copy()
    eval_workflow_kwargs["temperature"] = 0.6

    with PPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        trainer.train(
            workflow=config.workflow,
            eval_workflow=config.eval_workflow,
            workflow_kwargs=workflow_kwargs,
            eval_workflow_kwargs=eval_workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
