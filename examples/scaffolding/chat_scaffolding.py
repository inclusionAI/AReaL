"""
Multi-turn Chat Scaffolding Example.

This example demonstrates multi-turn chat-based RL training on GSM8K.
Each episode runs multiple generation turns with a reflection message
appended between turns to prompt the model to retry.

Usage:
    python examples/scaffolding/chat_scaffolding.py \
        --config examples/scaffolding/chat_scaffolding.yaml \
        +scheduler.type=local experiment_name=areal trial_name=chat_scaffolding
"""

import sys
from collections.abc import Callable
from typing import Any

import torch
from transformers import PreTrainedTokenizerFast

from areal.api.cli_args import GenerationHyperparameters, GRPOConfig, load_expr_config
from areal.api.engine_api import InferenceEngine
from areal.dataset import get_custom_dataset
from areal.experimental.scaffolding._compat import GenerationTask
from areal.experimental.scaffolding.workflow import ScaffoldingWorkflow
from areal.trainer import PPOTrainer
from areal.utils import logging
from areal.utils.hf_utils import load_hf_tokenizer

logger = logging.getLogger("ChatScaffoldingWorkflow")

DEFAULT_REFLECTION_MESSAGE = (
    "Your answer is either wrong or not parsable to the reward function. "
    "You may misunderstand the original question. "
    "Please carefully read the original question, check the previous errors, "
    "and try to answer it again."
)


class ChatScaffoldingWorkflow(ScaffoldingWorkflow):
    """ScaffoldingWorkflow for multi-turn chat with reflection retry.

    Each episode runs up to ``max_turns`` generation turns. After each
    non-final turn the ``reflection_message`` is appended as a user message
    to prompt the model to retry. Reward is computed on the final turn only.

    Generation and reward use the same direct-API approach as the base class
    (``_generate_via_worker`` + ``_compute_rewards_via_controller``).

    Parameters
    ----------
    reward_fn : Callable | str
        The reward function, or an importable string path.
    gconfig : GenerationHyperparameters
        Generation hyperparameters.
    tokenizer : PreTrainedTokenizerFast | str
        Tokenizer or path to load it.
    enable_thinking : bool
        Whether to enable thinking tokens.
    max_turns : int
        Maximum number of chat turns per episode.
    reflection_message : str
        Message appended after each non-final turn to prompt retry.
    """

    def __init__(
        self,
        reward_fn: Callable[..., Any] | str,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast | str,
        enable_thinking: bool = False,
        max_turns: int = 2,
        reflection_message: str = DEFAULT_REFLECTION_MESSAGE,
    ):
        super().__init__(
            reward_fn=reward_fn,
            gconfig=gconfig,
            tokenizer=tokenizer,
            enable_thinking=enable_thinking,
        )
        self.max_turns = max_turns
        self.reflection_message = reflection_message

    async def arun_episode(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        """Run a single multi-turn chat episode.

        Each turn: generate via SGLang OpenAI API, then optionally append a
        reflection message and retry.  After all turns, compute reward on the
        final completion and build tensor dicts identical to the base class.

        Parameters
        ----------
        engine : InferenceEngine
            The inference engine.
        data : dict[str, Any]
            Input data containing messages and ground truth.

        Returns
        -------
        dict[str, torch.Tensor]
            Trajectory tensors for PPO training.
        """
        if self.worker is None:
            self._lazy_init_scaffolding(engine)

        # Start from the original messages
        messages = list(data["messages"])
        last_output_str = ""
        all_output_tokens: list[int] = []

        for turn in range(self.max_turns):
            # Build prompt for this turn
            input_ids = list(
                self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=self.enable_thinking,
                )
            )
            prompt_str = self.tokenizer.decode(input_ids)

            # Generate via the SGLang OpenAI API (same as base class)
            gen_task = await self._generate_via_worker(prompt_str, input_ids)
            last_output_str = gen_task.output_str or ""
            all_output_tokens = list(gen_task.output_tokens or [])

            # Append the assistant response to messages
            messages.append({"role": "assistant", "content": last_output_str})

            # Append reflection message for non-final turns
            if turn < self.max_turns - 1:
                messages.append({"role": "user", "content": self.reflection_message})

        # Compute reward on the final turn's output (same pattern as base class)
        final_input_ids = list(
            self.tokenizer.apply_chat_template(
                data["messages"],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        )
        final_prompt_str = self.tokenizer.decode(final_input_ids)

        final_gen_task = GenerationTask(
            input_str=final_prompt_str,
            input_tokens=final_input_ids,
            output_str=last_output_str,
            output_tokens=all_output_tokens,
        )
        reward = await self._compute_rewards_via_controller(
            final_gen_task, final_prompt_str, data
        )

        # Build tensor dict (same as base class)
        seq = final_input_ids + all_output_tokens
        logprobs = [0.0] * len(seq)
        loss_mask = [0] * len(final_input_ids) + [1] * len(all_output_tokens)
        versions = [-1] * len(seq)

        res = {
            "input_ids": torch.tensor(seq, dtype=torch.int32),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.int32),
            "logprobs": torch.tensor(logprobs, dtype=torch.float32),
            "versions": torch.tensor(versions, dtype=torch.int32),
            "attention_mask": torch.ones(len(seq), dtype=torch.bool),
            "rewards": torch.tensor(reward, dtype=torch.float32),
        }
        return {k: v.unsqueeze(0) for k, v in res.items()}


def main(args):
    """Main entry point for multi-turn chat scaffolding training."""
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
        reward_fn="areal.reward.gsm8k.gsm8k_reward_fn",
        gconfig=config.gconfig,
        tokenizer=config.tokenizer_path,
        enable_thinking=False,
        max_turns=2,
        reflection_message=DEFAULT_REFLECTION_MESSAGE,
    )
    eval_workflow_kwargs = workflow_kwargs.copy()
    eval_workflow_kwargs["gconfig"] = config.gconfig.new(temperature=0.6)

    with PPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        trainer.train(
            workflow="examples.scaffolding.chat_scaffolding.ChatScaffoldingWorkflow",
            workflow_kwargs=workflow_kwargs,
            eval_workflow="examples.scaffolding.chat_scaffolding.ChatScaffoldingWorkflow",
            eval_workflow_kwargs=eval_workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
