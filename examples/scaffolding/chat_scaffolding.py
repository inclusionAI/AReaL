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
from areal.experimental.scaffolding._compat import (
    NativeGenerationController,
    ScaffoldingLlm,
)
from areal.experimental.scaffolding.controllers import (
    MultiTurnChatController,
    RLVRRewardController,
    TraceTrajectoryMaker,
)
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

    Generation and reward are delegated to ``scaffolding_llm`` which wraps a
    ``TraceTrajectoryMaker`` with a ``MultiTurnChatController``.

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

    def build_scaffolding_llm(self, engine: InferenceEngine) -> ScaffoldingLlm:
        """Build ScaffoldingLlm with MultiTurnChatController + TraceTrajectoryMaker.

        Parameters
        ----------
        engine : InferenceEngine
            The inference engine.

        Returns
        -------
        ScaffoldingLlm
            The constructed ScaffoldingLlm instance.
        """
        stop_strings = []
        if self.gconfig.stop_token_ids:
            for tid in self.gconfig.stop_token_ids:
                decoded = self.tokenizer.decode([tid])
                if decoded:
                    stop_strings.append(decoded)

        sampling_params = {
            "max_tokens": self.gconfig.max_new_tokens,
            "temperature": self.gconfig.temperature or 1.0,
        }
        if stop_strings:
            sampling_params["stop"] = stop_strings

        self.gen_controller = NativeGenerationController(
            sampling_params=sampling_params
        )
        self.reward_controller = RLVRRewardController(self.reward_fn)
        self.multi_turn_controller = MultiTurnChatController(
            generation_controller=self.gen_controller,
            max_turns=self.max_turns,
            reflection_message=self.reflection_message,
            tokenizer=self.tokenizer,
        )
        self.trajectory_maker = TraceTrajectoryMaker(
            rollout_controller=self.multi_turn_controller,
            reward_controller=self.reward_controller,
        )
        return ScaffoldingLlm(
            self.trajectory_maker,
            {NativeGenerationController.WorkerTag.GENERATION: self.worker},
        )

    async def arun_episode(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        """Run a single multi-turn chat episode.

        Delegates the full episode (multi-turn generation + reward) to
        ``self.scaffolding_llm``, which wraps a ``TraceTrajectoryMaker``
        with a ``MultiTurnChatController``.

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

        # Tokenize the original prompt (before multi-turn)
        input_ids = list(
            self.tokenizer.apply_chat_template(
                data["messages"],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        )
        prompt_str = self.tokenizer.decode(input_ids)

        # Configure per-episode data on multi-turn controller
        # (clone() in scaffolding_llm will deep-copy these)
        self.multi_turn_controller.messages = data["messages"]
        self.multi_turn_controller.input_tokens = input_ids

        # Run full pipeline via scaffolding_llm
        result = self.scaffolding_llm.generate_async(prompt_str)
        await result

        # Extract trace results from ScaffoldingOutput
        scaffolding_output = result.outputs[0]
        trace_results = scaffolding_output.data

        # Get the final output text from the last traced interaction
        if trace_results:
            last_interaction = list(trace_results.values())[-1]
            output_str = ""
            if last_interaction.completion is not None:
                output_str = (
                    last_interaction.completion.choices[0].message.content or ""
                )
        else:
            output_str = scaffolding_output.text or ""

        output_tokens = self.tokenizer.encode(output_str, add_special_tokens=False)

        # Compute reward on the final turn's output
        reward = float(
            self.reward_fn(prompt_str, output_str, input_ids, output_tokens, **data)
        )

        # Build tensor dict for PPO training
        seq = input_ids + output_tokens
        logprobs = [0.0] * len(seq)
        loss_mask = [0] * len(input_ids) + [1] * len(output_tokens)
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
