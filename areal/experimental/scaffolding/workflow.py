"""
ScaffoldingWorkflow - RolloutWorkflow with generation and reward via Scaffolding.

Architecture
------------
- Generation: via scaffolding Worker (SGLangWorker calls SGLang OpenAI API)
- Reward: via scaffolding RLVRRewardController
- Logprobs: placeholder (0.0) since recompute_logprob=true in training config
  causes the actor to recompute exact logprobs during PPO update.
- Worker & ScaffoldingLlm: lazily created from engine server addresses,
  exposed for subclasses (e.g., multi-turn workflows).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from transformers import PreTrainedTokenizerFast

from areal.api.cli_args import GenerationHyperparameters
from areal.api.engine_api import InferenceEngine
from areal.api.workflow_api import RolloutWorkflow
from areal import workflow_context
from areal.experimental.scaffolding._compat import (
    GenerationTask,
    NativeGenerationController,
    ScaffoldingLlm,
)
from areal.experimental.scaffolding.controllers import (
    PipelineTrajectoryMaker,
    RLVRRewardController,
)
from areal.experimental.scaffolding.task import RLVRRewardTask
from areal.experimental.scaffolding.worker import SGLangWorker
from areal.utils import logging, stats_tracker
from areal.utils.dynamic_import import import_from_string
from areal.utils.perf_tracer import session_context, trace_session

logger = logging.getLogger("ScaffoldingWorkflow")


class ScaffoldingWorkflow(RolloutWorkflow):
    """RolloutWorkflow with generation and reward via scaffolding components.

    Both generation and reward computation go through scaffolding:
    - Generation: SGLangWorker calls SGLang's OpenAI-compatible completions API
    - Reward: RLVRRewardController computes verifiable rewards

    Since the OpenAI API does not return per-token logprobs in AReaL's format,
    placeholder logprobs are used. Set ``recompute_logprob: true`` in the actor
    config so the training engine recomputes exact logprobs during PPO update.

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
    """

    def __init__(
        self,
        reward_fn: Callable[..., Any] | str,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast | str,
        enable_thinking: bool = False,
    ):
        if isinstance(reward_fn, str):
            reward_fn = import_from_string(reward_fn)
        self.reward_fn = reward_fn

        self.tokenizer = tokenizer
        if isinstance(self.tokenizer, str):
            from areal.utils.hf_utils import load_hf_tokenizer

            self.tokenizer = load_hf_tokenizer(self.tokenizer)
        self.gconfig = gconfig.new_with_stop_and_pad_token_ids(self.tokenizer)
        self.enable_thinking = enable_thinking

        # Scaffolding controllers
        self.reward_controller = RLVRRewardController(self.reward_fn)
        self.gen_controller = NativeGenerationController()

        # Lazily created from engine server addresses
        self.worker: SGLangWorker | None = None
        self.trajectory_maker: PipelineTrajectoryMaker | None = None
        self.scaffolding_llm: ScaffoldingLlm | None = None

    def _lazy_init_scaffolding(self, engine: InferenceEngine) -> None:
        """Create Worker, PipelineTrajectoryMaker, and ScaffoldingLlm."""
        import openai

        addr = engine.addresses[0]
        base_url = f"http://{addr}"
        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"

        async_client = openai.AsyncOpenAI(base_url=base_url, api_key="EMPTY")
        self.worker = SGLangWorker(
            async_client=async_client, model="default", engine=engine
        )

        self.trajectory_maker = PipelineTrajectoryMaker(
            self.gen_controller, self.reward_controller
        )

        self.scaffolding_llm = ScaffoldingLlm(
            self.trajectory_maker,
            {NativeGenerationController.WorkerTag.GENERATION: self.worker},
        )
        logger.info(f"Initialized scaffolding components with server at {addr}")

    async def _generate_via_worker(
        self, prompt_str: str, input_ids: list[int]
    ) -> GenerationTask:
        """Run generation through scaffolding Worker (SGLang OpenAI API).

        Parameters
        ----------
        prompt_str : str
            The prompt string.
        input_ids : list[int]
            The tokenized input IDs.

        Returns
        -------
        GenerationTask
            Completed task with output_str and output_tokens.
        """
        # Build generation params for SGLang completions API
        stop_strings = []
        if self.gconfig.stop_token_ids:
            for tid in self.gconfig.stop_token_ids:
                decoded = self.tokenizer.decode([tid])
                if decoded:
                    stop_strings.append(decoded)

        response = await self.worker.async_client.completions.create(
            model=self.worker.model,
            prompt=prompt_str,
            max_tokens=self.gconfig.max_new_tokens,
            temperature=self.gconfig.temperature or 1.0,
            stop=stop_strings or None,
        )

        output_str = response.choices[0].text
        # Tokenize to get output token IDs
        output_token_ids = self.tokenizer.encode(output_str, add_special_tokens=False)

        # Package as a GenerationTask (scaffolding data structure)
        gen_task = GenerationTask(
            input_str=prompt_str,
            input_tokens=input_ids,
            output_str=output_str,
            output_tokens=output_token_ids,
            finish_reason=response.choices[0].finish_reason,
        )
        return gen_task

    @trace_session("reward")
    async def _compute_rewards_via_controller(
        self,
        gen_task: GenerationTask,
        prompt_str: str,
        task_data: dict[str, Any],
    ) -> float:
        """Compute reward via scaffolding RLVRRewardController."""
        reward_task = RLVRRewardTask(
            prompt_str=prompt_str,
            completion_str=gen_task.output_str or "",
            input_tokens=list(gen_task.input_tokens or []),
            output_tokens=list(gen_task.output_tokens or []),
            task_data=task_data,
        )
        for _ in self.reward_controller.process([reward_task]):
            pass
        return float(reward_task.reward)

    @session_context()
    async def _collect_samples(
        self,
        prompt_str: str,
        input_ids: list[int],
        task_data: dict[str, Any],
    ) -> tuple[GenerationTask, float]:
        """Generate via Worker, compute reward via Controller."""
        gen_task = await self._generate_via_worker(prompt_str, input_ids)

        reward = await self._compute_rewards_via_controller(
            gen_task, prompt_str, task_data
        )
        stats_tracker.get(workflow_context.stat_scope()).scalar(reward=reward)

        return gen_task, reward

    async def arun_episode(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        """Run a single episode via scaffolding pipeline.

        1. Generation: SGLangWorker -> SGLang completions API
        2. Reward: RLVRRewardController.process()
        3. Output: tensor dict for PPO training

        Note: logprobs are placeholders (0.0). Set ``recompute_logprob: true``
        in actor config so the training engine computes exact logprobs.

        Parameters
        ----------
        engine : InferenceEngine
            The inference engine (used for server addresses on first call).
        data : dict[str, Any]
            Input data containing messages and ground truth.

        Returns
        -------
        dict[str, torch.Tensor]
            Trajectory tensors for PPO training.
        """
        if self.worker is None:
            self._lazy_init_scaffolding(engine)

        # Tokenize prompt
        input_ids = list(
            self.tokenizer.apply_chat_template(
                data["messages"],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        )
        prompt_str = self.tokenizer.decode(input_ids)

        # Scaffolding pipeline: Worker (generate) + Controller (reward)
        gen_task, reward = await self._collect_samples(prompt_str, input_ids, data)

        # Build tensor dict for PPO training
        output_tokens = list(gen_task.output_tokens or [])
        seq = input_ids + output_tokens
        # Placeholder logprobs — recompute_logprob=true will replace these
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
