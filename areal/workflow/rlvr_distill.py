"""Self-distillation RLVR workflow.

Extends :class:`RLVRWorkflow` with an additional feedback generation step.
After generating a response and computing its reward, the workflow generates
textual feedback conditioned on the prompt and response, then builds an
extended trajectory that encodes both student and teacher token positions.

The extended trajectory layout:

    [prompt | response | feedback_separator | response_copy]

where ``loss_mask`` marks the first response (student target) and
``feedback_mask`` marks feedback + response_copy (teacher context).
"""

import uuid
from collections.abc import Callable
from typing import Any

import torch
from transformers import PreTrainedTokenizerFast

from areal import workflow_context
from areal.api.cli_args import GenerationHyperparameters
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import ModelRequest, ModelResponse
from areal.api.reward_api import AsyncRewardWrapper
from areal.api.workflow_api import RolloutWorkflow
from areal.utils import logging, stats_tracker
from areal.utils.dynamic_import import import_from_string
from areal.utils.perf_tracer import atrace_session_phase, session_context, trace_session

from .rlvr import default_data_extract_prompt_fn, default_get_input_ids_fn

logger = logging.getLogger("SelfDistillRLVRWorkflow")


class SelfDistillRLVRWorkflow(RolloutWorkflow):

    def __init__(
        self,
        reward_fn: Callable[..., Any] | str,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast | str,
        feedback_template: str = (
            "\n\nPlease evaluate the above response and provide feedback "
            "on its correctness and quality.\n\nFeedback:"
        ),
        enable_thinking: bool = False,
        get_input_ids_fn: Callable[[Any, PreTrainedTokenizerFast, bool], list[int]]
        | str = default_get_input_ids_fn,
        data_extract_prompt_fn: Callable[[dict[str, Any]], Any]
        | str = default_data_extract_prompt_fn,
    ):
        self.reward_fn = reward_fn
        self.tokenizer = tokenizer
        if isinstance(self.tokenizer, str):
            from areal.utils.hf_utils import load_hf_tokenizer

            tokenizer = load_hf_tokenizer(self.tokenizer)
            self.tokenizer = tokenizer
        self.gconfig = gconfig.new_with_stop_and_pad_token_ids(self.tokenizer)
        self.feedback_template = feedback_template
        self.enable_thinking = enable_thinking
        if not isinstance(reward_fn, str):
            self.async_reward_fn = AsyncRewardWrapper(reward_fn)
        if isinstance(get_input_ids_fn, str):
            get_input_ids_fn = import_from_string(get_input_ids_fn)
        self.get_input_ids_fn = get_input_ids_fn
        if isinstance(data_extract_prompt_fn, str):
            data_extract_prompt_fn = import_from_string(data_extract_prompt_fn)
        self.data_extract_prompt_fn = data_extract_prompt_fn

    @trace_session("reward")
    async def _compute_rewards(
        self,
        resp: ModelResponse,
        prompt_str: str,
        task_data: dict[str, Any],
    ) -> float:
        """Decode completion and compute reward."""
        completions_str = self.tokenizer.decode(resp.output_tokens)
        reward = await self.async_reward_fn(
            prompt_str,
            completions_str,
            resp.input_tokens,
            resp.output_tokens,
            **task_data,
        )
        return reward

    @trace_session("feedback")
    async def _generate_feedback(
        self,
        engine: InferenceEngine,
        prompt_ids: list[int],
        response_ids: list[int],
    ) -> ModelResponse:
        """Generate feedback conditioned on prompt + response.

        Constructs a feedback prompt by appending the feedback template
        to the original prompt + response, then generates a completion.
        """
        feedback_prefix_ids = self.tokenizer.encode(
            self.feedback_template, add_special_tokens=False
        )
        feedback_input_ids = prompt_ids + response_ids + feedback_prefix_ids

        feedback_gconfig = self.gconfig.new(
            n_samples=1,
            max_new_tokens=min(self.gconfig.max_new_tokens, 512),
            temperature=0.7,
        )
        feedback_req = ModelRequest(
            rid=uuid.uuid4().hex,
            input_ids=feedback_input_ids,
            gconfig=feedback_gconfig,
            tokenizer=self.tokenizer,
        )

        async with atrace_session_phase("generate_feedback"):
            feedback_resp = await engine.agenerate(feedback_req)

        return feedback_resp

    @session_context()
    async def arun_episode(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        # Lazy-load reward function if given as string.
        if isinstance(self.reward_fn, str):
            self.reward_fn = import_from_string(self.reward_fn)
            self.async_reward_fn = AsyncRewardWrapper(self.reward_fn)

        input_ids = self.get_input_ids_fn(
            self.data_extract_prompt_fn(data),
            self.tokenizer,
            self.enable_thinking,
        )
        req = ModelRequest(
            rid=uuid.uuid4().hex,
            input_ids=input_ids,
            gconfig=self.gconfig.new(n_samples=1),
            tokenizer=self.tokenizer,
        )

        prompt_str = self.tokenizer.decode(input_ids)

        # Step 1: Generate response and compute reward.
        async with atrace_session_phase("generate"):
            resp = await engine.agenerate(req)
        reward = await self._compute_rewards(resp, prompt_str, data)

        stats_tracker.get(workflow_context.stat_scope()).scalar(reward=reward)

        prompt_ids = resp.input_tokens
        response_ids = resp.output_tokens
        prompt_len = resp.input_len
        response_len = resp.output_len

        feedback_resp = await self._generate_feedback(
            engine, prompt_ids, response_ids
        )

        # Extended trajectory:
        #   [prompt | response | feedback_template | feedback_output | response_copy]
        #
        # Masks:
        #   loss_mask: 1 on first response tokens (student target)
        #   feedback_mask: 1 on feedback + response_copy tokens (teacher context)

        feedback_prefix_ids = self.tokenizer.encode(
            self.feedback_template, add_special_tokens=False
        )
        feedback_output_ids = feedback_resp.output_tokens
        feedback_prefix_len = len(feedback_prefix_ids)
        feedback_output_len = feedback_resp.output_len

        # Build full sequence.
        seq = (
            prompt_ids
            + response_ids
            + feedback_prefix_ids
            + feedback_output_ids
            + response_ids  # response copy for teacher
        )
        total_len = len(seq)

        # Build logprobs (zeros for non-rollout tokens).
        logprobs = (
            [0.0] * prompt_len
            + resp.output_logprobs
            + [0.0] * (feedback_prefix_len + feedback_output_len + response_len)
        )

        # Build loss_mask: 1 on first response only.
        loss_mask = (
            [0] * prompt_len
            + [1] * response_len
            + [0] * (feedback_prefix_len + feedback_output_len + response_len)
        )

        # Build feedback_mask: 1 on feedback + response_copy.
        feedback_mask = [0] * (prompt_len + response_len) + [1] * (
            feedback_prefix_len + feedback_output_len + response_len
        )

        # Build versions.
        versions = (
            [-1] * prompt_len
            + resp.output_versions
            + [-1] * (feedback_prefix_len + feedback_output_len + response_len)
        )

        res = {
            "input_ids": torch.tensor(seq, dtype=torch.int32),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.int32),
            "feedback_mask": torch.tensor(feedback_mask, dtype=torch.int32),
            "logprobs": torch.tensor(logprobs, dtype=torch.float32),
            "versions": torch.tensor(versions, dtype=torch.int32),
            "attention_mask": torch.ones(total_len, dtype=torch.bool),
            "rewards": torch.tensor(reward, dtype=torch.float32),
        }
        return {k: v.unsqueeze(0) for k, v in res.items()}
