import asyncio
import uuid
from collections.abc import Callable
from typing import Any, cast

import torch
from transformers import AutoProcessor, PreTrainedTokenizerFast

from areal.api.cli_args import GenerationHyperparameters
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import ModelRequest, ModelResponse
from areal.core import workflow_context
from areal.utils import logging, stats_tracker
from areal.utils.data import concat_padded_tensors
from areal.utils.image import image2base64
from areal.utils.perf_tracer import (
    atrace_session_phase,
    session_context,
    trace_session,
)
from areal.workflow.rlvr import RLVRWorkflow

logger = logging.getLogger("VisionRLVRWorkflow")


class VisionRLVRWorkflow(RLVRWorkflow):
    def __init__(
        self,
        reward_fn: Callable[..., Any],
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast,
        processor: AutoProcessor | str,
        enable_thinking: bool,
    ):
        super().__init__(
            reward_fn,
            gconfig,
            tokenizer,
            enable_thinking,
        )
        if isinstance(processor, str):
            processor = AutoProcessor.from_pretrained(processor)
        self.processor = processor

    @trace_session("reward")
    async def _compute_rewards(
        self,
        resp: ModelResponse,
        prompt_str: str,
        task_data: dict[str, Any],
    ) -> float:
        """Decode completion and compute reward.

        Traces reward phase execution for SessionTracer. Decodes output tokens
        to string, calls async reward function with keyword arguments, and logs
        metric to stats tracker.

        Returns
        -------
        float
            Reward value.
        """

        completions_str = self.tokenizer.decode(resp.output_tokens)
        reward = await self.async_reward_fn(
            prompt=prompt_str,
            completions=completions_str,
            prompt_ids=resp.input_tokens,
            completion_ids=resp.output_tokens,
            **task_data,
        )

        return reward

    @session_context()
    async def _collect_samples(
        self,
        engine: InferenceEngine,
        req: ModelRequest,
        prompt_str: str,
        task_data: dict[str, Any],
    ) -> tuple[ModelResponse, float]:
        """Generate one sample and compute its reward.

        Registers a new session for this sample, calls engine.agenerate,
        computes reward, and logs metrics. SessionTracer automatically
        tracks generate and reward phases via @trace_session decorators.

        Returns
        -------
        tuple[ModelResponse, float]
            Model response and reward value.
        """
        async with atrace_session_phase("generate"):
            resp = await engine.agenerate(req)

        reward = await self._compute_rewards(resp, prompt_str, task_data)

        stats_tracker.get(workflow_context.stat_scope()).scalar(reward=reward)

        return resp, reward

    async def arun_episode(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        processor_callable = cast(Callable[..., dict[str, Any]], self.processor)
        processed_input = processor_callable(
            images=data["images"],
            text=data["messages"],
            padding=False,
            return_tensors="pt",
        )

        input_ids: list[int] = processed_input["input_ids"].tolist()[0]

        n_samples = self.gconfig.n_samples

        byte_images = image2base64(data["images"])
        req = ModelRequest(
            rid=uuid.uuid4().hex,
            input_ids=input_ids,
            image_data=byte_images,
            vision_msg_vllm=[data["messages_chat"]]
            if "messages_chat" in data
            else None,
            gconfig=self.gconfig.new(n_samples=1),
            tokenizer=self.tokenizer,
            processor=self.processor,
        )

        prompt_str = self.tokenizer.decode(input_ids)

        # Generate responses and collect rewards
        sample_results = await asyncio.gather(
            *[
                self._collect_samples(engine, req, prompt_str, data)
                for _ in range(n_samples)
            ]
        )
        if sample_results:
            resps, rewards = map(list, zip(*sample_results))
        else:
            resps, rewards = [], []

        # Build result tensors
        results = []
        for resp, reward in zip(resps, rewards):
            seq = resp.input_tokens + resp.output_tokens
            logprobs = [0.0] * resp.input_len + resp.output_logprobs
            loss_mask = [0] * resp.input_len + [1] * resp.output_len
            versions = [-1] * resp.input_len + resp.output_versions

            # Build multi-modal input for each data point
            multi_modal_input = [
                {
                    "pixel_values": processed_input["pixel_values"],
                }
            ]
            if "image_grid_thw" in processed_input:
                multi_modal_input[0]["image_grid_thw"] = processed_input[
                    "image_grid_thw"
                ]

            res = {
                "input_ids": torch.tensor(seq, dtype=torch.int32).unsqueeze(0),
                "loss_mask": torch.tensor(loss_mask, dtype=torch.int32).unsqueeze(0),
                "logprobs": torch.tensor(logprobs, dtype=torch.float32).unsqueeze(0),
                "multi_modal_input": multi_modal_input,
                "versions": torch.tensor(versions, dtype=torch.int32).unsqueeze(0),
                "attention_mask": torch.ones(len(seq), dtype=torch.bool).unsqueeze(0),
                "rewards": torch.tensor(reward, dtype=torch.float32).unsqueeze(0),
            }
            results.append(res)

        return concat_padded_tensors(results)
