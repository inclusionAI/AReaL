# Copyright 2025 Ant Group Inc.
# Licensed under the Apache License, Version 2.0

from datetime import datetime
from typing import Any, Callable, Dict, Optional

import torch

from arealite.api.cli_args import (
    GenerationHyperparameters,
    RolloutCollectorConfig,
    TrainingArgs,
)
from arealite.api.io_struct import LLMRequest, Trajectory, TrajStats
from arealite.api.llm_client_api import LLMClient
from arealite.api.rollout_api import RolloutCollector
from realhf.base import logging

logger = logging.getLogger(__file__)


class RlvrCollector(RolloutCollector):
    def __init__(
        self,
        args: TrainingArgs,
        config: RolloutCollectorConfig,
        reward_fn: Callable,
    ):
        super().__init__(args, config, None, None)
        self.reward_fn = reward_fn

    def run_episode(
        self,
        llm_client: LLMClient,
        gconfig: GenerationHyperparameters,
        env_option: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ) -> Trajectory:
        """Run a single episode and return the trajectory."""
        tik = datetime.now().timestamp()

        prompt_ids = env_option["input_ids"]
        query_id = env_option["query_id"]
        req = LLMRequest(input_ids=prompt_ids, gconfig=gconfig)
        resp = llm_client.generate(req)

        reward_kwargs = env_option.copy()
        reward_kwargs.pop("query_id")
        reward_kwargs.pop("prompt")
        reward = self.reward_fn(
            query_id=query_id,
            prompt=req.text,
            completion=resp.completion,
            prompt_ids=prompt_ids,
            completion_ids=resp.output_tokens,
            **reward_kwargs,
        )

        input_len = len(resp.input_tokens)
        output_len = len(resp.output_tokens)

        input_ids = resp.input_tokens + resp.output_tokens
        prompt_mask = [1] * input_len + [0] * output_len
        logprobs = [0.0] * input_len + resp.output_logprobs
        versions = [-1] * input_len + resp.output_versions

        return Trajectory(
            prompt=env_option,
            data=dict(
                # unsqueeze to add an additional batch dimension
                input_ids=torch.tensor(input_ids).unsqueeze(0),
                prompt_mask=torch.tensor(prompt_mask).unsqueeze(0),
                logprobs=torch.tensor(logprobs).unsqueeze(0),
                versions=torch.tensor(versions).unsqueeze(0),
                # reward
                rewards=torch.tensor([reward]),
            ),
            stats=TrajStats(
                start_time=tik,
                total_reward=reward,
                episode_length=1,
                info={},
            ),
        )

    async def arun_episode(
        self,
        llm_client: LLMClient,
        gconfig: GenerationHyperparameters,
        env_option: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ) -> Trajectory:
        """Async version of run_episode. Run a single episode and return the trajectory."""
        tik = datetime.now().timestamp()

        prompt_ids = env_option["input_ids"]
        query_id = env_option["query_id"]
        req = LLMRequest(input_ids=prompt_ids, gconfig=gconfig)

        # Use async LLM client
        resp = await llm_client.agenerate(req)

        # Run reward computation in executor to avoid blocking
        reward_kwargs = env_option.copy()
        reward_kwargs.pop("query_id")
        reward_kwargs.pop("prompt")
        reward = self.reward_fn(
            query_id=query_id,
            prompt=req.text,
            completion=resp.completion,
            prompt_ids=prompt_ids,
            completion_ids=resp.output_tokens,
            **reward_kwargs,
        )

        input_len = len(resp.input_tokens)
        output_len = len(resp.output_tokens)

        input_ids = resp.input_tokens + resp.output_tokens
        prompt_mask = [1] * input_len + [0] * output_len
        logprobs = [0.0] * input_len + resp.output_logprobs
        versions = [-1] * input_len + resp.output_versions

        logger.info(
            f"Prompt: {req.text}, reward: {reward}\nCompletion: {resp.completion}"
        )

        return Trajectory(
            prompt=env_option,
            data=dict(
                # unsqueeze to add an additional batch dimension
                input_ids=torch.tensor(input_ids).unsqueeze(0),
                prompt_mask=torch.tensor(prompt_mask).unsqueeze(0),
                logprobs=torch.tensor(logprobs).unsqueeze(0),
                versions=torch.tensor(versions).unsqueeze(0),
                # reward
                rewards=torch.tensor([reward]),
            ),
            stats=TrajStats(
                start_time=tik,
                total_reward=reward,
                episode_length=1,
                info={},
            ),
        )
