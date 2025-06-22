from datetime import datetime
from typing import Any, Callable, Optional

import torch

from arealite.api.cli_args import (
    GenerationHyperparameters,
    RolloutWorkflowConfig,
    TrainingArgs,
)
from arealite.api.io_struct import LLMRequest, Trajectory, TrajStats
from arealite.api.llm_client_api import LLMClient
from arealite.api.rollout_api import RolloutWorkflow


class RlvrWorkflow(RolloutWorkflow):
    def __init__(
        self,
        args: TrainingArgs,
        config: RolloutWorkflowConfig,
        llm_client: LLMClient,
        reward_fn: Callable,
    ):
        super().__init__(args, config, None, None)
        self.llm_client = llm_client
        self.reward_fn = reward_fn

    def run_episode(
        self,
        gconfig: GenerationHyperparameters,
        env_option: Optional[Any] = None,
        seed: Optional[int] = None,
    ) -> Trajectory:
        """Run a single episode and return the trajectory."""
        tik = datetime.now().timestamp()

        prompt_ids = env_option["input_ids"]
        query_id = env_option["query_id"]
        req = LLMRequest(input_ids=prompt_ids, gconfig=gconfig)
        resp = self.llm_client.generate(req)

        reward = self.reward_fn(
            query_ids=[query_id],
            prompts=[req.text],
            prompt_ids=[prompt_ids],
            completions=[resp.completion],
            completion_ids=[resp.output_tokens],
        )[0]

        input_len = len(resp.input_tokens)
        output_len = len(resp.output_tokens)

        input_ids = resp.input_tokens + resp.output_tokens
        prompt_mask = [1] * input_len + [0] * output_len
        logprobs = [0.0] * input_len + resp.output_logprobs
        versions = [-1] * input_len + resp.output_versions

        return Trajectory(
            prompt=env_option,
            data=dict(
                input_ids=torch.tensor(input_ids),
                prompt_mask=torch.tensor(prompt_mask),
                logprobs=torch.tensor(logprobs),
                versions=torch.tensor(versions),
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
        gconfig: GenerationHyperparameters,
        env_option: Optional[Any] = None,
        seed: Optional[int] = None,
    ) -> Trajectory:
        """Async version of run_episode. Run a single episode and return the trajectory."""
        tik = datetime.now().timestamp()

        prompt_ids = env_option["input_ids"]
        query_id = env_option["query_id"]
        req = LLMRequest(input_ids=prompt_ids, gconfig=gconfig)

        # Use async LLM client
        resp = await self.llm_client.agenerate(req)

        # Run reward computation in executor to avoid blocking
        reward = self.reward_fn(
            query_ids=[query_id],
            prompts=[req.text],
            prompt_ids=[prompt_ids],
            completions=[resp.completion],
            completion_ids=[resp.output_tokens],
        )[0]

        input_len = len(resp.input_tokens)
        output_len = len(resp.output_tokens)

        input_ids = resp.input_tokens + resp.output_tokens
        prompt_mask = [1] * input_len + [0] * output_len
        logprobs = [0.0] * input_len + resp.output_logprobs
        versions = [-1] * input_len + resp.output_versions

        return Trajectory(
            prompt=env_option,
            data=dict(
                input_ids=torch.tensor(input_ids),
                prompt_mask=torch.tensor(prompt_mask),
                logprobs=torch.tensor(logprobs),
                versions=torch.tensor(versions),
            ),
            stats=TrajStats(
                start_time=tik,
                total_reward=reward,
                episode_length=1,
                info={},
            ),
        )
