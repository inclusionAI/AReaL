from typing import Any, Optional

from arealite.api.cli_args import (
    GenerationHyperparameters,
    RolloutWorkflowConfig,
    TrainingArgs,
)
from arealite.api.io_struct import LLMRequest, Trajectory
from arealite.api.llm_client_api import LLMClient
from arealite.api.rollout_api import RolloutWorkflow


class RlvrWorkflow(RolloutWorkflow):
    def __init__(
        self,
        args: TrainingArgs,
        config: RolloutWorkflowConfig,
        llm_client: LLMClient,
    ):
        super().__init__(args, config, None, None)
        self.llm_client = llm_client

    def run_episode(
        self,
        gconfig: GenerationHyperparameters,
        env_option: Optional[Any] = None,
        seed: Optional[int] = None,
    ) -> Trajectory:
        """Run a single episode and return the trajectory."""
        prompt_ids = env_option["input_ids"]
        req = LLMRequest(input_ids=prompt_ids, gconfig=gconfig)
        resp = self.llm_client.generate(req)

        input_len = len(resp.input_tokens)
        output_len = len(resp.output_tokens)

        input_ids = resp.input_tokens + resp.output_tokens
        prompt_mask = [1] * input_len + [0] * output_len
        logprobs = [0.0] * input_len + resp.output_logprobs
        versions = [-1] * input_len + resp.output_versions

        return Trajectory(
            prompt=env_option,
            data=dict(
                input_ids=input_ids,
                prompt_mask=prompt_mask,
                logprobs=logprobs,
                versions=versions,
            ),
        )
