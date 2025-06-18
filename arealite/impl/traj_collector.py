import asyncio
from concurrent.futures import ProcessPoolExecutor
from typing import Any, List, Optional

import torch.multiprocessing as mp

from arealite.api.agent_api import AgentFactory
from arealite.api.cli_args import TrainingArgs, TrajCollectorConfig
from arealite.api.env_api import EnvFactory
from arealite.api.io_struct import Trajectory
from arealite.utils import pad_sequences_to_tensors


class TrajCollector:

    def __init__(self, args: TrainingArgs, config: TrajCollectorConfig):
        self.args = args
        self.config = config

        agent_factory = AgentFactory(args, config.llm_client)
        self.agent = agent_factory.make_agent(config.agent)
        env_factory = EnvFactory(args)
        self.env = env_factory.make_env(config.env)

    def run_episode(
        self,
        env_option: Optional[Any] = None,
        seed: Optional[int] = None,
    ) -> Trajectory:
        # Reset the environment and the agent's memory.
        obs, info = self.env.reset(options=env_option, seed=seed)
        self.agent.reset()

        data = []
        ret = 0.0

        done = False
        # Episode loop.
        while not done:
            # Take an action by sending a request to generation server.
            agent_infer_out = self.agent.act(obs)
            action = agent_infer_out.action

            # Advance one step in the environment.
            nex_obs, reward, terminated, truncated, info = self.env.step(action)

            # Collect the step data.
            resp = agent_infer_out.llm_resp
            input_len = len(resp.input_tokens)
            output_len = len(resp.output_tokens)

            input_ids = resp.input_tokens + resp.output_tokens
            prompt_mask = [1] * input_len + [0] * output_len
            logprobs = [0.0] * input_len + resp.output_logprobs
            versions = [-1] * input_len + resp.output_versions

            d = dict(
                input_ids=input_ids,
                prompt_mask=prompt_mask,
                logprobs=logprobs,
                versions=versions,
            )
            data.append(d)

            ret += reward

            # Prepare information for the next step.
            done = terminated or truncated
            obs = nex_obs

        return Trajectory(
            data=pad_sequences_to_tensors(data),
            stats=dict(ret=ret),
        )

    async def run_episode_async(
        self,
        env_option: Optional[Any] = None,
        seed: Optional[int] = None,
    ) -> Trajectory:
        """Run a single episode asynchronously and return trajectory."""
        # A trick to convert a sync function to async function
        return await asyncio.to_thread(self.run_episode(env_option, seed))


# application code

# 1. create a trimmed base trainer class for inheriance
# 2. use legacy CLI args
# 3. directly use huggingface Dataset
# 4. use huggingface.trainerstate
# TODO: how to do checkpointing?

# follow the signature of transformers.Trainer if possible

# distributed sampler
# process group init
