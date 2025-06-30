# Copyright 2025 Ant Group Inc.
# Licensed under the Apache License, Version 2.0

import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import Any, List, Optional, Tuple

import torch

from arealite.api.cli_args import GenerationHyperparameters, TrainingArgs
from arealite.api.io_struct import (
    AgentInferInput,
    AgentInferOutput,
    LLMRequest,
    Trajectory,
    TrajStats,
)
from arealite.api.llm_client_api import LLMClient
from arealite.api.rollout_api import Agent, Environment, RolloutCollector
from arealite.utils import pad_sequences_to_tensors
from functioncall.code.local_verify import code_verify as local_code_verify
from functioncall.code.verify import code_verify
from functioncall.math.verify import math_verify
from realhf.impl.dataset.math_code_dataset import load_metadata
from realhf.impl.dataset.math_parser import parse_lines_in_parallel

ENABLE_FUNCTION_CALL = True if os.getenv("FUNCTIONCALL_SERVICE_DOMAIN", "") else False
math_verify_call = math_verify if ENABLE_FUNCTION_CALL else parse_lines_in_parallel
code_verify_call = code_verify if ENABLE_FUNCTION_CALL else local_code_verify


@lru_cache(maxsize=128)
def _load_metadata_cached(dataset_path: str):
    """Cached version of load_metadata to avoid reloading metadata each time."""
    return load_metadata(dataset_path)


def extract_code(text, min_length=20):
    """Extract code blocks from text."""
    code_pattern = r"(?i)```(?:python|py|cpp|CPP)?\s*\n?(.*?)\n?```"
    code_blocks = re.findall(code_pattern, text, re.DOTALL)
    valid_blocks = []
    for block in code_blocks:
        clean_block = block.strip()
        if len(clean_block) < min_length:
            continue
        valid_blocks.append(clean_block)

    if not valid_blocks:
        return None
    # return the last code block
    return valid_blocks[-1]


@dataclass
class MathCodeAction:
    query_id: str
    answer: str


@dataclass
class MathCodeObs:
    query_id: str
    prompt_ids: List[int]


class MathCodeSingleStepEnv(Environment):
    """Math and Code single-step verification environment."""

    def __init__(self, args: TrainingArgs, solution_path: str):
        super().__init__(args)
        self.id2info, _ = _load_metadata_cached(solution_path)

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[Any, dict]:
        """Reset the environment."""
        super().reset(seed=seed)
        try:
            prompt_ids = options["input_ids"]
            query_id = options["query_id"]
        except KeyError:
            raise RuntimeError("`input_ids` and `query_id` must be set in env options.")
        # Return dummy observation and info
        return MathCodeObs(query_id=query_id, prompt_ids=prompt_ids), {}

    def step(
        self, action: MathCodeAction
    ) -> Tuple[MathCodeObs, float, bool, bool, dict]:
        """Execute one step in the environment."""
        query_id = action.query_id
        answer = action.answer

        query_id = query_id.split("@")[0]
        cur_task = self.id2info[query_id]["task"]

        if cur_task == "math":
            # Run math verification
            format_reward = math_verify_call(self.id2info, [answer], [query_id])[0]
        elif cur_task == "code":
            # Extract code blocks and run code verification
            extracted_answer = extract_code(answer)
            format_reward = code_verify_call(
                self.id2info, [extracted_answer], [query_id]
            )[0]
        else:
            raise NotImplementedError(f"Task type '{cur_task}' not implemented")

        # Return: observation, reward, terminated, truncated, info
        terminated = True  # Single step environment always terminates
        truncated = False
        info = {"task": cur_task, "query_id": query_id}

        return (
            None,
            format_reward,
            terminated,
            truncated,
            info,
        )


class MathCodeAgent(Agent):

    def act(self, inp: AgentInferInput) -> AgentInferOutput:
        """Given an observation, return an action."""
        # Extract information from observation
        obs: MathCodeObs = inp.obs
        query_id = obs.query_id
        prompt_ids = obs.prompt_ids

        # Create LLM request
        llm_req = LLMRequest(
            rid=str(query_id) + "-" + str(uuid.uuid4()),
            input_ids=prompt_ids,
            gconfig=inp.gconfig,
        )

        # Generate response using LLM client
        llm_resp = inp.llm_client.generate(llm_req)

        # Extract answers from completion
        answer = llm_resp.completion

        return AgentInferOutput(
            action=MathCodeAction(query_id=query_id, answer=answer),
            llm_req=llm_req,
            llm_resp=llm_resp,
        )

    async def aact(self, inp: AgentInferInput) -> AgentInferOutput:
        """Async version of act. Given an observation, return an action."""
        # Extract information from observation
        obs: MathCodeObs = inp.obs
        query_id = obs.query_id
        prompt_ids = obs.prompt_ids

        # Create LLM request
        llm_req = LLMRequest(
            rid=str(query_id) + "-" + str(uuid.uuid4()),
            input_ids=prompt_ids,
            gconfig=inp.gconfig,
        )

        # Generate response using async LLM client
        llm_resp = await inp.llm_client.agenerate(llm_req)

        # Extract answers from completion
        answer = llm_resp.completion

        return AgentInferOutput(
            action=MathCodeAction(query_id=query_id, answer=answer),
            llm_req=llm_req,
            llm_resp=llm_resp,
        )

    def reset(self):
        """Resets the agent's memory."""
        pass  # Stateless agent, no memory to reset

    async def areset(self):
        """Async version of reset. Resets the agent's memory."""
        pass  # Stateless agent, no memory to reset


class MathCodeSingleStepCollector(RolloutCollector):

    async def arun_episode(
        self,
        llm_client: LLMClient,
        gconfig: GenerationHyperparameters,
        env_option: Optional[Any] = None,
        seed: Optional[int] = None,
    ) -> Trajectory:
        """Async version of run_episode. Run a single episode and return the trajectory."""
        # Reset the environment and the agent's memory.
        obs, _ = self.env.reset(options=env_option, seed=seed)
        await self.agent.areset()

        data = []
        rewards = []
        tik = datetime.now().timestamp()
        ret = 0.0
        ep_len = 0

        done = False
        # Episode loop.
        while not done:
            # Take an action by sending a request to generation server.
            agent_infer_in = AgentInferInput(
                obs=obs, gconfig=gconfig, llm_client=llm_client
            )
            agent_infer_out = await self.agent.aact(agent_infer_in)
            action = agent_infer_out.action

            # Advance one step in the environment.
            nex_obs, reward, terminated, truncated, _ = self.env.step(action)

            # Collect the step data.
            resp = agent_infer_out.llm_resp
            input_len = len(resp.input_tokens)
            output_len = len(resp.output_tokens)

            input_ids = resp.input_tokens + resp.output_tokens
            prompt_mask = [1] * input_len + [0] * output_len
            logprobs = [0.0] * input_len + resp.output_logprobs
            versions = [-1] * input_len + resp.output_versions

            d = dict(
                input_ids=torch.tensor(input_ids, dtype=torch.long),
                prompt_mask=torch.tensor(prompt_mask, dtype=torch.bool),
                logprobs=torch.tensor(logprobs, dtype=torch.float32),
                versions=torch.tensor(versions, dtype=torch.long),
            )
            data.append(d)
            rewards.append(reward)

            ret += float(reward)
            ep_len += 1

            # Prepare information for the next step.
            done = terminated or truncated
            obs = nex_obs

        return Trajectory(
            prompt=env_option,
            data=dict(rewards=torch.tensor(rewards), **pad_sequences_to_tensors(data)),
            stats=TrajStats(
                start_time=tik,
                total_reward=ret,
                episode_length=ep_len,
                info={},
            ),
        )
