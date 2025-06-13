# Copyright 2025 Ant Group Inc.
# Licensed under the Apache License, Version 2.0 (the "License").

import uuid
from typing import TYPE_CHECKING

import torch
from gymnasium.core import ObsType

from arealite.api.agent_api import Agent
from arealite.api.cli_args import AgentConfig, LLMClientConfig, TrainingArgs
from arealite.api.io_struct import AgentInferOutput, LLMRequest, Message, Trajectory
from realhf.api.core.data_api import load_hf_tokenizer

if TYPE_CHECKING:
    from arealite.impl.environment.math_code_single_step_env import (
        MathCodeAction,
        MathCodeObs,
    )


class MathCodeSingleStepAgent(Agent):
    def __init__(
        self,
        args: TrainingArgs,
        client_config: LLMClientConfig,
        agent_config: AgentConfig,
    ):
        super().__init__(args, client_config, agent_config)

        config = agent_config.math_code_single_step

        self.gconfig = config.gconfig
        self.tokenizer = load_hf_tokenizer(config.tokenizer_path)

    def act(self, obs: MathCodeObs) -> AgentInferOutput:
        """Given an observation, return an action."""
        # Extract information from observation
        qid = obs.qid
        prompt_text = obs.prompt

        # Create prompt message
        messages = [Message(role="user", content=prompt_text, attachment=None)]

        # Create LLM request
        llm_req = LLMRequest(
            rid=str(qid) + "-" + str(uuid.uuid4()),
            model_id="actor",  # Default model ID for actor
            messages=messages,
            gconfig=self.gconfig,
        )

        # Generate response using LLM client
        llm_resp = self.llm_client.generate(llm_req)

        # Extract answers from completion
        answer = llm_resp.completion

        return AgentInferOutput(
            action=MathCodeAction(qid=qid, answer=answer),
            llm_req=llm_req,
            llm_resp=llm_resp,
        )

    def reset(self):
        """Resets the agent's memory."""
        pass  # Stateless agent, no memory to reset
