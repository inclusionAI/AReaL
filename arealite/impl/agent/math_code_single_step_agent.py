# Copyright 2025 Ant Group Inc.
# Licensed under the Apache License, Version 2.0 (the "License").

import uuid

from arealite.api.agent_api import Agent
from arealite.api.cli_args import AgentConfig, LLMClientConfig, TrainingArgs
from arealite.api.io_struct import AgentInferOutput, LLMRequest
from arealite.impl.environment.math_code_single_step_env import (
    MathCodeAction,
    MathCodeObs,
)
from realhf.api.core.data_api import load_hf_tokenizer


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

        # Create LLM request
        llm_req = LLMRequest(
            rid=str(qid) + "-" + str(uuid.uuid4()),
            text=prompt_text,
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
