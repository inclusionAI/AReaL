# Copyright 2025 Ant Group Inc.
# Licensed under the Apache License, Version 2.0 (the "License").

from typing import List

import torch
from gymnasium.core import ObsType

from arealite.api.agent_api import Agent
from arealite.api.cli_args import AgentConfig, TrainingArgs
from arealite.api.io_struct import AgentInferOutput, LLMRequest, Message, Trajectory
from realhf.api.core.data_api import load_hf_tokenizer


class MathCodeSingleStepAgent(Agent):
    def __init__(self, args: TrainingArgs, config: AgentConfig):
        super().__init__(args, config)
        agent_config = config.math_code_single_step
        self.gconfig = agent_config.gconfig
        self.tokenizer = load_hf_tokenizer(agent_config.tokenizer_path)
        self.success_rate_lb = agent_config.success_rate_lb
        self.success_rate_ub = agent_config.success_rate_ub
        self.reward_scaling = agent_config.reward_scaling
        self.reward_bias = agent_config.reward_bias

    def act(self, obs: ObsType) -> AgentInferOutput:
        """Given an observation, return an action."""
        # Extract information from observation
        qid, prompt_tokens, problem_data = obs
        
        # Create prompt message
        prompt_text = self.tokenizer.decode(prompt_tokens, skip_special_tokens=True)
        messages = [Message(role="user", content=prompt_text, attachment=None)]
        
        # Create LLM request
        llm_req = LLMRequest(
            rid=str(qid),
            model_id="actor",  # Default model ID for actor
            messages=messages,
            gconfig=self.gconfig,
            metadata={"problem_data": problem_data}
        )
        
        # Generate response using LLM client
        llm_resp = self.llm_client.generate(llm_req)
        
        # Extract answers from completion
        if isinstance(llm_resp.completion, list):
            answers = llm_resp.completion
        else:
            answers = [llm_resp.completion]
        
        # Create action (answers for environment)
        action = (qid, answers)
        
        return AgentInferOutput(
            action=action,
            llm_req=llm_req,
            llm_resp=llm_resp
        )

    def reset(self):
        """Resets the agent's memory."""
        pass  # Stateless agent, no memory to reset

    def create_trajectory(
        self, obs: ObsType, agent_output: AgentInferOutput, env_rewards: List[float]
    ) -> Trajectory:
        """Create trajectory data from observation, action, and rewards."""
        qid, prompt_tokens, _ = obs
        llm_resp = agent_output.llm_resp
        
        # Filter rewards based on success rate bounds
        mean_reward = sum(env_rewards) / len(env_rewards) if env_rewards else 0.0
        if mean_reward < self.success_rate_lb or mean_reward > self.success_rate_ub:
            return None  # Skip this trajectory
        
        # Apply reward scaling and bias
        scaled_rewards = [
            ((r - 0.5) * 2 - self.reward_bias) * self.reward_scaling
            for r in env_rewards
        ]
        
        # Create trajectory data structure
        n_samples = len(llm_resp.output_tokens) if isinstance(llm_resp.output_tokens[0], list) else 1
        prompt_len = len(prompt_tokens)
        
        # Build packed sequences
        if isinstance(llm_resp.output_tokens[0], list):
            all_seqs = [prompt_tokens + seq for seq in llm_resp.output_tokens]
            all_logprobs = llm_resp.output_logprobs
            seqlens = [len(seq) for seq in all_seqs]
        else:
            all_seqs = [prompt_tokens + llm_resp.output_tokens]
            all_logprobs = [llm_resp.output_logprobs]
            seqlens = [len(all_seqs[0])]
        
        data = {
            "packed_input_ids": torch.tensor(sum(all_seqs, []), dtype=torch.long),
            "packed_prompts": torch.tensor(prompt_tokens, dtype=torch.long),
            "packed_logprobs": torch.tensor(sum(all_logprobs, []), dtype=torch.float32),
            "prompt_mask": torch.tensor(
                sum(
                    [[1] * prompt_len + [0] * (seqlen - prompt_len) for seqlen in seqlens],
                    []
                ),
                dtype=torch.bool,
            ),
            "seq_no_eos_mask": torch.tensor([True] * n_samples, dtype=torch.bool),
            "rewards": torch.tensor(scaled_rewards, dtype=torch.float32),
            "version_start": torch.tensor(llm_resp.output_versions or [0] * n_samples, dtype=torch.int),
            "version_end": torch.tensor(llm_resp.output_versions or [0] * n_samples, dtype=torch.int),
        }
        
        stats = {
            "qid": qid,
            "mean_reward": mean_reward,
            "prompt_len": prompt_len,
            "seqlens": seqlens,
        }
        
        return Trajectory(data=data, stats=stats)