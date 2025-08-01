import uuid

import torch
from transformers import PreTrainedTokenizerFast

from areal.api.cli_args import GenerationHyperparameters
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import LLMRequest
from areal.api.workflow_api import RolloutWorkflow
from areal.utils.data import concat_padded_tensors


class MultiTurnWorkflow(RolloutWorkflow):
    def __init__(
        self,
        reward_fn,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast,
        max_turns: int,
        turn_discount: float,
    ):
        self.reward_fn = reward_fn
        self.gconfig = gconfig
        self.tokenizer = tokenizer
        self.max_turns = max_turns
        self.turn_discount = turn_discount

    async def arun_episode(self, engine: InferenceEngine, data):
        # Placeholders for the results
        seq, logprobs, loss_mask, versions = [], [], [], []
        messages = data["messages"]
        # Run multi-turn rollout until correct
        t = reward = 0
        discount = 1
        rid = uuid.uuid4().hex
        while reward == 0 and t < self.max_turns:
            # Amend a prompt if the previous answer is incorrect
            if t > 0:
                messages += [
                    {"role": "asistant", "content": completions_str},
                    {
                        "role": "user",
                        "content": "Your answer is not correct. Please try to answer it again.",
                    },
                ]
            # Convert the prompt into input_ids
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
            # Send generate request to get the response.
            req = LLMRequest(
                rid=rid,
                input_ids=input_ids,
                gconfig=self.gconfig.new(n_samples=1),
            )
            resp = await engine.agenerate(req)
            # compute reward: 1 for correct and 0 otherwise
            prompt_str = self.tokenizer.decode(input_ids)
            completions_str = self.tokenizer.decode(resp.output_tokens)
            reward = self.reward_fn(
                prompt=prompt_str,
                completions=completions_str,
                prompt_ids=resp.input_tokens,
                completion_ids=resp.output_tokens,
                **data,
            )
            # Amend results
            input_len = len(resp.input_tokens) - len(seq)
            seq += resp.input_tokens[-input_len:] + resp.output_tokens
            logprobs += [0.0] * input_len + resp.output_logprobs
            loss_mask += [0] * input_len + [1] * resp.output_len
            versions += [-1] * input_len + resp.output_versions
            # Increase counter
            t += 1
            discount *= self.turn_discount
        res = dict(
            seq=torch.tensor(seq),
            logprobs=torch.tensor(logprobs),
            loss_mask=torch.tensor(loss_mask),
            versions=torch.tensor(versions),
            rewards=torch.tensor([float(reward * discount)]),
            attetion_mask=torch.ones(len(seq), dtype=torch.bool),
        )
        res = {k: v.unsqueeze(0) for k, v in res.items()}
        return concat_padded_tensors([res])
