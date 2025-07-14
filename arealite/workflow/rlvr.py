import asyncio
import uuid

import torch
from tensordict import TensorDict
from transformers import PreTrainedTokenizerFast

from arealite.api.cli_args import GenerationHyperparameters
from arealite.api.io_struct import LLMRequest
from arealite.api.workflow_api import RolloutWorkflow
from arealite.utils.padding import concat_padded_tensors


class RLVRWorkflow(RolloutWorkflow):
    def __init__(
        self,
        reward_fn,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast,
        enable_thinking: bool,
    ):
        self.reward_fn = reward_fn
        self.gconfig = gconfig
        self.tokenizer = tokenizer
        self.enable_thinking = enable_thinking

    async def arun_episode(self, engine, data):
        input_ids = self.tokenizer.apply_chat_template(
            data["messages"],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        n_samples = self.gconfig.n_samples
        req = LLMRequest(
            rid=uuid.uuid4().hex,
            input_ids=input_ids,
            gconfig=self.gconfig.new(n_samples=1),
        )
        resps = await asyncio.gather(*[engine.agenerate(req) for _ in range(n_samples)])

        results = []
        for resp in resps:
            seq = resp.input_tokens + resp.output_tokens
            logprobs = [0] * resp.input_len + resp.output_logprobs
            prompt_mask = [1] * resp.input_len + [0] * resp.output_len
            versions = [-1] * resp.input_len + resp.output_versions

            reward = self.reward_fn(
                prompt=self.tokenizer.decode(input_ids),
                completions=self.tokenizer.decode(resp.output_tokens),
                prompt_ids=resp.input_tokens,
                completion_ids=resp.output_tokens,
                **data,
            )
            res = dict(
                # unsqueeze to add an additional batch dimension
                input_ids=torch.tensor(seq).unsqueeze(0),
                prompt_mask=torch.tensor(prompt_mask).unsqueeze(0),
                logprobs=torch.tensor(logprobs).unsqueeze(0),
                versions=torch.tensor(versions).unsqueeze(0),
                attention_mask=torch.ones(len(seq)).unsqueeze(0),
                # reward
                rewards=torch.tensor([reward]),
            )
            results.append(TensorDict(res, batch_size=[1]))

        return concat_padded_tensors(results)
