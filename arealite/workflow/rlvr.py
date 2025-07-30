import asyncio
import uuid

import torch
from tensordict import TensorDict
from transformers import PreTrainedTokenizerFast

from arealite.api.cli_args import GenerationHyperparameters
from arealite.api.io_struct import LLMRequest
from arealite.api.workflow_api import RolloutWorkflow
from realhf.api.core.data_api import RL_TASKS
from arealite.utils.padding import concat_padded_tensors


class RLVRWorkflow(RolloutWorkflow):
    def __init__(
        self,
        reward_fn,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast,
    ):
        self.reward_fn = reward_fn
        self.gconfig = gconfig
        self.tokenizer = tokenizer

    async def arun_episode(self, engine, data):
        # text = self.tokenizer.apply_chat_template(
        #     data["messages"], tokenize=False, add_generation_prompt=True
        # )
        text = data["prompt"][0]
        prompt_encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=None,
            padding=False,
            return_length=True,
            return_attention_mask=False)

        n_samples = self.gconfig.n_samples
        req = LLMRequest(
            rid=uuid.uuid4().hex,
            input_ids=prompt_encodings["input_ids"],
            gconfig=self.gconfig.new(n_samples=1),
        )
        resps = await asyncio.gather(*[engine.agenerate(req) for _ in range(n_samples)])

        results = []
        for resp in resps:
            seq = resp.input_tokens + resp.output_tokens
            logprobs = [0] * resp.input_len + resp.output_logprobs
            prompt_mask = [1] * resp.input_len + [0] * resp.output_len
            versions = [-1] * resp.input_len + resp.output_versions
            seq_no_eos_mask = resp.stop_reason == "stop"

            if "prompt" in data.keys():
                del data["prompt"]

            completion = self.tokenizer.decode(resp.output_tokens, clean_up_tokenization_spaces=False, skip_special_tokens=True)

            reward = self.reward_fn(
                prompt=text,
                completion=completion,
                prompt_ids=resp.input_tokens,
                completion_ids=resp.output_tokens,
                **data,
            )
            task_id = RL_TASKS.index(data["task"])
            res = dict(
                # unsqueeze to add an additional batch dimension
                input_ids=torch.tensor(seq).unsqueeze(0),  # seq=[10, 22, 33] => tensor([[10, 22, 33]])
                prompt_mask=torch.tensor(prompt_mask).unsqueeze(0),
                logprobs=torch.tensor(logprobs).unsqueeze(0),
                versions=torch.tensor(versions).unsqueeze(0),
                attention_mask=torch.ones(len(seq)).unsqueeze(0),
                rewards=torch.tensor([reward]),
                seqlen=torch.tensor([len(seq)]),
                task_ids=torch.tensor([task_id]),
                seq_no_eos_mask=torch.tensor([seq_no_eos_mask])
            )
            results.append(TensorDict(res, batch_size=[1]))

        return concat_padded_tensors(results)
    #  'prompt_mask': tensor([
    #         [1, 1, 0, 0, 0],
    #         [1, 1, 1, 0, 0],
    #     ]),  # Shape (2, 5)
