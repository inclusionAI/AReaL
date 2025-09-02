import asyncio
import uuid

import torch
from tensordict import TensorDict
from transformers import PreTrainedTokenizerFast

from arealite.api.cli_args import GenerationHyperparameters
from arealite.api.io_struct import LLMRequest
from arealite.api.workflow_api import RolloutWorkflow
from arealite.utils.padding import concat_padded_tensors
from realhf.api.core.data_api import RL_TASKS, load_hf_tokenizer
from realhf.base import logging

logger = logging.getLogger("RLVR")


class RLVRWorkflow(RolloutWorkflow):
    def __init__(
        self,
        reward_fn,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast = None,
        tokenizer_path: str = None,
    ):
        if tokenizer is None and tokenizer_path is None:
            raise ValueError("Either tokenizer or tokenizer_path must be provided")

        self.reward_fn = reward_fn
        self.gconfig = gconfig
        self.tokenizer = tokenizer if tokenizer is not None else None
        self.tokenizer_path = tokenizer_path

    async def arun_episode(self, engine, data):
        if self.tokenizer is None:
            self.tokenizer = load_hf_tokenizer(self.tokenizer_path)
        text = data["prompt"][0]
        prompt_encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=None,
            padding=False,
            return_length=True,
            return_attention_mask=False,
        )

        n_samples = self.gconfig.n_samples
        new_gconfig = self.gconfig.new(
            n_samples=1,
            max_new_tokens=self.gconfig.max_new_tokens,
            min_new_tokens=self.gconfig.min_new_tokens,
            greedy=self.gconfig.greedy,
            top_p=self.gconfig.top_p,
            top_k=self.gconfig.top_k,
            temperature=self.gconfig.temperature,
            stop_token_ids=self.gconfig.stop_token_ids,
        )

        req = LLMRequest(
            rid=uuid.uuid4().hex,
            text=text,
            input_ids=prompt_encodings["input_ids"],
            gconfig=new_gconfig,
        )
        resps = await asyncio.gather(*[engine.agenerate(req) for _ in range(n_samples)])
        results = []
        for resp in resps:
            seq = resp.input_tokens + resp.output_tokens
            logprobs = [0] * resp.input_len + resp.output_logprobs
            prompt_mask = [1] * resp.input_len + [0] * resp.output_len
            versions = [-1] * resp.input_len + resp.output_versions
            seq_no_eos_mask = (seq[-1] != self.tokenizer.eos_token_id) and (
                seq[-1] != self.tokenizer.pad_token_id
            )

            if "prompt" in data.keys():
                del data["prompt"]

            completion = self.tokenizer.decode(
                resp.output_tokens,
                clean_up_tokenization_spaces=False,
                skip_special_tokens=True,
            )

            reward = await self.reward_fn(
                prompt=text,
                completion=completion,
                prompt_ids=resp.input_tokens,
                completion_ids=resp.output_tokens,
                **data,
            )
            task_id = RL_TASKS.index(data["task"][0])
            print(
                f"[RLVRWorkflow] prompt: {text}, completion: {completion}, solutions: {data["solutions"][0]}"
            )
            res = dict(
                # unsqueeze to add an additional batch dimension
                input_ids=torch.tensor(seq).unsqueeze(
                    0
                ),  # seq=[10, 22, 33] => tensor([[10, 22, 33]])
                prompt_mask=torch.tensor(prompt_mask).unsqueeze(0),
                logprobs=torch.tensor(logprobs).unsqueeze(0),
                versions=torch.tensor(versions).unsqueeze(0),
                attention_mask=torch.ones(len(seq)).unsqueeze(0),
                rewards=torch.tensor([reward]),
                seqlen=torch.tensor([len(seq)]),
                task_ids=torch.tensor([task_id]),
                seq_no_eos_mask=torch.tensor([seq_no_eos_mask]),
            )
            results.append(TensorDict(res, batch_size=[1]))

        return concat_padded_tensors(results)

    async def arun_episodes(self, engine, data_list):
        if self.tokenizer is None:
            self.tokenizer = load_hf_tokenizer(self.tokenizer_path)

        n_samples = self.gconfig.n_samples
        new_gconfig = self.gconfig.new(
            n_samples=n_samples,
            max_new_tokens=self.gconfig.max_new_tokens,
            min_new_tokens=self.gconfig.min_new_tokens,
            greedy=self.gconfig.greedy,
            top_p=self.gconfig.top_p,
            top_k=self.gconfig.top_k,
            temperature=self.gconfig.temperature,
            stop_token_ids=self.gconfig.stop_token_ids,
        )

        reqs = []
        texts = []
        for data in data_list:
            text = data["prompt"][0]
            texts.append(text)
            prompt_encodings = self.tokenizer(
                text,
                truncation=True,
                max_length=None,
                padding=False,
                return_length=True,
                return_attention_mask=False,
            )

            req = LLMRequest(
                rid=uuid.uuid4().hex,
                text=text,
                input_ids=prompt_encodings["input_ids"],
                gconfig=new_gconfig,
            )
            reqs.append(req)

        resps = await engine.agenerate_batch(reqs)
        results = []
        for index, resp in enumerate(resps):
            seq = resp.input_tokens + resp.output_tokens
            logprobs = [0] * resp.input_len + resp.output_logprobs
            prompt_mask = [1] * resp.input_len + [0] * resp.output_len
            versions = [-1] * resp.input_len + resp.output_versions
            seq_no_eos_mask = (seq[-1] != self.tokenizer.eos_token_id) and (
                seq[-1] != self.tokenizer.pad_token_id
            )
            data = data_list[index // new_gconfig.n_samples]

            completion = self.tokenizer.decode(
                resp.output_tokens,
                clean_up_tokenization_spaces=False,
                skip_special_tokens=True,
            )

            text = texts[index // new_gconfig.n_samples]
            if "prompt" in data.keys():
                del data["prompt"]

            reward = await self.reward_fn(
                prompt=text,
                completion=completion,
                prompt_ids=resp.input_tokens,
                completion_ids=resp.output_tokens,
                **data,
            )
            task_id = RL_TASKS.index(data["task"][0])
            print(
                f"[RLVRWorkflow] prompt: {text}, completion: {completion}, solutions: {data['query_id'][0]}"
            )
            res = dict(
                # unsqueeze to add an additional batch dimension
                input_ids=torch.tensor(seq).unsqueeze(
                    0
                ),  # seq=[10, 22, 33] => tensor([[10, 22, 33]])
                prompt_mask=torch.tensor(prompt_mask).unsqueeze(0),
                logprobs=torch.tensor(logprobs).unsqueeze(0),
                versions=torch.tensor(versions).unsqueeze(0),
                attention_mask=torch.ones(len(seq)).unsqueeze(0),
                rewards=torch.tensor([reward]),
                seqlen=torch.tensor([len(seq)]),
                task_ids=torch.tensor([task_id]),
                seq_no_eos_mask=torch.tensor([seq_no_eos_mask]),
            )
            results.append(TensorDict(res, batch_size=[1]))

        return concat_padded_tensors(results)
