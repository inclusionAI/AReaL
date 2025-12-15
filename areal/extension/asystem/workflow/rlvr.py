import asyncio
import uuid

import torch
from transformers import PreTrainedTokenizerFast

from realhf.api.core.data_api import load_hf_tokenizer

from areal.api.cli_args import GenerationHyperparameters
from areal.api.io_struct import LLMRequest
from areal.api.workflow_api import RolloutWorkflow
from areal.extension.asystem.utils.util import RL_TASKS, worker_dump_rollout_output
from areal.utils import logging
from areal.utils.data import concat_padded_tensors

logger = logging.getLogger(__name__)


class RLVRWorkflow(RolloutWorkflow):
    def __init__(
        self,
        reward_fn,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast | str,
        exp_name: str = None,
        trial_name: str = None,
    ):
        self.reward_fn = reward_fn
        self.gconfig = gconfig
        self.tokenizer = tokenizer
        self._step = None
        self._rank = None
        self.exp_name = exp_name
        self.trial_name = trial_name

    async def arun_episode(self, engine, data):
        if isinstance(self.tokenizer, str):
            self.tokenizer = load_hf_tokenizer(self.tokenizer)

        from areal.extension.asystem.math_reward import reward_fn

        self.reward_fn = reward_fn

        text = data["prompt"]
        prompt_encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=None,
            padding=False,
            return_length=True,
            return_attention_mask=False,
        )

        n_samples = self.gconfig.n_samples
        # Validate max_new_tokens
        max_new_tokens = min(
            self.gconfig.max_tokens - len(prompt_encodings["input_ids"]), self.gconfig.max_new_tokens
        )
        if max_new_tokens <= 0:
            raise RuntimeError(
                f"max_new_tokens ({max_new_tokens}) is non-positive! "
                f"max_tokens={self.gconfig.max_tokens}, prompt_len={len(prompt_encodings["input_ids"])}, "
                f"max_new_tokens={self.gconfig.max_new_tokens}."
            )
        new_gconfig = self.gconfig.new(
            n_samples=1,
            max_new_tokens=max_new_tokens,
            max_tokens=self.gconfig.max_tokens,
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
            output_version = resp.output_version
            versions = [output_version] * (resp.input_len + resp.output_len)
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
            task_id = RL_TASKS.index(data["task"])

            sample_info = {
                "prompt": text,
                "completion": completion,
                "reward": reward,
                "solutions": data.get("solutions", []),
                "task": data["task"],
                "task_id": task_id,
                "input_tokens": resp.input_tokens,
                "output_tokens": resp.output_tokens,
                "output_logprobs": resp.output_logprobs,
                "seq_len": len(seq),
                "versions": versions,
                "query_id": data["query_id"],
                "stop_reason": resp.stop_reason,
            }

            worker_dump_rollout_output(sample_info=sample_info)

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
            results.append(res)

        return concat_padded_tensors(results)
