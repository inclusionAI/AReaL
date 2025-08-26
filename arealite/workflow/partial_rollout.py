import asyncio
import uuid
from typing import List, Dict, Any
import torch
from tensordict import TensorDict, NonTensorData
from transformers import PreTrainedTokenizerFast

from arealite.api.cli_args import GenerationHyperparameters
from arealite.api.io_struct import LLMRequest, LLMResponse
from arealite.api.workflow_api import RolloutWorkflow
from realhf.api.core.data_api import RL_TASKS, load_hf_tokenizer
from arealite.utils.padding import concat_padded_tensors


class PartialRolloutWorkflow(RolloutWorkflow):
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

    async def _run_new_prompt_task(self, engine, data: Dict[str, Any]) -> TensorDict:
        """
        Run a new prompt task.
        """
        data_name_id = f"q[{data['query_id'][0]}]i[{data['index_in_group'][0]}]"

        print(f"[PartialRolloutWorkflow] data {data_name_id} run_new_prompt_task with data: {data}")
        assert data.get("prompt") is not None
        assert data.get("previous_ids") is None
        
        prompt_text = data["prompt"][0]
        input_ids = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=None,
            padding=False,
            return_length=True,
            return_attention_mask=False)["input_ids"]

        assert self.gconfig.n_samples == 1, "in PartialRolloutWorkflow, n_samples must be 1"
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
        rid = uuid.uuid4().hex
        req = LLMRequest(
            rid=rid,
            input_ids=input_ids,
            gconfig=new_gconfig,
        )
        print(f"start req {rid}")
        resp: LLMResponse = await engine.agenerate(req)

        print(f"data {data_name_id}, finish req {rid}, resp is {resp}")

        """
        resp: LLMResponse, which contains the following fields:
            - input_tokens: List[int]
            - output_tokens: List[int]
            - output_logprobs: List[float]
            - output_versions: List[int]
            - stop_reason: Literal["length", "stop", "interrupt", "abort"] = "stop"
        """
        seq = resp.input_tokens + resp.output_tokens
        previous_logprobs = []
        logprobs = previous_logprobs + resp.output_logprobs
        prompt_mask = [1] * resp.input_len + [0] * resp.output_len
        output_version = resp.output_versions[0]
        versions = [output_version] * (resp.input_len + resp.output_len)

        # seq_no_eos_mask 的意义是，这个 sample 不需要 eos_mask，是有 EOS 的意思
        seq_no_eos_mask = seq[-1] == self.tokenizer.eos_token_id
        # seq_no_eos_mask = resp.stop_reason in ["stop", "interrupt", "abort"]

        if "prompt" in data.keys():
            del data["prompt"]

        completion = self.tokenizer.decode(resp.output_tokens, clean_up_tokenization_spaces=False, skip_special_tokens=True)

        reward = self.reward_fn(
            prompt=prompt_text,
            completion=completion,
            prompt_ids=resp.input_tokens,
            completion_ids=resp.output_tokens,
            **data,
        )
        task_id = RL_TASKS.index(data["task"][0])
        print(f"[PartialRolloutWorkflow][New Prompt] data {data_name_id}, task: {data['task'][0]}, input_ids {input_ids},  prompt: {prompt_text}, completion: {completion}, completion_tokens: {resp.output_tokens}, solutions: {data['solutions'][0]}, reward: {reward}")
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
            seq_no_eos_mask=torch.tensor([seq_no_eos_mask]),
            query_id=NonTensorData([data["query_id"][0]]),
            index_in_group=NonTensorData([data["index_in_group"][0]]),
            task=NonTensorData([data["task"][0]]),
            solutions=NonTensorData([data["solutions"][0]]),
        )
        return TensorDict(res, batch_size=[1])

    async def _run_reapply_task(self, engine, data: Dict[str, Any]) -> TensorDict:
        """
        Run a reapply task.
        """
        data_name_id = f"q[{data['query_id'][0]}]i[{data['index_in_group'][0]}]"
        print(f"[PartialRolloutWorkflow] data {data_name_id} run_reapply_task with data: {data}")
        assert data.get("previous_ids") is not None

        input_ids = data["previous_ids"][0]
        prompt_len = data["previous_prompt_len"][0]
        prompt_ids = input_ids[:prompt_len]
        prompt_text = self.tokenizer.decode(
            prompt_ids, clean_up_tokenization_spaces=False, skip_special_tokens=True
        )

        # seq_no_eos_mask 的意义是，这个 sample 不需要 eos_mask，是有 EOS 的意思
        is_sample_finished = data["previous_seq_no_eos_mask"][0]

        assert self.gconfig.n_samples == 1, "in PartialRolloutWorkflow, n_samples must be 1"
        
        if (not is_sample_finished) and (len(input_ids) < self.gconfig.max_tokens):
            # 续推
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
            rid = uuid.uuid4().hex
            req = LLMRequest(
                rid=rid,
                input_ids=input_ids,
                gconfig=new_gconfig,
            )
            print(f"start req {rid}")
            resp: LLMResponse = await engine.agenerate(req)
            print(f"data {data_name_id} finish req {rid}, resp is {resp}")

            """
            resp: LLMResponse, which contains the following fields:
                - input_tokens: List[int]
                - output_tokens: List[int]
                - output_logprobs: List[float]
                - output_versions: List[int]
                - stop_reason: Literal["length", "stop", "interrupt", "abort"] = "stop"
            """
            seq = resp.input_tokens + resp.output_tokens
            
            completion_len = len(seq) - prompt_len
            completion_ids = seq[prompt_len:]

            previous_logprobs = data["previous_logprobs"][0][:resp.input_len] # 这里要用 input_len 而不是 prompt_len，因为可能是中间打断的
            logprobs = previous_logprobs + resp.output_logprobs
            prompt_mask = [1] * prompt_len + [0] * completion_len
            versions = data["previous_version"][0][:resp.input_len] + resp.output_versions
            seq_no_eos_mask = seq[-1] == self.tokenizer.eos_token_id
            # seq_no_eos_mask = resp.stop_reason in ["stop", "interrupt", "abort"]
        else:
            # sample is already finished, we do not need to request a new rollout, just use the previous responses
            seq = input_ids
            completion_len = len(seq) - prompt_len
            completion_ids = seq[prompt_len:]

            logprobs = data["previous_logprobs"][0]
            prompt_mask = [1] * prompt_len + [0] * completion_len
            versions = data["previous_version"][0]
            seq_no_eos_mask = data["previous_seq_no_eos_mask"][0]


        if "prompt" in data.keys():
            del data["prompt"]
        if "previous_ids" in data.keys():
            del data["previous_ids"]
        if "previous_version" in data.keys():
            del data["previous_version"]
        if "previous_logprobs" in data.keys():
            del data["previous_logprobs"]

        completion = self.tokenizer.decode(completion_ids, clean_up_tokenization_spaces=False, skip_special_tokens=True)

        if not is_sample_finished:
            reward = self.reward_fn(
                prompt=prompt_text,
                completion=completion,
                prompt_ids=prompt_ids,
                completion_ids=completion_ids,
                **data,
            )
        else:
            reward = data["previous_rewards"][0]

        task_id = RL_TASKS.index(data["task"][0])
        print(f"[PartialRolloutWorkflow][Reapply] data {data_name_id}, task: {data['task'][0]}, input_ids {input_ids},  prompt: {prompt_text}, completion: {completion}, completion_tokens: {completion_ids}, solutions: {data['solutions'][0]}, reward: {reward}")
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
            seq_no_eos_mask=torch.tensor([seq_no_eos_mask]),
            query_id=NonTensorData([data["query_id"][0]]),
            index_in_group=NonTensorData([data["index_in_group"][0]]),
            task=NonTensorData([data["task"][0]]),
            solutions=NonTensorData([data["solutions"][0]]),
        )
        return TensorDict(res, batch_size=[1])

    async def arun_episode(self, engine, data: Dict[str, Any]) -> TensorDict:
        """
        :param data: A dictionary containing the following
            - "prompt": A list containing a single string prompt.

            - "previous_ids": If the key exists, state that this data is a sample that was not fully inferred in 
            the previous round. This field is a list containing a single input_ids, which is a list of token IDs 
            representing the prompt.
            - "previous_version": If the data is a reply sample, this field is a list containing a list of integers,
            which represents the versions of the tokens in the previous sample.
            - "previous_logprobs": If the data is a reply sample, this field is a list containing a list of floats,
            which represents the log probabilities of the tokens in the previous sample.
            - "previous_prompt_len": If the data is a reply sample, this field is a list containing an integer,
            which represents the lengths of the prompts in the previous sample.
            - "previous_seq_no_eos_mask": If the data is a reply sample, this field is a list containing an integer,
            which represents the sequence without end-of-sequence tokens in the previous sample(which means this sample
            is not finished yet).
            - "previous_rewards"

            - "task": A list containing a single task name, which is one of the RL_TASKS.
            - "solutions": A list containing a single solution string.
            - "query_id": A list containing a single query ID string, which is the id of the prompt, all samples 
            in the same group(has same prompt) should have the same query_id.
            - "index_in_group": A list containing a single integer, which is the index of this sample in the group. The "query_id" and "index_in_group" together uniquely identify a sample.
        """
        # text = self.tokenizer.apply_chat_template(
        #     data["messages"], tokenize=False, add_generation_prompt=True
        # )
        
        if self.tokenizer is None:
            self.tokenizer = load_hf_tokenizer(self.tokenizer_path)
        assert isinstance(data, dict), "data must be a dictionary"

        if data.get("previous_ids") is None:
            # new prompt
            return await self._run_new_prompt_task(engine, data)
        else:
            # reapply
            return await self._run_reapply_task(engine, data)

    async def arun_episodes(self, engine, data_list) -> List[TensorDict]:
        raise NotImplementedError
