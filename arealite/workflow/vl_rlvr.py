import asyncio
import uuid

import torch
from tensordict import TensorDict
from transformers import PreTrainedTokenizerFast,AutoProcessor
from arealite.api.cli_args import GenerationHyperparameters
from arealite.api.io_struct import VLMRequest
from arealite.workflow.rlvr import RLVRWorkflow
from arealite.utils.padding import concat_padded_tensors
from arealite.utils.image import image2base64

class VL_RLVRWorkflow(RLVRWorkflow):
    def __init__(
        self,
        reward_fn,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast,
        processor: AutoProcessor,
        enable_thinking: bool,
    ):
        super().__init__(reward_fn, gconfig, tokenizer, enable_thinking)
        self.processor = processor

    async def arun_episode(self, engine, data):
        # self.processor.tokenizer.add_generation_prompt=True

        processed_input = self.processor(
            images=data["images"],
            text=data["messages"],
            padding=False,
            return_tensors="pt",
        )
        input_ids=processed_input["input_ids"].tolist()[0]

        n_samples = self.gconfig.n_samples

        byte_images = image2base64(data["images"])

        req = VLMRequest(
            rid=uuid.uuid4().hex,
            input_ids=input_ids,
            image_data=byte_images,
            gconfig=self.gconfig.new(n_samples=1),
        )
        resps = await asyncio.gather(*[engine.agenerate(req) for _ in range(n_samples)])

        results = []
        for resp in resps:
            seq = resp.input_tokens + resp.output_tokens
            logprobs = [0.0] * resp.input_len + resp.output_logprobs
            loss_mask = [0] * resp.input_len + [1] * resp.output_len
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
                loss_mask=torch.tensor(loss_mask).unsqueeze(0),
                pixel_values=processed_input["pixel_values"].clone().detach().unsqueeze(0),
                image_grid_thw=processed_input["image_grid_thw"].clone().detach().unsqueeze(0),
                logprobs=torch.tensor(logprobs).unsqueeze(0),
                versions=torch.tensor(versions).unsqueeze(0),
                attention_mask=torch.ones(len(seq), dtype=torch.bool).unsqueeze(0),
                # reward
                rewards=torch.tensor([reward]),
            )
            results.append(TensorDict(res, batch_size=[1]))

        return concat_padded_tensors(results)
