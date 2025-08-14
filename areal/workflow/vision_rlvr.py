import asyncio
import functools
import os
import uuid

import colorama
import torch
from tensordict import TensorDict
from transformers import AutoProcessor, PreTrainedTokenizerFast
from areal.utils.qwen_vl_utils import process_vision_info

from areal.api.cli_args import GenerationHyperparameters
from areal.api.io_struct import VLMRequest
from areal.utils.data import concat_padded_tensors
from areal.utils.multimodal import image2base64
from areal.workflow.rlvr import RLVRWorkflow
from realhf.base import logging

logger = logging.getLogger("RLVR workflow")
REWARD_TIMEOUT_SECONDS = 30

class VisionRLVRWorkflow(RLVRWorkflow):
    def __init__(
        self,
        reward_fn,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast,
        processor: AutoProcessor,
        enable_thinking: bool,
        dump_dir: str | None = None,
    ):
        super().__init__(reward_fn, gconfig, tokenizer, enable_thinking, dump_dir)
        self.processor = processor

    async def arun_episode(self, engine, data):
        
        n_samples = self.gconfig.n_samples
        
        if data.get("videos", None) is not None:
            _, videos=process_vision_info(data["conversation"])

            processed_input = self.processor(
                videos=videos,
                text=data["messages"],
                padding=False,
                return_tensors="pt",
            )
            input_ids =self.processor.tokenizer(data["messages"],padding=False, return_tensors="pt")['input_ids'].tolist()[0]

            # Special extension and padding are applied for image but not video. Thus video input has to use tokenizer tokenized input_ids. Ref to https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py#L1295
            req = VLMRequest(
                rid=uuid.uuid4().hex,
                input_ids=input_ids,
                video_data=data["videos"],
                gconfig=self.gconfig.new(n_samples=1),
            )
            
        elif data.get("images", None) is not None:
            processed_input=self.processor(
                images=data["images"],
                text=data["messages"],
                padding=False,
                return_tensors="pt",
            )
            input_ids = processed_input["input_ids"].tolist()[0]
            byte_images= image2base64(data["images"]) 
            
            req = VLMRequest(
                rid=uuid.uuid4().hex,
                input_ids=input_ids,
                image_data=byte_images,
                gconfig=self.gconfig.new(n_samples=1),
            )

        resps = await asyncio.gather(*[engine.agenerate(req) for _ in range(n_samples)])
        version = engine.get_version()
        prompt_strs = []
        completions_strs = []
        rewards = []
        seqlens = []

        results = []
        loop = asyncio.get_event_loop()
        for resp in resps:
            seq = resp.input_tokens + resp.output_tokens
            logprobs = [0.0] * resp.input_len + resp.output_logprobs
            loss_mask = [0] * resp.input_len + [1] * resp.output_len
            versions = [-1] * resp.input_len + resp.output_versions

            prompt_str = self.tokenizer.decode(input_ids)
            completions_str = self.tokenizer.decode(resp.output_tokens)
            prompt_strs.append(prompt_str)
            completions_strs.append(completions_str)
            seqlens.append(len(seq))
            try:
                reward = await asyncio.wait_for(
                    loop.run_in_executor(
                        self.rw_executor,
                        functools.partial(
                            self.reward_fn,
                            prompt_str,
                            completions_str,
                            resp.input_tokens,
                            resp.output_tokens,
                            **data,
                        ),
                    ),
                    timeout=REWARD_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    f"Computing reward timeout after {REWARD_TIMEOUT_SECONDS}s. Set reward to 0."
                )
                reward = 0
            rewards.append(reward)
            res = dict(
                # unsqueeze to add an additional batch dimension
                input_ids=torch.tensor(seq).unsqueeze(0),
                loss_mask=torch.tensor(loss_mask).unsqueeze(0),
                logprobs=torch.tensor(logprobs).unsqueeze(0),
                versions=torch.tensor(versions).unsqueeze(0),
                attention_mask=torch.ones(len(seq), dtype=torch.bool).unsqueeze(0),
                # reward
                rewards=torch.tensor([reward]),
            )
            if "pixel_values" in processed_input and "image_grid_thw" in processed_input:
                res["pixel_values"]=processed_input["pixel_values"].unsqueeze(0)
                res["image_grid_thw"]=processed_input["image_grid_thw"].unsqueeze(0)
            if "pixel_values_videos" in processed_input and "video_grid_thw" in processed_input:
                res["pixel_values_videos"]=processed_input["pixel_values_videos"].unsqueeze(0)
                res["video_grid_thw"]=processed_input["video_grid_thw"].unsqueeze(0)
            results.append(TensorDict(res, batch_size=[1]))
        if self.dump_dir is not None:
            os.makedirs(os.path.join(self.dump_dir, str(version)), exist_ok=True)
            # Get the unique identifier for this prompt
            qid = None
            for key in ["query_id", "id", "qid"]:
                qid = data.get(key, None)
                if qid is not None:
                    break
            qid = qid or uuid.uuid4().hex

            # Dump rollout to file
            with open(
                os.path.join(self.dump_dir, str(version), f"{qid}.txt"), "a"
            ) as f:
                n_samples = self.gconfig.n_samples
                for i, (p, c, r, sl) in enumerate(
                    zip(prompt_strs, completions_strs, rewards, seqlens)
                ):
                    info = "\n".join(
                        [
                            f"idx: {i + 1} / {n_samples}, seqlen: {sl}, reward is {r}.",
                            f"prompt is \n{colorama.Fore.YELLOW + colorama.Style.DIM}{p}{colorama.Style.RESET_ALL}",
                            f"sequence is: \n{colorama.Fore.YELLOW + colorama.Style.DIM}{c}{colorama.Style.RESET_ALL}",
                        ]
                    )
                    f.write(info + "\n")

        return concat_padded_tensors(results)
