import os

import torch
from transformers import AutoProcessor, PreTrainedTokenizerFast

from areal.api.cli_args import GenerationHyperparameters
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import ModelRequest
from areal.utils import logging, stats_tracker
from areal.utils.image import image2base64
from areal.workflow.multi_turn import MultiTurnWorkflow

logger = logging.getLogger("Vision Multi-Turn workflow")


class VisionMultiTurnWorkflow(MultiTurnWorkflow):
    def __init__(
        self,
        reward_fn,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast,
        processor: AutoProcessor,
        max_turns: int,
        turn_discount: float,
        rollout_stat_scope: str = "rollout",
        dump_dir: str | None = None,
    ):
        super().__init__(
            reward_fn,
            gconfig,
            tokenizer,
            max_turns,
            turn_discount,
            rollout_stat_scope=rollout_stat_scope,
            dump_dir=dump_dir,
        )
        
        self.multi_modal = False
        self.processor=processor

    async def _run_one_episode(self, engine: InferenceEngine, data, rid):
        # Enforces `n_samples=1`
        # Placeholders for the results
        seq, logprobs, loss_mask, versions = [], [], [], []
        messages = data["messages"]
        
        if "images" in data.keys():
            byte_images = image2base64(data["images"])
            self.multi_modal = True
            processed_input = self.processor(
                images=data["images"],
                text=data["messages"],
                padding=False,
                return_tensors="pt",
            )
            #BUG: Sglang forces to convert input_ids to prompt when processing multimodal data, which leads to slightly different input_ids in rollout and training.
            input_ids = processed_input["input_ids"].tolist()[0] 
        else:
            input_ids = self.tokenizer.encode(messages, add_special_tokens=False)
            
        # Run multi-turn rollout until correct
        t = reward = 0
        discount = 1
        while reward == 0 and t < self.max_turns:
            # Send generate request to get the response.
            req = ModelRequest(
                rid=rid,
                input_ids=input_ids,
                gconfig=self.gconfig.new(n_samples=1),
                tokenizer=self.tokenizer,
            )
            if self.multi_modal:
                req.image_data=byte_images
                
            resp = await engine.agenerate(req)
            # compute reward: 1 for correct and 0 otherwise
            prompt_str = self.tokenizer.decode(input_ids)
            completions_str = self.tokenizer.decode(resp.output_tokens)
            reward = await self.async_reward_fn(
                prompt_str,
                completions_str,
                resp.input_tokens,
                resp.output_tokens,
                **data,
            )
            # Amend results
            input_len = len(resp.input_tokens) - len(seq)
            assert len(seq) == 0 or resp.input_tokens[:-input_len] == seq, (
                seq,
                resp.input_tokens[:-input_len],
                len(seq),
                len(resp.input_tokens[:-input_len]),
            )
            seq += resp.input_tokens[-input_len:] + resp.output_tokens
            logprobs += [0.0] * input_len + resp.output_logprobs
            loss_mask += [0] * input_len + [1] * resp.output_len
            versions += [-1] * input_len + resp.output_versions
            # Increase counter
            t += 1
            # Amend a prompt if the previous answer is incorrect
            if reward == 0 and t < self.max_turns:
                input_ids = input_ids + resp.output_tokens
                if resp.output_tokens[-1] != self.tokenizer.eos_token_id:
                    input_ids += [self.tokenizer.eos_token_id]
                input_ids += self.multi_turn_prompt_ids
                discount *= self.turn_discount

        reward = float(reward * discount)

        # Log reward.
        stats_tracker.get(self.rollout_stat_scope).scalar(reward=reward, num_turns=t)

        res = dict(
            input_ids=torch.tensor(seq),
            logprobs=torch.tensor(logprobs),
            loss_mask=torch.tensor(loss_mask),
            versions=torch.tensor(versions),
            rewards=torch.tensor(float(reward * discount)),
            attention_mask=torch.ones(len(seq), dtype=torch.bool),
        )
        res = {k: v.unsqueeze(0) for k, v in res.items()}
        if self.multi_modal:
            
            res["multi_modal_input"] = [
                {
                    "pixel_values": processed_input["pixel_values"],
                }
            ]
            if "image_grid_thw" in processed_input:
                res["multi_modal_input"][0]["image_grid_thw"] = processed_input[
                    "image_grid_thw"
                ]
            
        return (
            res,
            prompt_str,
            completions_str,
            reward,
            len(seq),
        )

