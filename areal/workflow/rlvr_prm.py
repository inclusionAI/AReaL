import asyncio
from dataclasses import asdict
import os
import re
import uuid

import aiofiles
import aiofiles.os
import colorama
import torch
from transformers import PreTrainedTokenizerFast, PreTrainedModel

from areal.api.cli_args import GenerationHyperparameters, PRMRewardHyperparameters
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import ModelRequest
from areal.api.reward_api import AsyncRewardWrapper
from areal.api.workflow_api import RolloutWorkflow
from areal.utils import logging, stats_tracker
from areal.utils.data import concat_padded_tensors

logger = logging.getLogger("RLVR workflow")


class PRMRLVRWorkflow(RolloutWorkflow):
    def __init__(
        self,
        reward_fn,
        reward_fn_prm,
        gconfig: GenerationHyperparameters,
        prmconfig: PRMRewardHyperparameters,
        tokenizer: PreTrainedTokenizerFast,
        # prm_model: PreTrainedModel,
        # prm_tokenizer: PreTrainedTokenizerFast,
        enable_thinking: bool,
        rollout_stat_scope: bool = "rollout",
        dump_dir: str | None = None,
    ):
        self.reward_fn = reward_fn
        self.gconfig = gconfig
        self.prmconfig = prmconfig
        self.tokenizer = tokenizer
        # self.prm_model = prm_model
        # self.prm_tokenizer = prm_tokenizer
        self.enable_thinking = enable_thinking
        self.dump_dir = dump_dir
        self.rollout_stat_scope = rollout_stat_scope
        self.async_reward_fn = AsyncRewardWrapper(reward_fn)
        self.async_reward_fn_prm = AsyncRewardWrapper(reward_fn_prm)
        if self.dump_dir is not None and not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir, exist_ok=True)

    async def arun_episode(self, engine: InferenceEngine, data):
        input_ids = self.tokenizer.apply_chat_template(
            data["messages"],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )

        n_samples = self.gconfig.n_samples
        req = ModelRequest(
            rid=uuid.uuid4().hex,
            input_ids=input_ids,
            gconfig=self.gconfig.new(n_samples=1),
            tokenizer=self.tokenizer,
        )
        resps = await asyncio.gather(*[engine.agenerate(req) for _ in range(n_samples)])

        version = engine.get_version()
        prompt_strs = []
        completions_strs = []
        rewards = []
        result_rewards = []
        prm_rewards = []
        reward_masks = []
        seqlens = []

        results = []
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
            result_reward = await self.async_reward_fn(
                prompt_str,
                completions_str,
                resp.input_tokens,
                resp.output_tokens,
                **data,
            )

            # separate steps
            full_str = self.tokenizer.decode(resp.output_tokens, clean_up_tokenization_spaces=False)
            raw_lines = full_str.split("\n")
            lines = [line for line in raw_lines if line.strip() != ""]
            ends = []
            pos = 0
            line_i = 0
            for raw_line in raw_lines:
                if raw_line.strip() == "":
                    pos += len(raw_line) + 1 
                    continue
                pos += len(raw_line)
                ends.append(pos)
                pos += 1  
                line_i += 1
            last_indices = [None] * len(lines)
            cur_len = 0
            seg_i = 0
            for idx, tok in enumerate(resp.output_tokens):
                piece = self.tokenizer.decode([tok], clean_up_tokenization_spaces=False)
                cur_len += len(piece)
                while seg_i < len(ends) and cur_len >= ends[seg_i]:
                    last_indices[seg_i] = idx
                    seg_i += 1
                if seg_i >= len(ends):
                    break
            if last_indices and last_indices[-1] != len(resp.output_tokens) - 2:
                last_indices[-1] = len(resp.output_tokens) - 2

            steps_str = "<extra_0>".join([line_text for line_text in lines])
            cr_pos = [resp.input_len+last_indice for last_indice in last_indices]

            prm_reward = await self.async_reward_fn_prm(
                prompt_str,
                steps_str,
                resp.input_tokens,
                resp.output_tokens,
                # self.prm_model,
                # self.prm_tokenizer,
                **data,
            )
            
            # Log reward.
            stats_tracker.get(self.rollout_stat_scope).scalar(reward=result_reward)

            rewards.append(prm_reward)         
            prm_rewards.append(prm_reward)
            result_rewards.append(result_reward)

            # step reward
            dense_reward = torch.zeros(len(seq), dtype=torch.float)
            # print(f"cr_pos: {cr_pos}")
            dense_reward[cr_pos] = torch.tensor(prm_reward, dtype=torch.float)
            reward_mask = torch.zeros(len(seq), dtype=torch.bool)
            reward_mask[cr_pos] = True
            reward_masks.append(reward_mask)

            res = dict(
                # unsqueeze to add an additional batch dimension
                input_ids=torch.tensor(seq).unsqueeze(0),
                loss_mask=torch.tensor(loss_mask).unsqueeze(0),
                logprobs=torch.tensor(logprobs).unsqueeze(0),
                versions=torch.tensor(versions).unsqueeze(0),
                attention_mask=torch.ones(len(seq), dtype=torch.bool).unsqueeze(0),
                # reward
                rewards=dense_reward.unsqueeze(0),
            )
            results.append(res)
        # print(f"original rewards: {results[0]["rewards"]}")
        # print(f"avg_prm_reward: {sum(prm_rewards[0]) / len(prm_rewards[0])}")
        # print(f"prm_reward: {prm_rewards[0]}, result_reward: {result_rewards[0]}")

        # clip mechanism
        if self.prmconfig.use_clip:
            for res, reward_mask, prm_reward in zip(results, reward_masks, prm_rewards):
                dense_reward = res["rewards"]
                if isinstance(prm_reward, list):
                    avg_prm_reward = sum(prm_reward) / len(prm_reward)
                else:
                    avg_prm_reward = prm_reward
                gt_mean = (dense_reward > avg_prm_reward) & reward_mask
                ls_mean = (dense_reward <= avg_prm_reward) & reward_mask
                res["rewards"][gt_mean] = 0  
                res["rewards"][ls_mean] -= avg_prm_reward
        # print(f"rewards after clip: {results[0]["rewards"]}")

        # delta mechanism
        if self.prmconfig.use_delta:
            for res, reward_mask in zip(results, reward_masks):
                rewards_1d = res["rewards"].squeeze(0)
                valid = rewards_1d[reward_mask]
                new_v = valid.clone()
                K = new_v.numel()
                new_v[-1] = 0
                if K > 1:
                    new_v[:-2] = valid[:-2] - valid[1:-1]
                out = rewards_1d.clone()
                out[reward_mask] = new_v
                res["rewards"] = out.unsqueeze(0) 
        # print(f"rewards after delta: {results[0]["rewards"]}")

        # success reward
        for res, result_reward in zip(results, result_rewards):
            seq_len = res["rewards"].shape[1]
            res["rewards"][:, seq_len-2] = result_reward
        # print(f"rewards add success reward: {results[0]["rewards"]}")

        if self.dump_dir is not None:
            dump_path = os.path.join(self.dump_dir, str(version))
            await aiofiles.os.makedirs(dump_path, exist_ok=True)
            # Get the unique identifier for this prompt
            qid = None
            for key in ["query_id", "id", "qid"]:
                qid = data.get(key, None)
                if qid is not None:
                    break
            qid = qid or uuid.uuid4().hex

            # Dump rollout to file
            file_path = os.path.join(dump_path, f"{qid}.txt")
            async with aiofiles.open(file_path, "a") as f:
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
                    await f.write(info + "\n")

        return concat_padded_tensors(results)
