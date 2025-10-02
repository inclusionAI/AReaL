import asyncio
from dataclasses import asdict
import os
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
        prm_rewards = []
        result_rewards = []
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
            prm_reward = await self.async_reward_fn(
                prompt_str,
                completions_str,
                resp.input_tokens,
                resp.output_tokens,
                # self.prm_model,
                # self.prm_tokenizer,
                **data,
            )
            
            # Log reward.
            stats_tracker.get(self.rollout_stat_scope).scalar(reward=prm_reward)

            rewards.append(prm_reward)
            prm_rewards.append(prm_reward)
            result_rewards.append(result_reward)

            res = dict(
                # unsqueeze to add an additional batch dimension
                input_ids=torch.tensor(seq).unsqueeze(0),
                loss_mask=torch.tensor(loss_mask).unsqueeze(0),
                logprobs=torch.tensor(logprobs).unsqueeze(0),
                versions=torch.tensor(versions).unsqueeze(0),
                attention_mask=torch.ones(len(seq), dtype=torch.bool).unsqueeze(0),
                # reward
                rewards=torch.tensor([float(prm_reward)]),
            )
            results.append(res)

        # clip mechanism
        if self.prmconfig.use_clip:
            avg_prm_reward = sum(prm_rewards) / len(prm_rewards)
            for i, val in enumerate(prm_rewards):
                if val > avg_prm_reward:
                    rewards[i] = 0
                else:
                    rewards[i] = rewards[i] - avg_prm_reward
        # delta mechanism
        if self.prmconfig.use_delta:
            for i, val in enumerate(rewards):
                rewards[i] = self.prmconfig.reward_shaping_alpha * rewards[i] + result_rewards[i]

        for res, r in zip(results, rewards):
            res["rewards"] = torch.tensor([float(r)])

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
