import asyncio
import os
import uuid

import colorama
import torch
from realhf.impl.environment import werewolf_env
from tensordict import TensorDict
from transformers import PreTrainedTokenizerFast

from areal.api.cli_args import GenerationHyperparameters
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import ModelRequest
from areal.api.reward_api import AsyncRewardWrapper
from areal.api.workflow_api import RolloutWorkflow
from areal.utils.data import concat_padded_tensors
from realhf.base import logging
from realhf.impl.environment.werewolf_env import WerewolfEnv

logger = logging.getLogger("Werewolf workflow")

class WerewolfWorkflow(RolloutWorkflow):
    """
    Workflow for running the werewolf game.
    """
    def __init__(
        self, 
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast,
        max_turns: int = 60,
        turn_discount: float = 1.0,
        dump_dir: str | None = None,
        env_kwargs: dict | None = None,
        role: str = "villager"
    ):
        self.gconfig = gconfig
        self.tokenizer = tokenizer
        self.max_turns = max_turns
        self.turn_discount = turn_discount
        self.dump_dir = dump_dir
        self.env_kwargs = env_kwargs
        self.role = role
        if self.dump_dir is not None and not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir, exist_ok=True)

    async def _run_one_episode(self, engine: InferenceEngine, data, rid):
        if self.env_kwargs:
            env = WerewolfEnv(**self.env_kwargs)
        else:
            env = WerewolfEnv()
        obs, _ = await env.reset()

        results = []
        prompt_strs = []
        completions_strs = []
        rewards = []
        seqlens = []
        vill_total = 0.0
        were_total = 0.0
        traj_len = 0

        for turn in range(self.max_turns):
            input_ids = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": obs}],
                tokenize=True,
                add_generation_prompt=True,
            )

            req = ModelRequest(
                rid=rid,
                input_ids=input_ids,
                gconfig=self.gconfig.new(n_samples=1),
                tokenizer=self.tokenizer,
            )
            resp = await engine.agenerate(req)

            seq = resp.input_tokens + resp.output_tokens
            logprobs = [0.0] * resp.input_len + resp.output_logprobs
            loss_mask = [0] * resp.input_len + [1] * resp.output_len
            versions = [-1] * resp.input_len + resp.output_versions

            prompt_str = self.tokenizer.decode(input_ids)
            completion_str = self.tokenizer.decode(resp.output_tokens)

            next_obs, reward_list, done, _, _ = await env.step(
                (data.get("query_id", ""), [completion_str])
            )
            vill_total += float(reward_list[0])
            were_total += float(reward_list[1])

            if self.role == "both":
                reward = reward_list[0] + reward_list[1]
            elif self.role == "werewolf":
                reward = reward_list[1]
            else:
                reward = reward_list[0]

            res = dict(
                input_ids=torch.tensor(seq).unsqueeze(0),
                loss_mask=torch.tensor(loss_mask).unsqueeze(0),
                logprobs=torch.tensor(logprobs).unsqueeze(0),
                versions=torch.tensor(versions).unsqueeze(0),
                attention_mask=torch.ones(len(seq), dtype=torch.bool).unsqueeze(0),
                rewards=torch.tensor([float(reward)]),
            )
            results.append(TensorDict(res, batch_size=[1]))
            prompt_strs.append(prompt_str)
            completions_strs.append(completion_str)
            rewards.append(reward)
            seqlens.append(len(seq))

            if done or (turn == self.max_turns - 1):
                logger.info(f"Trajectory ended with {turn + 1} turns, total reward: {sum(rewards)}.")
                break
            obs = next_obs
        
        stats = {}
        if hasattr(env, "get_stats"):
            try:
                stats = env.get_stats()
            except Exception:
                logger.error("No stats are available for this trajectory.")

        logging_vals = [
            len(results),
            traj_len,
            vill_total,
            were_total,
            stats.get("werewolf_kills", 0),
            stats.get("villager_correct_votes", 0),
            stats.get("villager_wrong_votes", 0),
            stats.get("witch_heals", 0),
            stats.get("witch_poisons", 0),
            stats.get("hunter_shots", 0),
        ]
        log_tensor = torch.tensor(logging_vals, dtype=torch.float32).unsqueeze(0)
        if results:
            results[0]["logging"] = log_tensor
            zero_log = torch.zeros_like(log_tensor)
            for i in range(1, len(results)):
                results[i]["logging"] = zero_log

        return results, prompt_strs, completions_strs, rewards, seqlens

    async def arun_episode(self, engine: InferenceEngine, data):
        rid = uuid.uuid4().hex
        tasks = [
            self._run_one_episode(engine, data, rid)
            for _ in range(self.gconfig.n_samples)
        ]
        episodes = await asyncio.gather(*tasks)

        results = []
        prompt_strs = []
        completions_strs = []
        rewards = []
        seqlens = []
        for res_list, p_list, c_list, r_list, sl_list in episodes:
            results.extend(res_list)
            prompt_strs.extend(p_list)
            completions_strs.extend(c_list)
            rewards.extend(r_list)
            seqlens.extend(sl_list)

        if self.dump_dir is not None and results:
            version = engine.get_version()
            os.makedirs(os.path.join(self.dump_dir, str(version)), exist_ok=True)
            qid = None
            for key in ["query_id", "id", "qid"]:
                qid = data.get(key, None)
                if qid is not None:
                    break
            qid = qid or uuid.uuid4().hex

            with open(
                os.path.join(self.dump_dir, str(version), f"{qid}.txt"), "a"
            ) as f:
                for i, (p, c, r, sl) in enumerate(
                    zip(prompt_strs, completions_strs, rewards, seqlens)
                ):
                    info = "\n".join(
                        [
                            f"idx: {i + 1}, seqlen: {sl}, reward is {r}.",
                            f"prompt is \n{colorama.Fore.YELLOW + colorama.Style.DIM}{p}{colorama.Style.RESET_ALL}",
                            f"sequence is: \n{colorama.Fore.YELLOW + colorama.Style.DIM}{c}{colorama.Style.RESET_ALL}",
                        ]
                    )
                    f.write(info + "\n")

        return concat_padded_tensors(results)