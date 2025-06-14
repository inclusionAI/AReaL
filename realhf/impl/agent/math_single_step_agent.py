# Copyright 2025 Ant Group Inc.
# Licensed under the Apache License, Version 2.0 (the "License").

import asyncio
import json
import os
from datetime import datetime
from typing import List

import colorama
import numpy as np
import torch

from realhf.api.core.agent_api import Agent, register_agent
from realhf.api.core.data_api import SequenceSample, load_hf_tokenizer
from realhf.api.core.env_api import EnvironmentService
from realhf.api.core.model_api import BundledGenerationOutputs
from realhf.base import constants, logging

logger = logging.getLogger("Math Code Agent")


class MathSingleStepAgent(Agent):
    def __init__(
        self,
        gconfig,
        tokenizer_path,
        answer_save_path,
        success_rate_lb,
        success_rate_ub,
        reward_scaling=1.0,
        reward_bias=0.0,
    ):
        self.gconfig = gconfig
        self.tokenizer = load_hf_tokenizer(tokenizer_path)
        self.answer_save_path = answer_save_path

        self.success_rate_lb = success_rate_lb
        self.success_rate_ub = success_rate_ub

        self.reward_scaling = reward_scaling
        self.reward_bias = reward_bias

    async def collect_trajectory(
        self,
        prompt: SequenceSample,
        env: EnvironmentService,
        obs_queue: asyncio.Queue,
        act_queue: asyncio.Queue,
    ) -> List[SequenceSample]:
        # reset does nothing, just to make it like multi-step environments
        await env.reset()

        assert prompt.bs == 1
        prompt_token_ids = prompt.data["packed_prompts"].cpu().numpy().tolist()
        qid = prompt.ids[0]
        birth_time = int(datetime.now().timestamp() * 1000)
        await obs_queue.put((qid, prompt_token_ids, self.gconfig))

        act: BundledGenerationOutputs = await act_queue.get()

        seq_strs = self.tokenizer.batch_decode(
            act.seqs,
            clean_up_tokenization_spaces=False,
            skip_special_tokens=True,
        )
        prompt_str = self.tokenizer.batch_decode(
            [act.prompt_ids],
            clean_up_tokenization_spaces=False,
            skip_special_tokens=True,
        )[0]

        answers = [seq_str.split(prompt_str)[1] for seq_str in seq_strs]

        # single-step env
        _, success, *_ = await env.step((qid, answers))
        rewards = [
            ((float(r) - 0.5) * 2 - self.reward_bias) * self.reward_scaling
            for r in success
        ]

        self.log_rewards_to_file(
            str(qid),
            prompt_str,
            seqlens=[len(s) for s in act.seqs],
            answers=answers,
            prompt_len=len(prompt_token_ids),
            rewards=rewards,
            success=success,
            version_starts=act.version_start,
            version_ends=act.version_end,
        )

        r = np.mean([float(s) for s in success])
        if r < self.success_rate_lb:
            logger.info(f"Query ID {qid} reward too low: {r} < {self.success_rate_lb}.")
            return []
        if r > self.success_rate_ub:
            logger.info(
                f"Query ID {qid} reward too high: {r} > {self.success_rate_ub}."
            )
            return []

        x = SequenceSample(
            keys=[
                "packed_input_ids",
                "prompt_mask",
                "packed_logprobs",
                "seq_no_eos_mask",
                "packed_prompts",
                "version_start",
                "version_end",
                "rewards",
                "birth_time",
            ],
            ids=[qid],
            dtypes=dict(
                packed_prompts=torch.long,
                packed_input_ids=torch.long,
                prompt_mask=torch.bool,
                seq_no_eos_mask=torch.bool,
                version_start=torch.int,
                version_end=torch.int,
                packed_logprobs=torch.float32,
                rewards=torch.float32,
                birth_time=torch.long,
            ),
            trailing_shapes=dict(
                packed_input_ids=(),
                prompt_mask=(),
                seq_no_eos_mask=(),
                packed_prompts=(),
                version_end=(),
                version_start=(),
                packed_logprobs=(),
                rewards=(),
                birth_time=(),
            ),
            seqlens=dict(
                packed_input_ids=[act.seqlens],
                packed_logprobs=[[s - 1 for s in act.seqlens]],
                packed_prompts=[[act.prompt_len]],
                prompt_mask=[act.seqlens],
                seq_no_eos_mask=[[1 for _ in range(self.gconfig.n)]],
                rewards=[[1 for _ in range(self.gconfig.n)]],
                version_start=[[1 for _ in range(self.gconfig.n)]],
                version_end=[[1 for _ in range(self.gconfig.n)]],
                birth_time=[[1]],
            ),
            data=dict(
                packed_prompts=torch.tensor(act.prompt_ids, dtype=torch.long),
                packed_logprobs=torch.tensor(
                    sum(act.logprobs, []), dtype=torch.float32
                ),
                packed_input_ids=torch.tensor(sum(act.seqs, []), dtype=torch.long),
                seq_no_eos_mask=torch.tensor(act.no_eos, dtype=torch.bool),
                rewards=torch.tensor(rewards, dtype=torch.float32),
                version_start=torch.tensor(act.version_start, dtype=torch.int),
                version_end=torch.tensor(act.version_end, dtype=torch.int),
                birth_time=torch.tensor([birth_time], dtype=torch.long),
                prompt_mask=torch.tensor(
                    sum(
                        [
                            [1] * act.prompt_len + [0] * (seqlen - act.prompt_len)
                            for seqlen in act.seqlens
                        ],
                        [],
                    ),
                    dtype=torch.bool,
                ),
            ),
        )
        if "task_ids" in prompt.keys:
            y = SequenceSample(
                keys=["task_ids"],
                ids=[qid],
                dtypes=dict(task_ids=torch.long),
                trailing_shapes=dict(task_ids=()),
                seqlens=dict(task_ids=[[1]]),
                data=dict(task_ids=prompt.data["task_ids"]),
            )
            x.update_(y)

        return [x]

    def log_rewards_to_file(
        self,
        qid: str,
        prompt: str,
        prompt_len: int,
        answers: List[str],
        seqlens: List[int],
        rewards: List[float],
        success: List[bool],
        version_starts: List[int],
        version_ends: List[int],
    ):
        group_size = len(answers)

        for group_idx in range(group_size):
            # NOTE: we can ensure that only one process is logging this query id
            gen_file_path = os.path.join(
                self.answer_save_path,
                str(version_starts[group_idx]),
                f"{qid}.txt",
            )
            os.makedirs(os.path.dirname(gen_file_path), exist_ok=True)

            version_start = version_starts[group_idx]
            version_end = version_ends[group_idx]
            reward = rewards[group_idx]
            answer = answers[group_idx]
            seqlen = seqlens[group_idx]
            with open(gen_file_path, "a") as _f:
                info = "\n".join(
                    [
                        f"idx: {group_idx + 1} / {group_size}, seqlen: {seqlen}, "
                        f"head version: {version_start}, tail version: {version_end}.",
                        f"reward is {reward}, prompt is {colorama.Fore.YELLOW + colorama.Style.DIM}{prompt}{colorama.Style.RESET_ALL}",
                        f"sequence is: {colorama.Fore.YELLOW + colorama.Style.DIM}{answer}{colorama.Style.RESET_ALL}.",
                    ]
                )
                _f.write(info + "\n")

            train_pass_monitor_file_path = os.path.join(
                self.answer_save_path,
                str(version_starts[group_idx]),
                f"{qid}.jsonl",
            )
            os.makedirs(os.path.dirname(train_pass_monitor_file_path), exist_ok=True)

            with open(train_pass_monitor_file_path, "a") as monitor_file:
                monitor_file.write(
                    json.dumps(
                        {
                            "version_start": int(version_start),
                            "version_end": int(version_end),
                            "success": bool(success),
                            "prompt_len": prompt_len,
                            "answer_len": seqlen - prompt_len,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )


register_agent("math-single-step", MathSingleStepAgent)
