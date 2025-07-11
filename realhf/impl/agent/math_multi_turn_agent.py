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


class MathMultiTurnAgent(Agent):
    """A multi-turn reasoning agent for mathematical tasks.

    In each turn the agent produces an answer and receives evaluation results from the environment.

    By default, we use 4 turns with a token budget=1K at each round.
    """

    def __init__(
        self,
        gconfig,
        tokenizer_path,
        answer_save_path,
        reward_scaling=1.0,
        reward_bias=0.0,
        turn_level_discount: float = 1.0,
        num_turns: int = 5,
    ):
        self.gconfig = gconfig.new(n=1)
        self.tokenizer = load_hf_tokenizer(tokenizer_path)
        self.answer_save_path = answer_save_path

        self.reward_scaling = reward_scaling
        self.reward_bias = reward_bias
        self.turn_level_discount = turn_level_discount

        self.num_turns = num_turns

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
        assert self.gconfig.n == 1

        prompt_token_ids = prompt.data["packed_prompts"].cpu().numpy().tolist()
        qid = prompt.ids[0]
        birth_time = int(datetime.now().timestamp() * 1000)

        prompt_str = self.tokenizer.batch_decode(
            [prompt_token_ids],
            clean_up_tokenization_spaces=False,
            skip_special_tokens=True,
        )[0]

        token_ids = prompt_token_ids
        all_rewards = []
        all_answers = []
        all_success = []
        x = dict(
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
                packed_input_ids=[[]],
                packed_logprobs=[[]],
                packed_prompts=[[len(prompt_token_ids)]],
                prompt_mask=[[]],
                seq_no_eos_mask=[[1 for _ in range(self.num_turns)]],
                rewards=[[1 for _ in range(self.num_turns)]],
                version_start=[[1 for _ in range(self.num_turns)]],
                version_end=[[1 for _ in range(self.num_turns)]],
                birth_time=[[1]],
            ),
            data=dict(
                packed_prompts=list(prompt_token_ids),
                packed_logprobs=[],
                packed_input_ids=[],
                seq_no_eos_mask=[],
                rewards=[],
                version_start=[],
                version_end=[],
                birth_time=torch.tensor([birth_time], dtype=torch.long),
                prompt_mask=[],
            ),
        )

        for turn in range(self.num_turns):
            await obs_queue.put((qid, token_ids, self.gconfig))

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

            # single-step env for evaluating generated solutions
            _, success, *_ = await env.step((qid, answers))
            rewards = [
                ((float(r) - 0.5) * 2 - self.reward_bias) * self.reward_scaling
                for r in success
            ]

            all_success.extend(success)
            all_answers.extend(answers)

            x["data"]["packed_input_ids"].extend(list(act.seqs[0]))
            x["data"]["packed_logprobs"].extend(list(act.logprobs[0]))
            x["data"]["seq_no_eos_mask"].append(act.no_eos[0])
            all_rewards.append(rewards[0])
            x["data"]["prompt_mask"].extend(
                [1] * act.prompt_len + [0] * (act.seqlens[0] - act.prompt_len)
            )

            x["data"]["version_start"].extend(list(act.version_start))
            x["data"]["version_end"].extend(list(act.version_end))

            x["seqlens"]["packed_input_ids"][0].append(act.seqlens[0])
            x["seqlens"]["packed_logprobs"][0].append(act.seqlens[0] - 1)
            x["seqlens"]["prompt_mask"][0].append(act.seqlens[0])

            token_ids = list(act.seqs[0])

            feedback = None
            if success[0]:
                feedback = "Congratulations! You are correct!"
            else:
                feedback = "Unfortunately your answer is wrong. Let's try again."

            feedback = "\n" + self.tokenizer.apply_chat_template(
                [dict(content=feedback, role="user")],
                add_generation_prompt=True,
                tokenize=False,
            )
            feedback = self.tokenizer(feedback)["input_ids"]
            token_ids.extend(feedback)

        self.log_rewards_to_file(
            str(qid),
            prompt_str,
            seqlens=x["seqlens"]["packed_input_ids"][0],
            answers=all_answers,
            prompt_len=len(prompt_token_ids),
            rewards=all_rewards,
            success=all_success,
            version_starts=x["data"]["version_start"],
            version_ends=x["data"]["version_end"],
        )

        for i in reversed(range(len(all_rewards) - 1)):
            all_rewards[i] = (
                all_rewards[i] + all_rewards[i + 1] * self.turn_level_discount
            )
        x["data"]["rewards"] = all_rewards

        for k in x["keys"]:
            if not isinstance(x["data"][k], torch.Tensor):
                x["data"][k] = torch.tensor(x["data"][k], dtype=x["dtypes"][k])

        x = SequenceSample(**x)

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


register_agent("math-multi-turn", MathMultiTurnAgent)
