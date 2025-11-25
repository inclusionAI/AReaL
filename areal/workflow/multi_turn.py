import asyncio
import os
import uuid
from collections.abc import Callable
from typing import Any

import aiofiles
import aiofiles.os
import colorama
import torch
from transformers import PreTrainedTokenizerFast

from areal.api.cli_args import GenerationHyperparameters
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import ModelRequest
from areal.api.reward_api import AsyncRewardWrapper
from areal.api.workflow_api import RolloutWorkflow
from areal.utils import logging, stats_tracker
from areal.utils.data import concat_padded_tensors

logger = logging.getLogger("Multi-Turn workflow")


class MultiTurnWorkflow(RolloutWorkflow):
    """Multi-attempt workflow that retries generation until the reward is positive."""

    def __init__(
        self,
        reward_fn: Callable[..., Any],
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast,
        max_turns: int,
        turn_discount: float,
        rollout_stat_scope: str = "rollout",
        dump_dir: str | None = None,
    ):
        if max_turns <= 0:
            raise ValueError("max_turns must be positive")
        if not (0.0 < turn_discount <= 1.0):
            raise ValueError("turn_discount must be in (0, 1].")

        self.reward_fn = reward_fn
        self.gconfig = gconfig.new_with_stop_and_pad_token_ids(tokenizer)
        self.tokenizer = tokenizer
        self.max_turns = max_turns
        self.turn_discount = turn_discount
        self.rollout_stat_scope = rollout_stat_scope
        self.async_reward_fn = AsyncRewardWrapper(reward_fn)
        self.dump_dir = dump_dir
        if self.dump_dir is not None and not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir, exist_ok=True)

        # Create tokens that should be amended if the answer is incorrect.
        # This method eliminates the encode-decode inconsistency issue and cancels system prompts.
        messages = [{"role": "assistant", "content": "some random message."}]
        s1 = list(self.tokenizer.apply_chat_template(messages, tokenize=True))
        messages += [
            {
                "role": "user",
                "content": "Your answer is either wrong or not parsable to the reward function. You may misunderstand the original question. "
                "Please carefully read the original question, check the previous errors, and try to answer it again.",
            }
        ]
        s2 = list(
            self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
        )
        self.multi_turn_prompt_ids = s2[len(s1) :]

    async def _run_one_episode(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> tuple[dict[str, torch.Tensor], str, str, float, int]:
        # Enforces `n_samples=1`
        # Placeholders for the results
        seq, logprobs, loss_mask, versions = [], [], [], []
        messages = data["messages"]
        # Convert the prompt into input_ids
        input_ids: list[int] = list(
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
        )
        # Run multi-turn rollout until correct
        t = 0
        reward = 0.0
        discount = 1.0
        prompt_str = ""
        completions_str = ""
        while reward == 0.0 and t < self.max_turns:
            # Send generate request to get the response.
            req = ModelRequest(
                rid=uuid.uuid4().hex,
                input_ids=input_ids,
                gconfig=self.gconfig.new(n_samples=1),
                tokenizer=self.tokenizer,
            )
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
            if reward == 0.0 and t < self.max_turns:
                input_ids = input_ids + resp.output_tokens
                if (
                    resp.output_tokens
                    and resp.output_tokens[-1] != self.tokenizer.eos_token_id
                ):
                    input_ids += [self.tokenizer.eos_token_id]
                input_ids += self.multi_turn_prompt_ids
                discount *= self.turn_discount

        reward = float(reward * discount)

        # Log reward.
        stats_tracker.get(self.rollout_stat_scope).scalar(reward=reward, num_turns=t)

        res = dict(
            input_ids=torch.tensor(seq, dtype=torch.int32),
            logprobs=torch.tensor(logprobs, dtype=torch.float32),
            loss_mask=torch.tensor(loss_mask, dtype=torch.int32),
            versions=torch.tensor(versions, dtype=torch.int32),
            rewards=torch.tensor(reward, dtype=torch.float32),
            attention_mask=torch.ones(len(seq), dtype=torch.bool),
        )
        res = {k: v.unsqueeze(0) for k, v in res.items()}
        return (
            res,
            prompt_str,
            completions_str,
            reward,
            len(seq),
        )

    async def arun_episode(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        tasks = [
            self._run_one_episode(engine, data) for _ in range(self.gconfig.n_samples)
        ]
        results = await asyncio.gather(*tasks)

        if self.dump_dir is not None:
            version = engine.get_version()
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
                for i, (_, p, c, r, sl) in enumerate(results):
                    info = "\n".join(
                        [
                            f"idx: {i + 1} / {n_samples}, seqlen: {sl}, reward is {r}.",
                            f"prompt is \n{colorama.Fore.YELLOW + colorama.Style.DIM}{p}{colorama.Style.RESET_ALL}",
                            f"sequence is: \n{colorama.Fore.YELLOW + colorama.Style.DIM}{c}{colorama.Style.RESET_ALL}",
                        ]
                    )
                    await f.write(info + "\n")

        data = [res[0] for res in results]
        return concat_padded_tensors(data)
