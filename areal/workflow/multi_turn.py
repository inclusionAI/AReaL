import asyncio
import functools
import os
import uuid
from concurrent.futures import ProcessPoolExecutor

import colorama
import torch
from tensordict import TensorDict
from transformers import PreTrainedTokenizerFast

from areal.api.cli_args import GenerationHyperparameters
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import ModelRequest
from areal.api.workflow_api import RolloutWorkflow
from areal.utils.data import concat_padded_tensors
from realhf.base import logging

logger = logging.getLogger("Multi-Turn workflow")

REWARD_TIMEOUT_SECONDS = 15


class MultiTurnWorkflow(RolloutWorkflow):
    def __init__(
        self,
        reward_fn,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast,
        max_turns: int,
        turn_discount: float,
        dump_dir: str | None = None,
    ):
        self.reward_fn = reward_fn
        self.gconfig = gconfig
        self.tokenizer = tokenizer
        self.max_turns = max_turns
        self.turn_discount = turn_discount
        self.rw_executor = ProcessPoolExecutor(max_workers=4)
        self.dump_dir = dump_dir
        if self.dump_dir is not None and not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir, exist_ok=True)

        # Create tokens that should be amended if the answer is incorrect.
        # This method eliminates the encode-decode inconsistency issue and cancels system prompts.
        messages = [{"role": "asistant", "content": "some random message."}]
        s1 = self.tokenizer.apply_chat_template(messages, tokenize=True)
        messages += [
            {
                "role": "user",
                "content": "\nYour answer is either wrong or not parsable to the reward function. Please try to answer it again.",
            }
        ]
        s2 = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        self.multi_turn_prompt_ids = s2[len(s1) :]

    async def _run_one_episode(self, engine: InferenceEngine, data, rid):
        # Enforces `n_samples=1`
        # Placeholders for the results
        seq, logprobs, loss_mask, versions = [], [], [], []
        messages = data["messages"]
        # Convert the prompt into input_ids
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
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
            resp = await engine.agenerate(req)
            # compute reward: 1 for correct and 0 otherwise
            prompt_str = self.tokenizer.decode(input_ids)
            completions_str = self.tokenizer.decode(resp.output_tokens)
            loop = asyncio.get_event_loop()
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
        res = dict(
            input_ids=torch.tensor(seq),
            logprobs=torch.tensor(logprobs),
            loss_mask=torch.tensor(loss_mask),
            versions=torch.tensor(versions),
            rewards=torch.tensor(float(reward * discount)),
            attention_mask=torch.ones(len(seq), dtype=torch.bool),
        )
        res = {k: v.unsqueeze(0) for k, v in res.items()}
        return (
            TensorDict(res, batch_size=[1]),
            prompt_str,
            completions_str,
            reward,
            len(seq),
        )

    async def arun_episode(self, engine: InferenceEngine, data):
        rid = uuid.uuid4().hex
        tasks = [
            self._run_one_episode(engine, data, rid)
            for _ in range(self.gconfig.n_samples)
        ]
        results = await asyncio.gather(*tasks)

        if self.dump_dir is not None:
            version = engine.get_version()
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
                for i, (_, p, c, r, sl) in enumerate(results):
                    info = "\n".join(
                        [
                            f"idx: {i + 1} / {n_samples}, seqlen: {sl}, reward is {r}.",
                            f"prompt is \n{colorama.Fore.YELLOW + colorama.Style.DIM}{p}{colorama.Style.RESET_ALL}",
                            f"sequence is: \n{colorama.Fore.YELLOW + colorama.Style.DIM}{c}{colorama.Style.RESET_ALL}",
                        ]
                    )
                    f.write(info + "\n")

        data = [res[0] for res in results]
        return concat_padded_tensors(data)
