import asyncio
import os
import uuid
from copy import deepcopy

import aiofiles
import aiofiles.os
import colorama
from transformers import PreTrainedTokenizerFast

from areal.api.cli_args import GenerationHyperparameters
from areal.api.engine_api import InferenceEngine
from areal.api.reward_api import AsyncRewardWrapper
from areal.api.workflow_api import RolloutWorkflow
from areal.experimental.openai import ArealOpenAI
from areal.utils import logging, stats_tracker

logger = logging.getLogger("Multi-Turn workflow")


class MultiTurnWorkflow(RolloutWorkflow):
    def __init__(
        self,
        reward_fn,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast,
        max_turns: int,
        turn_discount: float,
        rollout_stat_scope: str = "rollout",
        dump_dir: str | None = None,
    ):
        self.reward_fn = reward_fn
        self.gconfig = gconfig
        self.tokenizer = tokenizer
        self.max_turns = max_turns
        self.turn_discount = turn_discount
        self.async_reward_fn = AsyncRewardWrapper(reward_fn)
        self.dump_dir = dump_dir
        self.rollout_stat_scope = rollout_stat_scope
        if self.dump_dir is not None and not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir, exist_ok=True)

        self.reflection_msg = [
            {
                "role": "user",
                "content": "Your answer is either wrong or not parsable to the reward function. You may misunderstand the original question. "
                "Please carefully read the original question, check the preivous errors, and try to answer it again.",
            }
        ]

    async def _run_one_episode(self, engine: InferenceEngine, data, rid):
        client = ArealOpenAI(engine=engine, tokenizer=self.tokenizer)
        messages = deepcopy(data["messages"])
        # Run multi-turn rollout until correct
        t = reward = 0
        discount = 1
        while reward == 0 and t < self.max_turns:
            # Send generate request to get the response.
            _comp = await client.chat.completions.create(
                messages=messages,
                frequency_penalty=self.gconfig.frequency_penalty,
                max_completion_tokens=self.gconfig.max_new_tokens,
                stop=self.gconfig.stop,
                store=True,
                temperature=self.gconfig.temperature,
                top_p=self.gconfig.top_p,
            )
            # _comp is an openai ChatCompletion object
            # but we also need to fetch the saved token IDs
            comp = client.get_completions(_comp.id)
            reward = await self.async_reward_fn(
                self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                ),
                _comp.choices[0].message.content,
                comp.response.input_tokens,
                comp.response.output_tokens,
                **data,
            )
            # Increase counter
            t += 1
            # Amend a prompt if the previous answer is incorrect
            if reward == 0 and t < self.max_turns:
                messages += [
                    {
                        "role": "assistant",
                        "content": _comp.choices[0].message.content,
                    }
                ]
                messages += self.reflection_msg
                discount *= self.turn_discount

        reward = float(reward * discount)

        # Log reward.
        stats_tracker.get(self.rollout_stat_scope).scalar(reward=reward, num_turns=t)

        client.set_reward(_comp.id, reward)
        return client.export_completions(turn_discount=0.0), comp

    async def arun_episode(self, engine: InferenceEngine, data):
        rid = uuid.uuid4().hex
        tasks = [
            self._run_one_episode(engine, data, rid)
            for _ in range(self.gconfig.n_samples)
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
                for i, (_, comp) in enumerate(results):
                    sl = comp.response.input_len + comp.response.output_len
                    r = comp.reward
                    p = comp.messages
                    c = comp.completion.choices[0].message.content
                    info = "\n".join(
                        [
                            f"idx: {i + 1} / {n_samples}, seqlen: {sl}, reward is {r}.",
                            f"prompt is \n{colorama.Fore.YELLOW + colorama.Style.DIM}{p}{colorama.Style.RESET_ALL}",
                            f"sequence is: \n{colorama.Fore.YELLOW + colorama.Style.DIM}{c}{colorama.Style.RESET_ALL}",
                        ]
                    )
                    await f.write(info + "\n")

        data = [res[0] for res in results]
        ret = {}
        for d in data:
            ret.update(d)
        return ret
