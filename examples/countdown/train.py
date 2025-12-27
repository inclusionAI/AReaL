import asyncio
import os
import sys
import uuid

import aiofiles
import aiofiles.os
import colorama
import torch
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from reward_score import compute_score
from transformers import PreTrainedTokenizerFast

from areal.api.cli_args import GenerationHyperparameters, GRPOConfig, load_expr_config
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import ModelRequest
from areal.api.workflow_api import RolloutWorkflow
from areal.experimental.trainer import PPOTrainer
from areal.utils import logging, stats_tracker
from areal.utils.data import concat_padded_tensors
from areal.utils.stats_logger import StatsLogger

worker_id = uuid.uuid4().hex[:4]

logger = logging.getLogger(f"CountDown @ {worker_id}")


class CountDownWorkflow(RolloutWorkflow):
    def __init__(
        self,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast | str,
        rollout_stat_scope: str = "rollout",
        dump_dir: str | None = None,
    ):
        if isinstance(tokenizer, str):
            from areal.utils.hf_utils import load_hf_tokenizer

            tokenizer = load_hf_tokenizer(tokenizer)
        self.gconfig = gconfig.new_with_stop_and_pad_token_ids(tokenizer)
        self.tokenizer = tokenizer
        self.dump_dir = dump_dir
        self.rollout_stat_scope = rollout_stat_scope
        if self.dump_dir is not None and not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir, exist_ok=True)

    async def arun_episode(self, engine: InferenceEngine, data):
        input_ids = self.tokenizer.encode(data["query"], add_special_tokens=False)

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
            reward = compute_score(
                completions_str,
                data,
            )

            # Log reward.
            stats_tracker.get(self.rollout_stat_scope).scalar(reward=reward)

            rewards.append(reward)
            res = {
                # unsqueeze to add an additional batch dimension
                "input_ids": torch.tensor(seq).unsqueeze(0),
                "loss_mask": torch.tensor(loss_mask).unsqueeze(0),
                "logprobs": torch.tensor(logprobs).unsqueeze(0),
                "versions": torch.tensor(versions).unsqueeze(0),
                "attention_mask": torch.ones(len(seq), dtype=torch.bool).unsqueeze(0),
                # reward
                "rewards": torch.tensor([float(reward)]),
            }
            results.append(res)

        # logger.info(f"numbers: {data['numbers']} target: {data['target']} rewards: {rewards}")

        # if all([r<0.2 for r in rewards]):
        #     return None

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


def get_countdown_dataset(dataset_path, rank, world_size):
    dataset = load_dataset(
        path="json",
        split="train",
        data_files=dataset_path,
    )
    return split_dataset_by_node(dataset, rank=rank, world_size=world_size)


def main(args):
    config, _ = load_expr_config(args, GRPOConfig)

    train_dataset = load_dataset(
        path="json",
        split="train",
        data_files=config.train_dataset.path,
    )

    workflow_kwargs = dict(
        gconfig=config.gconfig,
        tokenizer=config.tokenizer_path,
        dump_dir=os.path.join(
            StatsLogger.get_log_path(config.stats_logger), "generated"
        ),
    )

    with PPOTrainer(config, train_dataset=train_dataset) as trainer:
        trainer.train(
            workflow="examples.countdown.train.CountDownWorkflow",
            workflow_kwargs=workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
