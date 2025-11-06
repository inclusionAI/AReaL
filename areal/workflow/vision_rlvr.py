import asyncio
import os
import uuid
from collections.abc import Callable
from typing import Any, cast

import aiofiles
import aiofiles.os
import colorama
import torch
from transformers import AutoProcessor, PreTrainedTokenizerFast

from areal.api.cli_args import GenerationHyperparameters
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import ModelRequest
from areal.api.workflow_api import WorkflowTaskInput
from areal.utils import logging, perf_tracer, stats_tracker
from areal.utils.data import concat_padded_tensors
from areal.utils.image import image2base64
from areal.workflow.rlvr import RLVRWorkflow

logger = logging.getLogger("RLVR workflow")


class VisionRLVRWorkflow(RLVRWorkflow):
    def __init__(
        self,
        reward_fn: Callable[..., Any],
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast,
        processor: AutoProcessor,
        enable_thinking: bool,
        rollout_stat_scope: str = "rollout",
        dump_dir: str | None = None,
    ):
        super().__init__(
            reward_fn,
            gconfig,
            tokenizer,
            enable_thinking,
            rollout_stat_scope=rollout_stat_scope,
            dump_dir=dump_dir,
        )
        self.processor = processor

    async def arun_episode(
        self,
        engine: InferenceEngine,
        task_input: WorkflowTaskInput,
    ) -> dict[str, torch.Tensor]:
        data = task_input.data
        request_id = task_input.request_id
        processor_callable = cast(Callable[..., dict[str, Any]], self.processor)
        processed_input = processor_callable(
            images=data["images"],
            text=data["messages"],
            padding=False,
            return_tensors="pt",
        )

        input_ids: list[int] = processed_input["input_ids"].tolist()[0]

        n_samples = self.gconfig.n_samples

        byte_images = image2base64(data["images"])
        req = ModelRequest(
            rid=uuid.uuid4().hex,
            input_ids=input_ids,
            image_data=byte_images,
            gconfig=self.gconfig.new(n_samples=1),
            tokenizer=self.tokenizer,
            processor=self.processor,
        )
        async with perf_tracer.atrace_request_phase(request_id, "generate"):
            resps = await asyncio.gather(
                *[engine.agenerate(req) for _ in range(n_samples)]
            )

        version = engine.get_version()
        prompt_str = self.tokenizer.decode(input_ids)
        prompt_strs = []
        completions_strs = []
        rewards = []
        seqlens = []

        # Record reward calculation timing
        perf_tracer.trace_request_event(request_id, "mark_reward_start")

        results = []
        for resp in resps:
            seq = resp.input_tokens + resp.output_tokens
            logprobs = [0.0] * resp.input_len + resp.output_logprobs
            loss_mask = [0] * resp.input_len + [1] * resp.output_len
            versions = [-1] * resp.input_len + resp.output_versions

            completions_str = self.tokenizer.decode(resp.output_tokens)
            prompt_strs.append(prompt_str)
            completions_strs.append(completions_str)
            seqlens.append(len(seq))
            reward = await self.async_reward_fn(
                prompt=prompt_str,
                completions=completions_str,
                prompt_ids=resp.input_tokens,
                completion_ids=resp.output_tokens,
                **data,
            )

            # Log reward.
            stats_tracker.get(self.rollout_stat_scope).scalar(reward=reward)

            rewards.append(reward)
            # We store multi_modal_input for each data point as a dict,
            multi_modal_input = [
                {
                    "pixel_values": processed_input["pixel_values"],
                }
            ]
            if "image_grid_thw" in processed_input:
                multi_modal_input[0]["image_grid_thw"] = processed_input[
                    "image_grid_thw"
                ]
            res = dict(
                # unsqueeze to add an additional batch dimension
                input_ids=torch.tensor(seq, dtype=torch.int32).unsqueeze(0),
                loss_mask=torch.tensor(loss_mask, dtype=torch.int32).unsqueeze(0),
                logprobs=torch.tensor(logprobs, dtype=torch.float32).unsqueeze(0),
                multi_modal_input=multi_modal_input,
                versions=torch.tensor(versions, dtype=torch.int32).unsqueeze(0),
                attention_mask=torch.ones(len(seq), dtype=torch.bool).unsqueeze(0),
                # reward
                rewards=torch.tensor([reward], dtype=torch.float32),
            )
            results.append(res)

        perf_tracer.trace_request_event(request_id, "mark_reward_end")

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
