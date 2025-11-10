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
from areal.api.io_struct import ModelRequest, ModelResponse
from areal.utils import logging, perf_tracer, stats_tracker
from areal.utils.data import concat_padded_tensors
from areal.utils.image import image2base64
from areal.utils.perf_tracer import atrace_session_phase, trace_perf, trace_session
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

    @trace_session("reward")
    async def _compute_rewards(
        self,
        resps: list[ModelResponse],
        prompt_str: str,
        task_data: dict[str, Any],
    ) -> tuple[list[float], list[str]]:
        """Compute rewards for all responses and decode completion strings.

        This method iterates through all model responses, decodes the output tokens
        into strings, and computes rewards for each completion. Each reward is logged
        to the stats tracker for monitoring.

        Parameters
        ----------
        resps : list[ModelResponse]
            List of ModelResponse objects from the inference engine.
        prompt_str : str
            The decoded prompt string for context.
        task_data : dict[str, Any]
            Original task data containing additional context (e.g., ground truth answer).

        Returns
        -------
        tuple[list[float], list[str]]
            A tuple containing:
            - rewards: List of computed reward values for each response.
            - completions_strs: List of decoded completion strings corresponding to each response.
        """
        rewards = []
        completions_strs = []

        for resp in resps:
            completions_str = self.tokenizer.decode(resp.output_tokens)
            completions_strs.append(completions_str)

            reward = await self.async_reward_fn(
                prompt=prompt_str,
                completions=completions_str,
                prompt_ids=resp.input_tokens,
                completion_ids=resp.output_tokens,
                **task_data,
            )
            stats_tracker.get(self.rollout_stat_scope).scalar(reward=reward)
            rewards.append(reward)

        return rewards, completions_strs

    @trace_perf("rlvr_workflow.arun_episode", category="compute")
    async def arun_episode(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        session_id = perf_tracer.get_session_id()
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

        # Generate responses
        async with atrace_session_phase(session_id, "generate"):
            resps = await asyncio.gather(
                *[engine.agenerate(req) for _ in range(n_samples)]
            )

        version = engine.get_version()
        prompt_str = self.tokenizer.decode(input_ids)
        prompt_strs = [prompt_str] * n_samples

        # Compute rewards
        rewards, completions_strs = await self._compute_rewards(resps, prompt_str, data)

        # Build result tensors
        results = []
        for resp, reward in zip(resps, rewards):
            seq = resp.input_tokens + resp.output_tokens
            logprobs = [0.0] * resp.input_len + resp.output_logprobs
            loss_mask = [0] * resp.input_len + [1] * resp.output_len
            versions = [-1] * resp.input_len + resp.output_versions

            # Build multi-modal input for each data point
            multi_modal_input = [
                {
                    "pixel_values": processed_input["pixel_values"],
                }
            ]
            if "image_grid_thw" in processed_input:
                multi_modal_input[0]["image_grid_thw"] = processed_input[
                    "image_grid_thw"
                ]

            res = {
                "input_ids": torch.tensor(seq, dtype=torch.int32).unsqueeze(0),
                "loss_mask": torch.tensor(loss_mask, dtype=torch.int32).unsqueeze(0),
                "logprobs": torch.tensor(logprobs, dtype=torch.float32).unsqueeze(0),
                "multi_modal_input": multi_modal_input,
                "versions": torch.tensor(versions, dtype=torch.int32).unsqueeze(0),
                "attention_mask": torch.ones(len(seq), dtype=torch.bool).unsqueeze(0),
                "rewards": torch.tensor([reward], dtype=torch.float32),
            }
            results.append(res)

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
            seqlens = [
                len(resp.input_tokens) + len(resp.output_tokens) for resp in resps
            ]
            async with aiofiles.open(file_path, "a") as f:
                for i, (prompt, completion, reward, seqlen) in enumerate(
                    zip(prompt_strs, completions_strs, rewards, seqlens)
                ):
                    info = "\n".join(
                        [
                            f"idx: {i + 1} / {n_samples}, seqlen: {seqlen}, reward is {reward}.",
                            f"prompt is \n{colorama.Fore.YELLOW + colorama.Style.DIM}{prompt}{colorama.Style.RESET_ALL}",
                            f"sequence is: \n{colorama.Fore.YELLOW + colorama.Style.DIM}{completion}{colorama.Style.RESET_ALL}",
                        ]
                    )
                    await f.write(info + "\n")

        return concat_padded_tensors(results)
