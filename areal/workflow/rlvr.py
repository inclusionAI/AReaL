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
from areal.api.io_struct import ModelRequest, ModelResponse
from areal.api.reward_api import AsyncRewardWrapper
from areal.api.workflow_api import RolloutWorkflow
from areal.utils import logging, perf_tracer, stats_tracker
from areal.utils.data import concat_padded_tensors
from areal.utils.perf_tracer import atrace_session_phase, trace_perf, trace_session

logger = logging.getLogger("RLVR workflow")


def default_get_input_ids_fn(
    data: Any,
    tokenizer: PreTrainedTokenizerFast,
    enable_thinking: bool,
) -> list[int]:
    input_ids = tokenizer.apply_chat_template(
        data,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    return list(input_ids)


def default_data_extract_prompt_fn(data: dict[str, Any]) -> Any:
    return data["messages"]


class RLVRWorkflow(RolloutWorkflow):
    """Single-turn reward learning workflow supporting optional thinking tokens."""

    def __init__(
        self,
        reward_fn: Callable[..., Any],
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast,
        enable_thinking: bool = False,
        rollout_stat_scope: str = "rollout",
        dump_dir: str | None = None,
        get_input_ids_fn: Callable[
            [Any, PreTrainedTokenizerFast, bool], list[int]
        ] = default_get_input_ids_fn,
        data_extract_prompt_fn: Callable[
            [dict[str, Any]], Any
        ] = default_data_extract_prompt_fn,
    ):
        self.reward_fn = reward_fn
        self.gconfig = gconfig
        self.tokenizer = tokenizer
        self.enable_thinking = enable_thinking
        self.dump_dir = dump_dir
        self.rollout_stat_scope = rollout_stat_scope
        self.async_reward_fn = AsyncRewardWrapper(reward_fn)
        self.get_input_ids_fn = get_input_ids_fn
        self.data_extract_prompt_fn = data_extract_prompt_fn
        if self.dump_dir is not None and not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir, exist_ok=True)

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
                prompt_str,
                completions_str,
                resp.input_tokens,
                resp.output_tokens,
                **task_data,
            )
            stats_tracker.get(self.rollout_stat_scope).scalar(reward=reward)
            rewards.append(reward)

        return rewards, completions_strs

    @trace_perf("rlvr_workflow.arun_episode", category="compute")
    async def arun_episode(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        input_ids = self.get_input_ids_fn(
            self.data_extract_prompt_fn(data),
            self.tokenizer,
            self.enable_thinking,
        )
        n_samples = self.gconfig.n_samples
        req = ModelRequest(
            rid=uuid.uuid4().hex,
            input_ids=input_ids,
            gconfig=self.gconfig.new(n_samples=1),
            tokenizer=self.tokenizer,
        )

        # Generate responses
        session_id = perf_tracer.get_session_id()
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

            res = {
                "input_ids": torch.tensor(seq, dtype=torch.int32),
                "loss_mask": torch.tensor(loss_mask, dtype=torch.int32),
                "logprobs": torch.tensor(logprobs, dtype=torch.float32),
                "versions": torch.tensor(versions, dtype=torch.int32),
                "attention_mask": torch.ones(len(seq), dtype=torch.bool),
                "rewards": torch.tensor(reward, dtype=torch.float32),
            }
            res = {k: v.unsqueeze(0) for k, v in res.items()}
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
