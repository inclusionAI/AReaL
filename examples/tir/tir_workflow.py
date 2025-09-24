import ast
import asyncio
import copy
import re
import uuid
from typing import Any, Dict, Optional, Tuple

import torch
from tensordict import TensorDict
from transformers import PreTrainedTokenizerFast

from areal.api.cli_args import (
    GenerationHyperparameters,
    GRPOConfig,
    dataclass,
    field,
)
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import ModelRequest, ModelResponse
from areal.api.reward_api import AsyncRewardWrapper
from areal.api.workflow_api import RolloutWorkflow
from areal.utils import logging, stats_tracker
from areal.utils.data import concat_padded_tensors

from .prompts import ANSWER, SYSTEM_PROMPT, TORL_PROMPT
from .tool_manager import ToolCallStatus, ToolManager

logger = logging.getLogger("TIR workflow")


@dataclass
class TIRConfig:
    max_turns: int = field(default=2)
    max_length: int = field(default=3000)
    tool_timeout: float = field(default=30)
    enable_tools: str = field(default="python")
    is_chat_model: bool = field(default=False)


@dataclass
class TIRGRPOConfig(GRPOConfig):
    tir: TIRConfig = field(default_factory=TIRConfig)


class TIRWorkflow(RolloutWorkflow):
    """Tool-Integrated Reasoning Workflow for multi-turn tool calling."""

    def __init__(
        self,
        reward_fn,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast,
        tir_config: TIRConfig,
        enable_thinking: bool = False,
        rollout_stat_scope: str = "rollout",
        dump_dir: Optional[str] = None,
    ):
        super().__init__()
        self.reward_fn = reward_fn
        self.gconfig = gconfig
        self.tokenizer = tokenizer
        self.tool_manager = ToolManager(
            tir_config.tool_timeout, tir_config.enable_tools, debug_mode=False
        )
        self.is_chat_model = tir_config.is_chat_model
        self.max_turns = tir_config.max_turns
        self.max_length = tir_config.max_length
        self.enable_thinking = enable_thinking
        self.rollout_stat_scope = rollout_stat_scope
        self.dump_dir = dump_dir
        self.async_reward_fn = AsyncRewardWrapper(reward_fn)

        self.start_markers = self.tool_manager.get_all_start_markers()
        self.end_markers = self.tool_manager.get_all_end_markers()

        logger.info(
            f"start markers: {self.start_markers}, end markers {self.end_markers}"
        )

    @staticmethod
    def _process_tool_result(tool_result) -> str:
        try:
            run_result, run_status = ast.literal_eval(tool_result)
        except Exception:
            run_result = tool_result
            run_status = "Done"
        res = run_result if run_status == "Done" else f"Error: {run_status}"
        return f"\n```output\n{res}\n```\n"

    async def arun_episode(
        self, engine: InferenceEngine, data: Dict[str, Any]
    ) -> TensorDict:
        """Run a complete TIR inference episode.
        :param engine: The inference engine.
        :param data: The input data.
        :return: The output tensor dict.
        """
        # Initialize conversation history
        messages = data["messages"]

        # Add system prompt with tool usage instructions
        system_prompt = SYSTEM_PROMPT.format(
            tool_descriptions=self.tool_manager.get_tool_descriptions_prompt()
        )
        if messages[0]["role"] == "system":
            messages[0]["content"] = system_prompt
        else:
            messages.insert(0, {"role": "system", "content": system_prompt})

        # Prepare input
        if self.is_chat_model:
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        else:
            input_ids = self.tokenizer.encode(
                TORL_PROMPT.format(prompt=messages[1]["content"]),
                add_special_tokens=False,
            )

        n_samples = self.gconfig.n_samples
        # Append conversation history
        results = await asyncio.gather(
            *[
                self._multi_round_response(engine, input_ids, data)
                for _ in range(n_samples)
            ]
        )

        return concat_padded_tensors(results)

    async def _multi_round_response(self, engine, prompt_ids, data):
        prompt_str = self.tokenizer.decode(prompt_ids)
        completions_str = ""
        has_tool = False
        tool_call_count = 0
        tool_success_count = 0
        stop_reason = None
        max_len = self.max_length
        turn = 0
        # State flag for each episode: whether waiting for tool start marker
        waiting_for_tool_start = True
        tool_start_idx = -1

        # initialize seq, logprobs, loss_mask, versions
        context_ids = copy.deepcopy(prompt_ids)
        seq = copy.deepcopy(prompt_ids)
        logprobs = [0.0] * len(context_ids)
        loss_mask = [0] * len(context_ids)
        versions = [-1] * len(context_ids)
        output_ids = []

        while turn <= self.max_turns:
            if len(context_ids) >= max_len:
                break

            # Generate response
            resp, stop_reason = await self._generate_response(
                engine, context_ids, max_len, waiting_for_tool_start
            )

            context_ids.extend(resp.output_tokens)
            seq.extend(resp.output_tokens)
            logprobs.extend(resp.output_logprobs)
            loss_mask.extend([1] * resp.output_len)
            versions.extend(resp.output_versions)

            cur_completions_str = self.tokenizer.decode(resp.output_tokens)
            completions_str += cur_completions_str
            output_ids.extend(resp.output_tokens)

            # End token, truncate
            if context_ids[-1] in [
                self.tokenizer.pad_token_id,
                self.tokenizer.eos_token_id,
            ]:
                break

            # If answer appears, truncate immediately
            if re.search(ANSWER, cur_completions_str):
                break

            # State transition logic: detect if tool start marker is encountered
            if waiting_for_tool_start and stop_reason == "stop":
                # Check if tool start marker is detected
                tool_start_marker = self._detect_tool_start_marker(cur_completions_str)
                if tool_start_marker:
                    waiting_for_tool_start = False
                    tool_start_idx = len(completions_str) - len(tool_start_marker)
                    # Continue generating until tool end marker
                    continue

            # If tool call is detected, execute tool call
            if (
                not waiting_for_tool_start
                and stop_reason == "stop"
                and tool_start_idx != -1
            ):
                tool_results, tool_status = self._execute_tools(
                    completions_str[tool_start_idx:]
                )
                if tool_status == ToolCallStatus.NOT_FOUND:
                    # No match found, continue generating until next tool end marker
                    continue
                turn += 1
                has_tool = True
                tool_call_count += 1  # Increment tool call count
                if (
                    tool_status == ToolCallStatus.SUCCESS
                    and "Error" not in tool_results
                ):
                    tool_success_count += 1
                tool_results = self._process_tool_result(tool_results)
                # Append tool response token IDs
                tool_rsp_token_ids = self.tokenizer.encode(
                    tool_results, add_special_tokens=False
                )
                # Concatenate to seq
                # Build tool mask
                context_ids.extend(tool_rsp_token_ids)
                seq.extend(tool_rsp_token_ids)
                logprobs.extend([0.0] * len(tool_rsp_token_ids))
                loss_mask.extend([0] * len(tool_rsp_token_ids))
                versions.extend([-1] * len(tool_rsp_token_ids))
                completions_str += tool_results

                # After tool execution completes, reset state flag to prepare for next tool call detection
                waiting_for_tool_start = True

        reward = await self.async_reward_fn(
            prompt_str,
            completions_str,
            prompt_ids,
            output_ids,
            tool_using=has_tool,
            tool_status=tool_call_count,
            **data,
        )

        # Record tool call count to stats_tracker
        stats_tracker.get(self.rollout_stat_scope).scalar(
            tool_call_count=tool_call_count, tool_success_count=tool_success_count
        )

        res = dict(
            input_ids=torch.tensor(seq[:max_len]).unsqueeze(0),
            logprobs=torch.tensor(logprobs[:max_len]).unsqueeze(0),
            loss_mask=torch.tensor(loss_mask[:max_len]).unsqueeze(0),
            versions=torch.tensor(versions[:max_len]).unsqueeze(0),
            attention_mask=torch.ones(len(seq[:max_len]), dtype=torch.bool).unsqueeze(
                0
            ),
            rewards=torch.tensor([float(reward)]),
        )
        return TensorDict(res, batch_size=[1])

    async def _generate_response(
        self,
        engine: InferenceEngine,
        input_ids: list[int],
        max_len: int,
        waiting_for_tool_start: bool,
    ) -> Tuple[ModelResponse, str]:
        """Generate response with tool call detection support"""

        # Select stop condition based on state flag
        if waiting_for_tool_start:
            # When waiting for tool start marker, use start_markers to stop
            stop_markers = [marker for marker in self.start_markers]
        else:
            # Tool start detected, use end_markers to stop
            stop_markers = [marker for marker in self.end_markers]

        # Set generation config, add tool call stop tokens
        gconfig = self.gconfig.new(
            n_samples=1, stop=[marker for marker in stop_markers]
        )

        # Generate response
        req = ModelRequest(
            rid=uuid.uuid4().hex,
            input_ids=input_ids,
            gconfig=gconfig,
            tokenizer=self.tokenizer,
        )

        resp = await engine.agenerate(req)
        return resp, resp.stop_reason

    def _detect_tool_start_marker(self, text: str) -> Optional[str]:
        """Detect if text ends with tool start marker"""
        for marker in self.start_markers:
            if text.endswith(marker):
                return marker
        return None

    def _execute_tools(self, response: str) -> Tuple[str, ToolCallStatus]:
        """Execute tool call"""
        # Call execute_tool_call
        tool_results = self.tool_manager.execute_tool_call(response)
        return tool_results
