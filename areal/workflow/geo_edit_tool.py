from __future__ import annotations

import json
import uuid
import re
from collections.abc import Callable
from typing import Any, Dict, List, Optional, Tuple, cast

import torch
from PIL import Image
from transformers import AutoProcessor, PreTrainedTokenizerFast

from areal.api.cli_args import GenerationHyperparameters
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import ModelRequest, ModelResponse
from areal.api.reward_api import AsyncRewardWrapper
from areal.api.workflow_api import RolloutWorkflow
from areal.utils import logging, stats_tracker
from areal.utils.data import concat_padded_tensors
from areal.utils.dynamic_import import import_from_string
from areal.utils.image import image2base64
from areal.utils.perf_tracer import atrace_session_phase, session_context, trace_session

from geo_edit.constants import (
    MAX_TOOL_CALLS,
    TOOL_EXECUTION_FAILURE_PROMPT,
    TOOL_EXECUTION_SUCCESS_PROMPT,
    VLLM_SYSTEM_PROMPT,
)
from geo_edit.environment.action import TOOL_FUNCTIONS, TOOL_FUNCTIONS_DECLARE
from geo_edit.environment.task.vision_qa_task import ToolCall

logger = logging.getLogger("GeoEdit Tool Workflow")


class GeoEditToolWorkflow(RolloutWorkflow):
    """Multi-turn tool-calling workflow for geo_edit VLM tasks."""

    _ACTION_PATTERN = re.compile(r"<action>(.*?)</action>", re.DOTALL | re.IGNORECASE)
    _ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)

    def __init__(
        self,
        reward_fn: Callable[..., Any] | str,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast,
        processor: AutoProcessor,
        system_prompt: Optional[str] = None,
        tool_functions: Optional[Dict[str, Any]] = None,
        enable_thinking: bool = True,
        max_tool_calls: int = MAX_TOOL_CALLS,
        rollout_stat_scope: str = "rollout",
        data_extract_prompt_fn: Optional[Callable[[Dict[str, Any]], str]] = None,
        data_extract_answer_fn: Optional[Callable[[Dict[str, Any]], str]] = None,
        data_extract_image_fn: Optional[Callable[[Dict[str, Any]], Optional[Image.Image]]] = None,
    ) -> None:
        self.reward_fn = reward_fn
        self.gconfig = gconfig.new_with_stop_and_pad_token_ids(tokenizer)
        self.tokenizer = tokenizer
        self.processor = processor
        self.enable_thinking = enable_thinking
        self.max_tool_calls = max_tool_calls
        self.rollout_stat_scope = rollout_stat_scope
        self.tool_functions = tool_functions or TOOL_FUNCTIONS
        self.system_prompt = self._build_system_prompt(system_prompt)

        if not isinstance(reward_fn, str):
            self.async_reward_fn = AsyncRewardWrapper(reward_fn)
        self.data_extract_prompt_fn = (
            data_extract_prompt_fn or self._default_extract_prompt_fn
        )
        self.data_extract_answer_fn = (
            data_extract_answer_fn or self._default_extract_answer_fn
        )
        self.data_extract_image_fn = (
            data_extract_image_fn or self._default_extract_image_fn
        )
        self._last_processed_input: Optional[Dict[str, Any]] = None

    def _build_system_prompt(self, override: Optional[str]) -> str:
        base_prompt = override if override is not None else VLLM_SYSTEM_PROMPT
        tool_prompt = self._build_tool_prompt()
        if base_prompt:
            base_prompt = base_prompt.strip()
        if tool_prompt:
            return f"{base_prompt}\n\n{tool_prompt}" if base_prompt else tool_prompt
        return base_prompt

    def _build_tool_prompt(self) -> str:
        if not self.tool_functions:
            return ""
        declarations = {d.get("name"): d for d in TOOL_FUNCTIONS_DECLARE}
        lines = ["Tool definitions:"]
        for tool_name in sorted(self.tool_functions.keys()):
            tool = declarations.get(tool_name, {"name": tool_name})
            description = " ".join(tool.get("description", "").split())
            if not description:
                description = "No description provided."
            lines.append(f"- {tool_name}: {description}")
            parameters = tool.get("parameters")
            if parameters:
                lines.append(
                    f"  parameters: {json.dumps(parameters, ensure_ascii=True)}"
                )
        lines.append(
            "Tool results are returned as <tool_result>{\"name\": \"ToolName\", "
            "\"output\": {...}} or <tool_result>{\"name\": \"ToolName\", "
            "\"error\": \"...\"}</tool_result>."
        )
        lines.append(
            "Call tools with JSON: <action>{\"name\":\"ToolName\",\"arguments\":{...}}</action>."
        )
        lines.append(
            "If a tool returns an image, the output includes image_ref for a new "
            "Observation and the image appears as the next user message."
        )
        return "\n".join(lines)

    def _default_extract_prompt_fn(self, data: Dict[str, Any]) -> str:
        for key in ("prompt", "question", "task_prompt"):
            if key in data and isinstance(data[key], str):
                return data[key]
        return ""

    def _default_extract_answer_fn(self, data: Dict[str, Any]) -> str:
        for key in ("answer", "ground_truth", "task_answer"):
            if key in data and isinstance(data[key], str):
                return data[key]
        return ""

    def _default_extract_image_fn(
        self, data: Dict[str, Any]
    ) -> Optional[Image.Image]:
        image = None
        if "images" in data and isinstance(data["images"], list) and data["images"]:
            image = data["images"][0]
        elif "image" in data:
            image = data["image"]
        elif "image_path" in data and isinstance(data["image_path"], str):
            image = Image.open(data["image_path"])
        if image is None:
            return None
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        return None

    def _resolve_image_token(self) -> str:
        processor = self.processor
        image_processor_type = processor.image_processor.image_processor_type.lower()
        if "qwen" in image_processor_type:
            return "<|vision_start|><|image_pad|><|vision_end|>"
        if "gemma3" in image_processor_type:
            return processor.boi_token
        return processor.image_token if processor is not None else "<image>"

    def _apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

    def _parse_action_body(self, action_body: str) -> Tuple[str, Dict[str, Any]]:
        payload = json.loads(action_body)
        tool_name = payload["name"]
        arguments = payload["arguments"]
        return tool_name, arguments

    def _parse_tool_calls_from_text(
        self, text: str, step: int
    ) -> Tuple[List[ToolCall], List[Dict[str, Any]]]:
        tool_calls: List[ToolCall] = []
        tool_call_records: List[Dict[str, Any]] = []
        if not text:
            return tool_calls, tool_call_records
        matches = list(self._ACTION_PATTERN.finditer(text))
        for index, match in enumerate(matches, start=1):
            action_body = match.group(1).strip()
            try:
                tool_name, arguments = self._parse_action_body(action_body)
            except Exception:
                continue
            call_id = f"call_{step}_{index}"
            tool_calls.append(
                ToolCall(name=tool_name, args=arguments, call_id=call_id)
            )
            tool_call_records.append(
                {"id": call_id, "name": tool_name, "args": arguments}
            )
        return tool_calls, tool_call_records

    def _parse_response(
        self, text: str, step: int
    ) -> Tuple[List[ToolCall], str]:
        tool_calls, _ = self._parse_tool_calls_from_text(text, step)
        answer_parts = [
            match.group(1).strip()
            for match in self._ANSWER_PATTERN.finditer(text)
        ]
        answer_text = "\n".join(part for part in answer_parts if part)
        return tool_calls, answer_text

    def _append_tool_result(
        self, messages: List[Dict[str, str]], payload: Dict[str, Any]
    ) -> None:
        text = f"<tool_result>{json.dumps(payload, ensure_ascii=True)}</tool_result>"
        messages.append({"role": "user", "content": text})

    def _append_tool_error(
        self, messages: List[Dict[str, str]], tool_call: ToolCall, error_msg: str
    ) -> None:
        payload = {"name": tool_call.name, "error": error_msg}
        self._append_tool_result(messages, payload)

    def _append_tool_image_for_calls(
        self,
        messages: List[Dict[str, str]],
        images_for_prompt: List[Image.Image],
        tool_calls: List[ToolCall],
        image: Image.Image,
        image_index: int,
    ) -> None:
        tool_name = tool_calls[-1].name if tool_calls else "tool"
        image_name = f"output_{image_index}.jpg"
        payload = {
            "name": tool_name,
            "output": {"image_ref": {f"Observation {image_index}": image_name}},
        }
        self._append_tool_result(messages, payload)
        image_token = self._resolve_image_token()
        messages.append(
            {
                "role": "user",
                "content": f"Observation {image_index}:\n{image_token}",
            }
        )
        images_for_prompt.append(image)

    def _check_function_calls_legal(
        self, tool_calls: List[ToolCall]
    ) -> Tuple[bool, Optional[str], Optional[str], Optional[int]]:
        if not tool_calls:
            return True, None, None, None
        first_call = tool_calls[0]
        expected_name = first_call.name
        expected_index = first_call.args.get("image_index")
        for call in tool_calls[1:]:
            if call.name != expected_name:
                return (
                    False,
                    "Function call names are inconsistent in the same action.",
                    None,
                    None,
                )
            call_index = call.args.get("image_index")
            if call_index != expected_index:
                return (
                    False,
                    "Function call image_index values are inconsistent in the same action.",
                    None,
                    None,
                )
        return True, None, expected_name, expected_index

    def _execute_tool_calls(
        self,
        tool_calls: List[ToolCall],
        messages: List[Dict[str, str]],
        image_pool: List[Image.Image],
        images_for_prompt: List[Image.Image],
    ) -> bool:
        if not tool_calls:
            return False
        if not image_pool:
            for call in tool_calls:
                self._append_tool_error(messages, call, "No image available for tool call.")
            messages.append(
                {"role": "user", "content": TOOL_EXECUTION_FAILURE_PROMPT}
            )
            return False

        is_legal, illegal_reason, expected_name, expected_index = (
            self._check_function_calls_legal(tool_calls)
        )
        if not is_legal:
            for call in tool_calls:
                self._append_tool_error(messages, call, illegal_reason or "Illegal tool call.")
            messages.append(
                {"role": "user", "content": TOOL_EXECUTION_FAILURE_PROMPT}
            )
            return False

        if expected_name == "image_crop":
            had_error = False
            for call in tool_calls:
                if call.name not in self.tool_functions:
                    self._append_tool_error(messages, call, f"Unknown function {call.name}")
                    had_error = True
                    continue
                try:
                    result = self.tool_functions[call.name](image_pool, **call.args)
                except Exception as exc:
                    self._append_tool_error(
                        messages,
                        call,
                        f"Function call {call.name} with args {call.args} failed with error: {exc}",
                    )
                    had_error = True
                    continue
                if isinstance(result, Image.Image):
                    image_pool.append(result.copy())
                    image_index = len(image_pool) - 1
                    self._append_tool_image_for_calls(
                        messages, images_for_prompt, [call], result, image_index
                    )
                else:
                    self._append_tool_error(
                        messages,
                        call,
                        f"Function call {call.name} with args {call.args} failed with error: {result}",
                    )
                    had_error = True
            messages.append(
                {
                    "role": "user",
                    "content": TOOL_EXECUTION_FAILURE_PROMPT
                    if had_error
                    else TOOL_EXECUTION_SUCCESS_PROMPT,
                }
            )
            return not had_error

        dynamic_image = None
        error_result: List[Tuple[ToolCall, str]] = []
        dynamic_image_index = expected_index

        for call in tool_calls:
            if call.name not in self.tool_functions:
                error_result.append((call, f"Unknown function {call.name}"))
                continue
            target_index = dynamic_image_index
            if dynamic_image is None and "image_index" in call.args:
                target_index = call.args["image_index"]
            dynamic_image_list = list(image_pool)
            if target_index is not None and 0 <= target_index < len(image_pool):
                if dynamic_image is not None:
                    dynamic_image_list[target_index] = dynamic_image.copy()
                else:
                    dynamic_image_list[target_index] = image_pool[target_index].copy()
            try:
                result = self.tool_functions[call.name](dynamic_image_list, **call.args)
                dynamic_image = result if isinstance(result, Image.Image) else None
                if dynamic_image_index is None and "image_index" in call.args:
                    dynamic_image_index = call.args["image_index"]
            except Exception as exc:
                error_result.append(
                    (
                        call,
                        f"Function call {call.name} with args {call.args} failed with error: {exc}",
                    )
                )

        if isinstance(dynamic_image, Image.Image):
            image_pool.append(dynamic_image.copy())
            image_index = len(image_pool) - 1
            self._append_tool_image_for_calls(
                messages, images_for_prompt, tool_calls, dynamic_image, image_index
            )
        else:
            for call, error_msg in error_result:
                self._append_tool_error(messages, call, error_msg)

        had_error = bool(error_result) or not isinstance(dynamic_image, Image.Image)
        messages.append(
            {
                "role": "user",
                "content": TOOL_EXECUTION_FAILURE_PROMPT
                if had_error
                else TOOL_EXECUTION_SUCCESS_PROMPT,
            }
        )
        return not had_error

    def _build_request(
        self,
        messages: List[Dict[str, str]],
        images_for_prompt: List[Image.Image],
    ) -> Tuple[ModelRequest, str, List[int]]:
        prompt_str = self._apply_chat_template(messages)
        processor_callable = cast(Callable[..., Dict[str, Any]], self.processor)
        processed_input = processor_callable(
            text=[prompt_str],
            images=images_for_prompt if images_for_prompt else None,
            padding=False,
            return_tensors="pt",
        )
        self._last_processed_input = processed_input
        input_ids: List[int] = processed_input["input_ids"].tolist()[0]
        image_data = image2base64(images_for_prompt) if images_for_prompt else None
        req = ModelRequest(
            rid=uuid.uuid4().hex,
            input_ids=input_ids,
            image_data=image_data,
            gconfig=self.gconfig.new(n_samples=1),
            tokenizer=self.tokenizer,
            processor=self.processor,
        )
        return req, prompt_str, input_ids

    @trace_session("reward")
    async def _compute_reward(
        self,
        prompt: str,
        predict_str_list: List[str],
        prompt_ids: List[int],
        completion_ids: List[int],
        task_data: Dict[str, Any],
    ) -> float:
        reward = await self.async_reward_fn(
            prompt,
            predict_str_list,
            prompt_ids,
            completion_ids,
            ground_truth=self.data_extract_answer_fn(task_data),
            reward_extra_info=task_data.get("reward_extra_info", None),
            **task_data,
        )
        return float(reward)

    @session_context()
    async def _collect_samples(
        self, engine: InferenceEngine, data: Dict[str, Any]
    ) -> Tuple[Dict[str, torch.Tensor], float]:
        if isinstance(self.reward_fn, str):
            self.reward_fn = import_from_string(self.reward_fn)
            self.async_reward_fn = AsyncRewardWrapper(self.reward_fn)

        prompt = self.data_extract_prompt_fn(data)
        initial_image = self.data_extract_image_fn(data)
        messages: List[Dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        images_for_prompt: List[Image.Image] = []
        image_pool: List[Image.Image] = []
        image_token = self._resolve_image_token()
        user_content = prompt
        if initial_image is not None:
            image_pool.append(initial_image)
            images_for_prompt.append(initial_image)
            user_content = f"{prompt}\nObservation 0:\n{image_token}"
        messages.append({"role": "user", "content": user_content})

        seq: List[int] = []
        logprobs: List[float] = []
        loss_mask: List[int] = []
        versions: List[int] = []
        predict_str_list: List[str] = []
        prompt_ids: List[int] = []

        step = 0
        force_prompt = (
            "Max tool calls reached. Please provide the final answer without further tool calls."
        )
        while True:
            async with atrace_session_phase("generate"):
                req, prompt_str, input_ids = self._build_request(
                    messages, images_for_prompt
                )
                if not prompt_ids:
                    prompt_ids = list(input_ids)
                resp = await engine.agenerate(req)

            assistant_text = self.tokenizer.decode(resp.output_tokens)
            predict_str_list.append(assistant_text)

            input_len = len(resp.input_tokens) - len(seq)
            if input_len < 0 or resp.input_tokens[: len(seq)] != seq:
                raise ValueError("Prompt tokens do not match accumulated sequence.")

            seq += resp.input_tokens[-input_len:] + resp.output_tokens
            logprobs += [0.0] * input_len + resp.output_logprobs
            loss_mask += [0] * input_len + [1] * resp.output_len
            versions += [-1] * input_len + resp.output_versions

            tool_calls, answer_text = self._parse_response(assistant_text, step + 1)
            messages.append({"role": "assistant", "content": assistant_text})

            if tool_calls:
                step += 1
                if step >= self.max_tool_calls:
                    messages.append({"role": "user", "content": force_prompt})
                    async with atrace_session_phase("generate"):
                        req, _, _ = self._build_request(messages, images_for_prompt)
                        resp = await engine.agenerate(req)
                    assistant_text = self.tokenizer.decode(resp.output_tokens)
                    predict_str_list.append(assistant_text)
                    input_len = len(resp.input_tokens) - len(seq)
                    if input_len < 0 or resp.input_tokens[: len(seq)] != seq:
                        raise ValueError(
                            "Prompt tokens do not match accumulated sequence."
                        )
                    seq += resp.input_tokens[-input_len:] + resp.output_tokens
                    logprobs += [0.0] * input_len + resp.output_logprobs
                    loss_mask += [0] * input_len + [1] * resp.output_len
                    versions += [-1] * input_len + resp.output_versions
                    messages.append({"role": "assistant", "content": assistant_text})
                    break

                self._execute_tool_calls(
                    tool_calls, messages, image_pool, images_for_prompt
                )
                continue

            if answer_text or not tool_calls:
                break

        completion_ids = seq[len(prompt_ids) :]
        reward = await self._compute_reward(
            prompt, predict_str_list, prompt_ids, completion_ids, data
        )
        stats_tracker.get(self.rollout_stat_scope).scalar(reward=reward)

        res = {
            "input_ids": torch.tensor(seq, dtype=torch.int32).unsqueeze(0),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.int32).unsqueeze(0),
            "logprobs": torch.tensor(logprobs, dtype=torch.float32).unsqueeze(0),
            "versions": torch.tensor(versions, dtype=torch.int32).unsqueeze(0),
            "attention_mask": torch.ones(len(seq), dtype=torch.bool).unsqueeze(0),
            "rewards": torch.tensor(reward, dtype=torch.float32).unsqueeze(0),
        }

        if self._last_processed_input and "pixel_values" in self._last_processed_input:
            multi_modal_input = [{"pixel_values": self._last_processed_input["pixel_values"]}]
            if "image_grid_thw" in self._last_processed_input:
                multi_modal_input[0]["image_grid_thw"] = self._last_processed_input[
                    "image_grid_thw"
                ]
            res["multi_modal_input"] = multi_modal_input

        return res, reward

    async def arun_episode(
        self, engine: InferenceEngine, data: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        results = [await self._collect_samples(engine, data)]
        return concat_padded_tensors([res for res, _ in results])
