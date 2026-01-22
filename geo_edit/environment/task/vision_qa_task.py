from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from .base import AbstractVLMTask
from ...constants import TOOL_EXECUTION_FAILURE_PROMPT, TOOL_EXECUTION_SUCCESS_PROMPT
from ...utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class ToolCall:
    name: str
    args: Dict[str, Any]
    call_id: Optional[str] = None


class VisionQATask(AbstractVLMTask):
    """Base vision QA task shared by API providers."""

    def __init__(
        self,
        task_id: str,
        task_prompt: str,
        task_answer: str,
        task_image_path: str | None,
        save_dir: Path | str,
        tool_functions: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(task_id)
        self.task_prompt = task_prompt
        self.task_answer = task_answer
        self.task_image_path = task_image_path
        self.tool_functions = tool_functions or {}
        self.state = True
        self.options = kwargs["options"] if "options" in kwargs else None

        self.image_path_map: Dict[int, str] = {}
        self.image_list: List[Image.Image] = []
        if self.task_image_path:
            image = Image.open(self.task_image_path).convert("RGB")
            self.image_list.append(image)
            self.image_path_map[id(image)] = self.task_image_path

        self.conversation_history: List[Dict[str, Any]] = []

        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.output_jsonl_path = os.path.join(self.save_dir, "output.jsonl")
        self.extra_info_jsonl_path = os.path.join(self.save_dir, "extra_info.jsonl")
        self.meta_info_jsonl_path = os.path.join(self.save_dir, "meta_info.jsonl")
        self.image_save_dir = os.path.join(self.save_dir, "images")
        os.makedirs(self.image_save_dir, exist_ok=True)

    def validate(
        self,
        chat_history: List[Dict],
        last_observation: Any,
        full_history: List[Any],
    ) -> Tuple[float, bool, Dict[str, Any]]:
        """verify the task"""
        return 0.0, False, {}

    def get_info(self) -> Dict[str, Any]:
        return {"task_id": self.task_id}

    def _prepare_tool_update(self) -> None:
        return

    def _append_tool_error(self, tool_call: ToolCall, error_msg: str) -> None:
        raise NotImplementedError

    def _append_tool_image_for_calls(
        self,
        tool_calls: List[ToolCall],
        image: Image.Image,
        image_name: str,
        image_path: str,
        image_bytes: bytes,
        image_index: int,
    ) -> None:
        raise NotImplementedError

    def append_prompt(self, prompt: str) -> None:
        raise NotImplementedError

    def _save_image(
        self, image: Image.Image
    ) -> Tuple[int, str, str, bytes]:
        self.image_list.append(image.copy())
        image_index = len(self.image_list) - 1
        image_name = f"output_{image_index}.jpg"
        image_path = os.path.join(self.image_save_dir, image_name)
        image.save(image_path)
        self.image_path_map[id(image)] = image_path
        image_bytes_io = io.BytesIO()
        image.save(image_bytes_io, format="JPEG")
        image_bytes = image_bytes_io.getvalue()
        return image_index, image_name, image_path, image_bytes

    def _check_function_calls_legal(
        self, tool_calls: List[ToolCall]
    ) -> Tuple[bool, Optional[str], Optional[str], Optional[int]]:
        if not tool_calls:
            logger.warning("No function calls found in the action.")
            return True, None, None, None
        first_call = tool_calls[0]
        expected_name = first_call.name
        expected_index = first_call.args["image_index"]
        for call in tool_calls[1:]:
            if call.name != expected_name:
                logger.warning("Inconsistent function call names: expected %s, got %s", expected_name, call.name)
                return False, "Function call names are inconsistent in the same action.", None, None
            call_index = call.args["image_index"]
            if call_index != expected_index:
                logger.warning("Inconsistent image_index values: expected %s, got %s", expected_index, call_index)
                return (False, "Function call image_index values are inconsistent in the same action.", None, None)
        return True, None, expected_name, expected_index

    def update_observation_from_action(self, tool_calls: List[ToolCall]) -> None:
        if not tool_calls:
            logger.warning("No function calls to process.")
            return
        if not self.image_list:
            for call in tool_calls:
                self._append_tool_error(call, "No image available for tool call.")
            self.append_prompt(TOOL_EXECUTION_FAILURE_PROMPT)
            return
        self._prepare_tool_update()
        is_legal, illegal_reason, expected_name, expected_index = (
            self._check_function_calls_legal(tool_calls)
        )
        dynamic_image = None
        error_result: List[Tuple[ToolCall, str]] = []
        dynamic_image_index = expected_index

        if not is_legal:
            logger.warning(illegal_reason)
            for call in tool_calls:
                self._append_tool_error(call, illegal_reason)
            self.append_prompt(TOOL_EXECUTION_FAILURE_PROMPT)
            return

        if expected_name == "image_crop":
            call_results: List[Tuple[str, ToolCall, Any]] = []
            for call in tool_calls:
                logger.info(
                    "Processing function call: %s with args: %s",
                    call.name,
                    call.args,
                )
                if call.name in self.tool_functions:
                    function_to_call = self.tool_functions[call.name]
                    try:
                        result = function_to_call(self.image_list, **call.args)
                        if isinstance(result, Image.Image):
                            call_results.append(("image", call, result))
                        else:
                            call_results.append(
                                ("error", call, (f"Function call {call.name} with args {call.args} failed with error: {result}"))
                            )
                    except Exception as exc:
                        call_results.append(
                            ("error", call, (f"Function call {call.name} with args {call.args} failed with error: {exc}"))
                        )
                else:
                    call_results.append(
                        ("error", call, f"Unknown function {call.name}")
                    )

            for result_type, call, payload in call_results:
                if result_type == "image":
                    image_index, image_name, image_path, image_bytes = self._save_image(payload)
                    self._append_tool_image_for_calls(
                        [call], payload, image_name, image_path, image_bytes, image_index,
                    )
                else:
                    self._append_tool_error(call, str(payload))
            had_error = any(result_type != "image" for result_type, _, _ in call_results)
            self.append_prompt(
                TOOL_EXECUTION_FAILURE_PROMPT
                if had_error
                else TOOL_EXECUTION_SUCCESS_PROMPT
            )
            return

        for call in tool_calls:
            logger.info(  "Processing function call: %s with args: %s", call.name, call.args)
            if call.name in self.tool_functions:
                function_to_call = self.tool_functions[call.name]
                target_index = dynamic_image_index
                if dynamic_image is None and "image_index" in call.args:
                    target_index = call.args["image_index"]
                dynamic_image_list = list(self.image_list)
                if target_index is not None and 0 <= target_index < len(self.image_list):
                    if dynamic_image is not None:
                        dynamic_image_list[target_index] = dynamic_image.copy()
                    else:
                        dynamic_image_list[target_index] = (
                            self.image_list[target_index].copy()
                        )
                try:
                    result = function_to_call(dynamic_image_list, **call.args)
                    dynamic_image = result
                    if dynamic_image_index is None and "image_index" in call.args:
                        dynamic_image_index = call.args["image_index"]
                except Exception as exc:
                    error_result.append(
                        (call, (f"Function call {call.name} with args {call.args} failed with error: {exc}"))
                    )
            else:
                error_result.append((call, f"Unknown function {call.name}"))

        if isinstance(dynamic_image, Image.Image):
            image_index, image_name, image_path, image_bytes = self._save_image(
                dynamic_image
            )
            self._append_tool_image_for_calls( 
                tool_calls, dynamic_image, image_name, image_path, image_bytes, image_index,
            )
        else:
            for call, error_msg in error_result:
                self._append_tool_error(call, error_msg)
        had_error = bool(error_result) or not isinstance(dynamic_image, Image.Image)
        self.append_prompt(
            TOOL_EXECUTION_FAILURE_PROMPT
            if had_error
            else TOOL_EXECUTION_SUCCESS_PROMPT
        )

    def save_trajectory(self) -> Dict[str, Any]:
        """save the trajectory to jsonl files"""
        extra_info_list = []
        function_call_total_count = 0
        function_call_each_count = {}
        function_call_per_step = []
        tokens_used_total = 0
        tokens_used_per_step = []

        for record in self.conversation_history:
            function_call = record["function_call"]
            if function_call:
                function_call_total_count += len(function_call)
                function_names = []
                for function_name, _ in function_call:
                    if function_name in function_call_each_count:
                        function_call_each_count[function_name] += 1
                    else:
                        function_call_each_count[function_name] = 1
                    function_names.append(function_name)
                function_call_per_step.append(function_names)
            else:
                function_call_per_step.append(None)
            if "tokens_used" in record["extra_info"]:
                tokens_used = record["extra_info"]["tokens_used"]
            else:
                tokens_used = 0
            tokens_used_total += tokens_used
            tokens_used_per_step.append(tokens_used)

        meta_info = {
            "question": self.task_prompt,
            "options": self.options,
            "answer": self.task_answer,
            "image_path": self.task_image_path,
            "function_call_total_count": function_call_total_count,
            "total_steps": len(self.conversation_history),
            "function_call_each_count": function_call_each_count,
            "function_call_per_step": function_call_per_step,
            "tokens_used_total": tokens_used_total,
            "tokens_used_per_step": tokens_used_per_step,
            "output_text": self.conversation_history[-1]["output_text"]
        }

        last_step_index = len(self.conversation_history) - 1
        for idx, record in enumerate(self.conversation_history):
            observation = record["observation"]
            extra_info_list.append(
                {
                    "step": record["step"],
                    "extra_info": record.pop("extra_info"),
                    "observation": observation,
                }
            )
            if idx != last_step_index:
                if "observation" in record:
                    record.pop("observation")

        with open(self.extra_info_jsonl_path, "w", encoding="utf-8") as f:
            for record in extra_info_list:
                f.write(json.dumps(record) + "\n")

        with open(self.output_jsonl_path, "w", encoding="utf-8") as f:
            for record in self.conversation_history:
                f.write(json.dumps(record) + "\n")

        with open(self.meta_info_jsonl_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(meta_info) + "\n")

        return meta_info
