from __future__ import annotations, print_function

import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from PIL import Image

from geo_edit.environment.task.vision_qa_task import ToolCall, VisionQATask
from geo_edit.utils.vision_task_utils import image_to_data_url
from geo_edit.utils.logger import setup_logger

logger = setup_logger(__name__)


class VLLMVisionQATask(VisionQATask):
    """Vision QA task for vLLM OpenAI-compatible Responses API with tool calls."""

    _THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
    _ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)

    def __init__(self, task_id: str, task_prompt: str, task_answer: str,
                 task_image_path: str | None, save_dir: Path | str,
                 tool_functions: Optional[Dict[str, Callable[..., Image.Image | str]]] = None,
                 system_prompt: Optional[str] = None, **kwargs):
        super().__init__(
            task_id=task_id,
            task_prompt=task_prompt,
            task_answer=task_answer,
            task_image_path=task_image_path,
            save_dir=save_dir,
            tool_functions=tool_functions,
            **kwargs,
        )
        self.system_prompt = system_prompt
        self.contents: Dict[str, List[Dict[str, Any]]] = {"input": []}
        if self.system_prompt:
            self.contents["input"].append({
                "role": "system",
                "content": [{"type": "input_text", "text": self.system_prompt}],
            })
        self._append_initial_observation()

    def _append_initial_observation(self) -> None:
        content = [{"type": "input_text", "text": self.task_prompt}]
        if self.image_list:
            image = self.image_list[0]
            image_url = image_to_data_url(image)
            if self.task_image_path:
                self.image_url_map[image_url] = self.task_image_path
            content.extend(
                [
                    {"type": "input_text", "text": "Observation 0:"},
                    {"type": "input_image", "image_url": image_url, "detail": "auto"},
                ]
            )
        self.contents["input"].append({"role": "user", "content": content})

    def _stringify_observation_item(self, item: Any) -> Any:
        if not isinstance(item, dict):
            return item

        if item.get("type") == "function_call_output":
            output = item.get("output")
            if isinstance(output, list):
                parts: List[Dict[str, str]] = []
                for part in output:
                    if part.get("type") == "input_image":
                        image_url = part["image_url"]
                        image_path = self.image_url_map.get(image_url, "")
                        parts.append({"type": "input_image", "image_path": image_path})
                    else:
                        parts.append(part)
                output = parts
            return {"type": "function_call_output", "call_id": item.get("call_id"), "output": output}
        if item.get("type") == "function_call":
            return {
                "type": "function_call", "call_id": item.get("call_id"),
                "name": item.get("name"), "arguments": item.get("arguments"),
            }

        output: Dict[str, Any] = {"role": item.get("role"), "content": []}

        content = item.get("content")
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    output["content"].append(part)
                    continue
                if part.get("type") == "input_image":
                    image_url = part["image_url"]
                    image_path = self.image_url_map.get(image_url, "")
                    output["content"].append({"type": "input_image", "image_path": image_path})
                else:
                    output["content"].append(part)
            return output

        output["content"] = content
        return output

    def parse_action(self, step: int, action: Any, extra_info: Dict[str, int | float | str | None]):
        contents_for_save = [
            self._stringify_observation_item(item) for item in self.contents["input"]
        ]

        if not action.output:
            raise ValueError("vLLM response contained no output.")

        raw_text = action.output_text
        if raw_text is None:
            raw_text = ""
        tool_calls: List[ToolCall] = []
        tool_call_records: List[Dict[str, Any]] = []
        tool_call_items: List[Dict[str, str]] = []

        for item in action.output:
            if item.type == "function_call":
                raw_arguments = item.arguments
                arguments: Dict[str, str | int] = {}
                arguments_payload = raw_arguments
                if isinstance(raw_arguments, str):
                    if raw_arguments:
                        try:
                            arguments = json.loads(raw_arguments)
                        except json.JSONDecodeError:
                            logger.warning("Failed to parse tool call arguments as JSON: %s", raw_arguments)
                            raise ValueError(f"Invalid JSON in tool call arguments: {raw_arguments}")
                elif isinstance(raw_arguments, dict):
                    arguments = raw_arguments
                    arguments_payload = json.dumps(raw_arguments, ensure_ascii=True)

                if hasattr(item, "call_id"):
                    call_id = item.call_id
                else:
                    call_id = item.id
                tool_calls.append(ToolCall(name=item.name, args=arguments, call_id=call_id))
                tool_call_records.append({"id": call_id, "name": item.name, "args": arguments})
                tool_call_items.append({
                    "type": "function_call", "name": item.name,
                    "arguments": arguments_payload, "call_id": call_id,
                })

        # Extract thinking from content
        thinking_parts = [m.group(1).strip() for m in self._THINK_PATTERN.finditer(raw_text)]
        thinking_process = "\n".join(part for part in thinking_parts if part)

        # Extract answer from content
        answer_parts = [m.group(1).strip() for m in self._ANSWER_PATTERN.finditer(raw_text)]
        output_text = "\n".join(part for part in answer_parts if part)

        if not tool_calls and not output_text:
            logger.error("vLLM response content: %s", raw_text)
            raise ValueError("vLLM response contained no tool call or final answer.")

        if tool_calls and output_text:
            logger.warning(
                "vLLM response contained tool call and final answer; "
                "keeping the answer and ignoring tool calls."
            )
            tool_calls = []
            tool_call_records = []
            tool_call_items = []

        if tool_call_items:
            self.contents["input"].extend(tool_call_items)

        action_record = {"text": output_text, "tool_calls": tool_call_records}
        self._record_conversation_history(
            step, contents_for_save, action_record, thinking_process, output_text, tool_calls, extra_info
        )
        return list(tool_calls)

    def _append_tool_result(
        self,
        call_id: str,
        payload: Dict[str, str | Dict[str, str]] | List[Dict[str, str]],
    ) -> None:
        output = payload if isinstance(payload, list) else json.dumps(payload, ensure_ascii=True)
        self.contents["input"].append({"type": "function_call_output", "call_id": call_id, "output": output})

    def _append_tool_error(self, tool_call: ToolCall, error_msg: str) -> None:
        payload = {"error": error_msg}
        self._append_tool_result(tool_call.call_id, payload)

    def _append_tool_image_for_calls(
        self,
        tool_calls: List[ToolCall],
        image: Image.Image,
        image_name: str,
        image_path: str,
        image_bytes: bytes,
        image_index: int,
    ) -> None:
        image_url = image_to_data_url(image)
        self.image_url_map[image_url] = image_path
        output = [
            {"type": "input_text", "text": f"Observation {image_index}:"},
            {"type": "input_image", "image_url": image_url, "detail": "auto"},
        ]
        for call in tool_calls:
            self._append_tool_result(call.call_id, output)

    def append_prompt(self, prompt: str) -> None:
        self.contents["input"].append({"role": "user", "content": [{"type": "input_text", "text": prompt}]})
