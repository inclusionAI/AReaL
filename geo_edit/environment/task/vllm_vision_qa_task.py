from __future__ import annotations, print_function

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from geo_edit.environment.task.vision_qa_task import ToolCall, VisionQATask
from geo_edit.utils.vision_task_utils import image_to_data_url
from geo_edit.utils.logger import setup_logger

logger = setup_logger(__name__)


class VLLMVisionQATask(VisionQATask):
    """Vision QA task for vLLM OpenAI-compatible chat completions with native tool calls."""

    _THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
    _ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)

    def __init__(self, task_id: str, task_prompt: str, task_answer: str,
                 task_image_path: str | None, save_dir: Path | str,
                 tool_functions: Optional[Dict[str, Any]] = None,
                 system_prompt: Optional[str] = None, **kwargs):
        super().__init__(task_id=task_id, task_prompt=task_prompt, task_answer=task_answer,
                         task_image_path=task_image_path, save_dir=save_dir,
                         tool_functions=tool_functions, **kwargs)
        self.system_prompt = system_prompt
        self.contents: List[Dict[str, Any]] = []
        if self.system_prompt:
            self.contents.append({"role": "system", "content": self.system_prompt})
        self._append_initial_observation()

    def _append_initial_observation(self) -> None:
        content = [{"type": "text", "text": self.task_prompt}]
        if self.image_list:
            image = self.image_list[0]
            image_url = image_to_data_url(image)
            if self.task_image_path:
                self.image_url_map[image_url] = self.task_image_path
            content.extend(
                [
                    {"type": "text", "text": "Observation 0:"},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ]
            )
        self.contents.append({"role": "user", "content": content})

    def _stringify_observation_item(self, item: Any) -> Any:
        if not isinstance(item, dict):
            return item

        output: Dict[str, Any] = {"role": item.get("role"), "content": []}

        content = item.get("content")
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    output["content"].append(part)
                    continue
                if part.get("type") == "image_url":
                    image_url = part["image_url"]
                    if isinstance(image_url, dict):
                        image_url = image_url["url"]
                    image_path = self.image_url_map.get(image_url, "")

                    output["content"].append(
                        {
                            "type": "image_url",
                            "image_path": image_path,
                        }
                    )
                else:
                    output["content"].append(part)
            return output

        output["content"] = content
        return output

    def parse_action(self, step: int, action: Any, extra_info: Dict[str, Any]):
        contents_for_save = [self._stringify_observation_item(item) for item in self.contents]

        if not action.choices:
            raise ValueError("vLLM response contained no choices.")

        message = action.choices[0].message

        content = message.content 
        native_tool_calls = message.tool_calls

        # Extract thinking from content
        thinking_parts = [m.group(1).strip() for m in self._THINK_PATTERN.finditer(content)]
        thinking_process = "\n".join(part for part in thinking_parts if part)

        # Extract answer from content
        answer_parts = [m.group(1).strip() for m in self._ANSWER_PATTERN.finditer(content)]
        output_text = "\n".join(part for part in answer_parts if part)

        # Parse native tool_calls
        tool_calls: List[ToolCall] = []
        tool_call_records: List[Dict[str, Any]] = []
        if native_tool_calls:
            for tc in native_tool_calls:
                args = tc.function.arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        logger.warning("Failed to parse tool call arguments as JSON: %s", args)
                        raise ValueError(f"Invalid JSON in tool call arguments: {args}")
                tool_calls.append(ToolCall(name=tc.function.name, args=args, call_id=tc.id))
                tool_call_records.append({"id": tc.id, "name": tc.function.name, "args": args})

        if not tool_calls and not output_text:
            logger.error("vLLM response content: %s", content)
            raise ValueError("vLLM response contained no tool call or final answer.")

        if tool_calls and output_text:
            logger.warning("vLLM response contained tool call and final answer; "
                           "keeping the answer and ignoring tool calls.")
            tool_calls = []
            tool_call_records = []

        # Build assistant message for conversation
        assistant_message: Dict[str, Any] = {"role": "assistant", "content": content}
        if native_tool_calls:
            assistant_message["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in native_tool_calls
            ]

        self.contents.append(assistant_message)

        action_record = {"text": output_text, "tool_calls": tool_call_records}
        self._record_conversation_history(
            step, contents_for_save, action_record, thinking_process, output_text, tool_calls, extra_info
        )
        return list(tool_calls)

    def _append_tool_result(self, call_id: str, payload: Dict[str, Any]) -> None:
        text = json.dumps(payload, ensure_ascii=True)
        self.contents.append(
            {"role": "tool", "tool_call_id": call_id, "content": text}
        )

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
        payload = {"image_ref": {f"Observation {image_index}": image_name}}
        for call in tool_calls:
            self._append_tool_result(call.call_id, payload)

        image_url = image_to_data_url(image)
        self.image_url_map[image_url] = image_path
        content = [
            {"type": "text", "text": f"Observation {image_index}:"},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
        self.contents.append({"role": "user", "content": content})

    def append_prompt(self, prompt: str) -> None:
        self.contents.append(
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        )
