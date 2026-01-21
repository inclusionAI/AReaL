from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from .vision_qa_task import ToolCall, VisionQATask
from ...utils.vision_task_utils import image_to_data_url
from ...utils.logger import setup_logger

logger = setup_logger(__name__)


class VLLMVisionQATask(VisionQATask):
    """Vision QA task for vLLM OpenAI-compatible chat completions."""

    def __init__(
        self,
        task_id: str,
        task_prompt: str,
        task_answer: str,
        task_image_path: str,
        save_dir: Path | str,
        tool_functions: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        **kwargs,
    ):
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
        self.image_url_map: Dict[str, str] = {}
        self.messages: List[Dict[str, Any]] = []
        if self.system_prompt:
            self.messages.append(
                {"role": "system", "content": self.system_prompt}
            )
        self._append_initial_observation()
        self.contents = self.messages

    def _append_initial_observation(self) -> None:
        image = self.image_list[0]
        image_url = image_to_data_url(image)
        self.image_url_map[image_url] = self.task_image_path
        content = [
            {"type": "text", "text": self.task_prompt},
            {"type": "text", "text": "Observation 0:"},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
        self.messages.append({"role": "user", "content": content})

    def _stringify_observation_item(self, item: Any) -> Any:
        if not isinstance(item, dict):
            return item

        if item.get("role") == "tool":
            output: Dict[str, Any] = {
                "role": "tool",
                "content": item.get("content", ""),
            }
            if "tool_call_id" in item:
                output["tool_call_id"] = item["tool_call_id"]
            return output

        output: Dict[str, Any] = {"role": item.get("role"), "content": []}
        if "tool_calls" in item:
            output["tool_calls"] = item["tool_calls"]

        content = item.get("content")
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    output["content"].append(part)
                    continue
                if part.get("type") == "image_url":
                    image_url = part.get("image_url", {})
                    if isinstance(image_url, dict):
                        image_url = image_url.get("url")
                    image_path = (
                        self.image_url_map.get(image_url, "")
                        if image_url
                        else ""
                    )
                    output["content"].append(
                        {
                            "type": "image_url",
                            "image_path": image_path or "<omitted>",
                        }
                    )
                else:
                    output["content"].append(part)
            return output

        output["content"] = content
        return output

    def _coerce_message_text(self, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") in {"text", "output_text"}:
                        parts.append(part.get("text", ""))
                elif isinstance(part, str):
                    parts.append(part)
            return "".join(parts)
        return str(content)

    def parse_action(self, step: int, action: Any, extra_info: Dict[str, Any]):
        output_text = ""
        tool_calls: List[ToolCall] = []

        contents_for_save = [
            self._stringify_observation_item(item) for item in self.messages
        ]

        if not action.choices:
            logger.warning("No choices found in vLLM response.")
            self.conversation_history.append(
                {
                    "step": step,
                    "observation": contents_for_save,
                    "action": {"text": "", "tool_calls": []},
                    "thinking_process": "",
                    "output_text": "",
                    "function_call": None,
                    "extra_info": extra_info,
                }
            )
            return tool_calls

        message = action.choices[0].message
        output_text = self._coerce_message_text(message.content)

        tool_call_records = []
        if message.tool_calls:
            for call in message.tool_calls:
                raw_arguments = call.function.arguments
                arguments: Dict[str, Any] = {}
                if isinstance(raw_arguments, str):
                    if raw_arguments:
                        try:
                            arguments = json.loads(raw_arguments)
                        except json.JSONDecodeError:
                            arguments = {}
                elif isinstance(raw_arguments, dict):
                    arguments = raw_arguments

                tool_calls.append(
                    ToolCall(
                        name=call.function.name,
                        args=arguments,
                        call_id=call.id,
                    )
                )
                tool_call_records.append(
                    {"id": call.id, "name": call.function.name, "args": arguments}
                )

        assistant_message: Dict[str, Any] = {"role": "assistant"}
        if message.content is not None:
            assistant_message["content"] = message.content
        if message.tool_calls:
            assistant_message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in message.tool_calls
            ]
        self.messages.append(assistant_message)

        function_call_list = (
            [(call.name, call.args) for call in tool_calls]
            if tool_calls
            else None
        )
        self.conversation_history.append(
            {
                "step": step,
                "observation": contents_for_save,
                "action": {
                    "text": output_text,
                    "tool_calls": tool_call_records,
                },
                "thinking_process": "",
                "output_text": output_text,
                "function_call": function_call_list,
                "extra_info": extra_info,
            }
        )

        return list(tool_calls)

    def _append_tool_error(self, tool_call: ToolCall, error_msg: str) -> None:
        payload = json.dumps({"error": error_msg}, ensure_ascii=True)
        message = {"role": "tool", "content": payload}
        if tool_call.call_id:
            message["tool_call_id"] = tool_call.call_id
        self.messages.append(message)

    def _append_tool_image_for_calls(
        self,
        tool_calls: List[ToolCall],
        image: Image.Image,
        image_name: str,
        image_path: str,
        image_bytes: bytes,
        image_index: int,
    ) -> None:
        payload = json.dumps(
            {"image_ref": {f"Observation {image_index}": image_name}},
            ensure_ascii=True,
        )
        for call in tool_calls:
            message = {"role": "tool", "content": payload}
            if call.call_id:
                message["tool_call_id"] = call.call_id
            self.messages.append(message)

        image_url = image_to_data_url(image)
        self.image_url_map[image_url] = image_path
        content = [
            {"type": "text", "text": f"Observation {image_index}:"},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
        self.messages.append({"role": "user", "content": content})

    def append_prompt(self, prompt: str) -> None:
        self.messages.append(
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        )
