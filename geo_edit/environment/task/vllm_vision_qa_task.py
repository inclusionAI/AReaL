from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from geo_edit.environment.task.vision_qa_task import ToolCall, VisionQATask
from geo_edit.environment.action.image_edition_tool import (
    bounding_box_function_declaration,
    draw_line_function_declaration,
    image_crop_function_declaration,
    image_label_function_declaration,
)
from geo_edit.utils.vision_task_utils import image_to_data_url
from geo_edit.utils.logger import setup_logger

logger = setup_logger(__name__)


class VLLMVisionQATask(VisionQATask):
    """Vision QA task for vLLM OpenAI-compatible chat completions."""

    _THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
    _ACTION_PATTERN = re.compile(r"<action>(.*?)</action>", re.DOTALL | re.IGNORECASE)
    _ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)

    def __init__(
        self,
        task_id: str,
        task_prompt: str,
        task_answer: str,
        task_image_path: str | None,
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
        base_prompt = system_prompt 
        tool_info = self._build_tool_prompt()
        if tool_info:
            self.system_prompt = f"{base_prompt}\n\n{tool_info}"
        else:
            self.system_prompt = base_prompt
            
        self.image_url_map: Dict[str, str] = {}
        self.messages: List[Dict[str, Any]] = []
        if self.system_prompt:
            self.messages.append(
                {"role": "system", "content": self.system_prompt}
            )
        self._append_initial_observation()
        self.contents = self.messages
        
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
        self.messages.append({"role": "user", "content": content})

    def _build_tool_prompt(self) -> str:
        if not self.tool_functions:
            return None
        declaration_pool = [
            image_crop_function_declaration,
            image_label_function_declaration,
            draw_line_function_declaration,
            bounding_box_function_declaration,
        ]
        declarations: Dict[str, Dict[str, Any]] = {}
        for tool in declaration_pool:
            name = tool.get("name")
            if name:
                declarations[name] = tool

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
                    image_path = self.image_url_map[image_url]

                    output["content"].append(
                        {
                            "type": "image_url",
                            "image_path": image_path ,
                        }
                    )
                else:
                    output["content"].append(part)
            return output

        output["content"] = content
        return output

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
            logger.warning("Empty text when parsing tool calls.")
            return tool_calls, tool_call_records
        matches = list(self._ACTION_PATTERN.finditer(text))
        for index, match in enumerate(matches, start=1):
            action_body = match.group(1).strip()
            try:
                tool_name, arguments = self._parse_action_body(action_body)
            except ValueError as e:
                logger.warning(f"Skipping invalid tool call: {e}")
                continue
            call_id = f"call_{step}_{index}"
            tool_calls.append(
                ToolCall(name=tool_name, args=arguments, call_id=call_id)
            )
            tool_call_records.append(
                {"id": call_id, "name": tool_name, "args": arguments}
            )
        return tool_calls, tool_call_records

    def parse_action(self, step: int, action: Any, extra_info: Dict[str, Any]):
        contents_for_save = [
            self._stringify_observation_item(item) for item in self.messages
        ]

        if not action.choices:
            raise ValueError("vLLM response contained no tool call or final answer.")

        content = action.choices[0].message.content

        thinking_parts = [
            match.group(1).strip()
            for match in self._THINK_PATTERN.finditer(content)
        ]
        thinking_process = "\n".join(part for part in thinking_parts if part)
        answer_parts = [
            match.group(1).strip()
            for match in self._ANSWER_PATTERN.finditer(content)
        ]
        output_text = "\n".join(part for part in answer_parts if part)
        tool_calls, tool_call_records = self._parse_tool_calls_from_text(
            content, step
        )

        if not tool_calls and not output_text:
            raise ValueError("vLLM response contained no tool call or final answer.")

        assistant_message: Dict[str, Any] = {"role": "assistant"}
        assistant_message["content"] = content
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
                "thinking_process": thinking_process,
                "output_text": output_text,
                "function_call": function_call_list,
                "extra_info": extra_info,
            }
        )

        return list(tool_calls)

    def _append_tool_result(self, payload: Dict[str, Any]) -> None:
        text = f"<tool_result>{json.dumps(payload, ensure_ascii=True)}</tool_result>"
        self.messages.append(
            {"role": "user", "content": [{"type": "text", "text": text}]}
        )

    def _append_tool_error(self, tool_call: ToolCall, error_msg: str) -> None:
        payload = {"name": tool_call.name, "error": error_msg}
        self._append_tool_result(payload)

    def _append_tool_image_for_calls(
        self,
        tool_calls: List[ToolCall],
        image: Image.Image,
        image_name: str,
        image_path: str,
        image_bytes: bytes,
        image_index: int,
    ) -> None:
        tool_name = tool_calls[-1].name if tool_calls else "tool"
        payload = {
            "name": tool_name,
            "output": {"image_ref": {f"Observation {image_index}": image_name}},
        }
        self._append_tool_result(payload)

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
