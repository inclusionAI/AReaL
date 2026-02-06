from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from PIL import Image

from geo_edit.environment.task.vision_qa_task import ToolCall, VisionQATask
from geo_edit.utils.vision_task_utils import image_to_data_url
from geo_edit.utils.logger import setup_logger

logger = setup_logger(__name__)

class OpenAIVisionQATask(VisionQATask):
    """vision qa task for OpenAI Responses API"""

    def __init__(self, task_id: str, task_prompt: str, task_answer: str,
                 task_image_path: str | None, save_dir: Path | str,
                 tool_functions: Optional[Dict[str, Callable[..., Image.Image | str]]] = None, **kwargs):
        super().__init__(task_id=task_id, task_prompt=task_prompt, task_answer=task_answer,
                         task_image_path=task_image_path, save_dir=save_dir,
                         tool_functions=tool_functions, **kwargs)

        input_items: List[Dict[str, Any]] = []
        if self.text_only:
            logger.info("Initializing OpenAIVisionQATask in text only mode.")
            content = [{"type": "input_text", "text": self.task_prompt}]
        else:
            image = self.image_list[0]
            image_url = image_to_data_url(image)
            self.image_url_map[image_url] = self.task_image_path
            content = [
                {"type": "input_text", "text": self.task_prompt},
                {"type": "input_text", "text": "Observation 0:"},
                {"type": "input_image", "image_url": image_url},
            ]
        input_items.append({"role": "user", "content": content})
        self.contents = {"input": input_items, "previous_response_id": None}

    def _append_tool_message(
        self,
        tool_call_id: Optional[str],
        payload: Dict[str, str | Dict[str, str]] | List[Dict[str, str]],
    ) -> None:
        output = payload if isinstance(payload, list) else json.dumps(payload, ensure_ascii=True)
        self.contents["input"].append({
            "type": "function_call_output", "call_id": tool_call_id, "output": output,
        })

    def append_prompt(self, text: str) -> None:
        self.contents["input"].append(
            {"role": "user", "content": [{"type": "input_text", "text": text}]}
        )

    def _stringify_observation_item(self, item: Any) -> Any:
        if not isinstance(item, dict):
            return item

        if item.get("type") == "function_call_output":
            output = item["output"]
            if isinstance(output, list):
                parts: List[Dict[str, str]] = []
                for part in output:
                    if part.get("type") == "input_image":
                        image_url = part["image_url"]
                        image_path = self.image_url_map[image_url] 
                        parts.append({"type": "input_image", "image_path": image_path or "<omitted>"})
                    else:
                        parts.append(part)
                output = parts
            return {"type": "function_call_output", "call_id": item["call_id"], "output": output}

        output: Dict[str, Any] = {"role": item["role"], "content": []}
        if "tool_call_id" in item:
            output["tool_call_id"] = item["tool_call_id"]
        for part in item["content"]:
            part_type = part["type"]
            if part_type == "input_image":
                image_url = part["image_url"]
                image_path = self.image_url_map[image_url]
                output["content"].append({"type": "input_image", "image_path": image_path or "<omitted>"})
            else:
                output["content"].append(part)
        return output

    def parse_action(self, step: int, action: Any, extra_info: Dict[str, int | float | str | None]):
        """update task contents from action"""
        output_text = ""
        thinking_process = ""
        tool_calls: List[ToolCall] = []

        for item in action.output:
            if item.type == "reasoning":
                # Extract thinking/reasoning content from OpenAI response
                if hasattr(item, "summary") and item.summary:
                    for summary_item in item.summary:
                        if hasattr(summary_item, "text") and summary_item.text:
                            thinking_process += summary_item.text
            elif item.type == "message":
                for part in item.content:
                    if part.type == "output_text":
                        output_text += part.text
            elif item.type == "function_call":
                raw_arguments = item.arguments
                arguments: Dict[str, str | int] = {}
                if isinstance(raw_arguments, str):
                    if raw_arguments:
                        try:
                            arguments = json.loads(raw_arguments)
                        except json.JSONDecodeError:
                            arguments = {}
                elif isinstance(raw_arguments, dict):
                    arguments = raw_arguments
                tool_calls.append(ToolCall(name=item.name, args=arguments, call_id=item.call_id))

        if output_text == "" and action.output_text is not None:
            output_text = action.output_text
            logger.info("Output text found in action.output_text field instead of parts.")

        contents_for_save = [self._stringify_observation_item(item) for item in self.contents["input"]]
        action_record = {
            "text": output_text,
            "tool_calls": [{"id": c.call_id, "name": c.name, "args": c.args} for c in tool_calls],
        }
        self._record_conversation_history(
            step, contents_for_save, action_record, thinking_process, output_text, tool_calls, extra_info
        )
        self.contents["previous_response_id"] = action.id
        return list(tool_calls)

    def _append_tool_error(self, tool_call: ToolCall, error_msg: str) -> None:
        self._append_tool_message(tool_call.call_id, {"error": error_msg})

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
            {"type": "input_image", "image_url": image_url},
        ]
        for call in tool_calls:
            self._append_tool_message(call.call_id, output)
