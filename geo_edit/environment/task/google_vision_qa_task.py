from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from geo_edit.environment.task.vision_qa_task import ToolCall, VisionQATask
from geo_edit.utils.logger import setup_logger

logger = setup_logger(__name__)

class GoogleVisionQATask(VisionQATask):
    """vision qa task for Google GenAI"""

    def __init__(self, task_id: str, task_prompt: str, task_answer: str,
                 task_image_path: str | None, save_dir: Path | str,
                 tool_functions: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(task_id=task_id, task_prompt=task_prompt, task_answer=task_answer,
                         task_image_path=task_image_path, save_dir=save_dir,
                         tool_functions=tool_functions, **kwargs)
        if self.text_only:
            logger.info("Initializing GoogleVisionQATask in text only mode.")
            self.contents = [self.task_prompt]
        else:
            self.contents = [self.task_prompt, "Observation 0:", self.image_list[0]]

    def _stringify_observation_item(self, item: Any) -> Any:
        from google.genai import types

        if isinstance(item, Image.Image):
            image_id = id(item)
            if image_id in self.image_path_map:
                return {"image_data": self.image_path_map[image_id]}
            logger.warning("Image data not found in image_path_map.")
            return {"image_data": None}
        if isinstance(item, types.Content):
            parts = item.parts
            listofdict_parts = []
            for part in parts:
                dict_part = {
                    "text": part.text if part.text else None,
                    "thought": part.thought,
                    "function_call": {
                        "name": part.function_call.name,
                        "args": part.function_call.args,
                    }
                    if part.function_call
                    else None,
                    "function_response": {
                        "name": part.function_response.name,
                        "response": part.function_response.response,
                    }
                    if part.function_response
                    else None,
                }
                listofdict_parts.append(dict_part)
            item = {"parts": listofdict_parts, "role": item.role}
        if isinstance(item, str) and item.startswith("parts=") and " role=" in item:
            parts_str, role_part = item.split(" role=", 1)
            role_part = role_part.strip()
            if role_part.startswith("'") and role_part.endswith("'"):
                role_part = role_part[1:-1]
            return {
                "parts": parts_str[len("parts="):],
                "role": role_part,
            }
        return item

    def parse_action(self, step: int, action: Any, extra_info: Dict[str, Any]):
        """update task contents from action"""
        self.contents.append(action)
        thinking_process = ""
        output_text = ""
        tool_calls: List[ToolCall] = []
        for part in action.parts:
            if part.thought:
                thinking_process += part.text
            elif part.function_call:
                tool_calls.append(ToolCall(name=part.function_call.name, args=part.function_call.args))
            elif part.text:
                output_text += part.text
        contents_for_save = [self._stringify_observation_item(item) for item in self.contents]
        self._record_conversation_history(
            step, contents_for_save, self._stringify_observation_item(action),
            thinking_process, output_text, tool_calls, extra_info
        )
        return tool_calls

    def _append_tool_error(self, tool_call: ToolCall, error_msg: str) -> None:
        from google.genai import types

        self.contents.append(
            types.Content(role="tool", parts=[types.Part.from_function_response(name=tool_call.name, response={"error": error_msg})])
        )

    def _append_tool_image_for_calls(
        self,
        tool_calls: List[ToolCall],
        image: Image.Image,
        image_name: str,
        image_path: str,
        image_bytes: bytes,
        image_index: int,
    ) -> None:
        from google.genai import types

        if not tool_calls:
            logger.warning("No tool calls provided to append tool image.")
            return
        tool_call = tool_calls[-1]
        function_response_data = {
            "image_ref": {f"Observation {image_index}": image_name},
        }
        function_response_multimodal_data = types.FunctionResponsePart(
            inline_data=types.FunctionResponseBlob(
                mime_type="image/jpeg",
                display_name=image_name,
                data=image_bytes,
            )
        )
        self.contents.append(
            types.Content(
                role="tool",
                parts=[
                    types.Part.from_function_response(
                        name=tool_call.name,
                        response=function_response_data,
                        parts=[function_response_multimodal_data],
                    )
                ],
            )
        )

    def append_prompt(self, prompt: str) -> None:
        self.contents.append(prompt)
