from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from PIL import Image

from geo_edit.environment.task.vision_qa_task import ToolCall, VisionQATask
from geo_edit.utils.logger import setup_logger

logger = setup_logger(__name__)


class GoogleVisionQATask(VisionQATask):
    """vision qa task for Google GenAI"""

    def __init__(
        self,
        task_id: str,
        task_prompt: str,
        task_answer: str,
        task_image_path: str | None,
        save_dir: Path | str,
        tool_functions: Optional[Dict[str, Callable[..., Image.Image | str]]] = None,
        **kwargs,
    ):
        super().__init__(
            task_id=task_id,
            task_prompt=task_prompt,
            task_answer=task_answer,
            task_image_path=task_image_path,
            save_dir=save_dir,
            tool_functions=tool_functions,
            api_mode="responses",  # Google uses its own native API, api_mode is ignored
            **kwargs,
        )
        self._init_contents()

    def _init_contents(self) -> None:
        """Initialize contents to initial state."""
        if self.text_only:
            logger.info("Initializing GoogleVisionQATask in text only mode.")
            self.contents = [self.task_prompt]
        else:
            self.contents = ["Observation 0:", self.image_list[0], self.task_prompt]

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the task to initial state."""
        obs, info = super().reset(seed=seed, options=options)
        self._init_contents()
        return self._get_observation(), info

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
                "parts": parts_str[len("parts=") :],
                "role": role_part,
            }
        return item

    def parse_action(self, step: int, action: Any, extra_info: Dict[str, int | float | str | None]):
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
        self._record_conversation_history(step, contents_for_save, self._stringify_observation_item(action), thinking_process, output_text, tool_calls, extra_info)
        return tool_calls

    def _append_tool_error(self, tool_call: ToolCall, error_msg: str) -> None:
        from google.genai import types

        self.contents.append(types.Content(role="tool", parts=[types.Part.from_function_response(name=tool_call.name, response={"error": error_msg})]))

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

    def _append_tool_text_for_calls(self, tool_calls: List[ToolCall], text: str) -> None:
        from google.genai import types

        for call in tool_calls:
            self.contents.append(
                types.Content(
                    role="tool",
                    parts=[
                        types.Part.from_function_response(
                            name=call.name,
                            response={"analysis": text},
                        )
                    ],
                )
            )

    def append_prompt(self, prompt: str) -> None:
        self.contents.append(prompt)

    def append_system_prompt(self, prompt: str) -> None:
        """Append a system instruction to contents."""
        from google.genai import types
        # For Google API, system instructions are typically in config, but we can add as user message with special formatting
        self.contents.append(
            types.Content(role="user", parts=[types.Part.from_text(text=f"[SYSTEM INSTRUCTION]\n{prompt}")])
        )

    def append_assistant_message(self, text: str) -> None:
        """Append an assistant message to contents."""
        from google.genai import types
        self.contents.append(
            types.Content(role="model", parts=[types.Part.from_text(text=text)])
        )

    def _build_sft_messages(self) -> List[Dict[str, Any]]:
        """Build SFT-format messages for training.

        Uses self.contents which has the complete conversation including phase 1 reasoning
        and tool_calls information (added by parse_action).
        """
        import json

        messages = []

        # Convert all contents to SFT format
        for item in self.contents:
            if isinstance(item, str):
                # Text message (user prompt or observation label)
                if messages and messages[-1]["role"] == "user":
                    # Append to last user message
                    if isinstance(messages[-1]["content"], str):
                        messages[-1]["content"] = [{"type": "text", "text": messages[-1]["content"]}]
                    if isinstance(messages[-1]["content"], list):
                        messages[-1]["content"].append({"type": "text", "text": item})
                else:
                    messages.append({"role": "user", "content": item})
            elif isinstance(item, Image.Image):
                # Image
                image_id = id(item)
                image_path = self.image_path_map.get(image_id, "")
                if image_path:
                    if messages and messages[-1]["role"] == "user":
                        if isinstance(messages[-1]["content"], str):
                            messages[-1]["content"] = [{"type": "text", "text": messages[-1]["content"]}]
                        messages[-1]["content"].append({
                            "type": "image_url",
                            "image_url": {"url": f"file://{image_path}"}
                        })
                    else:
                        messages.append({
                            "role": "user",
                            "content": [{"type": "image_url", "image_url": {"url": f"file://{image_path}"}}]
                        })
            else:
                # Google Content object (assistant or tool message)
                from google.genai import types
                if isinstance(item, types.Content):
                    msg = self._convert_google_content_to_sft_from_object(item)
                    if msg:
                        messages.append(msg)

        return messages

    def _convert_google_content_to_sft_from_object(self, content: Any) -> Dict[str, Any] | None:
        """Convert Google Content object to SFT format."""
        import json
        from google.genai import types

        if not isinstance(content, types.Content):
            return None

        role = content.role
        if not role:
            return None

        # Convert role: "model" -> "assistant"
        sft_role = "assistant" if role == "model" else role

        parts = content.parts
        if not parts:
            return None

        msg = {"role": sft_role, "content": []}
        tool_calls = []

        for part in parts:
            # Handle text
            if part.text and not part.thought:
                msg["content"].append({"type": "text", "text": part.text})

            # Handle function_call
            if part.function_call:
                tool_calls.append({
                    "id": f"call_{part.function_call.name}",
                    "type": "function",
                    "function": {
                        "name": part.function_call.name,
                        "arguments": json.dumps(dict(part.function_call.args), ensure_ascii=False) if part.function_call.args else "{}"
                    }
                })

            # Handle function_response (tool role)
            if part.function_response and sft_role == "tool":
                return {
                    "role": "tool",
                    "tool_call_id": f"call_{part.function_response.name}",
                    "content": json.dumps(dict(part.function_response.response), ensure_ascii=False) if part.function_response.response else "{}"
                }

        # Add tool_calls if present
        if tool_calls:
            msg["tool_calls"] = tool_calls

        # Simplify content if it's just one text part
        if len(msg["content"]) == 1 and msg["content"][0]["type"] == "text":
            msg["content"] = msg["content"][0]["text"]
        elif not msg["content"]:
            msg["content"] = ""

        return msg

    def _convert_google_content_to_sft(self, content_dict: Dict[str, Any]) -> Dict[str, Any] | None:
        """Convert stringified Google Content to SFT format."""
        import json

        role = content_dict.get("role")
        if not role:
            return None

        # Convert role: "model" -> "assistant"
        sft_role = "assistant" if role == "model" else role

        parts = content_dict.get("parts", [])
        if not parts:
            return None

        msg = {"role": sft_role, "content": []}
        tool_calls = []

        for part in parts:
            if not isinstance(part, dict):
                continue

            # Handle text
            text = part.get("text")
            thought = part.get("thought")
            if text and not thought:
                msg["content"].append({"type": "text", "text": text})

            # Handle function_call
            function_call = part.get("function_call")
            if function_call:
                tool_calls.append({
                    "id": f"call_{function_call['name']}",
                    "type": "function",
                    "function": {
                        "name": function_call["name"],
                        "arguments": json.dumps(function_call["args"], ensure_ascii=False) if isinstance(function_call["args"], dict) else str(function_call["args"])
                    }
                })

            # Handle function_response (tool role)
            function_response = part.get("function_response")
            if function_response and sft_role == "tool":
                return {
                    "role": "tool",
                    "tool_call_id": f"call_{function_response['name']}",
                    "content": json.dumps(function_response["response"], ensure_ascii=False)
                }

        # Add tool_calls if present
        if tool_calls:
            msg["tool_calls"] = tool_calls

        # Simplify content if it's just one text part
        if len(msg["content"]) == 1 and msg["content"][0]["type"] == "text":
            msg["content"] = msg["content"][0]["text"]
        elif not msg["content"]:
            msg["content"] = ""

        return msg
