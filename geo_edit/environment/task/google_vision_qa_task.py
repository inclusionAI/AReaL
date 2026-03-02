from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

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

    def append_assistant_message(self, text: str) -> None:
        """Append an assistant message to contents."""
        from google.genai import types
        self.contents.append(
            types.Content(role="model", parts=[types.Part.from_text(text=text)])
        )

    def _build_sft_messages(self) -> List[Dict[str, Any]]:
        """Build SFT-format messages for training.

        Combines observation (user/tool/image messages) with action records (assistant messages with tool_calls).
        """
        import json

        if not self.conversation_history:
            return []

        messages = []
        added_items = set()

        for idx, record in enumerate(self.conversation_history):
            step = record["step"]
            observation = record.get("observation", [])

            # Add observation items (user messages, images, tool responses)
            for obs_idx, item in enumerate(observation):
                item_id = (step, obs_idx, json.dumps(item, sort_keys=True) if isinstance(item, dict) else str(item))
                if item_id in added_items:
                    continue

                if isinstance(item, dict):
                    if "parts" in item and "role" in item:
                        # Stringified Content - only add if it's NOT an assistant/model message
                        # (assistant messages come from action_record)
                        role = item.get("role")
                        if role not in ("model", "assistant"):
                            msg = self._convert_google_content_to_sft(item)
                            if msg:
                                messages.append(msg)
                                added_items.add(item_id)
                    elif "image_data" in item:
                        image_path = item["image_data"]
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
                            added_items.add(item_id)
                elif isinstance(item, str):
                    if messages and messages[-1]["role"] == "user" and isinstance(messages[-1]["content"], list):
                        messages[-1]["content"].append({"type": "text", "text": item})
                    else:
                        messages.append({"role": "user", "content": item})
                    added_items.add(item_id)

            # Add assistant message from action_record
            action_record = record.get("action", {})
            thinking_process = record.get("thinking_process", "")
            output_text = record.get("output_text", "")
            function_calls = record.get("function_call")

            # Parse action_record which is a stringified Content
            if isinstance(action_record, dict) and "parts" in action_record:
                assistant_msg = self._convert_google_content_to_sft(action_record)
                if assistant_msg:
                    # Override content with thinking + output if available
                    content_parts = []
                    if thinking_process:
                        content_parts.append(thinking_process)
                    if output_text:
                        content_parts.append(output_text)
                    if content_parts:
                        assistant_msg["content"] = "\n".join(content_parts)

                    messages.append(assistant_msg)

        return messages

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
                        "arguments": json.dumps(function_call["args"]) if isinstance(function_call["args"], dict) else str(function_call["args"])
                    }
                })

            # Handle function_response (tool role)
            function_response = part.get("function_response")
            if function_response and sft_role == "tool":
                return {
                    "role": "tool",
                    "tool_call_id": f"call_{function_response['name']}",
                    "content": json.dumps(function_response["response"])
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
