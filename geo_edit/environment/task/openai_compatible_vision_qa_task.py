from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional
from PIL import Image
from geo_edit.environment.task.vision_qa_task import ToolCall, VisionQATask
from geo_edit.utils.image_utils import image_to_data_url
from geo_edit.utils.logger import setup_logger

logger = setup_logger(__name__)

class OpenAICompatibleVisionQATask(VisionQATask):
    """Unified Vision QA task for OpenAI-compatible APIs.

    Supports:
    - OpenAI (Responses API)
    - vLLM (OpenAI-compatible, supports both responses and chat_completions)
    - SGLang (OpenAI-compatible, chat_completions only)

    Key differences handled:
    - api_mode: "responses" vs "chat_completions" affects message format
    - Response parsing: OpenAI extracts from response object, vLLM/SGLang use <think>/<answer> tags
    - previous_response_id: Only supported by OpenAI
    """

    _THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
    _ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)

    def __init__(
        self,
        task_id: str,
        task_prompt: str,
        task_answer: str,
        task_image_path: str | None,
        save_dir: Path | str,
        tool_functions: Optional[Dict[str, Callable[..., Image.Image | str]]] = None,
        model_type: Literal["openai", "vllm", "sglang"] = "openai",
        api_mode: Literal["responses", "chat_completions"] = "responses",
        **kwargs,
    ):
        super().__init__(
            task_id=task_id,
            task_prompt=task_prompt,
            task_answer=task_answer,
            task_image_path=task_image_path,
            save_dir=save_dir,
            tool_functions=tool_functions,
            model_type=model_type,
            api_mode=api_mode,
            **kwargs,
        )

        # Determine content format based on api_mode
        # responses API: uses "input_text", "input_image"
        # chat_completions API: uses "text", "image_url"
        self._use_responses_format = (api_mode == "responses")

        # Initialize contents structure
        if self._use_responses_format:
            # OpenAI/vLLM responses API format
            self.contents: Dict[str, Any] = {"input": [], "previous_response_id": None}
        else:
            # SGLang/vLLM chat_completions format
            self.contents: List[Dict[str, Any]] = []

        self._append_initial_observation()

    def _append_initial_observation(self) -> None:
        """Append initial user message with prompt and image."""
        if self._use_responses_format:
            content = [{"type": "input_text", "text": self.task_prompt}]
            if self.image_list:
                image = self.image_list[0]
                image_url = image_to_data_url(image)
                if self.task_image_path:
                    self.image_url_map[image_url] = self.task_image_path
                content.extend([
                    {"type": "input_text", "text": "Observation 0:"},
                    {"type": "input_image", "image_url": image_url, "detail": "auto"},
                ])
            self.contents["input"].append({"role": "user", "content": content})
        else:
            content = [{"type": "text", "text": self.task_prompt}]
            if self.image_list:
                image = self.image_list[0]
                image_url = image_to_data_url(image)
                if self.task_image_path:
                    self.image_url_map[image_url] = self.task_image_path
                content.extend([
                    {"type": "text", "text": "Observation 0:"},
                    {"type": "image_url", "image_url": {"url": image_url, "detail": "auto"}},
                ])
            self.contents.append({"role": "user", "content": content})

    def _stringify_observation_item(self, item: Any) -> Any:
        """Convert observation item to JSON-serializable format."""
        if not isinstance(item, dict):
            return item

        if self._use_responses_format:
            return self._stringify_responses_format(item)
        else:
            return self._stringify_chat_completions_format(item)

    def _stringify_responses_format(self, item: Dict[str, Any]) -> Any:
        """Stringify item for responses API format."""
        if item.get("type") == "function_call_output":
            output = item.get("output")
            if isinstance(output, list):
                parts: List[Dict[str, str]] = []
                for part in output:
                    if part.get("type") == "input_image":
                        image_url = part["image_url"]
                        image_path = self.image_url_map.get(image_url, "")
                        parts.append({"type": "input_image", "image_path": image_path or "<omitted>"})
                    else:
                        parts.append(part)
                output = parts
            return {"type": "function_call_output", "call_id": item.get("call_id"), "output": output}

        if item.get("type") == "function_call":
            return {
                "type": "function_call",
                "call_id": item.get("call_id"),
                "name": item.get("name"),
                "arguments": item.get("arguments"),
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
                    output["content"].append({"type": "input_image", "image_path": image_path or "<omitted>"})
                else:
                    output["content"].append(part)
            return output
        output["content"] = content
        return output

    def _stringify_chat_completions_format(self, item: Dict[str, Any]) -> Any:
        """Stringify item for chat_completions API format."""
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
                    output["content"].append({"type": "image_url", "image_path": image_path})
                else:
                    output["content"].append(part)
            return output

        output["content"] = content
        return output

    def parse_action(self, step: int, action: Any, extra_info: Dict[str, int | float | str | None]):
        """Parse action from model response."""
        if self._use_responses_format:
            return self._parse_responses_action(step, action, extra_info)
        else:
            return self._parse_chat_completions_action(step, action, extra_info)

    def _parse_responses_action(self, step: int, action: Any, extra_info: Dict[str, int | float | str | None]):
        """Parse action from responses API format."""
        contents_for_save = [self._stringify_observation_item(item) for item in self.contents["input"]]

        if not action.output:
            raise ValueError("Response contained no output.")

        raw_text = getattr(action, "output_text", "") or ""
        tool_calls: List[ToolCall] = []
        tool_call_records: List[Dict[str, Any]] = []
        tool_call_items: List[Dict[str, str]] = []
        thinking_process = ""
        output_text = ""

        for item in action.output:
            if item.type == "reasoning":
                # OpenAI reasoning format
                if hasattr(item, "summary") and item.summary:
                    for summary_item in item.summary:
                        if hasattr(summary_item, "text") and summary_item.text:
                            thinking_process += summary_item.text
            elif item.type == "message":
                # OpenAI message format
                for part in item.content:
                    if part.type == "output_text":
                        output_text += part.text
            elif item.type == "function_call":
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

                call_id = getattr(item, "call_id", None) or getattr(item, "id", None)
                tool_calls.append(ToolCall(name=item.name, args=arguments, call_id=call_id))
                tool_call_records.append({"id": call_id, "name": item.name, "args": arguments})
                tool_call_items.append({
                    "type": "function_call",
                    "name": item.name,
                    "arguments": arguments_payload,
                    "call_id": call_id,
                })

        # For vLLM: extract thinking/answer from tags if not found in response object
        if self.model_type == "vllm" and not thinking_process and not output_text:
            thinking_parts = [m.group(1).strip() for m in self._THINK_PATTERN.finditer(raw_text)]
            thinking_process = "\n".join(part for part in thinking_parts if part)
            answer_parts = [m.group(1).strip() for m in self._ANSWER_PATTERN.finditer(raw_text)]
            output_text = "\n".join(part for part in answer_parts if part)

        # Fallback to output_text if no message content found
        if not output_text and raw_text and not tool_calls:
            output_text = raw_text

        if not tool_calls and not output_text:
            logger.error("Response content: %s", raw_text)
            raise ValueError("Response contained no tool call or final answer.")

        if tool_calls and output_text:
            logger.warning("Response contained tool call and final answer; keeping the answer and ignoring tool calls.")
            tool_calls = []
            tool_call_records = []
            tool_call_items = []

        if tool_call_items:
            self.contents["input"].extend(tool_call_items)

        # Update previous_response_id for OpenAI
        if self.model_type == "openai" and hasattr(action, "id"):
            self.contents["previous_response_id"] = action.id

        action_record = {"text": output_text, "tool_calls": tool_call_records}
        self._record_conversation_history(step, contents_for_save, action_record, thinking_process, output_text, tool_calls, extra_info)
        return list(tool_calls)

    def _parse_chat_completions_action(self, step: int, action: Any, extra_info: Dict[str, int | float | str | None]):
        """Parse action from chat_completions API format."""
        contents_for_save = [self._stringify_observation_item(item) for item in self.contents]

        if not action.choices:
            raise ValueError("Response contained no choices.")

        message = action.choices[0].message
        content = message.content or ""
        native_tool_calls = message.tool_calls

        # Extract structured reasoning from OpenAI/matrixllm response
        thinking_process = ""
        if hasattr(message, "reasoning_content") and message.reasoning_content:
            thinking_process = message.reasoning_content

        # Fallback: extract thinking from <think> tags
        if not thinking_process:
            thinking_parts = [m.group(1).strip() for m in self._THINK_PATTERN.finditer(content)]
            thinking_process = "\n".join(part for part in thinking_parts if part)

        answer_parts = [m.group(1).strip() for m in self._ANSWER_PATTERN.finditer(content)]
        output_text = "\n".join(part for part in answer_parts if part)

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
            logger.error("Response content: %s", content)
            raise ValueError("Response contained no tool call or final answer.")

        if tool_calls and output_text:
            logger.warning("Response contained tool call and final answer; keeping the answer and ignoring tool calls.")
            tool_calls = []
            tool_call_records = []

        # Append assistant message
        assistant_message: Dict[str, Any] = {"role": "assistant", "content": content}
        if native_tool_calls:
            assistant_message["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in native_tool_calls
            ]
        self.contents.append(assistant_message)

        action_record = {"text": output_text, "tool_calls": tool_call_records}
        self._record_conversation_history(step, contents_for_save, action_record, thinking_process, output_text, tool_calls, extra_info)
        return list(tool_calls)

    def _append_tool_result(self, call_id: str, payload: Dict[str, str | Dict[str, str]] | List[Dict[str, str]]) -> None:
        """Append tool result to contents."""
        if self._use_responses_format:
            output = payload if isinstance(payload, list) else json.dumps(payload, ensure_ascii=True)
            self.contents["input"].append({"type": "function_call_output", "call_id": call_id, "output": output})
        else:
            text = json.dumps(payload, ensure_ascii=True) if isinstance(payload, dict) else json.dumps(payload)
            self.contents.append({"role": "tool", "tool_call_id": call_id, "content": text})

    def _append_tool_error(self, tool_call: ToolCall, error_msg: str) -> None:
        self._append_tool_result(tool_call.call_id, {"error": error_msg})

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

        if self._use_responses_format:
            output = [
                {"type": "input_text", "text": f"Observation {image_index}:"},
                {"type": "input_image", "image_url": image_url, "detail": "auto"},
            ]
            for call in tool_calls:
                self._append_tool_result(call.call_id, output)
        else:
            # For chat_completions, append tool result then user message with image
            payload = {"image_ref": {f"Observation {image_index}": image_name}}
            for call in tool_calls:
                self._append_tool_result(call.call_id, payload)
            content = [
                {"type": "text", "text": f"Observation {image_index}:"},
                {"type": "image_url", "image_url": {"url": image_url, "detail": "auto"}},
            ]
            self.contents.append({"role": "user", "content": content})

    def _append_tool_text_for_calls(self, tool_calls: List[ToolCall], text: str) -> None:
        payload = {"analysis": text}
        for call in tool_calls:
            self._append_tool_result(call.call_id, payload)

    def append_prompt(self, prompt: str) -> None:
        if self.api_mode == "responses":
            self.contents["input"].append({"role": "user", "content": [{"type": "input_text", "text": prompt}]})
        else:
            self.contents.append({"role": "user", "content": [{"type": "text", "text": prompt}]})

    def append_assistant_message(self, text: str) -> None:
        """Append an assistant message to contents."""
        if self.api_mode == "responses":
            self.contents["input"].append({
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}]
            })
        else:
            self.contents.append({"role": "assistant", "content": text})

    def _build_sft_messages(self) -> List[Dict[str, Any]]:
        """Build SFT-format messages for training.

        Combines observation (user/tool messages) with action records (assistant messages with tool_calls).
        """
        if not self.conversation_history:
            return []

        messages = []
        added_observations = set()

        # Process each step in conversation history
        for idx, record in enumerate(self.conversation_history):
            step = record["step"]
            observation = record.get("observation", [])

            # Add observation messages only once (avoid duplicates from multiple steps)
            for obs_idx, item in enumerate(observation):
                obs_id = (step, obs_idx, json.dumps(item, sort_keys=True) if isinstance(item, dict) else str(item))
                if obs_id not in added_observations and isinstance(item, dict):
                    msg = self._convert_observation_item_to_sft(item)
                    if msg and msg.get("role") in ("user", "tool"):
                        messages.append(msg)
                        added_observations.add(obs_id)

            # Add assistant message from action record
            action_record = record.get("action", {})
            thinking_process = record.get("thinking_process", "")
            output_text = record.get("output_text", "")
            function_calls = record.get("function_call")

            assistant_msg = {"role": "assistant"}

            # Build content from thinking + output
            content_parts = []
            if thinking_process:
                content_parts.append(thinking_process)
            if output_text:
                content_parts.append(output_text)

            assistant_msg["content"] = "\n".join(content_parts) if content_parts else (action_record.get("text", ""))

            # Add tool_calls from action_record
            if function_calls:
                tool_calls = []
                tool_call_records = action_record.get("tool_calls", [])

                for func_name, func_args in function_calls:
                    matching_call = next((tc for tc in tool_call_records if tc.get("name") == func_name), None)

                    if matching_call:
                        tool_calls.append({
                            "id": matching_call.get("id", f"call_{func_name}"),
                            "type": "function",
                            "function": {
                                "name": func_name,
                                "arguments": matching_call.get("arguments", json.dumps(func_args) if isinstance(func_args, dict) else str(func_args))
                            }
                        })
                    else:
                        tool_calls.append({
                            "id": f"call_{step}_{func_name}",
                            "type": "function",
                            "function": {
                                "name": func_name,
                                "arguments": json.dumps(func_args) if isinstance(func_args, dict) else str(func_args)
                            }
                        })

                assistant_msg["tool_calls"] = tool_calls

            messages.append(assistant_msg)

        return messages

    def _convert_observation_item_to_sft(self, item: Dict[str, Any]) -> Dict[str, Any] | None:
        """Convert a single observation item to SFT format."""
        role = item.get("role")
        if not role:
            return None

        msg = {"role": role}

        # Handle content
        content = item.get("content")
        if isinstance(content, str):
            msg["content"] = content
        elif isinstance(content, list):
            formatted_content = []
            for part in content:
                if isinstance(part, dict):
                    part_type = part.get("type")
                    # Handle image_path (already stringified by _stringify_observation_item)
                    if part_type == "image_url" and "image_path" in part:
                        image_path = part["image_path"]
                        if image_path and image_path != "<omitted>":
                            formatted_content.append({
                                "type": "image_url",
                                "image_url": {"url": f"file://{image_path}"}
                            })
                        else:
                            formatted_content.append({"type": "text", "text": "[IMAGE]"})
                    # Handle image_url that hasn't been stringified yet
                    elif part_type == "image_url" and "image_url" in part:
                        image_url_data = part["image_url"]
                        if isinstance(image_url_data, dict):
                            url = image_url_data.get("url", "")
                        else:
                            url = image_url_data
                        # Try to map URL to path
                        image_path = self.image_url_map.get(url, "")
                        if image_path:
                            formatted_content.append({
                                "type": "image_url",
                                "image_url": {"url": f"file://{image_path}"}
                            })
                        else:
                            # Keep original URL if no mapping found
                            formatted_content.append(part)
                    elif part_type in ("text", "input_text", "output_text"):
                        text = part.get("text", "")
                        if text:
                            formatted_content.append({"type": "text", "text": text})
                    else:
                        formatted_content.append(part)
                else:
                    formatted_content.append({"type": "text", "text": str(part)})

            msg["content"] = formatted_content if formatted_content else ""
        else:
            msg["content"] = str(content) if content else ""

        # Handle tool_calls (for assistant messages)
        if "tool_calls" in item:
            msg["tool_calls"] = item["tool_calls"]

        # Handle tool_call_id (for tool messages)
        if "tool_call_id" in item:
            msg["tool_call_id"] = item["tool_call_id"]

        # Handle name (for tool messages)
        if "name" in item:
            msg["name"] = item["name"]

        return msg
