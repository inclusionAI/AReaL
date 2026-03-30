from __future__ import annotations

import copy
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
from PIL import Image

from geo_edit.environment.task.base import AbstractVLMTask
from geo_edit.prompts import TOOL_EXECUTION_FAILURE_PROMPT, TOOL_EXECUTION_SUCCESS_PROMPT
from geo_edit.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class ToolCall:
    name: str
    args: Dict[str, str | int]
    call_id: Optional[str] = None


class VisionQATask(AbstractVLMTask, gym.Env):
    """Base vision QA task shared by API providers.

    Implements Gymnasium interface for RL training while maintaining backwards
    compatibility with existing inference scripts.

    Gym Interface:
        - reset(): Resets task state and returns (observation, info)
        - step(action): Executes action and returns (obs, reward, terminated, truncated, info)

    Legacy Interface (still supported):
        - parse_action(): Parse model response into tool calls
        - update_observation_from_action(): Execute tool calls and update state
    """

    # Gymnasium metadata
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        task_id: str,
        task_prompt: str,
        task_answer: str,
        task_image_path: str | None,
        save_dir: Path | str,
        model_type: Literal["google", "openai", "vllm", "sglang"] = "openai",
        api_mode: Literal["responses", "chat_completions"] = "responses",
        tool_functions: Optional[Dict[str, Callable[..., Image.Image | str]]] = None,
        tool_return_types: Optional[Dict[str, str]] = None,
        max_steps: int = 10,
        **kwargs,
    ):
        AbstractVLMTask.__init__(self, task_id)
        gym.Env.__init__(self)

        self.task_prompt = task_prompt
        self.task_answer = task_answer
        self.task_image_path = task_image_path
        self.model_type = model_type
        self.api_mode = api_mode
        self.tool_functions = tool_functions or {}
        self.tool_return_types = tool_return_types or {}
        self.state = True
        self.options = kwargs["options"] if "options" in kwargs else None
        self.meta_info_extra = kwargs.get("meta_info_extra")
        self.max_steps = max_steps
        self._step_count = 0

        self.image_path_map: Dict[int, str] = {}
        self.image_url_map: Dict[str, str] = {}  # For OpenAI/VLLM: maps data URL to file path
        self.image_list: List[Image.Image] = []
        if self.task_image_path:
            image = Image.open(self.task_image_path).convert("RGB")
            self.image_list.append(image)
            self.image_path_map[id(image)] = self.task_image_path

        # Store initial image state for reset
        self._initial_image_list = [img.copy() for img in self.image_list]
        self._initial_image_path_map = dict(self.image_path_map)

        self.text_only = kwargs.get("text_only", False) or not self.image_list
        self.conversation_history: List[Dict[str, Any]] = []

        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.output_jsonl_path = os.path.join(self.save_dir, "output.jsonl")
        self.extra_info_jsonl_path = os.path.join(self.save_dir, "extra_info.jsonl")
        self.meta_info_jsonl_path = os.path.join(self.save_dir, "meta_info.jsonl")
        self.trajectory_jsonl_path = os.path.join(self.save_dir, "trajectory.jsonl")
        self.image_save_dir = os.path.join(self.save_dir, "images")
        os.makedirs(self.image_save_dir, exist_ok=True)

        # Gymnasium spaces - contents is API-specific so we use a flexible Dict space
        self.observation_space = spaces.Dict({
            "contents": spaces.Sequence(spaces.Text(max_length=100000)),
            "step": spaces.Discrete(max_steps + 1),
        })
        self.action_space = spaces.Text(max_length=100000)

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

    def _append_tool_text_for_calls(self, tool_calls: List[ToolCall], text: str) -> None:
        raise NotImplementedError

    def append_prompt(self, prompt: str) -> None:
        raise NotImplementedError

    def _build_sft_messages(self) -> List[Dict[str, Any]]:
        """Build SFT-format messages from conversation history.

        Should be implemented by subclasses to convert internal representation
        to standard chat format with roles (system/user/assistant/tool).
        """
        raise NotImplementedError

    def _record_conversation_history(
        self,
        step: int,
        contents_for_save: List[Any],
        action_record: Any,
        thinking_process: str,
        output_text: str,
        tool_calls: List[ToolCall],
        extra_info: Dict[str, int | float | str | None],
    ) -> None:
        """Record a step in conversation history."""
        self.conversation_history.append(
            {
                "step": step,
                "observation": contents_for_save,
                "action": action_record,
                "thinking_process": thinking_process,
                "output_text": output_text,
                "function_call": [(c.name, c.args) for c in tool_calls] if tool_calls else None,
                "extra_info": extra_info,
            }
        )

    def _save_image(self, image: Image.Image) -> Tuple[int, str, str, bytes]:
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

    def _check_function_calls_legal(self, tool_calls: List[ToolCall]) -> Tuple[bool, Optional[str], Optional[str], Optional[int]]:
        if not tool_calls:
            logger.warning("No function calls found in the action.")
            return True, None, None, None

        first = tool_calls[0]
        if "image_index" not in first.args:
            return False, "Function call must include image_index.", None, None
        expected_index = first.args["image_index"]
        text_tool_names = {c.name for c in tool_calls if self.tool_return_types.get(c.name) == "text"}
        image_tool_names = {c.name for c in tool_calls if self.tool_return_types.get(c.name, "image") == "image"}
        if text_tool_names and image_tool_names:
            return (
                False,
                "Cannot mix image-producing tools with text-analysis tools in one action.",
                None,
                None,
            )
        has_crop = (first.name == "image_crop") or any(c.name == "image_crop" for c in tool_calls[1:])

        for c in tool_calls[1:]:
            if "image_index" not in c.args:
                return False, "Function call must include image_index.", None, None
            if c.args["image_index"] != expected_index:
                logger.warning("Inconsistent image_index values: expected %s, got %s", expected_index, c.args["image_index"])
                return False, "Function call image_index values are inconsistent in the same action.", None, None
            if has_crop and c.name != "image_crop":
                logger.warning("Inconsistent function call names: expected %s, got %s", "image_crop", c.name)
                return False, "Function call names are inconsistent in the same action.", None, None

        return True, None, ("image_crop" if has_crop else None), expected_index

    @staticmethod
    def _is_error_text(result: str) -> bool:
        return result.strip().lower().startswith("error:")

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
        is_legal, illegal_reason, expected_name, expected_index = self._check_function_calls_legal(tool_calls)
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
            call_results: List[Tuple[str, ToolCall, Image.Image | str]] = []
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
                        elif isinstance(result, str):
                            if self._is_error_text(result):
                                call_results.append(("error", call, (f"Function call {call.name} with args {call.args} failed with error: {result}")))
                            else:
                                call_results.append(("text", call, result))
                        else:
                            call_results.append(("error", call, (f"Function call {call.name} with args {call.args} failed with error: {result}")))
                    except Exception as exc:
                        call_results.append(("error", call, (f"Function call {call.name} with args {call.args} failed with error: {exc}")))
                else:
                    call_results.append(("error", call, f"Unknown function {call.name}"))

            for result_type, call, payload in call_results:
                if result_type == "image":
                    image_index, image_name, image_path, image_bytes = self._save_image(payload)
                    self._append_tool_image_for_calls(
                        [call], payload, image_name, image_path, image_bytes, image_index,
                    )
                elif result_type == "text":
                    self._append_tool_text_for_calls([call], payload)
                else:
                    logger.warning("Tool execution failed: %s", payload)
                    self._append_tool_error(call, str(payload))
            had_error = any(result_type == "error" for result_type, _, _ in call_results)
            self.append_prompt(TOOL_EXECUTION_FAILURE_PROMPT if had_error else TOOL_EXECUTION_SUCCESS_PROMPT)
            return

        all_text_tools = all(self.tool_return_types.get(call.name) == "text" for call in tool_calls)
        if all_text_tools:
            success_count = 0
            error_result: List[Tuple[ToolCall, str]] = []
            for call in tool_calls:
                logger.info(
                    "Processing text-analysis tool call: %s with args: %s",
                    call.name,
                    call.args,
                )
                if call.name not in self.tool_functions:
                    error_result.append((call, f"Unknown function {call.name}"))
                    continue
                function_to_call = self.tool_functions[call.name]
                try:
                    result = function_to_call(self.image_list, **call.args)
                    if isinstance(result, str) and not self._is_error_text(result):
                        self._append_tool_text_for_calls([call], result)
                        success_count += 1
                    else:
                        error_msg = result if isinstance(result, str) else f"Unsupported tool output type: {type(result)}"
                        error_result.append(
                            (
                                call,
                                f"Function call {call.name} with args {call.args} failed with error: {error_msg}",
                            )
                        )
                except Exception as exc:
                    error_result.append(
                        (
                            call,
                            f"Function call {call.name} with args {call.args} failed with error: {exc}",
                        )
                    )

            for call, error_msg in error_result:
                logger.warning("Tool execution failed: %s", error_msg)
                self._append_tool_error(call, error_msg)
            had_error = bool(error_result) or success_count == 0
            self.append_prompt(TOOL_EXECUTION_FAILURE_PROMPT if had_error else TOOL_EXECUTION_SUCCESS_PROMPT)
            return

        for call in tool_calls:
            logger.info("Processing function call: %s with args: %s", call.name, call.args)
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
                        dynamic_image_list[target_index] = self.image_list[target_index].copy()
                try:
                    result = function_to_call(dynamic_image_list, **call.args)
                    dynamic_image = result
                    if dynamic_image_index is None and "image_index" in call.args:
                        dynamic_image_index = call.args["image_index"]
                except Exception as exc:
                    logger.warning("Tool execution raised exception: %s(%s) -> %s", call.name, call.args, exc)
                    error_result.append((call, (f"Function call {call.name} with args {call.args} failed with error: {exc}")))
            else:
                error_result.append((call, f"Unknown function {call.name}"))

        if isinstance(dynamic_image, Image.Image):
            image_index, image_name, image_path, image_bytes = self._save_image(dynamic_image)
            self._append_tool_image_for_calls(
                tool_calls,
                dynamic_image,
                image_name,
                image_path,
                image_bytes,
                image_index,
            )
        else:
            for call, error_msg in error_result:
                logger.warning("Tool execution failed: %s", error_msg)
                self._append_tool_error(call, error_msg)
        had_error = bool(error_result) or not isinstance(dynamic_image, Image.Image)
        self.append_prompt(TOOL_EXECUTION_FAILURE_PROMPT if had_error else TOOL_EXECUTION_SUCCESS_PROMPT)

    def save_trajectory(self) -> Dict[str, Any]:
        """save the trajectory to jsonl files"""
        extra_info_list = []
        function_call_total_count = 0
        function_call_each_count = {}
        function_call_per_step = []
        tokens_total_per_step = []
        tokens_input_per_step = []
        tokens_output_per_step = []
        tokens_output_total = 0

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
            extra_info = record.get("extra_info", {})
            tokens_used = extra_info.get("tokens_used")
            tokens_input = extra_info.get("tokens_input")
            tokens_output = extra_info.get("tokens_output")
            tokens_thoughts = extra_info.get("tokens_thoughts", None)

            tokens_total_per_step.append(tokens_used)

            tokens_input_per_step.append(tokens_input)

            if isinstance(tokens_output, (int, float)) and isinstance(tokens_thoughts, (int, float)):
                tokens_output_per_step.append(tokens_output + tokens_thoughts)
            elif isinstance(tokens_output, (int, float)):
                tokens_output_per_step.append(tokens_output)
            else:
                tokens_output_per_step.append(None)

        tokens_used_total = tokens_total_per_step[-1]

        # Token calculation based on api_mode
        # - responses API: total_tokens is cumulative, input = total - output
        # - chat_completions API: prompt_tokens is cumulative input, total = input + output
        if self.model_type == "google":
            # Google uses its own native API
            tokens_output_total = sum(t for t in tokens_output_per_step if isinstance(t, (int, float)))
            tokens_input_total = None
            if isinstance(tokens_used_total, (int, float)) and isinstance(tokens_output_total, (int, float)):
                tokens_input_total = float(tokens_used_total) - float(tokens_output_total)
                if tokens_input_total < 0:
                    raise ValueError("Calculated tokens_input_total is negative.")
        elif self.api_mode == "chat_completions":
            # Chat Completions API: prompt_tokens is cumulative input tokens
            tokens_output_total = sum(t for t in tokens_output_per_step if isinstance(t, (int, float)))
            tokens_input_total = tokens_input_per_step[-1]
            tokens_used_total = tokens_output_total + tokens_input_total
        elif self.api_mode == "responses":
            # Responses API: total_tokens is cumulative
            tokens_output_total = sum(t for t in tokens_output_per_step if isinstance(t, (int, float)))
            tokens_used_total = tokens_total_per_step[-1]
            tokens_input_total = tokens_used_total - tokens_output_total
            if tokens_input_total < 0:
                raise ValueError("Calculated tokens_input_total is negative.")

        meta_info = {
            "id": self.task_id,
            "question": self.task_prompt,
            "options": self.options,
            "answer": self.task_answer,
            "image_path": self.task_image_path,
            "function_call_total_count": function_call_total_count,
            "total_steps": len(self.conversation_history),
            "function_call_each_count": function_call_each_count,
            "function_call_per_step": function_call_per_step,
            "tokens_used_total": tokens_used_total,
            "tokens_output_per_step": tokens_output_per_step,
            "tokens_output_total": tokens_output_total,
            "tokens_input_total": tokens_input_total,
            "tokens_input_per_step": tokens_input_per_step,
            "tokens_total_per_step": tokens_total_per_step,
            "output_text": self.conversation_history[-1]["output_text"],
        }
        if isinstance(self.meta_info_extra, dict):
            meta_info.update(self.meta_info_extra)

        last_step_index = len(self.conversation_history) - 1
        for idx, record in enumerate(self.conversation_history):
            observation = record.get("observation")
            extra_info_list.append(
                {
                    "step": record["step"],
                    "extra_info": record.get("extra_info", {}),
                    "observation": observation,
                }
            )

        with open(self.extra_info_jsonl_path, "w", encoding="utf-8") as f:
            for record in extra_info_list:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        with open(self.output_jsonl_path, "w", encoding="utf-8") as f:
            for record in self.conversation_history:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        with open(self.meta_info_jsonl_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(meta_info, ensure_ascii=False) + "\n")

        # Generate trajectory.json for SFT training (formatted JSON, not JSONL)
        try:
            sft_messages = self._build_sft_messages()
            trajectory_json_path = self.trajectory_jsonl_path.replace('.jsonl', '.json')
            with open(trajectory_json_path, "w", encoding="utf-8") as f:
                json.dump(sft_messages, f, indent=2, ensure_ascii=False)
        except NotImplementedError:
            # Subclass hasn't implemented _build_sft_messages yet
            pass

        return meta_info

    def save_state(self) -> Dict[str, Any]:
        """Save current task state snapshot for later restoration.

        Returns:
            Dictionary containing restorable state
        """
        return {
            "contents": copy.deepcopy(self.contents),
            "conversation_history": copy.deepcopy(self.conversation_history),
            "image_list": [img.copy() for img in self.image_list],
            "state": self.state,
            "step_count": self._step_count,
        }

    def restore_state(self, state: Dict[str, Any]) -> None:
        """Restore task to a previously saved state.

        Args:
            state: State dictionary from save_state()
        """
        self.contents = copy.deepcopy(state["contents"])
        self.conversation_history = copy.deepcopy(state["conversation_history"])
        self.image_list = [img.copy() for img in state["image_list"]]
        self.state = state["state"]
        if "step_count" in state:
            self._step_count = state["step_count"]

    # =========================================================================
    # Gymnasium Interface
    # =========================================================================

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the task to initial state.

        This method is called at the start of each episode. It resets all mutable
        state while keeping task configuration (prompt, answer, etc.) intact.

        Args:
            seed: Optional random seed (unused, kept for Gym compatibility)
            options: Optional dict that may contain:
                - task_prompt: Override task prompt
                - task_answer: Override expected answer

        Returns:
            observation: Dict containing 'contents' and 'step'
            info: Dict with task metadata
        """
        super().reset(seed=seed)

        # Reset step counter
        self._step_count = 0
        self.state = True

        # Reset image list to initial state
        self.image_list = [img.copy() for img in self._initial_image_list]
        self.image_path_map = dict(self._initial_image_path_map)
        self.image_url_map = {}

        # Reset conversation history
        self.conversation_history = []

        # Handle options override
        if options:
            if "task_prompt" in options:
                self.task_prompt = options["task_prompt"]
            if "task_answer" in options:
                self.task_answer = options["task_answer"]

        # Contents will be re-initialized by subclass _append_initial_observation()
        # This reset() is called by subclasses that override it

        observation = self._get_observation()
        info = self.get_info()
        return observation, info

    def step(
        self,
        action: Any,
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment.

        This combines parse_action() and update_observation_from_action() into
        a single Gymnasium-compatible step.

        Args:
            action: Model response object (API-specific format)
            extra_info: Optional dict with token usage etc.

        Returns:
            observation: Current state after action
            reward: Reward value (0.0 during episode, computed at end)
            terminated: True if episode ended (final answer or error)
            truncated: True if max steps reached
            info: Dict with additional information including 'tool_calls'
        """
        self._step_count += 1
        extra_info = extra_info or {}

        # Parse the action to extract tool calls
        try:
            tool_calls = self.parse_action(
                step=self._step_count,
                action=action,
                extra_info=extra_info,
            )
        except Exception as e:
            logger.error("Failed to parse action: %s", e)
            self.state = False
            return (
                self._get_observation(),
                0.0,
                True,  # terminated
                False,
                {"error": str(e), "tool_calls": []},
            )

        # If no tool calls, episode is done (final answer given)
        if not tool_calls:
            reward = self._calculate_reward()
            return (
                self._get_observation(),
                reward,
                True,  # terminated
                False,
                {"tool_calls": [], "final_answer": True},
            )

        # Execute tool calls
        self.update_observation_from_action(tool_calls)

        # Check if truncated (max steps)
        truncated = self._step_count >= self.max_steps

        return (
            self._get_observation(),
            0.0,  # reward only at episode end
            False,  # not terminated
            truncated,
            {"tool_calls": [(tc.name, tc.args) for tc in tool_calls]},
        )

    def _get_observation(self) -> Dict[str, Any]:
        """Get current observation in Gymnasium format.

        Returns:
            Dict with 'contents' and 'step' keys
        """
        return {
            "contents": self.contents,
            "step": self._step_count,
        }

    def _calculate_reward(self) -> float:
        """Calculate reward at episode end.

        Override this in subclasses for task-specific reward computation.
        Default implementation returns 0.0.

        Returns:
            Reward value
        """
        return 0.0

    def render(self) -> None:
        """Render the environment (optional, for debugging)."""
        pass

    def close(self) -> None:
        """Clean up resources."""
        pass
