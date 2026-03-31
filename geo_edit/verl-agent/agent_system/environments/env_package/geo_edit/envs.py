"""GeoEdit environment wrapper for verl-agent training.

Wraps multiple VisionQATask instances, bypassing Gymnasium step() and
directly using update_observation_from_action() for tool execution.
"""

import os
import tempfile
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from geo_edit.environment.task.vision_qa_task import ToolCall, VisionQATask
from geo_edit.environment.task.openai_compatible_vision_qa_task import OpenAICompatibleVisionQATask
from geo_edit.prompts.system_prompts import (
    TOOL_EXECUTION_FAILURE_PROMPT,
    TOOL_EXECUTION_SUCCESS_PROMPT,
)
from geo_edit.tool_definitions.router import ToolRouter


def _image_to_numpy(image: Image.Image) -> np.ndarray:
    """Convert PIL Image to numpy array (H, W, 3) uint8."""
    return np.array(image.convert("RGB"), dtype=np.uint8)


class GeoEditMultiProcessEnv:
    """Vectorized environment wrapping multiple VisionQATask instances.

    Unlike the Gymnasium-based VecVisionQAEnv, this class is designed for
    verl-agent's EnvironmentManager pattern:
    - reset() accepts task metadata via kwargs
    - step() accepts pre-parsed actions (tool_calls or final answers)
    - Returns numpy arrays for rewards/dones
    """

    def __init__(
        self,
        tool_router: ToolRouter,
        max_tool_calls: int = 5,
        save_dir: Optional[str] = None,
    ):
        self.tool_router = tool_router
        self.tool_functions = tool_router.get_available_tools()
        self.tool_return_types = tool_router.get_tool_return_types()
        self.tool_declarations = tool_router.get_available_declarations()
        self.max_tool_calls = max_tool_calls
        self.save_dir = save_dir or tempfile.mkdtemp(prefix="geo_edit_")

        self.tasks: List[VisionQATask] = []
        self.task_types: List[str] = []
        self.step_counts: List[int] = []
        self.num_envs = 0

    def reset(self, kwargs: Optional[Any] = None) -> tuple:
        """Reset environments with task data from kwargs.

        Args:
            kwargs: List of dicts (one per env), each with keys:
                - task_id: str
                - task_prompt: str
                - task_answer: str
                - task_image_path: str (optional)
                - task_type: str (optional, default "exact")

        Returns:
            (text_obs: List[str], infos: List[Dict])
        """
        if kwargs is None:
            kwargs = []
        # Handle both list-of-dicts (verl-agent pattern) and single dict formats
        if isinstance(kwargs, dict):
            # Convert single dict with parallel lists to list-of-dicts
            n = len(kwargs.get("task_ids", []))
            kwargs = [
                {
                    "task_id": kwargs["task_ids"][i],
                    "task_prompt": kwargs["task_prompts"][i],
                    "task_answer": kwargs["task_answers"][i],
                    "task_image_path": kwargs.get("task_image_paths", [None] * n)[i],
                    "task_type": kwargs.get("task_types", ["exact"] * n)[i],
                }
                for i in range(n)
            ]

        self.num_envs = len(kwargs)
        self.tasks = []
        self.task_types = []
        self.step_counts = [0] * self.num_envs

        text_obs = []
        infos = []

        for i, kw in enumerate(kwargs):
            task_id = kw.get("task_id", str(i))
            task_prompt = kw.get("task_prompt", "")
            task_answer = kw.get("task_answer", "")
            task_image_path = kw.get("task_image_path", None)
            task_type = kw.get("task_type", "exact")

            task_save_dir = os.path.join(self.save_dir, f"task_{task_id}")
            os.makedirs(task_save_dir, exist_ok=True)

            task = OpenAICompatibleVisionQATask(
                task_id=task_id,
                task_prompt=task_prompt,
                task_answer=task_answer,
                task_image_path=task_image_path,
                save_dir=task_save_dir,
                model_type="vllm",
                api_mode="chat_completions",
                tool_functions=self.tool_functions,
                tool_return_types=self.tool_return_types,
                max_steps=self.max_tool_calls,
            )
            self.tasks.append(task)
            self.task_types.append(task_type)

            info = {
                "task_id": task_id,
                "task_answer": task_answer,
                "task_type": task_type,
                "tool_calling": False,
                "won": False,
            }
            infos.append(info)

            text_obs.append(task_prompt)

        return text_obs, infos

    def step(self, parsed_actions: List[Dict[str, Any]]) -> tuple:
        """Execute parsed actions on each task.

        Args:
            parsed_actions: List of dicts from GeoEditEnvironmentManager._parse_text_actions.
                Each dict has:
                - type: "tool_call" | "answer" | "none"
                - tool_calls: List[ToolCall]
                - answer_text: str

        Returns:
            (text_obs, rewards, dones, infos)
            - text_obs: List[str] - tool execution feedback or empty
            - rewards: np.ndarray of shape (num_envs,)
            - dones: np.ndarray of shape (num_envs,)
            - infos: List[Dict]
        """
        text_obs = []
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=np.bool_)
        infos = []

        for i, action in enumerate(parsed_actions):
            task = self.tasks[i]
            self.step_counts[i] += 1

            info = {
                "task_id": task.task_id,
                "task_answer": task.task_answer,
                "task_type": self.task_types[i] if i < len(self.task_types) else "exact",
                "tool_calling": False,
                "won": False,
                "is_action_valid": 1,
            }

            if action["type"] == "tool_call":
                # Execute tool calls via VisionQATask
                info["tool_calling"] = True
                prev_image_count = len(task.image_list)
                try:
                    task.update_observation_from_action(action["tool_calls"])
                    feedback = self._get_last_tool_feedback(task)
                    # Add image observation info if new images were produced
                    new_image_count = len(task.image_list)
                    if new_image_count > prev_image_count:
                        feedback = f"New image produced: Observation {new_image_count - 1}\n{feedback}"
                    text_obs.append(feedback)
                except Exception as e:
                    text_obs.append(f"Tool execution error: {e}")

                # Check if max steps reached
                if self.step_counts[i] >= self.max_tool_calls:
                    dones[i] = True

            elif action["type"] == "answer":
                # Final answer - compute reward
                from agent_system.environments.env_package.geo_edit.reward import compute_reward

                reward = compute_reward(
                    action["answer_text"],
                    task.task_answer,
                    task_type=info.get("task_type", "exact"),
                )
                rewards[i] = reward
                dones[i] = True
                info["won"] = reward > 0
                info["predicted_answer"] = action["answer_text"]
                text_obs.append("")

            else:
                # Invalid/empty action
                info["is_action_valid"] = 0
                text_obs.append("No valid action detected. Please use <action> for tool calls or <answer> for your final answer.")
                if self.step_counts[i] >= self.max_tool_calls:
                    dones[i] = True

            infos.append(info)

        return text_obs, rewards, dones, infos

    def get_images(self) -> List[Optional[np.ndarray]]:
        """Get the latest image from each task as numpy arrays."""
        images = []
        for task in self.tasks:
            if task.image_list:
                images.append(_image_to_numpy(task.image_list[-1]))
            else:
                images.append(None)
        return images

    def get_tool_declarations_text(self) -> str:
        """Format tool declarations as text for inclusion in prompts."""
        from geo_edit.tool_definitions.router import format_tool_declarations_text
        return format_tool_declarations_text(self.tool_declarations)

    def _get_last_tool_feedback(self, task: OpenAICompatibleVisionQATask) -> str:
        """Extract tool execution feedback from task's contents.

        After update_observation_from_action(), the task appends tool results
        and a success/failure prompt to self.contents. We extract the last
        user message (which contains the feedback prompt) as the text feedback.
        """
        if not hasattr(task, "contents") or not task.contents:
            return ""

        # For chat_completions format, contents is a list of messages
        # The last message appended by update_observation_from_action is the
        # success/failure prompt (a user message with text content)
        contents = task.contents
        if isinstance(contents, list) and contents:
            last_msg = contents[-1]
            if last_msg.get("role") == "user":
                content = last_msg.get("content", "")
                if isinstance(content, list):
                    # Extract text parts
                    texts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
                    return "\n".join(t for t in texts if t)
                elif isinstance(content, str):
                    return content

        return ""

    def close(self):
        """Clean up resources."""
        self.tasks = []
        self.step_counts = []


def build_geo_edit_envs(
    seed: int,
    env_num: int,
    env_config: Dict[str, Any],
) -> GeoEditMultiProcessEnv:
    """Factory function to create GeoEditMultiProcessEnv.

    Args:
        seed: Random seed (unused currently, reserved for future).
        env_num: Number of parallel environments.
        env_config: Environment config dict with keys:
            - tools: List of tool names/categories to enable (None = use config.yaml)
            - max_tool_calls: Max tool calls per episode
            - ray_address: Ray cluster address
            - node_resource: Ray resource name for tool agent scheduling

    Returns:
        GeoEditMultiProcessEnv instance.
    """
    tool_router = ToolRouter(
        tool_mode="auto",
        enable_tools=env_config.get("tools", None),
        ray_address=env_config.get("ray_address", "auto"),
        node_resource=env_config.get("node_resource", "tool_agent"),
    )

    return GeoEditMultiProcessEnv(
        tool_router=tool_router,
        max_tool_calls=env_config.get("max_tool_calls", 5),
    )
