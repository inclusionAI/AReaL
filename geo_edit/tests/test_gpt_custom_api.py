from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pytest

from geo_edit.agents.base import AgentConfig
from geo_edit.agents.api_agent import APIBasedAgent
from geo_edit.config import build_api_agent_configs
from geo_edit.constants import MAX_TOOL_CALLS
from geo_edit.prompts import get_system_prompt
from geo_edit.environment.task.openai_compatible_vision_qa_task import OpenAICompatibleVisionQATask
from geo_edit.tool_definitions import ToolRouter
from geo_edit.utils.logger import setup_logger

logger = setup_logger(__name__)

DEFAULT_PROMPT = """
A circle $K$ is inscribed in a quarter circle with radius 6 as shown in the figure. What is the radius of circle $K$?<image1>

option[
    A."$\\frac{6-\\sqrt{2}}{2}$",
    B."$\\frac{3 \\sqrt{2}}{2}$",
    C."2.5",
    D."3",
    E."$6(\\sqrt{2}-1)$"
]
"""


@dataclass(frozen=True, slots=True)
class Args:
    api_base: str | None
    api_key: str
    model_name: str
    image_path: Path
    prompt: str
    system_prompt: str | None
    tool_mode: str
    temperature: float
    max_tokens: int
    max_steps: int


def test_gpt_custom_api_base_tool_call() -> None:
    """Test GPT with custom API base for tool calling."""
    api_key = os.environ.get("API_KEY")
    api_base = "https://matrixllm.alipay.com/v1"
    model_name = "gpt-5-2025-08-07"

    # matrixllm API only supports chat_completions mode and does not support temperature
    is_matrixllm = api_base and "matrixllm" in api_base

    args = Args(
        api_base=api_base if api_base else None,
        api_key=api_key,
        model_name=model_name,
        image_path=Path("./geo_edit/images/173.jpg"),
        prompt=DEFAULT_PROMPT,
        system_prompt=None,
        tool_mode="force",
        temperature=0.0 if is_matrixllm else 0.2,  # matrixllm does not support temperature
        max_tokens=16384,
        max_steps=MAX_TOOL_CALLS,
    )

    # matrixllm only supports chat_completions mode
    api_mode = "chat_completions" if is_matrixllm else "responses"

    save_dir = Path("./gpt_custom_api_test")
    save_dir.mkdir(parents=True, exist_ok=True)

    use_tools = args.tool_mode != "direct"
    tool_router = ToolRouter(tool_mode=args.tool_mode)
    tool_functions = tool_router.get_available_tools() if use_tools else {}
    tool_return_types = tool_router.get_tool_return_types() if use_tools else {}

    if args.system_prompt is not None:
        system_prompt = args.system_prompt
    elif use_tools:
        system_prompt = get_system_prompt("OpenAI", args.tool_mode)
    else:
        system_prompt = "You are a helpful assistant. Think step by step and provide the final answer in <answer>...</answer>."

    agent_configs = build_api_agent_configs(
        tool_router,
        api_mode=api_mode,
        max_output_tokens=args.max_tokens,
        temperature=args.temperature,
        system_prompt=system_prompt,
    )

    config = AgentConfig(
        model_type="OpenAI",
        model_name=args.model_name,
        api_key=args.api_key,
        api_base=args.api_base,
        generate_config=agent_configs.generate_config,
        n_retry=3,
        api_mode=api_mode,
    )

    agent = APIBasedAgent(config)
    agent.reset()

    task = OpenAICompatibleVisionQATask(
        task_id="gpt_custom_api_test",
        task_prompt=args.prompt,
        task_answer="",
        task_image_path=str(args.image_path),
        save_dir=save_dir,
        tool_functions=tool_functions,
        tool_return_types=tool_return_types,
        model_type="openai",
        api_mode=api_mode,
    )

    logger.info("Model: %s", args.model_name)
    logger.info("API base: %s", args.api_base or "default")
    logger.info("Image: %s", args.image_path)
    logger.info("Tool mode: %s", args.tool_mode)

    for step in range(1, args.max_steps + 1):
        action, extra_info = agent.act(task.contents)
        assert not isinstance(action, str)

        tool_calls = task.parse_action(step=step, action=action, extra_info=extra_info)
        if not tool_calls:
            output_text = task.conversation_history[-1]["output_text"]
            assert output_text
            print(f"Final answer: {output_text}")
            break
        task.update_observation_from_action(tool_calls)
    else:
        raise AssertionError("Reached max steps without a final answer.")

    task.save_trajectory()
    logger.info("GPT custom API base test completed successfully.")


if __name__ == "__main__":
    print("GPT Custom API Base Tool Call Test")
    print("=" * 50)
    print("\nEnvironment variables:")
    print("  OPENAI_API_KEY    - Required: API key for authentication")
    print("  OPENAI_API_BASE   - Optional: Custom API base URL")
    print("  OPENAI_MODEL_NAME - Optional: Model name (default: gpt-4o)")
    print("\nRunning test...")
    test_gpt_custom_api_base_tool_call()
