"""Test script for separated reasoning generation with matrixllm (Gemini/GPT via chat_completions)."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pytest

from geo_edit.agents.base import AgentConfig
from geo_edit.agents.api_agent import APIBasedAgent
from geo_edit.config import (
    build_api_agent_configs,
    derive_api_config,
)
from geo_edit.constants import MAX_TOOL_CALLS
from geo_edit.prompts import get_system_prompt
from geo_edit.prompts.system_prompts import (
    SEPARATED_REASONING_ONLY_PROMPT,
    SEPARATED_TOOL_CALL_ONLY_PROMPT,
    SEPARATED_FINAL_ANSWER_PROMPT,
)
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
    model_type: str
    image_path: Path
    prompt: str
    temperature: float
    max_tokens: int
    max_steps: int


def _run_separated_reasoning_test(
    model_name: str,
    model_type: Literal["SGLang", "OpenAI"],
    test_name: str,
) -> None:
    """Common test logic for separated three-phase generation."""
    api_key = os.environ.get("API_KEY")
    api_base = "https://matrixllm.alipay.com/v1"

    if not api_key:
        pytest.skip("API_KEY environment variable not set")

    args = Args(
        api_base=api_base,
        api_key=api_key,
        model_name=model_name,
        model_type=model_type,
        image_path=Path("./geo_edit/images/173.jpg"),
        prompt=DEFAULT_PROMPT,
        temperature=1.0,
        max_tokens=16384,
        max_steps=MAX_TOOL_CALLS,
    )

    save_dir = Path(f"./{test_name}")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Force mode only
    tool_router = ToolRouter(tool_mode="force")
    tool_functions = tool_router.get_available_tools()
    tool_return_types = tool_router.get_tool_return_types()

    system_prompt = get_system_prompt(model_type, "force")

    # Build configs for chat_completions API
    agent_configs = build_api_agent_configs(
        tool_router,
        api_mode="chat_completions",
        max_output_tokens=args.max_tokens,
        temperature=args.temperature,
        reasoning_level="low",
        system_prompt=system_prompt,
    )

    base = agent_configs.generate_config
    reasoning_only_config = derive_api_config(
        base, api_mode="chat_completions", system_prompt=SEPARATED_REASONING_ONLY_PROMPT, tool_choice="none"
    )
    tool_call_only_config = derive_api_config(
        base, api_mode="chat_completions", system_prompt=SEPARATED_TOOL_CALL_ONLY_PROMPT
    )
    final_answer_config = derive_api_config(
        base, api_mode="chat_completions", system_prompt=SEPARATED_FINAL_ANSWER_PROMPT, tool_choice="none"
    )

    config = AgentConfig(
        model_type=model_type,
        model_name=args.model_name,
        api_key=args.api_key,
        api_base=args.api_base,
        generate_config=agent_configs.generate_config,
        n_retry=3,
        api_mode="chat_completions",
    )

    agent = APIBasedAgent(config)
    agent.reset()

    task = OpenAICompatibleVisionQATask(
        task_id=test_name,
        task_prompt=args.prompt,
        task_answer="E",
        task_image_path=str(args.image_path),
        save_dir=save_dir,
        tool_functions=tool_functions,
        tool_return_types=tool_return_types,
        model_type=model_type.lower(),
        api_mode="chat_completions",
    )

    logger.info("Model: %s", args.model_name)
    logger.info("Model type: %s", model_type)
    logger.info("API base: %s", args.api_base)
    logger.info("Image: %s", args.image_path)

    original_generate_config = agent.config.generate_config
    answer_pattern = re.compile(r"<answer>", re.IGNORECASE)

    for step in range(1, args.max_steps + 1):
        logger.info("=== Step %d ===", step)

        # ===== Phase 1: Generate reasoning =====
        logger.info("Phase 1: Generating reasoning...")
        agent.config.generate_config = reasoning_only_config
        reasoning_action, reasoning_extra = agent.act(task.contents)

        # Extract reasoning text from chat_completions response
        reasoning_text = reasoning_action.choices[0].message.content or ""
        logger.info("Reasoning text (first 500 chars): %s", reasoning_text[:500])
        logger.info("Reasoning extra: %s", reasoning_extra)

        # Check for <answer> - should not appear in reasoning phase
        if answer_pattern.search(reasoning_text):
            raise ValueError("Reasoning phase should not generate <answer>. Model violated the protocol.")

        # ===== Phase 2: Generate tool call =====
        logger.info("Phase 2: Generating tool call...")
        task.append_assistant_message(reasoning_text)

        agent.config.generate_config = tool_call_only_config
        tool_action, tool_extra = agent.act(task.contents)

        agent.config.generate_config = original_generate_config

        # Merge reasoning_extra into tool_extra
        merged_extra = {"reasoning_" + k: v for k, v in reasoning_extra.items()}
        merged_extra.update(tool_extra)
        logger.info("Tool extra: %s", tool_extra)
        logger.info("Merged extra: %s", merged_extra)

        # Verify merged_extra contains both reasoning and tool info
        assert any(k.startswith("reasoning_") for k in merged_extra), "merged_extra should contain reasoning info"

        # Parse action
        tool_calls = task.parse_action(step=step, action=tool_action, extra_info=merged_extra)

        if not tool_calls:
            output_text = task.conversation_history[-1]["output_text"]
            assert output_text, "Should have output text when no tool calls"
            logger.info("Final answer: %s", output_text)
            break

        logger.info("Tool calls: %s", [(tc.name, tc.args) for tc in tool_calls])
        task.update_observation_from_action(tool_calls)

    # ===== Phase 3: Generate final answer =====
    if task.state:
        logger.info("Phase 3: Generating final answer...")
        agent.config.generate_config = final_answer_config
        action, extra_info = agent.act(task.contents)
        agent.config.generate_config = original_generate_config
        task.parse_action(step=args.max_steps + 1, action=action, extra_info=extra_info)

    meta_info = task.save_trajectory()
    logger.info("Test completed successfully!")
    logger.info("Total steps: %d", meta_info["total_steps"])
    logger.info("Total tool calls: %d", meta_info["function_call_total_count"])
    logger.info("Total tokens: %d", meta_info["tokens_used_total"])

    # Verify trajectory.json was created
    trajectory_path = save_dir / "trajectory.json"
    if trajectory_path.exists():
        logger.info("trajectory.json created successfully")
        with open(trajectory_path, "r", encoding="utf-8") as f:
            trajectory_data = json.load(f)
            logger.info("Trajectory contains %d messages", len(trajectory_data))

            # Log first few messages for inspection
            for i, msg in enumerate(trajectory_data[:5]):
                role = msg.get("role")
                has_tool_calls = "tool_calls" in msg
                content_preview = str(msg.get("content", ""))[:100]
                logger.info("  Message %d: role=%s, has_tool_calls=%s, content=%s...",
                           i, role, has_tool_calls, content_preview)

                # Log tool_calls if present
                if has_tool_calls:
                    tool_calls = msg.get("tool_calls", [])
                    for tc in tool_calls:
                        logger.info("    Tool call: %s(%s)",
                                   tc.get("function", {}).get("name", "unknown"),
                                   tc.get("function", {}).get("arguments", "")[:50])
    else:
        logger.warning("trajectory.json was not created")


def test_separated_reasoning_gpt() -> None:
    """Test separated reasoning generation with GPT via matrixllm."""
    _run_separated_reasoning_test(
        model_name="gpt-5-2025-08-07",
        model_type="OpenAI",
        test_name="separated_reasoning_gpt_test",
    )

