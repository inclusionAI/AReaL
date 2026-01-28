from __future__ import annotations
#TODO: 使vllm force tool call有效化
import os
from dataclasses import dataclass
from pathlib import Path

from geo_edit.agents.base import AgentConfig
from geo_edit.agents.vllm_agent import VLLMBasedAgent
from geo_edit.config import build_vllm_agent_configs
from geo_edit.constants import MAX_TOOL_CALLS, get_system_prompt
from geo_edit.environment.action import TOOL_FUNCTIONS
from geo_edit.environment.task.vllm_vision_qa_task import VLLMVisionQATask
from geo_edit.utils.logger import setup_logger

logger = setup_logger(__name__)

DEFAULT_PROMPT = "Describe the image and put the answer in <answer>...</answer>."


@dataclass(frozen=True, slots=True)
class Args:
    api_base: str
    model_name: str
    image_path: Path
    prompt: str
    system_prompt: str | None
    tool_mode: str
    temperature: float
    max_tokens: int
    max_steps: int


def _ensure_answer_tag(prompt: str) -> str:
    if "<answer>" in prompt.lower():
        return prompt
    return f"{prompt.rstrip()}\nAnswer inside <answer>...</answer>."


def test_vllm_task(tmp_path: Path) -> None:
    args = Args(
        api_base="http://127.0.0.1:8000",
        model_name=os.environ["VLLM_MODEL_NAME"],
        image_path=Path(__file__).resolve().parents[1] / "images" / "input_image.png",
        prompt=DEFAULT_PROMPT,
        system_prompt=None,
        tool_mode="direct",
        temperature=0.2,
        max_tokens=512,
        max_steps=MAX_TOOL_CALLS,
    )

    api_base = args.api_base
    model_name = args.model_name

    save_dir = tmp_path / "vllm_task_test"
    save_dir.mkdir(parents=True, exist_ok=True)

    use_tools = args.tool_mode != "direct"
    tool_functions = TOOL_FUNCTIONS if use_tools else {}

    if args.system_prompt is not None:
        system_prompt = args.system_prompt
    elif use_tools:
        system_prompt = get_system_prompt("vLLM")
    else:
        system_prompt = (
            "You are a helpful assistant. Think step by step in <think> tags and "
            "provide the final answer in <answer>...</answer>."
        )

    prompt = _ensure_answer_tag(args.prompt)

    agent_configs = build_vllm_agent_configs(
        max_output_tokens=args.max_tokens,
        temperature=args.temperature,
        tool_mode=args.tool_mode,
        system_prompt=system_prompt,
    )

    config = AgentConfig(
        model_type="vLLM",
        model_name=model_name,
        api_base=api_base,
        generate_config=agent_configs.generate_config,
        n_retry=3,
    )

    agent = VLLMBasedAgent(config)
    agent.reset()

    task = VLLMVisionQATask(
        task_id="vllm_task_test",
        task_prompt=prompt,
        task_answer="",
        task_image_path=str(args.image_path),
        save_dir=save_dir,
        tool_functions=tool_functions,
        system_prompt=system_prompt,
    )

    logger.info("Model: %s", model_name)
    logger.info("API base: %s", api_base)
    logger.info("Image: %s", args.image_path)
    logger.info("Tool mode: %s", args.tool_mode)

    for step in range(1, args.max_steps + 1):
        action, extra_info = agent.act(task.contents)
        assert not isinstance(action, str)

        tool_calls = task.parse_action(step=step, action=action, extra_info=extra_info)
        if not tool_calls:
            output_text = task.conversation_history[-1]["output_text"]
            assert output_text
            print(output_text)
            break
        task.update_observation_from_action(tool_calls)
    else:
        raise AssertionError("Reached max steps without a final answer.")

    task.save_trajectory()
