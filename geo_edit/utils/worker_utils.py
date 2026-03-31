"""Shared worker utilities for generation scripts.

This module provides common worker initialization patterns used across
separated_reasoning_generate.py and iterative_sampling_generate.py.
"""
from __future__ import annotations

import os
import signal
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

from geo_edit.agents.api_agent import AgentConfig, APIBasedAgent
from geo_edit.config import (
    build_google_agent_configs,
    build_api_agent_configs,
    derive_google_config,
    derive_api_config,
)
from geo_edit.prompts import get_system_prompt
from geo_edit.prompts.system_prompts import (
    SIMPLIFIED_TOOL_SELECTION_PROMPT,
    SEPARATED_TOOL_CALL_ONLY_PROMPT,
    SEPARATED_FINAL_ANSWER_PROMPT,
    MULTI_ROUND_TOOL_SELECTION_PROMPT,
    CHAIN_TOOL_SELECTION_PROMPT,
)
from geo_edit.tool_definitions import ToolRouter
from geo_edit.environment.task.google_vision_qa_task import GoogleVisionQATask
from geo_edit.environment.task.openai_compatible_vision_qa_task import OpenAICompatibleVisionQATask
from geo_edit.utils.logger import setup_logger

logger = setup_logger(__name__)


def worker_init_signal_handler():
    """Ignore SIGINT in worker processes - let main process handle it."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


@dataclass
class SeparatedReasoningConfigs:
    """Configuration container for separated reasoning phase configs."""
    reasoning_only: Any = None
    multi_round_reasoning: Any = None
    chain_reasoning: Any = None  # For iterative sampling chain tool selection
    tool_call_only: Any = None
    final_answer: Any = None
    extended_reasoning: Any = None  # Optional, for iterative sampling
    extra_configs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkerContext:
    """Container for worker state."""
    agent: APIBasedAgent
    agent_configs: Any
    api_mode: str
    task_class: Type
    tool_router: ToolRouter
    phase_configs: SeparatedReasoningConfigs
    output_path: Optional[str] = None


def connect_to_ray_agents(tool_router: ToolRouter, enabled_agent_names: List[str]) -> None:
    """Connect worker to existing Ray actors."""
    if not enabled_agent_names:
        return
    from geo_edit.environment.tool_agents import get_manager
    manager = get_manager()
    agent_configs = tool_router.get_enabled_agent_configs()
    manager.connect_to_existing_agents(enabled_agent_names, configs=agent_configs)
    logger.info(f"Worker (PID: {os.getpid()}) connected to {len(enabled_agent_names)} Ray actors")


def build_phase_configs(
    model_type: str,
    base_config: Any,
    extra_prompts: Optional[Dict[str, str]] = None,
) -> SeparatedReasoningConfigs:
    """Build phase-specific configs for separated reasoning.

    Args:
        model_type: "Google" or "OpenAI"/"SGLang".
        base_config: Base generation config to derive from.
        extra_prompts: Optional dict mapping config name to system prompt.
            Example: {"extended_reasoning": ITERATIVE_EXTENDED_REASONING_PROMPT}

    Returns:
        SeparatedReasoningConfigs with all phase configs.
    """
    configs = SeparatedReasoningConfigs()

    if model_type == "Google":
        configs.reasoning_only = derive_google_config(
            base_config, system_prompt=SIMPLIFIED_TOOL_SELECTION_PROMPT, tool_mode="NONE"
        )
        configs.multi_round_reasoning = derive_google_config(
            base_config, system_prompt=MULTI_ROUND_TOOL_SELECTION_PROMPT, tool_mode="NONE"
        )
        configs.chain_reasoning = derive_google_config(
            base_config, system_prompt=CHAIN_TOOL_SELECTION_PROMPT, tool_mode="NONE"
        )
        configs.tool_call_only = derive_google_config(
            base_config, system_prompt=SEPARATED_TOOL_CALL_ONLY_PROMPT
        )
        configs.final_answer = derive_google_config(
            base_config, system_prompt=SEPARATED_FINAL_ANSWER_PROMPT, tool_mode="NONE"
        )
        # Build extra configs if provided
        if extra_prompts:
            for name, prompt in extra_prompts.items():
                config = derive_google_config(base_config, system_prompt=prompt, tool_mode="NONE")
                if name == "extended_reasoning":
                    configs.extended_reasoning = config
                else:
                    configs.extra_configs[name] = config
    else:
        # OpenAI/SGLang via chat_completions
        configs.reasoning_only = derive_api_config(
            base_config, api_mode="chat_completions",
            system_prompt=SIMPLIFIED_TOOL_SELECTION_PROMPT, tool_choice="none"
        )
        configs.multi_round_reasoning = derive_api_config(
            base_config, api_mode="chat_completions",
            system_prompt=MULTI_ROUND_TOOL_SELECTION_PROMPT, tool_choice="none"
        )
        configs.chain_reasoning = derive_api_config(
            base_config, api_mode="chat_completions",
            system_prompt=CHAIN_TOOL_SELECTION_PROMPT, tool_choice="none"
        )
        configs.tool_call_only = derive_api_config(
            base_config, api_mode="chat_completions",
            system_prompt=SEPARATED_TOOL_CALL_ONLY_PROMPT
        )
        configs.final_answer = derive_api_config(
            base_config, api_mode="chat_completions",
            system_prompt=SEPARATED_FINAL_ANSWER_PROMPT, tool_choice="none"
        )
        # Build extra configs if provided
        if extra_prompts:
            for name, prompt in extra_prompts.items():
                config = derive_api_config(
                    base_config, api_mode="chat_completions",
                    system_prompt=prompt, tool_choice="none"
                )
                if name == "extended_reasoning":
                    configs.extended_reasoning = config
                else:
                    configs.extra_configs[name] = config

    return configs


def init_worker_base(
    api_key: str,
    model_name_or_path: str,
    model_type: str,
    api_base: Optional[str],
    port: Optional[int],
    output_path: str,
    enabled_agent_names: List[str],
    enable_tools: Optional[List[str]],
    extra_prompts: Optional[Dict[str, str]] = None,
) -> WorkerContext:
    """Initialize worker with agent and configs.

    This is the shared initialization logic for generation scripts.

    Args:
        api_key: API key for the model.
        model_name_or_path: Model name or path.
        model_type: "Google", "OpenAI", or "SGLang".
        api_base: Optional API base URL.
        port: Optional port number.
        output_path: Output directory path.
        enabled_agent_names: List of Ray agent names to connect to.
        enable_tools: List of tools to enable.
        extra_prompts: Optional extra prompts for additional phase configs.

    Returns:
        WorkerContext containing all initialized components.
    """
    worker_init_signal_handler()

    # Create ToolRouter WITHOUT initializing Ray actors
    tool_router = ToolRouter(tool_mode="force", enable_tools=enable_tools, skip_agent_init=True)

    # Connect to existing Ray actors
    connect_to_ray_agents(tool_router, enabled_agent_names)

    if model_type == "Google" and not api_key:
        raise ValueError("API key must be provided for Google models.")

    system_prompt = get_system_prompt(model_type, "force")

    # Build configs based on model type
    if model_type == "Google":
        api_mode = "google"
        agent_configs = build_google_agent_configs(
            tool_router,
            thinking_level="low",
            include_thoughts=True,
            temperature=1.0,
            system_prompt=system_prompt,
        )
        task_class = GoogleVisionQATask
    else:
        api_mode = "chat_completions"
        agent_configs = build_api_agent_configs(
            tool_router,
            api_mode="chat_completions",
            temperature=1.0,
            reasoning_level="low",
            system_prompt=system_prompt,
        )
        task_class = OpenAICompatibleVisionQATask

    # Build phase-specific configs
    phase_configs = build_phase_configs(
        model_type, agent_configs.generate_config, extra_prompts
    )

    # Create agent
    config = AgentConfig(
        model_type=model_type,
        model_name=model_name_or_path,
        api_key=api_key,
        api_base=api_base,
        port=port,
        generate_config=agent_configs.generate_config,
        n_retry=3,
        api_mode=api_mode,
    )
    agent = APIBasedAgent(config)

    logger.info(f"Worker initialized with {model_type} agent (PID: {os.getpid()})")

    return WorkerContext(
        agent=agent,
        agent_configs=agent_configs,
        api_mode=api_mode,
        task_class=task_class,
        tool_router=tool_router,
        phase_configs=phase_configs,
        output_path=output_path,
    )
