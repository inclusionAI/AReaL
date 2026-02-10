"""Tool Agent Configuration - following agents/base.py AgentConfig pattern."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

# Default config path constant
DEFAULT_CONFIG_PATH = "tool_agent_config.json"


@dataclass
class ToolAgentConfig:
    """Configuration for a single Tool Agent.

    Attributes:
        model_name_or_path: Model name or path to load.
        max_model_len: Maximum model length for vLLM.
        gpu_memory_utilization: GPU memory utilization (0-1).
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
        num_gpus: Number of GPUs required.
        resources: Custom Ray resource labels for node scheduling.
            Example: {"tool_agent_gpu": 1} to run on nodes with this label.
            Start worker nodes with: ray start --resources='{"tool_agent_gpu": 1}'
    """

    model_name_or_path: str
    max_model_len: int = 8192
    gpu_memory_utilization: float = 0.8
    temperature: float = 0.0
    max_tokens: int = 1024
    num_gpus: int = 1
    resources: Optional[Dict[str, float]] = None


def load_configs_from_json(config_path: Path = DEFAULT_CONFIG_PATH) -> Dict[str, Dict[str, Any]]:
    """Load tool agent configurations from JSON file.

    Args:
        config_path: Path to the config file.

    Returns:
        Dict mapping tool names to their configurations.
    """
    with config_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["tool_agents"] if "tool_agents" in payload else payload


def load_configs(config_path: Path = DEFAULT_CONFIG_PATH) -> Dict[str, ToolAgentConfig]:
    """Load configurations from JSON file as ToolAgentConfig objects."""
    raw_configs = load_configs_from_json(config_path)
    return {name: ToolAgentConfig(**cfg) for name, cfg in raw_configs.items()}
