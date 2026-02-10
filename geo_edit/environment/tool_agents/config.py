"""Tool Agent Configuration - Ray Actor common settings."""

from dataclasses import dataclass
from typing import Dict, Optional

# Common default settings for all agents
DEFAULT_MAX_MODEL_LEN = 8192
DEFAULT_GPU_MEMORY_UTILIZATION = 0.8
DEFAULT_TEMPERATURE = 0.0
DEFAULT_NUM_GPUS = 1


@dataclass
class ToolAgentConfig:
    """Configuration for a single Tool Agent.

    Attributes:
        model_path: Model name or path to load.
        max_tokens: Maximum tokens to generate (from tool definition).
        max_model_len: Maximum model length for vLLM.
        gpu_memory_utilization: GPU memory utilization (0-1).
        temperature: Sampling temperature.
        num_gpus: Number of GPUs required.
        resources: Custom Ray resource labels for node scheduling.
    """
    model_path: str
    max_tokens: int
    max_model_len: int = DEFAULT_MAX_MODEL_LEN
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION
    temperature: float = DEFAULT_TEMPERATURE
    num_gpus: int = DEFAULT_NUM_GPUS
    resources: Optional[Dict[str, float]] = None


def build_agent_configs() -> Dict[str, ToolAgentConfig]:
    """Build agent configs from tool_definitions/agents/ definitions."""
    from geo_edit.tool_definitions.agents import AGENT_CONFIGS

    return {
        name: ToolAgentConfig(
            model_path=cfg["model_path"],
            max_tokens=cfg["max_tokens"],
        )
        for name, cfg in AGENT_CONFIGS.items()
    }
