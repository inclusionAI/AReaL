"""Tool Router - central registry for all tools.

Controls tool availability based on:
1. config.yaml - static enable/disable per tool
2. tool_mode - runtime mode ("auto", "force", "direct")
   - "auto"/"force": use enabled tools from config
   - "direct": disable ALL tools (no tool system prompt)
3. enable_tools parameter - runtime override to enable specific tools
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional

import yaml
from PIL import Image
from geo_edit.tool_definitions.functions import FUNCTION_TOOLS
from geo_edit.tool_definitions.agents import (
    AGENT_DECLARATIONS,
    AGENT_RETURN_TYPES,
    AGENT_CONFIGS,
    MULTI_TOOL_DECLARATIONS,
)
from geo_edit.utils.logger import setup_logger

logger = setup_logger(__name__)

# =============================================================================
# Configuration
# =============================================================================

_CONFIG_PATH = Path(__file__).parent / "config.yaml"
with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
    _TOOL_CONFIG: Dict[str, bool] = yaml.safe_load(f)


# =============================================================================
# Tool Category Presets
# =============================================================================

TOOL_CATEGORIES = {
    "general": ["image_crop", "image_label", "draw_line", "draw_path", "bounding_box", "image_highlight",
                "text_ocr", "auto_segment", "bbox_segment", "text_segment", "exemplar_segment",
                "concept_count", "presence_check", "grounding_dino"],
    "math": ["math_latex_ocr", "math_image_describe", "formula_ocr", "gllava", "multimath"],
    "table": ["table_ocr"],
    "chart": ["chart_reasoning", "chart_data_extract", "chart_trend_analysis", "chart_text_ocr", "chartr1"],
    "map": ["text_spotting", "map_text_ocr"],
    "document": ["seal_ocr"],
    "ocr": ["text_ocr", "table_ocr", "formula_ocr", "chart_text_ocr", "text_spotting", "seal_ocr", "map_text_ocr"],
    "segment": ["auto_segment", "bbox_segment", "text_segment", "exemplar_segment", "concept_count", "presence_check"],
    "reasoning": ["visual_reasoning"],
}


def get_all_tool_names() -> List[str]:
    """Get all available tool names."""
    return list(_build_tool_registry().keys())


def expand_tool_names(tool_specs: List[str]) -> List[str]:
    """Expand tool specifications (names or categories) to actual tool names.

    Args:
        tool_specs: List of tool names or category names (e.g., ["math", "text_ocr", "chart"])

    Returns:
        List of expanded tool names.
    """
    expanded = set()
    for spec in tool_specs:
        spec_lower = spec.lower().strip()
        if spec_lower in TOOL_CATEGORIES:
            expanded.update(TOOL_CATEGORIES[spec_lower])
        else:
            expanded.add(spec_lower)
    return list(expanded)


def _make_agent_execute(
    agent_name: str,
    fixed_params: Optional[Dict[str, Any]] = None,
    return_type: str = "text",
) -> Callable[..., Any]:
    """Create an execute function for a specific agent.

    Args:
        agent_name: Name of the base agent to call.
        fixed_params: Optional dict of fixed parameters to inject (e.g., fixed_task, fixed_prompt, filter_map).
        return_type: Return type of the tool ("text" or "image"). When "image", the base64
            string returned by the actor is decoded back to a PIL.Image.Image.
    """
    import base64
    import json
    from io import BytesIO

    def execute(image_list: List[Image.Image], image_index: int, **kwargs) -> Any:
        from geo_edit.environment.tool_agents import call_agent

        # Inject fixed parameters if specified
        if fixed_params:
            # Handle fixed_task for PaddleOCR
            if "fixed_task" in fixed_params:
                kwargs["task"] = fixed_params["fixed_task"]
            # Handle fixed_prompt for Multimath/ChartMoE
            if "fixed_prompt" in fixed_params:
                kwargs["question"] = fixed_params["fixed_prompt"]
            # Handle fixed_mode for SAM3
            if "fixed_mode" in fixed_params:
                mode = fixed_params["fixed_mode"]
                kwargs["mode"] = mode
                if mode == "auto":
                    kwargs.pop("bounding_box", None)
                    kwargs.pop("text_prompt", None)
                elif mode == "bbox":
                    kwargs.pop("text_prompt", None)
                elif mode == "exemplar_segment":
                    kwargs.pop("text_prompt", None)  # Uses bounding_box as exemplar
                elif mode in ("text_segment", "concept_count", "presence_check"):
                    kwargs.pop("bounding_box", None)  # Only needs text_prompt

        result = call_agent(agent_name, image_list, image_index, **kwargs)

        # Post-process for map_text_ocr: filter text results
        if fixed_params and fixed_params.get("filter_map"):
            try:
                result_json = json.loads(result)
                if "text" in result_json:
                    if isinstance(result_json["text"], list):
                        # Spotting task: list of {text, bbox}
                        from geo_edit.tool_definitions.agents.paddleocr_tool import process_map_ocr_result
                        result_json["text"] = process_map_ocr_result(result_json["text"])
                    elif isinstance(result_json["text"], str):
                        # OCR task: plain text string
                        from geo_edit.tool_definitions.agents.paddleocr_tool import filter_map_text_string
                        result_json["text"] = filter_map_text_string(result_json["text"])
                    result = json.dumps(result_json, ensure_ascii=False)
            except (json.JSONDecodeError, ImportError) as e:
                logger.warning(f"Failed to post-process map OCR result: {e}")

        # Convert base64 string to PIL.Image for image-returning agents
        if return_type == "image" and isinstance(result, str) and not result.startswith("Error"):
            try:
                image_bytes = base64.b64decode(result)
                return Image.open(BytesIO(image_bytes))
            except Exception as e:
                logger.warning(f"Failed to decode agent image result: {e}")
                return result

        return result
    return execute


def _build_tool_registry() -> Dict[str, tuple]:
    """Build the complete tool registry from functions, agents, and multi-tools."""
    registry = dict(FUNCTION_TOOLS)

    # Add legacy agent tools with dynamically created execute functions
    for name, declaration in AGENT_DECLARATIONS.items():
        ret_type = AGENT_RETURN_TYPES[name]
        registry[name] = (
            declaration,
            _make_agent_execute(name, return_type=ret_type),
            "agent",
            ret_type,
        )

    # Add multi-tool declarations (fine-grained tools)
    for tool_name, tool_info in MULTI_TOOL_DECLARATIONS.items():
        decl = tool_info["declaration"]
        base_agent = tool_info["base_agent"]

        # Extract fixed parameters from declaration
        fixed_params = {}
        if "fixed_task" in decl:
            fixed_params["fixed_task"] = decl["fixed_task"]
        if "fixed_prompt" in decl:
            fixed_params["fixed_prompt"] = decl["fixed_prompt"]
        if "fixed_mode" in decl:
            fixed_params["fixed_mode"] = decl["fixed_mode"]
        if "filter_map" in decl:
            fixed_params["filter_map"] = decl["filter_map"]

        # Create clean declaration without internal fields
        clean_decl = {k: v for k, v in decl.items() if k not in ("fixed_task", "fixed_prompt", "fixed_mode", "filter_map", "return_type")}

        tool_return_type = decl.get("return_type", "text")
        registry[tool_name] = (
            clean_decl,
            _make_agent_execute(base_agent, fixed_params if fixed_params else None, return_type=tool_return_type),
            "agent",
            tool_return_type,
        )

    return registry


_TOOL_REGISTRY: Dict[str, tuple] = _build_tool_registry()


def format_tool_declarations_text(declarations: List[Dict]) -> str:
    """Format tool declarations as human-readable text for inclusion in prompts.

    This is the single source of truth for tool-list formatting used across
    SFT data conversion, RL environment prompts, and inference evaluation.
    """
    lines = []
    for decl in declarations:
        name = decl["name"]
        desc = decl.get("description", "").strip()
        params = decl.get("parameters", {}).get("properties", {})
        param_strs = []
        for pname, pinfo in params.items():
            ptype = pinfo.get("type", "string")
            pdesc = pinfo.get("description", "")
            param_strs.append(f"    - {pname} ({ptype}): {pdesc}")
        params_text = "\n".join(param_strs) if param_strs else "    (no parameters)"
        lines.append(f"- {name}: {desc}\n  Parameters:\n{params_text}")
    return "\n\n".join(lines)


# =============================================================================
# ToolRouter Class
# =============================================================================

class ToolRouter:
    """Router for dynamic tool selection and agent lifecycle management.

    Args:
        tool_mode: "auto"/"force" to use enabled tools, "direct" to disable all tools.
        enable_tools: List of tool names/categories to enable (overrides config.yaml).
            Supports category names: "general", "math", "table", "chart", "map", "document", "ocr", "segment".
        node_resource: Ray custom resource name to schedule agents on specific nodes.
            E.g., "tool_agent" will add {"tool_agent": 1} to each agent's resources.
        ray_address: Ray cluster address for agent initialization.
        skip_agent_init: If True, skip automatic agent initialization (useful for worker processes
            that should connect to existing agents without re-initializing them).
    """

    def __init__(
        self,
        tool_mode: Literal["auto", "force", "direct"] = "auto",
        enable_tools: Optional[List[str]] = None,
        node_resource: str = "tool_agent",
        ray_address: str = "auto",
        skip_agent_init: bool = False,
    ):
        self.tool_mode = tool_mode
        self._agents: Dict[str, Any] = {}

        # Override config with enable_tools if provided
        self._tool_override: Optional[List[str]] = None
        if enable_tools:
            self._tool_override = expand_tool_names(enable_tools)
            logger.info(f"Tool override enabled: {self._tool_override}")

        # Auto-initialize agents if any are enabled (unless explicitly skipped)
        if not skip_agent_init and self.get_enabled_agents():
            self._agents = self._create_agents(ray_address, node_resource)

    def _get_enabled_tool_names(self) -> List[str]:
        """Get names of all enabled tools (respects tool_mode, config.yaml, and overrides)."""
        if self.tool_mode == "direct":
            return []

        # If override is set, use only those tools
        if self._tool_override is not None:
            return [name for name in self._tool_override if name in _TOOL_REGISTRY]

        # Otherwise use config.yaml
        return [name for name in _TOOL_REGISTRY if _TOOL_CONFIG.get(name, False)]

    def _create_agents(
        self,
        ray_address: str,
        node_resource: Optional[str],
    ) -> Dict[str, Any]:
        """Create Ray Actors for enabled agent tools."""
        from geo_edit.environment.tool_agents import get_manager

        agent_configs = self.get_enabled_agent_configs()
        if not agent_configs:
            return {}

        # Add node resource to each agent config if specified
        if node_resource:
            for name in agent_configs:
                agent_configs[name] = agent_configs[name].copy()
                agent_configs[name]["resources"] = {node_resource: 1}

        logger.info("Initializing Tool Agents (Ray Actors)...")
        agents = get_manager().create_agents(agent_configs, ray_address)
        logger.info(f"Created {len(agents)} tool agents: {list(agents.keys())}")
        return agents

    def get_available_declarations(self) -> List[Dict]:
        """Get tool declarations for available tools."""
        return [_TOOL_REGISTRY[name][0] for name in self._get_enabled_tool_names()]

    def get_available_tools(self) -> Dict[str, Callable[..., Image.Image | str]]:
        """Get tool functions for available tools."""
        return {name: _TOOL_REGISTRY[name][1] for name in self._get_enabled_tool_names()}

    def get_tool_return_types(self) -> Dict[str, str]:
        """Get return types for all available tools."""
        return {name: _TOOL_REGISTRY[name][3] for name in self._get_enabled_tool_names()}

    def get_enabled_agents(self) -> List[str]:
        """Get list of unique base agent names for Ray Actor initialization.

        Maps fine-grained tool names to their base agents and deduplicates.
        E.g., [text_ocr, text_spotting, auto_segment] -> [paddleocr, sam3]
        """
        base_agents = set()
        for name in self._get_enabled_tool_names():
            if _TOOL_REGISTRY[name][2] != "agent":
                continue
            # Check if this is a multi-tool with a base_agent
            if name in MULTI_TOOL_DECLARATIONS:
                base_agents.add(MULTI_TOOL_DECLARATIONS[name]["base_agent"])
            elif name in AGENT_CONFIGS:
                # Legacy single-tool agent (e.g., gllava, ovr)
                base_agents.add(name)
        return list(base_agents)

    def get_enabled_agent_configs(self) -> Dict[str, dict]:
        """Get configs for enabled base agents.

        Returns:
            Dict mapping base agent name to its config dict.
        """
        return {
            name: AGENT_CONFIGS[name]
            for name in self.get_enabled_agents()
            if name in AGENT_CONFIGS
        }

    def is_tool_enabled(self) -> bool:
        """Check if any tools are enabled."""
        return bool(self._get_enabled_tool_names())

    def is_agent_enabled(self) -> bool:
        """Check if any agent tools are enabled and initialized."""
        return bool(self._agents)

    def shutdown_agents(self, tool_names: Optional[List[str]] = None):
        """Shutdown Tool Agents.

        Args:
            tool_names: List of agent names to shutdown. If None, shutdown all.
        """
        if not self._agents:
            return

        from geo_edit.environment.tool_agents import get_manager

        logger.info("Shutting down Tool Agents...")
        get_manager().shutdown(tool_names)
        self._agents = {}
        logger.info("Tool Agents shutdown complete.")
