"""
Shared base class for geo_edit tool types.

Provides common logic for action parsing, environment (image list) management,
image encode/decode, and ToolRouter-based agent execution.

Subclasses only need to set:
    tool_type   – unique tool type name for registration
    agent_name  – base agent name (e.g. "paddleocr"), or None for function-only
    enable_tools – list of tool names this tool type handles
"""

import io
import json
import base64
import logging
import os
import re
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

from PIL import Image

from .base import BaseTool

logger = logging.getLogger(__name__)

# Ensure geo_edit package is importable
_AREAL_ROOT = os.environ.get(
    "AREAL_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")),
)
if _AREAL_ROOT not in sys.path:
    sys.path.insert(0, _AREAL_ROOT)

_ACTION_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL | re.IGNORECASE)


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def encode_image_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


def decode_image_url(url: str) -> Image.Image:
    if url.startswith("data:image"):
        b64 = url.split("base64,", 1)[1]
    else:
        b64 = url
    return Image.open(io.BytesIO(base64.b64decode(b64)))


# ---------------------------------------------------------------------------
# Base tool class for all geo_edit tools
# ---------------------------------------------------------------------------

class GeoEditToolBase(BaseTool):
    """Base class for geo_edit tool types (both function and agent)."""

    tool_type: str = ""           # subclass MUST override
    stop_tokens = ["</action>"]
    enable_tools: List[str] = []  # tool names this type handles

    # Populated by _init_tools()
    function_tools: Dict[str, tuple] = {}

    def get_usage_inst(self):
        return f"Tools: {', '.join(sorted(self.function_tools.keys()))}"

    def parse_action(self, action: str) -> Tuple[dict, bool]:
        match = _ACTION_RE.search(action)
        if not match:
            return {}, False
        try:
            parsed = json.loads(match.group(1).strip())
            if "name" not in parsed:
                return {}, False
            if parsed["name"] not in self.function_tools:
                logger.warning(
                    f"[{self.tool_type}] Tool name '{parsed['name']}' not in registered function_tools. "
                    f"Available tools: {sorted(self.function_tools.keys())}"
                )
                return {}, False
            return parsed, True
        except (json.JSONDecodeError, KeyError):
            return {}, False

    def load_env(self, trajectory_id):
        env = self.env_cache.get(trajectory_id)
        if env is None:
            env = {
                "trajectory_id": trajectory_id,
                "metadata": {"turns": 0},
                "previous_obs": [],
                "images": [],
                "images_initialized": False,
            }
        return env

    def conduct_action(self, trajectory_id, action, extra_field):
        parsed, valid = self.parse_action(action)
        env = self.load_env(trajectory_id)

        # Initialize images from extra_field on first action
        if not env["images_initialized"] and extra_field.get("images"):
            for img_source in extra_field["images"]:
                if isinstance(img_source, str):
                    if os.path.exists(img_source):
                        env["images"].append(Image.open(img_source).convert("RGB"))
                    else:
                        env["images"].append(decode_image_url(img_source))
                elif isinstance(img_source, Image.Image):
                    env["images"].append(img_source.copy())
            env["images_initialized"] = True

        if not valid:
            observation = (
                "Error: Could not parse action. Expected format: "
                '<action>{"name": "tool_name", "arguments": {...}}</action>'
            )
            self.update_env(trajectory_id, env, parsed, False, extra_field, observation)
            self.save_env(trajectory_id, env)
            return observation, False, False

        tool_name = parsed["name"]
        tool_args = parsed.get("arguments", {})
        tool_fn = self.function_tools[tool_name][1]

        try:
            if "image_index" in tool_args:
                tool_args["image_index"] = int(tool_args["image_index"])
            result = tool_fn(env["images"], **tool_args)
        except Exception as e:
            observation = f"Error executing {tool_name}: {str(e)}"
            self.update_env(trajectory_id, env, parsed, False, extra_field, observation)
            self.save_env(trajectory_id, env)
            return observation, False, False

        if isinstance(result, Image.Image):
            env["images"].append(result.copy())
            idx = len(env["images"]) - 1
            encoded = encode_image_url(result)
            observation = {
                "obs": f"Tool executed successfully.\nObservation {idx}:\n<image>",
                "image": encoded,
            }
        elif isinstance(result, str):
            if result.startswith("Error"):
                observation = result
                self.update_env(trajectory_id, env, parsed, False, extra_field, observation)
                self.save_env(trajectory_id, env)
                return observation, False, False
            else:
                observation = {"obs": f"Tool executed successfully.\nResult: {result}"}
        else:
            observation = {"obs": f"Tool returned: {str(result)}"}

        self.update_env(trajectory_id, env, parsed, True, extra_field, observation)
        self.save_env(trajectory_id, env)
        return observation, False, True


# ---------------------------------------------------------------------------
# Agent tool base (loads tools via ToolRouter + Ray)
# ---------------------------------------------------------------------------

class GeoEditAgentToolBase(GeoEditToolBase):
    """Base class for agent-based geo_edit tools that require GPU + Ray."""

    agent_name: str = ""  # subclass MUST override (e.g. "paddleocr")

    def __init__(self, num_workers=1):
        super().__init__(num_workers)
        self._router = None
        self._load_agent_tools()

    def _load_agent_tools(self):
        """Load tools via ToolRouter, initializing only the agent this type needs."""
        try:
            import ray

            if not ray.is_initialized():
                ray.init(address="auto", ignore_reinit_error=True)
                logger.info(f"Ray initialized: {ray.cluster_resources()}")

            from geo_edit.tool_definitions.router import ToolRouter

            self._router = ToolRouter(
                tool_mode="auto",
                enable_tools=list(self.enable_tools),
                skip_agent_init=False,
                ray_address="auto",
            )

            tools = self._router.get_available_tools()
            declarations = {d["name"]: d for d in self._router.get_available_declarations()}
            return_types = self._router.get_tool_return_types()

            self.function_tools = {}
            for name, fn in tools.items():
                decl = declarations.get(name, {"name": name})
                ret_type = return_types.get(name, "text")
                self.function_tools[name] = (decl, fn, "agent", ret_type)

            logger.info(
                f"{self.__class__.__name__} loaded {len(self.function_tools)} tools "
                f"via agent '{self.agent_name}': {sorted(self.function_tools.keys())}"
            )
        except Exception as e:
            logger.error(
                f"{self.__class__.__name__} failed to load agent '{self.agent_name}': {e}",
                exc_info=True,
            )
            self.function_tools = {}
