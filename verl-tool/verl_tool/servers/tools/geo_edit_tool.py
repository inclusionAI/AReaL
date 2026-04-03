import io
import json
import base64
import logging
import re
import os
import sys
from typing import Tuple, List, Dict, Any, Union
from PIL import Image

from .base import BaseTool, register_tool

logger = logging.getLogger(__name__)

# Ensure geo_edit is importable (AReaL root)
_AREAL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if _AREAL_ROOT not in sys.path:
    sys.path.insert(0, _AREAL_ROOT)

_ACTION_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL | re.IGNORECASE)


def _encode_image_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


def _decode_image_url(url: str) -> Image.Image:
    if url.startswith("data:image"):
        b64 = url.split("base64,", 1)[1]
    else:
        b64 = url
    return Image.open(io.BytesIO(base64.b64decode(b64)))


def _load_function_tools():
    # Load each function tool module directly via importlib to bypass
    # geo_edit.tool_definitions.__init__ → router → agents → ray dependency chain
    import importlib.util
    tools_dir = os.path.join(_AREAL_ROOT, "geo_edit", "tool_definitions", "functions")
    modules = {
        "image_crop": "crop.py",
        "image_label": "label.py",
        "draw_line": "draw_line.py",
        "draw_path": "draw_path.py",
        "bounding_box": "bbox.py",
        "image_highlight": "highlight.py",
    }
    result = {}
    for tool_name, filename in modules.items():
        filepath = os.path.join(tools_dir, filename)
        spec = importlib.util.spec_from_file_location(filename[:-3], filepath)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        result[tool_name] = (mod.DECLARATION, mod.execute, "function", mod.RETURN_TYPE)
    return result


@register_tool
class GeoEditTool(BaseTool):
    tool_type = "geo_edit_tool"
    stop_tokens = ["</action>"]

    def __init__(self, num_workers=1):
        super().__init__(num_workers)
        self.function_tools = _load_function_tools()
        logger.info(f"GeoEditTool loaded {len(self.function_tools)} function tools: {list(self.function_tools.keys())}")

    def get_usage_inst(self):
        return "Vision tool agent supporting crop, draw_line, draw_path, label, highlight, bbox and more."

    def parse_action(self, action: str) -> Tuple[dict, bool]:
        match = _ACTION_RE.search(action)
        if not match:
            return {}, False
        try:
            parsed = json.loads(match.group(1).strip())
            if "name" not in parsed:
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

        if not env["images_initialized"] and extra_field.get("images"):
            for img_source in extra_field["images"]:
                if isinstance(img_source, str):
                    if os.path.exists(img_source):
                        env["images"].append(Image.open(img_source).convert("RGB"))
                    else:
                        env["images"].append(_decode_image_url(img_source))
                elif isinstance(img_source, Image.Image):
                    env["images"].append(img_source.copy())
            env["images_initialized"] = True

        if not valid:
            observation = "Error: Could not parse action. Expected format: <action>{\"name\": \"tool_name\", \"arguments\": {...}}</action>"
            self.update_env(trajectory_id, env, parsed, False, extra_field, observation)
            self.save_env(trajectory_id, env)
            return observation, False, False

        tool_name = parsed["name"]
        tool_args = parsed.get("arguments", {})

        if tool_name not in self.function_tools:
            observation = f"Error: Unknown tool '{tool_name}'. Available: {list(self.function_tools.keys())}"
            self.update_env(trajectory_id, env, parsed, False, extra_field, observation)
            self.save_env(trajectory_id, env)
            return observation, False, False

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
            encoded = _encode_image_url(result)
            observation = {
                "obs": f"Tool executed successfully.\nObservation {idx}: <image>",
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
