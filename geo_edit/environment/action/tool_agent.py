from __future__ import annotations

import base64
import json
import os
import subprocess
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

from PIL import Image

from geo_edit.utils.image_utils import image_to_data_url
from geo_edit.utils.logger import setup_logger

logger = setup_logger(__name__)

_TOOL_AGENT_REGISTRY: Dict[str, "ToolAgent"] = {}
_TOOL_AGENT_ENDPOINTS: Dict[str, Dict[str, Any]] = {}
_ROOT_DIR = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG_PATH = _ROOT_DIR / "scripts" / "tool_agent_config.json"
_DEFAULT_LAUNCH_SCRIPT = _ROOT_DIR / "scripts" / "launch_vllm_tool_agent.sh"


class ToolAgent:
    """OpenAI-compatible client wrapper for a single remote tool-agent server."""

    def __init__(self, server_ip: str, server_port: int):
        from openai import OpenAI

        self.server_ip = server_ip
        self.server_port = server_port
        self.base_url = f"http://{server_ip}:{server_port}/v1"
        self.client = OpenAI(base_url=self.base_url, api_key="none")
        self.model = self._resolve_model_name()

    def _resolve_model_name(self) -> str:
        models = self.client.models.list()
        return models.data[0].id

    def _build_request_payload(self, tool_name: str, image: Image.Image, question: str) -> Dict[str, Any]:
        instruction = (
            "You are a tool agent. Analyze the image and answer the question. "
            "Return JSON with at least one field in {analysis, text, result, error}."
        )
        user_text = json.dumps({"tool_name": tool_name, "question": question}, ensure_ascii=True)
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": instruction},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {"type": "image_url", "image_url": {"url": image_to_data_url(image)}},
                    ],
                },
            ],
            "temperature": 0.0,
        }

    @staticmethod
    def _extract_content_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                elif hasattr(part, "text") and isinstance(part.text, str):
                    parts.append(part.text)
            return "".join(parts)
        return ""

    @staticmethod
    def _decode_image(payload: str) -> Image.Image:
        encoded = payload.split(",", 1)[1] if payload.startswith("data:") else payload
        image_bytes = base64.b64decode(encoded)
        return Image.open(BytesIO(image_bytes)).convert("RGB")

    def _parse_response(self, response: Any) -> Image.Image | str:
        if not response.choices:
            return "Error: Tool agent response is empty."
        message = response.choices[0].message
        content_text = self._extract_content_text(message.content)
        if not content_text:
            return "Error: Tool agent returned empty content."

        try:
            payload = json.loads(content_text)
        except json.JSONDecodeError:
            return content_text

        error_text = payload.get("error")
        if isinstance(error_text, str) and error_text:
            return f"Error: {error_text}"
        output_type = payload.get("output_type")
        if output_type == "image":
            encoded = payload.get("image_base64") or payload.get("image")
            if isinstance(encoded, str) and encoded:
                return self._decode_image(encoded)
            return "Error: Tool agent image payload is empty."
        if isinstance(payload.get("analysis"), str):
            return payload["analysis"]
        if isinstance(payload.get("text"), str):
            return payload["text"]
        if isinstance(payload.get("result"), str):
            return payload["result"]
        return content_text

    def generate(self, tool_name: str, image_list: List[Image.Image], image_index: int, question: str) -> Image.Image | str:
        if image_index < 0 or image_index >= len(image_list):
            return "Error: Invalid image index."
        request_payload = self._build_request_payload(tool_name, image_list[image_index], question)
        response = self.client.chat.completions.create(**request_payload)
        return self._parse_response(response)


def _resolve_tool_agent_config_path(config_path: Optional[str] = None) -> Path:
    if config_path:
        return Path(config_path)
    env_path = os.environ.get("GEO_EDIT_TOOL_AGENT_CONFIG_PATH")
    if env_path:
        return Path(env_path)
    return _DEFAULT_CONFIG_PATH


def load_tool_configs_from_json_path(config_path: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    path = _resolve_tool_agent_config_path(config_path)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["tool_agents"] if "tool_agents" in payload else payload


def _run_launch_command(command: List[str]) -> None:
    subprocess.run(command, check=True)


def _launch_command_with_ray(command: List[str], node_ip: str) -> None:
    try:
        import ray
    except ImportError:
        _run_launch_command(command)
        return

    if not ray.is_initialized():
        _run_launch_command(command)
        return

    resource_key = f"node:{node_ip}"
    cluster_resources = ray.cluster_resources()
    if resource_key not in cluster_resources:
        logger.warning("Ray cluster has no resource %s, fallback to local launch.", resource_key)
        _run_launch_command(command)
        return

    remote_runner = ray.remote(num_cpus=0)(_run_launch_command)
    ray.get(remote_runner.options(resources={resource_key: 0.001}).remote(command))


def launch_tool_agent_servers(tool_names: List[str], node_ip: str, config_path: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    config_map = load_tool_configs_from_json_path(config_path)
    endpoints: Dict[str, Dict[str, Any]] = {}
    for tool_name in tool_names:
        tool_config = config_map[tool_name]
        tool_node_ip = str(tool_config.get("node_ip", node_ip))
        launch_script = Path(tool_config.get("launch_script", _DEFAULT_LAUNCH_SCRIPT))
        if not launch_script.is_absolute():
            launch_script = _ROOT_DIR / "scripts" / launch_script
        command = [
            "bash",
            str(launch_script),
            "--tool-name",
            tool_name,
            "--model-name-or-path",
            tool_config["model_name_or_path"],
            "--port",
            str(tool_config["port"]),
            "--gpu-id",
            str(tool_config["gpu_id"]),
            "--node-ip",
            tool_node_ip,
        ]
        optional_arg_map = {
            "host": "--host",
            "max_model_len": "--max-model-len",
            "dtype": "--dtype",
            "gpu_memory_utilization": "--gpu-memory-utilization",
            "startup_timeout_s": "--startup-timeout-s",
            "log_dir": "--log-dir",
        }
        for config_key, arg_name in optional_arg_map.items():
            if config_key in tool_config:
                command.extend([arg_name, str(tool_config[config_key])])
        _launch_command_with_ray(command, tool_node_ip)
        endpoints[tool_name] = {"ip": tool_node_ip, "port": int(tool_config["port"])}
        logger.info("Launched tool agent server for %s at %s:%s", tool_name, tool_node_ip, tool_config["port"])

    os.environ["GEO_EDIT_TOOL_AGENT_ENDPOINTS"] = json.dumps(endpoints, ensure_ascii=True)
    global _TOOL_AGENT_ENDPOINTS
    _TOOL_AGENT_ENDPOINTS = endpoints
    return endpoints


def initialize_tool_agent_registry(tool_endpoints: Dict[str, Dict[str, Any]]) -> Dict[str, ToolAgent]:
    registry: Dict[str, ToolAgent] = {}
    for tool_name, endpoint in tool_endpoints.items():
        registry[tool_name] = ToolAgent(server_ip=endpoint["ip"], server_port=int(endpoint["port"]))
    global _TOOL_AGENT_REGISTRY, _TOOL_AGENT_ENDPOINTS
    _TOOL_AGENT_REGISTRY = registry
    _TOOL_AGENT_ENDPOINTS = tool_endpoints
    return registry


def call_tool_agent_text(tool_name: str, image_list: List[Image.Image], image_index: int, question: str) -> str:
    tool_agent = _TOOL_AGENT_REGISTRY.get(tool_name)
    if tool_agent is None:
        return f"Error: Tool agent {tool_name} is not initialized."
    try:
        output = tool_agent.generate(tool_name=tool_name, image_list=image_list, image_index=image_index, question=question)
    except Exception as exc:
        logger.error("Tool agent call failed for %s: %s", tool_name, exc)
        return f"Error: Tool agent {tool_name} failed with error: {exc}"
    return output


def initialize_tool_agent_registry_from_env() -> Dict[str, ToolAgent]:
    endpoints_text = os.environ.get("GEO_EDIT_TOOL_AGENT_ENDPOINTS")
    if not endpoints_text:
        return {}
    endpoints = json.loads(endpoints_text)
    return initialize_tool_agent_registry(endpoints)


def shutdown_tool_agent(tool_names: Optional[List[str]] = None) -> None:
    endpoints = _TOOL_AGENT_ENDPOINTS
    if not endpoints:
        endpoints_text = os.environ.get("GEO_EDIT_TOOL_AGENT_ENDPOINTS")
        if endpoints_text:
            endpoints = json.loads(endpoints_text)
    if not endpoints:
        return
    selected_names = tool_names if tool_names is not None else list(endpoints.keys())
    for tool_name in selected_names:
        endpoint = endpoints[tool_name]
        shutdown_url = f"http://{endpoint['ip']}:{endpoint['port']}/shutdown"
        try:
            request = Request(shutdown_url, method="POST")
            with urlopen(request, timeout=3.0):
                pass
            logger.info("Shutdown request sent to tool agent %s at %s", tool_name, shutdown_url)
        except (URLError, TimeoutError) as exc:
            logger.warning("Failed to shutdown tool agent %s via %s: %s", tool_name, shutdown_url, exc)
