"""Tool Agent Manager - lifecycle management for Ray Actors."""

import base64
from io import BytesIO
from typing import Any, Dict, List, Optional, Type

import ray
from PIL import Image

from geo_edit.prompts import get_tool_agent_prompt
from geo_edit.tool_definitions.agents import get_actor_class
from geo_edit.utils.logger import setup_logger

logger = setup_logger(__name__)

_MANAGER: Optional["ToolAgentManager"] = None


def get_manager() -> "ToolAgentManager":
    """Get the singleton ToolAgentManager instance."""
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = ToolAgentManager()
    return _MANAGER


class ToolAgentManager:
    """Manages Tool Agent lifecycle: creation, calling, and shutdown."""

    def __init__(self):
        self._actors: Dict[str, ray.actor.ActorHandle] = {}
        self._configs: Dict[str, Dict[str, Any]] = {}

    def create_agents(
        self,
        configs: Dict[str, Dict[str, Any]],
        ray_address: str = "auto",
    ) -> Dict[str, ray.actor.ActorHandle]:
        """Create Tool Agents from configuration dict.

        Args:
            configs: Dict mapping tool names to their configurations.
                Each config must contain:
                - model_name_or_path: str
                - max_model_len: int
                - gpu_memory_utilization: float
                - temperature: float
                - max_tokens: int
                Optional:
                - num_gpus: int (default 1)
                - resources: Dict[str, float] for Ray scheduling
            ray_address: Ray cluster address.

        Returns:
            Dict mapping tool names to their actor handles.
        """
        if not ray.is_initialized():
            ray.init(address=ray_address, ignore_reinit_error=True)

        for name, cfg in configs.items():
            if name in self._actors:
                logger.info("Agent %s already exists, skipping", name)
                continue

            self._configs[name] = cfg

            # Build actor options
            num_gpus = cfg.get("num_gpus", 1)
            resources = cfg.get("resources")

            actor_options = {
                "name": f"tool_agent_{name}",
                "lifetime": "detached",
            }
            if resources:
                actor_options["resources"] = resources

            # Get the Actor class for this agent
            ActorClass = get_actor_class(name)

            # Get system prompt from registry
            system_prompt = get_tool_agent_prompt(name)

            # Create Ray remote actor with the specific Actor class
            RemoteActorClass = ray.remote(num_gpus=num_gpus)(ActorClass)
            actor = RemoteActorClass.options(**actor_options).remote(
                model_name=cfg["model_name_or_path"],
                max_model_len=cfg["max_model_len"],
                gpu_memory_utilization=cfg["gpu_memory_utilization"],
                system_prompt=system_prompt,
            )

            self._actors[name] = actor
            logger.info("Created tool agent: %s -> %s", name, cfg["model_name_or_path"])

        return self._actors

    def get_actor(self, tool_name: str) -> Optional[ray.actor.ActorHandle]:
        """Get actor handle by tool name, reconnecting if necessary."""
        actor = self._actors.get(tool_name)
        if actor is None:
            try:
                actor = ray.get_actor(f"tool_agent_{tool_name}")
                self._actors[tool_name] = actor
            except ValueError:
                pass
        return actor

    def call(
        self,
        tool_name: str,
        image_list: List[Image.Image],
        image_index: int,
        question: str,
    ) -> str:
        """Call a Tool Agent to analyze an image.

        Args:
            tool_name: Name of the tool agent to call.
            image_list: List of PIL images.
            image_index: Index of the image to analyze.
            question: Question to ask about the image.

        Returns:
            Analysis result as string.
        """
        actor = self.get_actor(tool_name)
        if actor is None:
            return f"Error: Tool agent '{tool_name}' not found"

        if image_index < 0 or image_index >= len(image_list):
            return f"Error: Invalid image index {image_index}"

        # Convert image to base64
        image = image_list[image_index]
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # Get config for this agent
        cfg = self._configs[tool_name]

        try:
            result = ray.get(
                actor.analyze.remote(
                    image_b64,
                    question,
                    cfg["temperature"],
                    cfg["max_tokens"],
                )
            )
            return result
        except Exception as e:
            logger.error("Tool agent %s call failed: %s", tool_name, e)
            return f"Error: {e}"

    def shutdown(self, tool_names: Optional[List[str]] = None):
        """Shutdown Tool Agents.

        Args:
            tool_names: List of tool names to shutdown. If None, shutdown all.
        """
        names = tool_names or list(self._actors.keys())
        for name in names:
            actor = self._actors.pop(name, None)
            self._configs.pop(name, None)
            if actor:
                try:
                    ray.kill(actor)
                    logger.info("Killed tool agent: %s", name)
                except Exception as e:
                    logger.warning("Failed to kill %s: %s", name, e)

    def status(self) -> Dict[str, Dict[str, Any]]:
        """Get health status of all Tool Agents.

        Returns:
            Dict mapping tool names to their status.
        """
        status = {}
        for name, actor in self._actors.items():
            try:
                health = ray.get(actor.health_check.remote())
                status[name] = {"status": "healthy", **health}
            except Exception as e:
                status[name] = {"status": "error", "error": str(e)}
        return status