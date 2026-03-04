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

    def connect_to_existing_agents(
        self,
        agent_names: List[str],
        configs: Optional[Dict[str, Dict[str, Any]]] = None,
        ray_address: str = "auto",
    ) -> Dict[str, ray.actor.ActorHandle]:
        """Connect to existing Ray actors by name.

        This is useful for worker processes that need to access actors
        created in the main process.

        Args:
            agent_names: List of agent names to connect to.
            configs: Optional dict mapping agent names to their configs.
                     If not provided, will use default configs from agents module.
            ray_address: Ray cluster address.

        Returns:
            Dict mapping tool names to their actor handles.
        """
        if not ray.is_initialized():
            ray.init(address=ray_address, namespace="tool_agent", ignore_reinit_error=True)

        # Import AGENT_CONFIGS if configs not provided
        if configs is None:
            from geo_edit.tool_definitions.agents import AGENT_CONFIGS
            configs = AGENT_CONFIGS

        for name in agent_names:
            if name in self._actors:
                continue

            try:
                # Get actor by name from Ray namespace
                actor = ray.get_actor(name, namespace="tool_agent")
                self._actors[name] = actor
                # Store config for this agent (needed for call() method)
                if name in configs:
                    self._configs[name] = configs[name]
                logger.debug(f"Connected to existing actor: {name}")
            except ValueError:
                logger.warning(f"Actor {name} not found in Ray cluster")

        return self._actors

    def create_agents(
        self,
        configs: Dict[str, Dict[str, Any]],
        ray_address: str = "auto",
        wait_for_ready: bool = True,
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
            wait_for_ready: If True, wait for all agents to finish loading models.

        Returns:
            Dict mapping tool names to their actor handles.
        """
        if not ray.is_initialized():
            ray.init(address=ray_address, namespace="tool_agent", ignore_reinit_error=True)

        new_actors = []
        for name, cfg in configs.items():
            if name in self._actors:
                logger.info("Agent %s already exists, skipping", name)
                continue

            self._configs[name] = cfg

            # Build actor options
            num_gpus = cfg.get("num_gpus", 1)
            resources = cfg.get("resources")

            actor_options = {}
            if resources:
                actor_options["resources"] = resources

            # Get the Actor class for this agent
            ActorClass = get_actor_class(name)

            # Get system prompt from registry
            system_prompt = get_tool_agent_prompt(name)

            # Create Ray remote actor with the specific Actor class
            # Use name= to make it discoverable via ray.get_actor()
            RemoteActorClass = ray.remote(num_gpus=num_gpus)(ActorClass)
            actor = RemoteActorClass.options(name=name, **actor_options).remote(
                model_name=cfg["model_name_or_path"],
                max_model_len=cfg["max_model_len"],
                gpu_memory_utilization=cfg["gpu_memory_utilization"],
                system_prompt=system_prompt,
            )

            self._actors[name] = actor
            new_actors.append((name, actor))
            logger.info("Created tool agent: %s -> %s", name, cfg["model_name_or_path"])

        # Wait for all new agents to finish initialization
        if wait_for_ready and new_actors:
            logger.info("Waiting for %d tool agents to load models...", len(new_actors))
            health_refs = [(name, actor.health_check.remote()) for name, actor in new_actors]
            for name, ref in health_refs:
                try:
                    ray.get(ref)
                    logger.info("Tool agent %s is ready", name)
                except Exception as e:
                    logger.error("Tool agent %s failed to initialize: %s", name, e)
            logger.info("All tool agents are ready")

        return self._actors

    def get_actor(self, tool_name: str) -> Optional[ray.actor.ActorHandle]:
        """Get actor handle by tool name."""
        return self._actors.get(tool_name)

    def call(
        self,
        tool_name: str,
        image_list: List[Image.Image],
        image_index: int,
        **kwargs,
    ) -> str:
        """Call a Tool Agent to analyze an image.

        Args:
            tool_name: Name of the tool agent to call.
            image_list: List of PIL images.
            image_index: Index of the image to analyze.
            **kwargs: Tool-specific parameters (e.g., question, text_prompt, mode, etc.).

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
                    cfg["temperature"],
                    cfg["max_tokens"],
                    **kwargs,
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