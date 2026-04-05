"""Tool Agent Manager - lifecycle management for Ray Actors."""

import base64
import inspect
import itertools
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
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = ToolAgentManager()
    return _MANAGER


def _replica_names(base_name: str, num_replicas: int) -> List[str]:
    if num_replicas <= 1:
        return [base_name]
    return [f"{base_name}_replica_{i}" for i in range(num_replicas)]


class ToolAgentManager:

    def __init__(self):
        self._actors: Dict[str, ray.actor.ActorHandle] = {}
        self._configs: Dict[str, Dict[str, Any]] = {}
        # base_name -> list of replica actor names (for round-robin dispatch)
        self._replica_map: Dict[str, List[str]] = {}
        self._robin: Dict[str, itertools.cycle] = {}

    def connect_to_existing_agents(
        self,
        agent_names: List[str],
        configs: Optional[Dict[str, Dict[str, Any]]] = None,
        ray_address: str = "auto",
    ) -> Dict[str, ray.actor.ActorHandle]:
        if not ray.is_initialized():
            ray.init(address=ray_address, namespace="tool_agent", ignore_reinit_error=True)

        if configs is None:
            from geo_edit.tool_definitions.agents import AGENT_CONFIGS
            configs = AGENT_CONFIGS

        for actor_name in agent_names:
            if actor_name in self._actors:
                continue
            try:
                actor = ray.get_actor(actor_name, namespace="tool_agent")
                self._actors[actor_name] = actor

                base_name = actor_name
                for suffix_check in configs:
                    if actor_name == suffix_check or actor_name.startswith(f"{suffix_check}_replica_"):
                        base_name = suffix_check
                        break

                if base_name in configs:
                    self._configs[base_name] = configs[base_name]
                if base_name not in self._replica_map:
                    self._replica_map[base_name] = []
                self._replica_map[base_name].append(actor_name)

                logger.debug(f"Connected to existing actor: {actor_name}")
            except ValueError:
                logger.warning(f"Actor {actor_name} not found in Ray cluster")

        for base_name, replicas in self._replica_map.items():
            self._robin[base_name] = itertools.cycle(replicas)

        return self._actors

    def create_agents(
        self,
        configs: Dict[str, Dict[str, Any]],
        ray_address: str = "auto",
        wait_for_ready: bool = True,
    ) -> Dict[str, ray.actor.ActorHandle]:
        if not ray.is_initialized():
            ray.init(address=ray_address, namespace="tool_agent", ignore_reinit_error=True)

        new_actors = []
        for name, cfg in configs.items():
            if name in self._replica_map:
                logger.info("Agent %s already exists, skipping", name)
                continue

            self._configs[name] = cfg

            num_gpus = cfg.get("num_gpus", 1)
            num_replicas = cfg.get("num_replicas", 1)
            resources = cfg.get("resources")

            actor_options = {}
            if resources:
                actor_options["resources"] = resources

            ActorClass = get_actor_class(name)

            init_sig = inspect.signature(ActorClass.__init__)
            init_params = set(init_sig.parameters.keys()) - {'self', 'kwargs'}

            system_prompt = get_tool_agent_prompt(name)

            actor_kwargs = {}
            param_mapping = {
                'model_name': cfg.get("model_name_or_path"),
                'max_model_len': cfg.get("max_model_len"),
                'gpu_memory_utilization': cfg.get("gpu_memory_utilization"),
                'temperature': cfg.get("temperature"),
                'max_tokens': cfg.get("max_tokens"),
                'system_prompt': system_prompt,
            }

            for param_name, param_value in param_mapping.items():
                if param_name in init_params and param_value is not None:
                    actor_kwargs[param_name] = param_value

            RemoteActorClass = ray.remote(num_gpus=num_gpus)(ActorClass)

            replica_names = _replica_names(name, num_replicas)
            self._replica_map[name] = replica_names

            for replica_name in replica_names:
                actor = RemoteActorClass.options(name=replica_name, **actor_options).remote(**actor_kwargs)
                self._actors[replica_name] = actor
                new_actors.append((replica_name, actor))

            logger.info(
                "Created tool agent: %s -> %s (x%d replicas)",
                name, cfg["model_name_or_path"], num_replicas,
            )

        if wait_for_ready and new_actors:
            logger.info("Waiting for %d tool agents to load models...", len(new_actors))
            health_refs = [(n, a.health_check.remote()) for n, a in new_actors]
            for n, ref in health_refs:
                try:
                    ray.get(ref)
                    logger.info("Tool agent %s is ready", n)
                except Exception as e:
                    logger.error("Tool agent %s failed to initialize: %s", n, e)
            logger.info("All tool agents are ready")

        for base_name, replicas in self._replica_map.items():
            self._robin[base_name] = itertools.cycle(replicas)

        return self._actors

    def get_actor(self, tool_name: str) -> Optional[ray.actor.ActorHandle]:
        if tool_name in self._robin:
            replica_name = next(self._robin[tool_name])
            return self._actors.get(replica_name)
        return self._actors.get(tool_name)

    def get_all_actor_names(self) -> List[str]:
        return list(self._actors.keys())

    def call(
        self,
        tool_name: str,
        image_list: List[Image.Image],
        image_index: int,
        **kwargs,
    ) -> str:
        actor = self.get_actor(tool_name)
        if actor is None:
            return f"Error: Tool agent '{tool_name}' not found"

        if image_index < 0 or image_index >= len(image_list):
            return f"Error: Invalid image index {image_index}"

        image = image_list[image_index]
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        cfg = self._configs.get(tool_name, {})

        try:
            if "temperature" in cfg and "max_tokens" in cfg:
                result = ray.get(
                    actor.analyze.remote(
                        image_b64,
                        cfg["temperature"],
                        cfg["max_tokens"],
                        **kwargs,
                    )
                )
            else:
                result = ray.get(
                    actor.analyze.remote(
                        image_b64,
                        **kwargs,
                    )
                )
            return result
        except Exception as e:
            logger.error("Tool agent %s call failed: %s", tool_name, e)
            return f"Error: {e}"

    def shutdown(self, tool_names: Optional[List[str]] = None):
        names_to_kill = []
        if tool_names:
            for name in tool_names:
                names_to_kill.extend(self._replica_map.pop(name, [name]))
                self._configs.pop(name, None)
                self._robin.pop(name, None)
        else:
            names_to_kill = list(self._actors.keys())
            self._replica_map.clear()
            self._configs.clear()
            self._robin.clear()

        for actor_name in names_to_kill:
            actor = self._actors.pop(actor_name, None)
            if actor:
                try:
                    ray.kill(actor)
                    logger.info("Killed tool agent: %s", actor_name)
                except Exception as e:
                    logger.warning("Failed to kill %s: %s", actor_name, e)

    def status(self) -> Dict[str, Dict[str, Any]]:
        result = {}
        for name, actor in self._actors.items():
            try:
                health = ray.get(actor.health_check.remote())
                result[name] = {"status": "healthy", **health}
            except Exception as e:
                result[name] = {"status": "error", "error": str(e)}
        return result