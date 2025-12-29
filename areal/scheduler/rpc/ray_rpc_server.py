import os
import traceback
from concurrent.futures import Future
from typing import Any

import ray

from areal.api.cli_args import BaseExperimentConfig
from areal.api.engine_api import InferenceEngine, TrainEngine
from areal.scheduler.rpc.rtensor import RTensor
from areal.utils import logging, name_resolve, seeding
from areal.utils.data import (
    broadcast_tensor_container,
    tensor_container_to,
)
from areal.utils.dynamic_import import import_from_string


@ray.remote
class RayRPCServer:
    """
    Ray engine container. Represents either:
    - one training world rank, or
    - one rollout instance

    Supports multiple named engines per worker for colocation scenarios.

    Placement group scheduling is controlled by the scheduler.
    The actor is only responsible for the engine lifecycle and method calls
    within this process.
    """

    def __init__(self):
        self._engines: dict[str, TrainEngine | InferenceEngine] = {}
        self._default_engine_name: str | None = None  # For backward compatibility
        self.logger = logging.getLogger("RayRPCServer")

    def _get_device(self):
        # lazy resolve the device inside worker process
        from areal.platforms import current_platform

        return current_platform.current_device()

    def ping(self) -> str:
        return "ok"

    def configure(self, config: BaseExperimentConfig, role: str, rank: int) -> None:
        name_resolve.reconfigure(config.cluster.name_resolve)
        # Set seed for any TrainEngine instances
        for engine in self._engines.values():
            if isinstance(engine, TrainEngine):
                seeding.set_random_seed(config.seed, key=f"{role}{rank}")
                break
        self.logger.info(f"RayRPCServer configured for role role={role}, rank={rank}")

    def set_env(self, env: dict[str, str]) -> None:
        for k, v in env.items():
            os.environ[str(k)] = str(v)

    def create_engine(
        self,
        engine_path: str,
        *init_args,
        engine_name: str | None = None,
        **init_kwargs,
    ) -> None:
        try:
            engine_class = import_from_string(engine_path)
            self.logger.debug(f"Initializing engine {engine_class}")
            if not issubclass(engine_class, (TrainEngine, InferenceEngine)):
                raise TypeError(
                    f"Engine class must be a TrainEngine or InferenceEngine, but got {engine_class}"
                )
            engine = engine_class(*init_args, **init_kwargs)

            # Use engine_name if provided, otherwise generate a default name
            if engine_name is None:
                engine_name = f"engine_{len(self._engines)}"
            self._engines[engine_name] = engine

            # Track first engine as default for backward compatibility
            if self._default_engine_name is None:
                self._default_engine_name = engine_name

            self.logger.info(
                f"RayRPCServer Engine '{engine_path}' instantiated as '{engine_name}'!"
            )
        except Exception as e:
            self.logger.error(
                f"RayRPCServer failed to create engine '{engine_path}' : {e}\n"
                f"{traceback.format_exc()}"
            )
            raise

    def call(self, method: str, *args, engine_name: str | None = None, **kwargs) -> Any:
        self.logger.debug(
            f"Calling {method} on engine '{engine_name}' with arguments {args=} {kwargs=}"
        )

        # Resolve engine: use provided name, fallback to default, or error
        if engine_name is None:
            engine_name = self._default_engine_name
        if engine_name is None or engine_name not in self._engines:
            raise RuntimeError(
                f"Engine '{engine_name}' not found. "
                f"Available engines: {list(self._engines.keys())}. "
                "Call create_engine() first."
            )
        engine = self._engines[engine_name]

        raw_args = list(args)
        raw_kwargs = kwargs.copy()
        # fetch remote tensors if any
        args = RTensor.localize(raw_args)
        kwargs = RTensor.localize(raw_kwargs)

        should_broadcast = kwargs.pop("should_broadcast", True)

        # keep broadcast behavior the same as RPCServer
        try:
            if should_broadcast and isinstance(engine, TrainEngine):
                device = self._get_device()

                args = tensor_container_to(args, device)
                args = broadcast_tensor_container(
                    args,
                    src_rank=engine.current_data_parallel_head(),
                    group=engine.context_and_model_parallel_group,
                )
                kwargs = tensor_container_to(kwargs, device)
                kwargs = broadcast_tensor_container(
                    kwargs,
                    src_rank=engine.current_data_parallel_head(),
                    group=engine.context_and_model_parallel_group,
                )
        except Exception as e:
            self.logger.error(
                f"RayRPCServer broadcast failed for '{method}': {e}\n"
                f"{traceback.format_exc()}"
            )
            raise

        try:
            fn = getattr(engine, method)
            result = fn(*args, **kwargs)
            if isinstance(result, Future):
                result = result.result()
            # Convert all tensors to RTensors and store the tensor locally
            layout = RTensor.extract_layout(
                result, layouts=dict(args=raw_args, kwargs=raw_kwargs), node_addr=""
            )
            if layout is not None:
                result = RTensor.remotize(result, layout, node_addr="")
            # put back to cpu to mimic RPCServer encode/decode
            result = tensor_container_to(result, "cpu")
            self.logger.debug(f"Successfully completed RayRPCServer call {result}")
            return result
        except Exception as e:
            self.logger.error(
                f"RayRPCServer Engine method '{method}' failed: {e}\n"
                f"{traceback.format_exc()}"
            )
            raise

    def destroy(self) -> None:
        # Destroy all engines
        for engine_name, engine in list(self._engines.items()):
            try:
                engine.destroy()
                self.logger.info(f"RayRPCServer Engine '{engine_name}' destroyed")
            except Exception as e:
                self.logger.error(
                    f"RayRPCServer error destroying engine '{engine_name}': {e}"
                )
        self._engines.clear()
        self._default_engine_name = None
        ray.actor.exit_actor()
