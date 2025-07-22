import abc
import asyncio
from typing import Any, Dict, List, Tuple

from realhf.api.core.config import EnvServiceAbstraction


class EnvironmentService(abc.ABC):

    # TODO: import gymnasium, use its types and signatures
    async def step(self, action: Any) -> Tuple[Any, Any, bool, bool, Dict]:
        # obs, reward, terminated, truncated, info
        raise NotImplementedError()

    async def reset(self, seed=None, options=None) -> Tuple[Any, Dict]:
        # obs, info
        raise NotImplementedError()


ALL_ENV_CLASSES = {}


def register_environment(name, env_cls):
    assert name not in ALL_ENV_CLASSES
    assert "/" not in name
    ALL_ENV_CLASSES[name] = env_cls


class NullEnvironment:

    async def step(self, action):
        await asyncio.sleep(1)
        # obs, reward, terminated, truncated, info
        return None, 0.0, True, False, {}

    async def reset(self, seed=None, options=None) -> Tuple[Any, Dict]:
        await asyncio.sleep(0.1)
        return None, {}


register_environment("null", NullEnvironment)


def make_env(
    cfg: EnvServiceAbstraction,
) -> EnvironmentService:
    return ALL_ENV_CLASSES[cfg.type_](**cfg.args)
