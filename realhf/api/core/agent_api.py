# Copyright 2025 Ant Group Inc.
# Licensed under the Apache License, Version 2.0 (the "License").

# Experimental APIs of RL agents.

import asyncio
from abc import ABC
from typing import List

from realhf.api.core.config import AgentAbstraction
from realhf.api.core.data_api import SequenceSample
from realhf.api.core.env_api import EnvironmentService


class Agent(ABC):
    # TODO: implement type checking inside each queue.
    async def collect_trajectory(
        self,
        prompt: SequenceSample,
        env: EnvironmentService,
        obs_queue: asyncio.Queue,
        act_queue: asyncio.Queue,
    ) -> List[SequenceSample]:
        raise NotImplementedError()


ALL_AGNETS = {}


def register_agent(name, cls_):
    assert name not in ALL_AGNETS
    ALL_AGNETS[name] = cls_


def make_agent(cfg: AgentAbstraction) -> Agent:
    return ALL_AGNETS[cfg.type_](**cfg.args)
