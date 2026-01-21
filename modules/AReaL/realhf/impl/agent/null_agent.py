# Copyright 2025 Ant Group Inc.
# Licensed under the Apache License, Version 2.0 (the "License").

# A null agent used for testing
import asyncio
import copy
import random
from typing import List

from realhf.api.core.agent_api import Agent, register_agent
from realhf.api.core.data_api import SequenceSample
from realhf.api.core.env_api import EnvironmentService
from realhf.api.core.model_api import (
    BundledGenerationOutputs,
    GenerationHyperparameters,
)
from realhf.base import logging, testing

logger = logging.getLogger("Null Agent")


class NullAgent(Agent):
    OBS_PUT_CNT = 0
    ACT_GET_CNT = 0

    def __init__(self, episode_length: int = 1, traj_size: int = 1):
        self.episode_len = episode_length
        self.traj_size = traj_size

    async def collect_trajectory(
        self,
        prompt: SequenceSample,
        env: EnvironmentService,
        obs_queue: asyncio.Queue,
        act_queue: asyncio.Queue,
    ) -> List[SequenceSample]:

        qid = prompt.ids[0]
        prompt_token_ids = [
            random.randint(0, testing.TESTING_MODEL_VOCAB_SIZE - 1)
            for _ in range(random.randint(0, 64))
        ]
        for step in range(self.episode_len):
            await obs_queue.put((qid, prompt_token_ids, GenerationHyperparameters()))
            self.OBS_PUT_CNT += 1
            act = await act_queue.get()
            self.ACT_GET_CNT += 1
            assert isinstance(act, BundledGenerationOutputs)

        ids = [str(qid) + f"-{idx}" for idx in range(self.traj_size)]
        traj = [copy.deepcopy(prompt) for _ in range(self.traj_size)]
        for t, i in zip(traj, ids):
            t.ids[0] = i
        return traj


register_agent("null", NullAgent)
