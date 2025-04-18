# Copyright 2025 Ant Group Inc.

import asyncio
import os
from typing import List, Tuple

from functioncall.math.verify import math_verify
from realhf.api.core.env_api import EnvironmentService, register_environment
from realhf.base import logging
from realhf.impl.dataset.math_code_dataset import load_metadata
from realhf.impl.dataset.math_parser import parse_lines_in_parallel

ENABLE_FUNCTION_CALL = True if os.getenv("FUNCTIONCALL_SERVICE_DOMAIN", "") else False
math_verify_call = math_verify if ENABLE_FUNCTION_CALL else parse_lines_in_parallel

logger = logging.getLogger("Math Single Step Environment")


class MathSingleStepEnv(EnvironmentService):
    def __init__(self, dataset_path: str):
        self.id2info, _ = load_metadata(dataset_path)

    async def reset(self, seed=None, options=None):
        return None, {}

    async def step(self, action: Tuple[str, List[str]]):
        qid, answers = action
        group_size = len(answers)
        format_rewards = await asyncio.to_thread(
            math_verify_call,
            self.id2info,
            answers,
            [qid for _ in range(group_size)],
        )
        return None, format_rewards, True, False, {}


register_environment("math-single-step", MathSingleStepEnv)
