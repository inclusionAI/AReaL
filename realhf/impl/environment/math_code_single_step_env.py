# Copyright 2025 Ant Group Inc.

import asyncio
import os
import re
from typing import List, Tuple

from functioncall.code.local_verify import code_verify as local_code_verify
from functioncall.code.verify import code_verify
from functioncall.math.verify import math_verify
from realhf.api.core.env_api import EnvironmentService, register_environment
from realhf.base import logging
from realhf.impl.dataset.math_code_dataset import load_metadata
from realhf.impl.dataset.math_parser import parse_lines_in_parallel

ENABLE_FUNCTION_CALL = True if os.getenv("FUNCTIONCALL_SERVICE_DOMAIN", "") else False
math_verify_call = math_verify if ENABLE_FUNCTION_CALL else parse_lines_in_parallel
code_verify_call = code_verify if ENABLE_FUNCTION_CALL else local_code_verify

logger = logging.getLogger("Math Single Step Environment")


def extract_code(text, min_length=20):
    code_pattern = r"(?i)```(?:python|py|cpp|CPP)?\s*\n?(.*?)\n?```"
    code_blocks = re.findall(code_pattern, text, re.DOTALL)
    valid_blocks = []
    for block in code_blocks:
        clean_block = block.strip()
        if len(clean_block) < min_length:
            continue

        valid_blocks.append(clean_block)

    if not valid_blocks:
        # logger.warning(f"failed to extract python code from {text}")
        return None
    # return the last code block
    return valid_blocks[-1]


class MathCodeSingleStepEnv(EnvironmentService):
    def __init__(self, dataset_path: str):
        self.id2info, _ = load_metadata(dataset_path)

    async def reset(self, seed=None, options=None):
        return None, {}

    async def step(self, action: Tuple[str, List[str]]):
        qid, answers = action
        group_size = len(answers)
        qid = qid.split("@")[0]
        cur_task = self.id2info[qid]["task"]

        if cur_task == "math":
            format_rewards = await asyncio.to_thread(
                math_verify_call,
                self.id2info,
                answers,
                [qid for _ in range(group_size)],
                max_workers=1,
            )
        elif cur_task == "code":
            answers = [extract_code(x) for x in answers]
            format_rewards = await asyncio.to_thread(
                code_verify_call,
                self.id2info,
                answers,
                [qid for _ in range(group_size)],
                max_workers=1,
            )
        else:
            raise NotImplementedError()

        return None, format_rewards, True, False, {}


register_environment("math-code-single-step", MathCodeSingleStepEnv)
