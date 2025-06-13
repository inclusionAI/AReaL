# Copyright 2025 Ant Group Inc.
# Licensed under the Apache License, Version 2.0 (the "License").

import asyncio
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, List, Optional, Tuple

import gymnasium as gym
import numpy as np

from arealite.api.cli_args import EnvConfig, TrainingArgs
from arealite.api.env_api import Environment
from functioncall.code.local_verify import code_verify as local_code_verify
from functioncall.code.verify import code_verify
from functioncall.math.verify import math_verify
from realhf.base import logging
from realhf.impl.dataset.math_code_dataset import load_metadata
from realhf.impl.dataset.math_parser import parse_lines_in_parallel

ENABLE_FUNCTION_CALL = True if os.getenv("FUNCTIONCALL_SERVICE_DOMAIN", "") else False
math_verify_call = math_verify if ENABLE_FUNCTION_CALL else parse_lines_in_parallel
code_verify_call = code_verify if ENABLE_FUNCTION_CALL else local_code_verify

logger = logging.getLogger("Math Code Single Step Environment")


@lru_cache(maxsize=128)
def _load_metadata_cached(dataset_path: str):
    """Cached version of load_metadata to avoid reloading metadata each time."""
    return load_metadata(dataset_path)


def extract_code(text, min_length=20):
    """Extract code blocks from text."""
    code_pattern = r"(?i)```(?:python|py|cpp|CPP)?\s*\n?(.*?)\n?```"
    code_blocks = re.findall(code_pattern, text, re.DOTALL)
    valid_blocks = []
    for block in code_blocks:
        clean_block = block.strip()
        if len(clean_block) < min_length:
            continue
        valid_blocks.append(clean_block)

    if not valid_blocks:
        return None
    # return the last code block
    return valid_blocks[-1]


@dataclass
class MathCodeAction:
    qid: str
    answer: str


@dataclass
class MathCodeObs:
    qid: str
    prompt: str


class MathCodeSingleStepEnv(Environment):
    """Math and Code single-step verification environment."""

    def __init__(self, args: TrainingArgs, config: EnvConfig):
        super().__init__(args, config)
        env_config = config.math_code_single_step
        self.dataset_path = env_config.dataset_path
        self.id2info, _ = _load_metadata_cached(self.dataset_path)

        # TODO: define observation and action spaces

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[Any, dict]:
        """Reset the environment."""
        super().reset(seed=seed)
        prompt = options["prompt"]
        qid = options["qid"]
        # Return dummy observation and info
        return MathCodeObs(qid=qid, prompt=prompt), {}

    def step(
        self, action: MathCodeAction
    ) -> Tuple[MathCodeObs, float, bool, bool, dict]:
        """Execute one step in the environment."""
        qid = action.qid
        answer = action.answer

        qid = qid.split("@")[0]
        cur_task = self.id2info[qid]["task"]

        if cur_task == "math":
            # Run math verification
            format_rewards = math_verify_call(self.id2info, [answer], [qid])
        elif cur_task == "code":
            # Extract code blocks and run code verification
            extracted_answer = extract_code(answer)
            format_rewards = code_verify_call(self.id2info, [extracted_answer], [qid])
        else:
            raise NotImplementedError(f"Task type '{cur_task}' not implemented")

        # Return: observation, reward, terminated, truncated, info
        terminated = True  # Single step environment always terminates
        truncated = False
        info = {"task": cur_task, "qid": qid}

        return None, format_rewards, terminated, truncated, info
