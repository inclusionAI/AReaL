# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import time
import json
import hashlib
import random
import os
import json
import subprocess
import time
import regex as re
from pathlib import Path
import uuid
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from typing import Tuple
from tqdm import tqdm
import torch
from collections import defaultdict
from verl import DataProto
from verl.protocol import collate_fn
from .reward_score import _default_compute_score
from verl.workers.reward_manager import register
from verl_tool.servers.tools.utils.sql_executor import score as sql_score_func

THINK_START, THINK_END = "<think>", "</think>"
SQL_START, SQL_END = "<sql>", "</sql>"
SOLUTION_START, SOLUTION_END = "<solution>", "</solution>"
OBS_START, OBS_END = "<observation>", "</observation>"

def parse_action(action: str, tag_type: str = "sql") -> Tuple[str, bool]:
    """
    Parse the raw action string to extract SQL code from either <sql></sql> or <solution></solution> tags.
    
    Args:
        action: Raw action string containing SQL code
        tag_type: Type of tag to extract ("sql" or "solution")
        
    Returns:
        Tuple containing the extracted code and a validity flag
    """
    tag_start_map = {
        "sql": SQL_START,
        "solution": SOLUTION_START
    }
    tag_end_map = {
        "sql": SQL_END,
        "solution": SOLUTION_END
    }

    # Find the last occurrence of the start tag
    start_tag = tag_start_map[tag_type]
    end_tag = tag_end_map[tag_type]
    
    sql_code_start_idx = action.rfind(start_tag)
    if sql_code_start_idx == -1:
        return "", False
    
    # Find the corresponding end tag after the start tag
    sql_code_end_idx = action.find(end_tag, sql_code_start_idx + len(start_tag))
    if sql_code_end_idx == -1:
        return "", False
    
    # Extract the content between the tags
    sql_code = action[sql_code_start_idx + len(start_tag):sql_code_end_idx].strip()
    return sql_code, True

# Copied from SkyRL-SQL/skyrl_gym/envs/sql/utils.py
def verify_format_and_extract(output: str, action_list: list) -> Tuple[str, bool]:
    """
    Verify the format of the output and extract thoughts, solution, and SQL code.
    Args:
        output (str): The output string to verify and extract from.
    Returns:
        solution (str): The extracted solution SQL code.
        is_correct_format (bool): Whether the output format is correct.
    """
    is_correct_format = True
    # verify the <solution> tags in the last action
    
    if not re.search(rf"{SOLUTION_START}.*?{SOLUTION_END}", output, re.S):
        is_correct_format = False

    # verify the <think> tags in as starts in each action
    for action in action_list:
        if not (action.startswith(THINK_START) and re.search(rf"{THINK_START}.*?{THINK_END}", action, re.S)):
            is_correct_format = False
            break
    
    solution, found_solution = parse_action(output, "solution")
    
    if not found_solution:
        solution, found_solution = parse_action(output, "sql")
        
    return solution, is_correct_format

def hash_string(s):
    return hashlib.sha256(s.encode()).hexdigest()

@register("sqlcoder")
class SQLCoderRewardManager:
    def __init__(
        self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", **kwargs) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score if compute_score else _default_compute_score
        self.reward_fn_key = reward_fn_key
        self.step = 0

    def __call__(self, data: DataProto, return_dict=False):
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        
        # reward extra info every key of it is a default len(data) list filled with None
        prompt_ids = data.batch['prompts']
        prompt_length = prompt_ids.shape[-1]
        response_ids = data.batch['responses']
        valid_prompt_length = data.batch['attention_mask'][:, :prompt_length].sum(dim=-1)
        valid_response_length = data.batch['attention_mask'][:, prompt_length:].sum(dim=-1)
        reward_extra_info = defaultdict(list)
        
        scores = []
        for i in tqdm(range(len(data)), desc="Processing SQLCoder responses", total=len(data), disable=True):
            # Get the entire response for format checking
            valid_response_length_i = valid_response_length[i].item()
            response = self.tokenizer.decode(
                response_ids[i][:valid_response_length_i], skip_special_tokens=False
            )
            # Get database and ground truth information
            extra_info = data[i].non_tensor_batch.get('extra_info', {})
            meta = {
                "db_id": extra_info.get("db_id"),
                "gold_sql": extra_info.get("gt_sql"),
                "cmp_method": "bird",
                "db_path": extra_info.get("db_path")
            }
            score = {}
            action_list = [x.get('action', "") for x in data[i].non_tensor_batch['tool_interact_info']]
            
            parsed_solution, is_format_correct = verify_format_and_extract(response, action_list)
            if is_format_correct:
                score['is_format_correct'] = 1.
            else:
                score['is_format_correct'] = 0.
                
            execution_score = sql_score_func(parsed_solution, meta)[0] if parsed_solution else 0.0
            score['accuracy'] = execution_score
            
            score['score'] = score['accuracy'] if is_format_correct else -1.0 # final score
            
            scores.append(score)

            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
                if self.num_examine == 1:
                    reward = score["accuracy"] # for validation
            else:
                if self.num_examine == 1:
                    reward = score if score > 0 else 0.0
                else:
                    reward = score

            reward_tensor[i, valid_response_length_i - 1] = reward
            
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor