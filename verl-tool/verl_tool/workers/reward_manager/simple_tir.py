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
import numpy as np
from pathlib import Path
from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
import torch
from collections import defaultdict
import regex as re
from contextlib import redirect_stderr, redirect_stdout
def extract_box_contents(text):
    """
    Extract contents from \box{} commands in a string.
    
    Args:
        text (str): Input string containing \box{} commands
        
    Returns:
        list: List of contents found inside \box{} commands
    """
    # Pattern to match \box{content} with proper brace matching
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    
    # Find all matches
    matches = re.findall(pattern, text)
    return matches[-1] if matches else ""

def extract_answer(text):
    """
    Extract the final answer from the text based on the mode.
    
    Args:
        text (str): Input string containing the answer
        mode (str): Mode of extraction ('math' or 'lcb_code')
    Returns:
        str: Extracted answer
    """
    
    all_valid_python_code = re.findall(r"```\n?python(.*?)```", text, re.DOTALL)
    if not all_valid_python_code:
        all_valid_python_code = re.findall(r"```\n?(.*?)```", text, re.DOTALL)
    last_code_block = all_valid_python_code[-1] if all_valid_python_code else ""
    if "final_answer" in last_code_block:
        final_answer_func = "_final_answer_result = None\ndef final_answer(answer):\n    global _final_answer_result\n    _final_answer_result = answer\n"
        # execute the code block to get the return value of final_answer
        local_vars = {}
        global_vars = {}
        try:
            with redirect_stdout(None), redirect_stderr(None):
                exec(final_answer_func + last_code_block, global_vars, local_vars)
            return str(local_vars.get('_final_answer_result', global_vars.get('_final_answer_result', "")))
        except Exception as e:
            return ""
    else:
        return extract_box_contents(text)

    
@register("simpletir")
class SimpleTIRRewardManager:
    """The reward manager.
    """
    name="simpletir"

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key='data_source', **kwargs) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        # self.compute_score = compute_score if compute_score else _default_compute_score
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.step = None
        self.add_format_think_penalty = False # -0.5 if not begines with <think> and end with </think>
        self.add_format_answer_penalty = False # -0.5 if not having <answer> </answer>
        self.add_valid_action_penalty = False # -0.25 if num turns > 0 not action not valid
        self.add_unfinished_traj_penalty = False # -0.25 if the traj is not finished
        self.add_no_tool_interact_penalty = False # -0.25 if the traj's num turn is 0, no interaction at all
        self.add_code_exec_penalty = False # -0.25 if the execution has an error.

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""
        # check the last step index
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        for i in range(len(data)):
            score = {}
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            extra_info = data_item.non_tensor_batch.get('extra_info', None)
            extracted_answer = extract_answer(response_str)
            
            _score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            ) # 1 or -1
            score['accuracy'] = 1. if _score > 0 else 0.
            score['score'] = _score
            score['has_answer'] = 1. if extracted_answer else 0.

            if score['accuracy'] > 0:
                reward_extra_info['correct_response_length'].append(valid_response_length)
            else:
                reward_extra_info['wrong_response_length'].append(valid_response_length)

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

            reward_tensor[i, valid_response_length - 1] = reward 

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print(f"[score]", score)
                
        correct_response_length_mean = np.mean(reward_extra_info['correct_response_length']) if reward_extra_info['correct_response_length'] else None
        wrong_response_length_mean = np.mean(reward_extra_info['wrong_response_length']) if reward_extra_info['wrong_response_length'] else None
        reward_extra_info['correct_response_length'] = [correct_response_length_mean] * len(reward_tensor)
        reward_extra_info['wrong_response_length'] = [wrong_response_length_mean] * len(reward_tensor)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": dict(sorted(reward_extra_info.items()))
            }
        else:
            return reward_tensor
