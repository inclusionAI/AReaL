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
from .reward_score.torl_math import compute_score as torl_compute_score
from verl.workers.reward_manager import register
from .torl import ToRLRewardManager
import regex as re

from typing import Union, List
def deepsearch_compute_score(solution_str, ground_truth: Union[List[str], str]):
    if isinstance(ground_truth, str):
        ground_truth = [ground_truth]
    score = 0.0
    for gt in ground_truth:
        score = max(score, torl_compute_score(solution_str, gt))
    return score

@register("deepsearch")
class PixelReasonerRewardManager(ToRLRewardManager):
    """
    A reward manager for the Pixel Reasoner.
    It uses the TORL framework to compute rewards based on the outputs of the model.
    """
    name = "deepsearch"
    
    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key='data_source', **kwargs) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = deepsearch_compute_score
        self.reward_fn_key = reward_fn_key
        self.step = None
        self.add_tool_call_reward = True # +0.1 if the response contains a tool call
        self.add_format_penalty = True # -0.5 if the response does not start with <think> and end with </think>

    def add_additional_penalties(self, response: str, data_i, scores_i: dict):
        # 1.4 format penalty
        if self.add_format_penalty:
            # check if <think> exists in the response
            #  and if \\boxed{} exists in the response
            think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
            answer_match = re.search(r"\\boxed\{.*?\}", response)
            if not think_match or not answer_match:
                scores_i['score'] = -1
                scores_i['format_penalty'] = 1
            else:
                scores_i['format_penalty'] = 0
        
        scores_i['score'] = scores_i['accuracy']
        
        if "tool_interact_info" in data_i.non_tensor_batch:
            if self.add_tool_call_reward:
                tool_interact_info = data_i.non_tensor_batch.get('tool_interact_info', None)
                num_turn = len(tool_interact_info) if tool_interact_info is not None else 0
                num_valid_action = sum([1 for t in tool_interact_info if t.get('valid_action', False)]) if tool_interact_info is not None else 0
                if num_valid_action > 0:
                    scores_i['score'] += 0.1
                    scores_i['tool_call_reward'] = 1
                else:
                    scores_i['tool_call_reward'] = 0
        
        return scores_i