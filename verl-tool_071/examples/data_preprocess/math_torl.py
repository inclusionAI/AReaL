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
"""
Preprocess the GSM8k dataset to parquet format
"""

import os
import datasets
import fire
from verl.utils.hdfs_io import copy, makedirs
import argparse

from verl.utils.reward_score.math import remove_boxed, last_boxed_only_string
from verl.utils.reward_score import prime_math

def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))

system_prompt1 = '''A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. Please integrate natural language reasoning with programs to solve the problem above, and put your final answer within \\boxed{}.:
'''

system_prompt2 = '''A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. User: Please integrate natural language reasoning with programs to solve the problem above. If you want to test any python code, writing it inside <python> and  </python> tags following with <output>. Please put your final answer within \\boxed{}.:
'''

system_prompt3 = '''A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. User: Please integrate natural language reasoning with programs to solve the problem above. If you want to run any python code, write code in the python markdown code block and the execution will be appended in an output code block like "```python\nyou code here\n```\n```output\nresult here\n```". Please put your final answer within \\boxed{}.'''

system_prompt4 = '''A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. User: Please put your final answer within \\boxed{}.'''

system_prompt5 = """\
A conversation between user and assistant. The user asks a question, and the assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. Please integrate natural language reasoning with programs to solve the problem. That means during the thinking, the assistant can run any python code by writing in the python markdown code block, then the stdout and stderr result will be appended in an output code block like "```python\nyou code here\n```\n```output\nresult here\n```". Please put your final answer within \\boxed{}."""

system_prompt6 = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. User: Please integrate natural language reasoning with programs to solve the problem above. If you want to run any code, include "<|calling system for feedback|>" at the end of your response for the current turn. Then the system will execute the code in the markdown block and provide the standard output and error. If you think the solution is complete and don't need to test, don't include "<|calling system for feedback|>" in the response and put your final answer within \\boxed{}. 
"""

system_prompt7 = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. User: Please integrate natural language reasoning with programs to solve the problem. That means during the thinking, the assistant can run any python code by writing in the python markdown code block, then the stdout and stderr result will be appended in an output code block like "```python\nyou code here\n```\n```output\nresult here\n```". Please put your final answer within \\boxed{}.
"""

def main(
    data_source='DigitalLearningGmbH/MATH-lighteval',
    local_dir='~/data/math_torl',
    hdfs_dir=None,
    level:str = 'hard',
    sys_prompt_version: str = 'v1',
):
    
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(data_source, trust_remote_code=True)
    train_dataset = dataset['train']
    test_dataset = dataset['test']
    
    global system_prompt
    v_idx = sys_prompt_version.split('v')[-1]
    system_prompt_version = int(v_idx)
    system_prompt = eval(f'system_prompt{system_prompt_version}')
    print(f"Using system prompt version {system_prompt_version}...", flush=True)
    
    # easy: level 1
    # medium: level 1-4
    # hard: level 3-5
    
    if level == 'easy':
        level_range = (1, 2)
    elif level == 'medium':
        level_range = (1, 5)
    elif level == 'hard':
        level_range = (3, 6)
    else:
        raise ValueError(f"Unknown level: {level}. Please choose from easy, medium, or hard.")
    train_dataset = train_dataset.filter(lambda x: x['level'] in [f"Level {i}" for i in range(level_range[0], level_range[1])])
    test_dataset = test_dataset.filter(lambda x: x['level'] in [f"Level {i}" for i in range(level_range[0], level_range[1])])
    math500_test_dataset = datasets.load_dataset('HuggingFaceH4/MATH-500', split='test')
    
    # add a row to each data item that represents a unique id
    def make_map_fn(split, data_source):

        def process_fn(example, idx):
            question = example.pop('problem')
            answer = example.pop('solution')
            solution = extract_solution(answer)
            
            data = {
                "data_source": data_source,
                "prompt": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": question
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'question': question,
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train', data_source), with_indices=True, remove_columns=train_dataset.column_names)
    test_dataset = test_dataset.map(function=make_map_fn('test', data_source), with_indices=True, remove_columns=test_dataset.column_names)
    math500_test_dataset = math500_test_dataset.map(function=make_map_fn('test', 'HuggingFaceH4/MATH-500'), with_indices=True, remove_columns=math500_test_dataset.column_names)
    
    print(train_dataset)
    print(train_dataset[0])
    

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
    math500_test_dataset.to_parquet(os.path.join(local_dir, 'math500_test.parquet'))
    
    # aime24
    aime24_dataset = datasets.load_dataset('Maxwell-Jia/AIME_2024', split='train') # actually test set
    def make_map_fn(split, data_source):

        def process_fn(example, idx):
            question = example.pop('Problem')
            answer = str(example.pop('Answer'))
            solution = example.pop('Solution')
            
            data = {
                "data_source": data_source,
                "prompt": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": question
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'question': question,
                }
            }
            return data

        return process_fn
    
    aime24_dataset = aime24_dataset.map(function=make_map_fn('test', 'aime24'), with_indices=True, remove_columns=aime24_dataset.column_names)
    aime24_dataset.to_parquet(os.path.join(local_dir, 'aime24_test.parquet'))
    print(aime24_dataset)
    print(aime24_dataset[0])
    
    # aime25
    aime25_dataset = datasets.load_dataset('opencompass/AIME2025', 'AIME2025-I', split='test') # actually test set
    aime25_dataset2 = datasets.load_dataset('opencompass/AIME2025', 'AIME2025-II', split='test') # actually test set
    # concatenate the two datasets
    aime25_dataset = datasets.concatenate_datasets([aime25_dataset, aime25_dataset2])
    
    def make_map_fn(split, data_source):

        def process_fn(example, idx):
            question = example.pop('question')
            answer = str(example.pop('answer'))
            
            data = {
                "data_source": data_source,
                "prompt": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": question
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'question': question,
                }
            }
            return data

        return process_fn

    aime25_dataset = aime25_dataset.map(function=make_map_fn('test', 'aime25'), with_indices=True)
    aime25_dataset.to_parquet(os.path.join(local_dir, 'aime25_test.parquet'))
    print(aime25_dataset)
    print(aime25_dataset[0])
    
    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
    

if __name__ == '__main__':
    fire.Fire(main)
    
"""
python examples/data_preprocess/math_torl.py --data_source DigitalLearningGmbH/MATH-lighteval --local_dir data/math_torl
python examples/data_preprocess/math_torl.py --data_source DigitalLearningGmbH/MATH-lighteval --local_dir data/math_torl_v2 --sys_prompt_version v2
python examples/data_preprocess/math_torl.py --data_source DigitalLearningGmbH/MATH-lighteval --local_dir data/math_torl_v3 --sys_prompt_version v3
python examples/data_preprocess/math_torl.py --data_source DigitalLearningGmbH/MATH-lighteval --local_dir data/math_torl_v4 --sys_prompt_version v4
python examples/data_preprocess/math_torl.py --data_source DigitalLearningGmbH/MATH-lighteval --local_dir data/math_torl_v5 --sys_prompt_version v5
python examples/data_preprocess/math_torl.py --data_source DigitalLearningGmbH/MATH-lighteval --local_dir data/math_torl_v6 --sys_prompt_version v6
python examples/data_preprocess/math_torl.py --data_source DigitalLearningGmbH/MATH-lighteval --local_dir data/math_torl_v7 --sys_prompt_version v7
"""