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
from pathlib import Path
import datasets
import os
import fire
import numpy as np
from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math import remove_boxed, last_boxed_only_string



execution_prompt = """\
Answer the given coding question. You must conduct reasoning inside <think> and </think> first before you can finally output the final program. During the thinking, you can test your program by writing it inside ```python and ``` tags following with "```output". The code will be executed, and the terminal output (standard output and standard error) will be returned between <output> and </output>. Each program between ```python and ``` tags are independent program. You can test Python codes as many times as you want. If you find no further code execution needed, you can then give the final program in a markdown code block like this: ```python\nyour code here\n``` without appending anything,. The final program will be evaluated against the hidden test cases. If the final program passes all the test cases, you will get a reward. If the final program fails any of the test cases, you will get a penalty.
"""

naive_instruction = "Let's think step by step and generate the final program in a markdown code block like this: ```python\nyour code here\n```."
naive_execution_prompt = """\
Let's think step by step and generate the correct program for this coding question. You are able to run the python code in the markdown code block and the output will be provided to you in the ````output` block. Put your final program in a markdown code block like this: ```python\nyour code here\n```.
"""

coder_instruction = """\
Let's think step by step and generate the correct program for this coding question. You should attempt multiple times before give the final program.
In each attempt, you should 
- test your program by reviewing the code syntax and logic, and fix any potential issues in the next attempt.
- imagine a set of test cases based on your understanding of the problem and the constraints. 
- You then need to test your program with these test cases. Since you are not able to run the program in a real environment, you need to use text to simulate the program running and think loudly to describe how each variable changes during the execution. Finally, see whether the program produces the expected output.
- if the program fails any of the test cases, you need to debug the program and fix the issues in the next attempt.
- if the program passes all the test cases, you can then give the final program in a markdown code block like this: ```python\nyour code here\n```.

You are also allowed to analyze the problem with any other domain-specific knowledge you have, like math, physics, etc to help you solve the problem.

Now start thinking and generate the final program in a markdown code block like this: ```python\nyour code here\n```.
"""

math_system_prompt = '''A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. User: Please integrate natural language reasoning with programs to solve the problem above, and put your final answer within \\boxed{}.
'''

mathcoder_system_prompt = '''A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. User: Please integrate natural language reasoning with programs to solve the problem above. The execution result for a code block will be put in the a "output" markdown block if you want to run it. For math problems, please put your final answer within \\boxed{}. For code problems, please put your final answer in a markdown code block like this: ```python\nyour code here\n```.
'''

mathcoder_system_prompt_v2 = '''A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. Please integrate natural language reasoning with programs to solve the problem above. If you want to run any python code, write code in the python markdown code block and the execution will be appended in an output code block like "```python\nyour code here\n```\n```output\nresult here\n```". For math problems, please put your final answer within \\boxed{}. For code problems, please put your final answer in a markdown code block like this: ```python\nyour code here\n```.
'''

mathcoder_system_prompt_v3 = '''A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. Please integrate natural language reasoning with programs to solve the problem above. For math problems, please put your final answer within \\boxed{}. For code problems, please put your final answer in a markdown code block like this: ```python\nyour code here\n```.
'''

### Utils ###
def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


### Main Preprocessing Function ###
def main(
    code_dataset_path: str = 'CodeDPO/AceCoderV2-mini-processed',
    math_dataset_path: str = 'DigitalLearningGmbH/MATH-lighteval',
    local_dir: str = 'data/mathcoder',
    hdfs_dir: str = None,
    level: str = 'hard',
    add_execution_prompt: bool = False,
    detaield_instruction: bool = False,
    sys_prompt_version: str = 'v1',
):
    global mathcoder_system_prompt
    if sys_prompt_version == 'v2':
        mathcoder_system_prompt = mathcoder_system_prompt_v2
    elif sys_prompt_version == 'v3':
        mathcoder_system_prompt = mathcoder_system_prompt_v3
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    _process_acecoder(code_dataset_path, local_dir, add_execution_prompt, detaield_instruction)
    _process_math(math_dataset_path, local_dir, hdfs_dir, level)
    _process_torl_math(code_dataset_path, local_dir)

### AceCoder Logic ###
def _process_acecoder(dataset_path, local_dir, add_execution_prompt, detaield_instruction):
    print(f"Loading AceCoder from {dataset_path}")
    dataset = datasets.load_dataset(dataset_path, split='train')

    # 500 examples for testing
    
    dataset = dataset.train_test_split(test_size=500, seed=42)
    train_dataset = dataset['train']
    test_dataset = dataset['test']

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            question_raw = example.pop('question')

            # if not add_execution_prompt:
            #     if not detaield_instruction:
            #         question = question_raw + ' ' + naive_instruction
            #     else:
            #         question = question_raw + ' ' + coder_instruction
            # else:
            #     question = question_raw + ' ' + execution_prompt
            
            tests = example.pop('tests')
            data = {
                "data_source": "acecoder",
                "prompt": [
                    {
                        "role": "system",
                        "content": mathcoder_system_prompt,
                    },
                    {
                        "role": "user",
                        "content": question_raw,
                    }
                ],
                "ability": "code",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": None,
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'id': example['id'],
                    "question": question_raw,
                    "test_cases": tests,
                    "inputs_outputs": None,
                }
            }
            return data
        return process_fn

    train_dataset = train_dataset.map(make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(make_map_fn('test'), with_indices=True)
    print(train_dataset)
    print(train_dataset[0])

    train_dataset.to_parquet(local_dir / 'code_train.parquet')
    test_dataset.to_parquet(local_dir / 'code_test.parquet')
    print(f"Saved AceCoder data to {local_dir}")

def _process_torl_math(dataset_path, local_dir):
    dataset = datasets.load_dataset("VerlTool/ToRL-Math", split="train")
    def make_map_fn(item, idx):
        assert item['prompt'][0]['role'] == 'system', f"Expected system role, got {item['prompt'][0]['role']}"
        item['prompt'][0]['system'] = mathcoder_system_prompt
        item['extra_info'].update({
            'split': 'train',
            'index': idx,
            'id': item['extra_info']['id'],
            'question': item['prompt'][1]['content'],
            'ground_truth': item['reward_model']['ground_truth']
        })
        return item
    dataset = dataset.map(make_map_fn, with_indices=True)
    dataset.to_parquet(local_dir / 'torl_math_train.parquet')
    print(f"Saved ToRL-Math data to {local_dir}")

### Math Dataset Logic ###
def _process_math(data_source, local_dir, hdfs_dir, level):
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(data_source, trust_remote_code=True)
    train_dataset = dataset['train']
    test_dataset = dataset['test']
    
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
                    "content": mathcoder_system_prompt
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
    

    train_dataset.to_parquet(os.path.join(local_dir, 'math_train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'math_test.parquet'))
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
                    "content": mathcoder_system_prompt
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
                    "content": mathcoder_system_prompt
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
python examples/data_preprocess/mathcoder.py --code_dataset_path chiruan/CodeDPO-AceCoderV2-150K-processed-Qwen32B-inference --local_dir data/mathcoder --add_execution_prompt 
python examples/data_preprocess/mathcoder.py --code_dataset_path chiruan/CodeDPO-AceCoderV2-150K-processed-Qwen32B-inference --local_dir data/mathcoder_v2 --add_execution_prompt --sys_prompt_version v2
python examples/data_preprocess/mathcoder.py --code_dataset_path chiruan/CodeDPO-AceCoderV2-150K-processed-Qwen32B-inference --local_dir data/mathcoder_v3 --add_execution_prompt --sys_prompt_version v3
"""