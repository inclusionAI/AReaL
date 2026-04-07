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
import fire
import os
import datasets
import json
from pathlib import Path
from tqdm import tqdm

execution_prompt = """\
Answer the given coding question. You must conduct reasoning about the problem and then provide the final program as answer. 
During the thinking process, you can write test cases or test your current solutions using a testing tool. if you want to test any python code, writing it inside ```python and ``` tags following with "```output". 
The code between "```python" and "``````output" will then be executed, and the terminal output (standard output and standard error) will be provided to you. 
Each program between ```python and ``` tags are independent program. You can test Python codes as many times as you want. 
If you find no further code execution needed, you can then give your final solution in a markdown code block like this: ```python\nyour code here\n``` without appending anything. 
The final program will be evaluated against the hidden test cases. If the final program passes all the test cases, you will get a reward. If the final program fails any of the test cases, you will get a penalty.
"""

naive_instruction = "Let's think step by step and generate the final program in a markdown code block like this: ```python\nyour code here\n```."
naive_execution_prompt = """
A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The Assistant can reason with the help of Python code. If the Assistant wants to run any Python code, it writes it inside ```python and ``` tags, and makes sure to follow it with "```output", meaning that it is requesting the code to be executed. Then the result of execution will be provided to the Assistant between "```output" and "```" for the python code block that it follows. The Assistant can test Python codes as many times as it wants. If the Assistant finds no further code execution needed, it can then give the final solution in a markdown code block like this: ```python\nyour code here\n``` without appending anything.
"""
complex_execution_prompt = """
A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The Assistant can reason with the help of Python code. If the Assistant wants to run any Python code, it writes it inside ```python and ``` tags, and makes sure to follow it with "```output", meaning that it is requesting the code to be executed. Then the result of execution will be provided to the Assistant between "```output" and "```" for the python code block that it follows. 

Coding questions can ask various forms of program solutions:
- If the coding question has a starter code, you may use the starter code to write the solution to the problem.
- Elif the coding question has a function signature, you may use the function signature to write the solution to the problem.
- Else you may write a program that reads the input from standard input and writes the output to standard output. (do not directly test on the sample inputs)

The Assistant can test Python codes as many times as it wants. If the Assistant finds no further code execution needed, it can then give the final solution in a markdown code block like this: ```python\nyour code here\n``` without appending anything. 
"""
# naive_execution_prompt = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. User: Please integrate natural language reasoning with programs to solve the coding problems below. If you want to test any python code, writing it inside <python> and </python> tags, results will be inside <output> and </output>. Please put your final answer in a markdown code block like this: python\nyour code here\n``` without appending anything."""

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
naive_coder_instruction = """\
A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. 

Let's think step by step and generate the final program in a markdown code block like this: ```python\nyour code here\n```.
"""

complex_coder_instruction = """\
A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. 

Coding questions can ask various forms of program solutions:
- If the coding question has a starter code, you may use the starter code to write the solution to the problem.
- Elif the coding question has a function signature, you may use the function signature to write the solution to the problem.
- Else you may write a program that reads the input from standard input and writes the output to standard output. (do not directly test on the sample inputs)

Let's think step by step and generate the final program in a markdown code block like this: ```python\nyour code here\n```.
"""

def main(
    dataset_path: str = 'agentica-org/DeepCoder-Preview-Dataset',
    subset='all',
    local_dir: str = 'data/deepcoder',
    add_execution_prompt: bool = False,
    propmt_type='complex',
    add_public_tests: bool = False,
):
    all_subsets = ['lcbv5', 'taco', 'codeforces', 'primeintellect']
    assert subset in all_subsets + ['all'], f"Invalid subset {subset}, please choose from {all_subsets} or 'all'"
    
    local_dir = Path(local_dir)
    local_dir_post_fix = ""
    if add_execution_prompt:
        local_dir_post_fix = "-with-execution-prompt"
    if add_public_tests:
        local_dir_post_fix += "-with-public-tests"
    local_dir_post_fix += f"-{propmt_type}"
    local_dir = local_dir / (subset + local_dir_post_fix)
    local_dir.mkdir(parents=True, exist_ok=True)

    if subset == 'all':
        train_data = []
        test_data = []
        for _subset in all_subsets:
            dataset = datasets.load_dataset(dataset_path, _subset)
            if "train" in dataset:
                train_data.extend([
                    {
                        'problem': example['problem'],
                        'tests': example['tests'],
                        'data_source': _subset,
                        'metadata': example.get('metadata', None),
                        'starter_code': example.get('starter_code', None)
                    }
                for example in dataset['train']])
            if "test" in dataset:
                test_data.extend([
                    {
                        'problem': example['problem'],
                        'tests': example['tests'],
                        'data_source': _subset,
                        'metadata': example.get('metadata', None),
                        'starter_code': example.get('starter_code', None)
                    }
                for example in dataset['test']])
        train_dataset = datasets.Dataset.from_list(train_data)
        test_dataset = datasets.Dataset.from_list(test_data)
    else:
        dataset = datasets.load_dataset(dataset_path, subset)
        if "train" in dataset:
            train_dataset = dataset['train']
        else:
            train_dataset = None
        if "test" in dataset:
            test_dataset = dataset['test']
        else:
            test_dataset = None

    
    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            
            if propmt_type == 'complex':
                system_instruction = complex_execution_prompt if add_execution_prompt else complex_coder_instruction
            elif propmt_type == 'naive':
                system_instruction = naive_execution_prompt if add_execution_prompt else naive_coder_instruction
            else:
                raise ValueError(f"Unknown propmt_type: {propmt_type}")
            
            question_raw = example.pop('problem')
            inputs_outputs = example.pop('tests')
            data_source = example.get('data_source', f"{dataset_path}-{subset}")
            if "primeintellect" in data_source or 'codeforces' in data_source:
                # special process of primeintellect dataset
                inputs_outputs = json.loads(inputs_outputs)
                fn_name = inputs_outputs[0].get("fn_name")
                inputs_outputs = {
                    "type": inputs_outputs[0].get("type"),
                    "inputs": [inputs_outputs[j]['input'] for j in range(len(inputs_outputs))],
                    "outputs": [inputs_outputs[j]['output'] for j in range(len(inputs_outputs))],
                }
                if fn_name:
                    inputs_outputs["fn_name"] = fn_name
                inputs_outputs = json.dumps(inputs_outputs)
            if 'lcbv5' in data_source:
                starter_code = example.get('starter_code')
                if starter_code:
                    question_raw += "\n\nHere is the starter code:\n" + example.get('starter_code', '')
                tests = inputs_outputs
                tests = json.loads(tests)
                new_tests = {
                    "type": tests[0]['testtype'],
                    "inputs": [tests[j]['input'] for j in range(len(tests))],
                    "outputs": [tests[j]['output'] for j in range(len(tests))]
                }
                if example['metadata']['func_name']:
                    assert new_tests['type'] == "functional"
                    func_name = example['metadata']['func_name']
                    new_tests['fn_name'] = func_name
                inputs_outputs = json.dumps(new_tests)
            
            if add_public_tests:
                public_tests = json.loads(inputs_outputs)
                public_tests['inputs'] = public_tests['inputs'][:3]
                public_tests['outputs'] = public_tests['outputs'][:3]
                public_tests = json.dumps(public_tests)
            else:
                public_tests = None
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "system",
                        "content": system_instruction,
                    },
                    {
                        "role": "user",
                        "content": question_raw,
                    }
                ],
                "ability": "code",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": ""
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'id': f"{data_source}:{idx}",
                    "question": question_raw,
                    "test_cases": None,
                    "inputs_outputs": inputs_outputs,
                    "public_tests": public_tests,
                }
            }
            return data

        return process_fn
    batch_size=50 # this is required for processing lcbv5
    features = {
        'data_source': datasets.Value(dtype='string', id=None),
        'prompt': [
            {
                'content': datasets.Value(dtype='string', id=None),
                'role': datasets.Value(dtype='string', id=None)
            }
        ],
        'ability': datasets.Value(dtype='string', id=None),
        'reward_model': {
            'ground_truth': datasets.Value(dtype='string', id=None),
            'style': datasets.Value(dtype='string', id=None)
        },
        'extra_info': {
            'id': datasets.Value(dtype='string', id=None),
            'index': datasets.Value(dtype='int64', id=None),
            'inputs_outputs': datasets.Value(dtype='large_string', id=None),
            'question': datasets.Value(dtype='string', id=None),
            'split': datasets.Value(dtype='string', id=None),
            'test_cases': datasets.Value(dtype='null', id=None)
        }
    }
    features = datasets.Features(features)
    if train_dataset is not None:
        train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True, remove_columns=train_dataset.column_names, features=features)
        print(f"Loaded {len(train_dataset)} training samples")
        print(f"Example of a training sample:")
        print(train_dataset[0])
        train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'), batch_size=batch_size)
        print(f"Saved to {len(train_dataset)} training samples to {local_dir}/train.parquet")
    if test_dataset is not None:
        test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True, remove_columns=test_dataset.column_names, features=features)
        print(f"Loaded {len(test_dataset)} testing samples")
        print(f"Example of a test sample:")
        print(test_dataset[0])
        test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'), batch_size=batch_size)
        print(f"Saved to {len(test_dataset)} testing samples to {local_dir}/test.parquet")

if __name__ == '__main__':
    fire.Fire(main)
    
"""
python examples/data_preprocess/deepcoder.py --dataset_path agentica-org/DeepCoder-Preview-Dataset --subset lcbv5 --local_dir data/deepcoder --add_execution_prompt
python examples/data_preprocess/deepcoder.py --dataset_path agentica-org/DeepCoder-Preview-Dataset --subset taco --local_dir data/deepcoder --add_execution_prompt
python examples/data_preprocess/deepcoder.py --dataset_path agentica-org/DeepCoder-Preview-Dataset --subset codeforces --local_dir data/deepcoder --add_execution_prompt
python examples/data_preprocess/deepcoder.py --dataset_path agentica-org/DeepCoder-Preview-Dataset --subset primeintellect --local_dir data/deepcoder --add_execution_prompt
python examples/data_preprocess/deepcoder.py --dataset_path agentica-org/DeepCoder-Preview-Dataset --subset all --local_dir data/deepcoder --add_execution_prompt
"""