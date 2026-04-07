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
from pathlib import Path

system_prompt1 = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. User: Please integrate natural language reasoning with programs to solve the coding problems below. If you want to test any python code, writing it inside <python> and </python> tags following with <output>. Please put your final answer in a markdown code block like this: python\nyour code here\n``` without appending anything.
"""

system_prompt2 = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. User: Please integrate natural language reasoning with programs to solve the coding problems below. If the you want to run any python code, execution result will be in the output markdown block like "```output\nexecution result here\n```" following the code block. Please put your final answer in a markdown code block like this: python\nyour code here\n``` without appending anything.
"""

system_prompt3 = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. User: Please solve the coding problems below and put your final answer in a markdown code block like this: python\nyour code here\n``` without appending anything.
"""

system_prompt4 = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. User: Please integrate natural language reasoning with programs to solve the coding problems below. If the you want to run any python code, execution result will be in the output markdown block like "```output\nstdout and stderr\n```" following the code block. Please put your final answer in a markdown code block like this: python\nyour code here\n``` without appending anything. make sure you also write test cases for the code you write so you can get meaningful execution results for debugging.
"""

system_prompt5 = """A conversation between user and assistant. The user asks a question, and the assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. During the thinking, the assistant can test its solution code with self-written test cases in the following format:
```python
solution code and test cases here
```
```output
execution stdout and stderr result
```
Make sure you always append test cases for the code you write so you can get meaningful execution results. Please put your final solution code ready to submit in the last markdown code block like ```python\nyour code here\n``` without appending anything. 
"""
    
system_prompt6 = """A conversation between user and assistant. The user asks a question, and the assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. During the thinking, the assistant can test its solution code with self-written test cases in the following format:
```
<python>
solution code and test cases here
</python>
<output>
execution stdout and stderr result
</output>
```
Make sure you always append test cases for the code you write so you can get meaningful execution results. Please put your final solution code ready to submit in the last markdown code block like ```python\nyour code here\n``` without appending anything. 
"""

system_prompt7 = """A conversation between user and assistant. The user asks a question, and the assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. During the thinking, the assistant can call a python interpreter tool by writing code in "<tool_call> code here </tool_call>", then the standard output and error will be given in the a markdown block. You can use this to check if your code is correct in terms of syntax, logic, and your self-written test cases. Please put your final solution code ready to submit in the <answer> </answer> tags, where you write a markdown code block like "<answer>```python\nyour code here\n```</answer>" without appending anything.
"""

system_prompt8 = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. User: Please integrate natural language reasoning with programs to solve the coding problems below. If you want to test the code of your solution, include "<|calling system for feedback|>" at the end of your response for the current turn. Then the system will execute the code in the markdown block and provide the standard output and error. Make sure you also write test cases for the code you write so you can get non-empty execution results for debugging. If you think the solution is complete and don't need to test, don't include "<|calling system for feedback|>" in the response and put your final answer in a markdown code block like this: ```python\nyour code here\n``` without appending anything. 
"""

system_prompt9 = """A conversation between user and assistant. The user asks a question, and the assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. You can test your solution by appending "<|calling system for feedback|>" after your solution. Then the system will execute the code in the markdown block and provide the standard output and error. Make sure you also write test cases for the code you write so you can get non-empty execution results for debugging. Don't "<|calling system for feedback|>" if the system confirms the solution passes all test cases, and then put your final solution like this ```python\nyour code here\n``` without appending anything. 
"""
    
system_prompt10 = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. User: Please integrate natural language reasoning with programs to solve the coding problems below. If you want to test the code of your solution, include "<|calling system for feedback|>" at the end of your response for the current turn. Then the system will execute the code in the markdown block and provide the standard output and error. If there is an error, then you debug it. If there is no error, finalize your solution in a markdown code block ```python\nyour code here\n``` without appending anything or any other analysis.
"""

system_prompt11 = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. Please integrate natural language reasoning to solve the coding problems below. **For every code solution, you should also write test cases to assert that the solution actually works as you expected**. To run the code adn test cases, include "<|calling system for feedback|>" at the end of your response for the current turn. Then the system will execute the code in the markdown block and provide the standard output and error. If there is an error, then you debug it. If there is no error, finalize your solution in a markdown code block ```python\nyour code here\n``` without appending anything or any other analysis.
"""

system_prompt12 = """Solve the following coding problem. For each of your code solution, you **must** also write some test cases using the `assert` statement to test the correctness of your code, and the solution code and test cases **must be in a single code block**. Execution result will be given by the system and you should self-debug your code based on the execution result until all the test cases pass. If you think the solution is complete and don't need to test, put your final answer in a markdown code block like this: ```python\nyour code here\n``` without appending anything."""

system_prompt13 = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. Please solve the following coding problem and verify the correctness by writing test cases in `assert` statements. Execution results will be put into the in output markdown block: "```output ```". Put your final answer in a markdown code block like this: python\nyour code here\n``` without appending anything."""

public_test_template = """\
### Public Test Cases
Here are some public test cases where you can use to test your program.
```python
{test_cases}
```
"""
def main(
    dataset_path: str = 'VerlTool/AceCoderV2-122K',
    local_dir: str = 'data/acecoder',
    system_prompt_idx: int = 1,
    add_public_tests: bool = False,
    add_public_tests_all: bool = False,
):
    local_dir = Path(local_dir)
    local_dir_post_fix = f"-system-prompt-{system_prompt_idx}"
    if add_public_tests:
        local_dir_post_fix += f"-pub-tests"
    if add_public_tests_all:
        local_dir_post_fix += f"-all-tests"
    local_dir = local_dir / (dataset_path.split('/')[-1] + local_dir_post_fix)
    local_dir.mkdir(parents=True, exist_ok=True)
    
    dataset = datasets.load_dataset(dataset_path, split='train')

    # 500 examples for testing
    dataset = dataset.train_test_split(test_size=500, seed=42)
    train_dataset = dataset['train']
    test_dataset = dataset['test']
    
    system_instruction = eval(f'system_prompt{system_prompt_idx}')

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            question_raw = example.pop('question')
            
            tests = example.pop('tests')
            
            if add_public_tests:
                public_tests = example.get('public_tests', None)
            elif add_public_tests_all:
                public_tests = tests
            else:
                public_tests = None
            data = {
                "data_source": dataset_path,
                "prompt": [
                    {
                        "role": "system",
                        "content": system_instruction.strip(' \n'),
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
                    'id': str(example['id']),
                    "question": question_raw,
                    "public_tests": public_tests,
                    "test_cases": tests,
                    "inputs_outputs": None,
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True, remove_columns=train_dataset.column_names)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True, remove_columns=test_dataset.column_names)
    
    print(f"Loaded {len(train_dataset)} training samples")
    print(f"Loaded {len(test_dataset)} testing samples")
    print(f"Example of a training sample:")
    print(train_dataset[0])

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
    print(f"Saved to {len(train_dataset)} training samples to {local_dir}/train.parquet")
    print(f"Saved to {len(test_dataset)} testing samples to {local_dir}/test.parquet")

if __name__ == '__main__':
    fire.Fire(main)
    
"""
python examples/data_preprocess/acecoder_custom.py --dataset_path VerlTool/AceCoderV2-69K --local_dir data/acecoder_custom --system_prompt_idx 1
python examples/data_preprocess/acecoder_custom.py --dataset_path VerlTool/AceCoderV2-69K --local_dir data/acecoder_custom --system_prompt_idx 2
python examples/data_preprocess/acecoder_custom.py --dataset_path VerlTool/AceCoderV2-69K --local_dir data/acecoder_custom --system_prompt_idx 3
python examples/data_preprocess/acecoder_custom.py --dataset_path VerlTool/AceCoderV2-69K --local_dir data/acecoder_custom --system_prompt_idx 4
python examples/data_preprocess/acecoder_custom.py --dataset_path VerlTool/AceCoderV2-69K --local_dir data/acecoder_custom --system_prompt_idx 5
python examples/data_preprocess/acecoder_custom.py --dataset_path VerlTool/AceCoderV2-69K --local_dir data/acecoder_custom --system_prompt_idx 6
python examples/data_preprocess/acecoder_custom.py --dataset_path VerlTool/AceCoderV2-69K --local_dir data/acecoder_custom --system_prompt_idx 7
python examples/data_preprocess/acecoder_custom.py --dataset_path VerlTool/AceCoderV2-69K --local_dir data/acecoder_custom --system_prompt_idx 8
python examples/data_preprocess/acecoder_custom.py --dataset_path VerlTool/AceCoderV2-69K --local_dir data/acecoder_custom --system_prompt_idx 9
python examples/data_preprocess/acecoder_custom.py --dataset_path VerlTool/AceCoderV2-69K --local_dir data/acecoder_custom --system_prompt_idx 10
python examples/data_preprocess/acecoder_custom.py --dataset_path VerlTool/AceCoderV2-69K --local_dir data/acecoder_custom --system_prompt_idx 11
python examples/data_preprocess/acecoder_custom.py --dataset_path VerlTool/AceCoderV2-69K --local_dir data/acecoder_custom --system_prompt_idx 11 --add_public_tests True
python examples/data_preprocess/acecoder_custom.py --dataset_path VerlTool/AceCoderV2-69K --local_dir data/acecoder_custom --system_prompt_idx 11 --add_public_tests_all True
python examples/data_preprocess/acecoder_custom.py --dataset_path VerlTool/AceCoderV2-69K --local_dir data/acecoder_custom --system_prompt_idx 12
python examples/data_preprocess/acecoder_custom.py --dataset_path VerlTool/AceCoderV2-69K --local_dir data/acecoder_custom --system_prompt_idx 13
python examples/data_preprocess/acecoder_custom.py --dataset_path VerlTool/AceCoderV2-69K --local_dir data/acecoder_custom --system_prompt_idx 13 --add_public_tests True
python examples/data_preprocess/acecoder_custom.py --dataset_path VerlTool/AceCoderV2-69K --local_dir data/acecoder_custom --system_prompt_idx 13 --add_public_tests_all True
python examples/data_preprocess/acecoder_custom.py --dataset_path VerlTool/AceCoderV2-69K-cleaned --local_dir data/acecoder_custom --system_prompt_idx 13
python examples/data_preprocess/acecoder_custom.py --dataset_path VerlTool/AceCoderV2-69K-cleaned --local_dir data/acecoder_custom --system_prompt_idx 13 --add_public_tests True
python examples/data_preprocess/acecoder_custom.py --dataset_path VerlTool/AceCoderV2-69K-cleaned --local_dir data/acecoder_custom --system_prompt_idx 13 --add_public_tests_all True
python examples/data_preprocess/acecoder_custom.py --dataset_path VerlTool/AceCoderV2-122K --local_dir data/acecoder_custom --system_prompt_idx 1
python examples/data_preprocess/acecoder_custom.py --dataset_path VerlTool/AceCoderV2-122K --local_dir data/acecoder_custom --system_prompt_idx 2
"""