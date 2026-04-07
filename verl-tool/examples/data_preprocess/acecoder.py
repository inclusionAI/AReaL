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


naive_instruction = "Let's think step by step and generate the final program in a markdown code block like this: ```python\nyour code here\n```."

def main(
    dataset_path: str = 'VerlTool/AceCoderV2-122K',
    local_dir: str = 'data/acecoder',
):
    local_dir = Path(local_dir)
    local_dir_post_fix = ""
    local_dir = local_dir / (dataset_path.split('/')[-1] + local_dir_post_fix)
    local_dir.mkdir(parents=True, exist_ok=True)
    
    dataset = datasets.load_dataset(dataset_path, split='train')

    # 500 examples for testing
    
    dataset = dataset.train_test_split(test_size=500, seed=42)
    train_dataset = dataset['train']
    test_dataset = dataset['test']

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            question_raw = example.pop('question')
                    
            tests = example.pop('tests')
            data = {
                "data_source": dataset_path,
                "prompt": [
                    {
                        "role": "user",
                        "content": question_raw + naive_instruction
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
dataset_path="TIGER-Lab/AceCode-87K" # V1 dataset
dataset_path="TIGER-Lab/AceCode-V2-122K" # V2 dataset
python examples/data_preprocess/acecoder.py --dataset_path ${dataset_path} --local_dir data/acecoder
"""