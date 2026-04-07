"""
Preprocess the MMAU-Pro dataset to parquet format
"""

import os

import datasets
import fire
from pathlib import Path

from verl.utils.hdfs_io import copy, makedirs

# -------------------
# prompts

system_prompt = """You are a helpful audio & language assistant with external tools.

You have access to the following tool(s) for audio(s) analysis:
- Tool #1:
{"type": "function", "function": {"name": "audio_crop", "description": "Crop the specified audio based on the time window in seconds.", "parameters": {"type": "object", "properties": {"time_window": {"type": "array", "description": "A tuple of two numbers indicating the start and end time (in seconds) of the audio segment to crop.", "items": {"type": "number"}}, "target_audio": {"type": "number", "description": "The index of the audio to crop. Index from 1 to the number of audios. Choose 1 to operate on original audio."}}, "required": ["time_window", "target_audio"]}}}

To invoke a tool, please use the following format (ensure to use proper JSON formatting and you have to put it between <tool_call> and </tool_call>):
<tool_call>
{"name": "audio_crop", "arguments": {"time_window": [start_seconds, end_seconds], "target_audio": audio_index}}
</tool_call>

Now, you are connected with the user."""

guideline = """
RESPONSE GUIDELINE:
1. [Given the audio, question and possible choices, expand on your reasoning step by step.]
2. [Use tools, with correct tool-call format, to help you better understand the audios.]
3. [Put your final answer in \boxed{exact_answer_text}]"""
# -------------------

def concat_question_and_choices(question:str, choices:list):
    """
    Concatenate the question with possible choices.
    Args:
        question (str): The question to concatenate.
        choices (list): The choices to concatenate.
    Returns:
        str: The concatenated complete question with possible choices.
    """
    if choices is None or len(choices) == 0:
        # if there are no choices available, we assume the question itself is complete
        return question
    return "Given the audio(s) information, answer the following question:\n<question>" \
        + question \
        + '\n</question>\nPossible choices:\n<choices>' \
        + '\n'.join(choices) \
        + "\n</choices>"


def main(
    data_source="gamma-lab-umd/MMAU-Pro", # the repo_id on huggingface dataset, https://huggingface.co/datasets/gamma-lab-umd/MMAU-Pro
    local_dir='/data/zhenxiong/data/mmau_pro',
    audio_dir="/data/zhenxiong/data/mmau_pro/audio",
    hdfs_dir=None,
    audio_sep="<audio>",
):
    
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(data_source, cache_dir="/data/zhenxiong/hf_cache/", token=HF_TOKEN)["test"]
    num_samples = len(dataset)
    dataset = dataset.train_test_split(test_size=int(num_samples * 0.1), seed=42)
    train_dataset = dataset['train']
    test_dataset = dataset['test']
    tiny_dataset = train_dataset.select(range(10)) # randomly select 10 samples from training set for debugging
    
    # add a row to each data item that represents a unique id
    def make_train_map_fn(split, data_source):

        def process_fn(example, idx):
            question_raw = example.pop('question')
            choices = example.pop('choices')
            question_raw = concat_question_and_choices(question_raw, choices)
            # Add guideline to the question
            question_raw += f"\n\n{guideline}"
            
            answer = example.pop('answer')
            category = example.pop('category')
            audio_paths = [(Path(audio_dir) / audio_path.split("/")[-1]).absolute().as_posix() for audio_path in example.pop('audio_path')]
            
            # Build multimodal content with audio placeholder
            mm_content = audio_sep * len(audio_paths) + question_raw
            
            # Build prompt following the format in prepare_train.py
            prompt = [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": mm_content,
                }
            ]
            
            data = {
                "data_source": data_source,
                "prompt": prompt,
                "audios": [{"audio": audio_path} for audio_path in audio_paths],
                "ability": "audio_reasoning",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer
                },
                "extra_info": {
                    'split': f"{split}_{idx}",
                    'index': idx,
                    'category': category,
                    'audios': [{"audio": audio_path} for audio_path in audio_paths],
                }
            }
            return data

        return process_fn
    
    train_dataset = train_dataset.map(function=make_train_map_fn('train', data_source), with_indices=True, remove_columns=train_dataset.column_names)
    test_dataset = test_dataset.map(function=make_train_map_fn('test', data_source), with_indices=True, remove_columns=test_dataset.column_names)
    tiny_dataset = tiny_dataset.map(function=make_train_map_fn('tiny', data_source), with_indices=True, remove_columns=tiny_dataset.column_names)

    print(f"Loaded {len(train_dataset)} training samples")
    print(f"Loaded {len(test_dataset)} test samples")
    print(f"Loaded {len(tiny_dataset)} tiny samples")
    print("Example of a training sample:")
    print(train_dataset[0])
    
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
    tiny_dataset.to_parquet(os.path.join(local_dir, 'tiny.parquet'))
    print(f"Saved {len(train_dataset)} training samples to {local_dir}/train.parquet")
    print(f"Saved {len(test_dataset)} test samples to {local_dir}/test.parquet")
    print(f"Saved {len(tiny_dataset)} tiny samples to {local_dir}/tiny.parquet")

    
    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
    

if __name__ == '__main__':
    fire.Fire(main)