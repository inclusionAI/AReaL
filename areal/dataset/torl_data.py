from io import BytesIO
from typing import Any, Dict, Optional, Union

from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from PIL import Image
from PIL.Image import Image as ImageObject
from torchvision import transforms

"""
Data columns (total 5 columns):
 #   Column        Non-Null Count  Dtype 
---  ------        --------------  ----- 
 0   data_source   1275 non-null   object
 1   prompt        1275 non-null   object
 2   ability       1275 non-null   object
 3   reward_model  1275 non-null   object
 4   extra_info    1275 non-null   object
"""

def get_torl_data_rl_dataset(
    path: str,
    split: str,
    tokenizer,
    rank: int,
    world_size: int,
    max_length: Optional[int] = None,
):
    # Load parquet dataset instead of json
    dataset = load_dataset("parquet", data_files=path, split='train')

    def process(sample):
        # Handle the prompt content - it might be a list of messages or a string
        answer = sample['reward_model']['ground_truth']    
        answer = f"\\boxed{{{answer}}}"    
        return {"messages": sample['prompt'], "answer": answer}

    dataset = dataset.map(process).remove_columns(["prompt", "reward_model"])

    # Filter out sequences longer than max_length if tokenizer and max_length are provided
    if max_length is not None:

        def filter_length(sample):
            # Tokenize the user content to check length
            content = sample["messages"][0]["content"]
            tokens = tokenizer.encode(content)
            return len(tokens) <= max_length

        dataset = dataset.filter(filter_length)

    dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)
    return dataset
