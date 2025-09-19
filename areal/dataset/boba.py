from typing import Optional

from datasets import load_dataset
from datasets.distributed import split_dataset_by_node

def get_boba_rl_dataset(
    path: str,
    split: str,
    tokenizer,
    rank: int,
    world_size: int,
    max_length: Optional[int] = None,
):
    dataset = load_dataset("json", data_files=path, split="train")

    def process(sample):
        messages = [
            {
                "role": "user",
                "content": sample["prompt"].replace('<｜User｜>', '')\
                    .replace(r'put your final answer within \boxed{}.<｜Assistant｜><think>', r'\nput your final answer within \boxed{}.')
            }
        ]
        return {"messages": messages, "answer": sample["solutions"][0]}

    dataset = dataset.map(process).remove_columns(["prompt"])

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
