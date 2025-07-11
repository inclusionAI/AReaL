from datasets import Dataset, load_dataset
from datasets.distributed import split_dataset_by_node

def get_gsm8k_sft_dataset(path, split, tokenizer, rank, world_size):
    dataset = load_dataset(path=path, name="main", split=split)
    dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)
    def process(sample):
        seq_token = tokenizer.encode(
            sample["question"] + sample["answer"] + tokenizer.eos_token
        )
        prompt_token = tokenizer.encode(sample["question"])
        prompt_mask = [1] * len(prompt_token) + [0] * (
            len(seq_token) - len(prompt_token)
        )
        return {"input_ids": seq_token, "prompt_mask": prompt_mask}

    dataset = dataset.map(process).remove_columns(["question", "answer"])
    return dataset


