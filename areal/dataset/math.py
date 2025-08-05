from datasets import load_dataset
from datasets.distributed import split_dataset_by_node


def get_math_sft_dataset(path, split, tokenizer, rank, world_size):
    dataset = load_dataset(path=path, split=split)
    dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)

    def process(sample):
        seq_token = tokenizer.encode(
            sample["problem"] + sample["solution"] + tokenizer.eos_token
        )
        prompt_token = tokenizer.encode(sample["problem"])
        loss_mask = [0] * len(prompt_token) + [1] * (len(seq_token) - len(prompt_token))
        return {"input_ids": seq_token, "loss_mask": loss_mask}

    dataset = dataset.map(process).remove_columns(["problem", "solution"])
    return dataset


def get_math_rl_dataset(path, split, rank, world_size):
    dataset = load_dataset(path=path, split=split)
    dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)

    def process(sample):
        messages = [{"role": "user", "content": sample["problem"]}]
        return {"messages": messages, "answer": sample["solution"]}

    dataset = dataset.map(process).remove_columns(["problem", "solution"])
    return dataset 