from datasets import Dataset


def process_rl_dataset(dataset: Dataset):
    def process(sample):
        messages = [{"role": "user", "content": sample["prompt"]}]
        return {"messages": messages}

    dataset = dataset.map(process).remove_columns(["prompt"])
    return dataset
