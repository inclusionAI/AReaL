from datasets import Dataset


def process_gsm8k_rl_dataset(dataset: Dataset):
    def process(sample):
        messages = [{"role": "user", "content": sample["question"]}]
        return {"messages": messages}

    dataset = dataset.map(process).remove_columns(["question"])
    return dataset
