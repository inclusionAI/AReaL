from datasets import Dataset


def process_gsm8k_dataset(dataset: Dataset, tokenizer, reward_mode):
    def process_example(example, idx):
        # Add query_id column
        example["query_id"] = str(idx)
        example["prompt"] = example["question"]

        # used by the reward function
        example["method"] = reward_mode
        return example

    dataset = dataset.map(
        lambda example, idx: process_example(example, idx),
        with_indices=True,
    )
    return dataset.map(
        lambda x: tokenizer(x["question"], return_attention_mask=False), batched=True
    )
