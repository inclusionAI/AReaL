from datasets import Dataset


def process_areal_dataset(dataset: Dataset, tokenizer):
    return dataset.map(
        lambda x: tokenizer(x["prompt"], return_attention_mask=False), batched=True
    )
