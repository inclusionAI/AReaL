from datasets import Dataset


def process_gsm8k_rl_dataset(dataset: Dataset, tokenizer, reward_mode):
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


def process_gsm8k_sft_dataset(dataset: Dataset, tokenizer):
    def _tokenize(example, idx):
        # Add query_id column
        example["query_id"] = str(idx)
        example["prompt"] = example["question"]
        example["seq"] = example["prompt"] + example["answer"] + tokenizer.eos_token

        tokenized_prompt = tokenizer(
            example["question"],
            return_attention_mask=False,
        )["input_ids"]
        tokenized_seq = tokenizer(
            example["prompt"] + example["answer"] + tokenizer.eos_token,
            return_attention_mask=False,
        )["input_ids"]
        seq_len = len(tokenized_seq)
        prompt_len = len(tokenized_prompt)

        return {"seq": tokenized_seq, "prompt_len": prompt_len, "seq_len": seq_len}

    dataset = dataset.map(
        lambda example, idx: _tokenize(example, idx),
        with_indices=True,
    )
    return dataset
