from datasets import load_dataset


def get_hhrlhf_rw_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: int | None = None,
):
    dataset = load_dataset(path=path, split=split)

    def process(sample):
        chosen_seq_token = tokenizer.encode(sample["chosen"] + tokenizer.eos_token)
        rejected_seq_token = tokenizer.encode(sample["rejected"] + tokenizer.eos_token)
        return {"chosen_ids": chosen_seq_token, "rejected_ids": rejected_seq_token}

    dataset = dataset.map(process).remove_columns(["chosen", "rejected"])

    if max_length is not None:
        # Filter out sequences longer than max_length
        dataset = dataset.filter(
            lambda x: (len(x["chosen_ids"]) <= max_length)
            and (len(x["rejected_ids"]) <= max_length)
        )

    return dataset
