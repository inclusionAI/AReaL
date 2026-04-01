from datasets import load_dataset
import os

def get_math500_rl_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: int | None = None,
):
    # ---- Detect raw JSON / JSONL dataset ----
    data_file = os.path.join(path, f"{split}.jsonl")
    dataset = load_dataset(
        "json",
        data_files={split: data_file},
        split=split,
    )

    def process(sample):
        messages = [{
            "role": "user",
            "content": sample["problem"] + "\nPlease put your final answer within \\boxed{}.",
        }]
        return {"messages": messages, "answer": sample["answer"]}
    
    dataset = dataset.map(process).remove_columns(["problem"])

    if max_length is not None:
        def filter_length(sample):
            return (
                len(tokenizer.encode(sample["messages"][0]["content"]))
                <= max_length
            )

        dataset = dataset.filter(filter_length)

    return dataset
