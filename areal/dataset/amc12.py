from datasets import load_dataset
import os

def get_amc12_rl_dataset(
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
            "content": (
                sample["question"]
                + "\n\nReturn ONLY the final answer in LaTeX as: \\boxed{...}."
                + "\nDo not include any other text."
                + "\nAfter the box, output the token <END> and then stop."
            ),
        }]
        return {"messages": messages, "answer": sample["answer"]}
    
    dataset = dataset.map(process).remove_columns(["question"])

    if max_length is not None:
        def filter_length(sample):
            return (
                len(tokenizer.encode(sample["messages"][0]["content"]))
                <= max_length
            )

        dataset = dataset.filter(filter_length)

    return dataset