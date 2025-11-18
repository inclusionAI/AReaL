from typing import Optional

from datasets import load_dataset


def get_gsm8k_sft_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: Optional[int] = None,
):
    dataset = load_dataset(path=path, name="main", split=split)

    def process(sample):
        seq_token = tokenizer.encode(
            sample["question"] + sample["answer"] + tokenizer.eos_token
        )
        prompt_token = tokenizer.encode(sample["question"])
        loss_mask = [0] * len(prompt_token) + [1] * (len(seq_token) - len(prompt_token))
        return {"input_ids": seq_token, "loss_mask": loss_mask}

    dataset = dataset.map(process).remove_columns(["question", "answer"])

    if max_length is not None:
        # Filter out sequences longer than max_length
        dataset = dataset.filter(lambda x: len(x["input_ids"]) <= max_length)

    return dataset


def get_gsm8k_rl_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: Optional[int] = None,
):
    dataset = load_dataset(path=path, name="main", split=split)

    def process(sample):
        messages = [
            {
                "role": "user",
                "content": sample["question"]
                + "\nPlease put your final answer within \\boxed{}.",
            }
        ]
        return {"messages": messages}

    dataset = dataset.map(process).remove_columns(["question"])

    # Filter out sequences longer than max_length if tokenizer and max_length are provided
    if max_length is not None:

        def filter_length(sample):
            # Tokenize the user content to check length
            content = sample["messages"][0]["content"]
            tokens = tokenizer.encode(content)
            return len(tokens) <= max_length

        dataset = dataset.filter(filter_length)

    return dataset


def get_gsm8k_reasoning_rl_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: Optional[int] = None,
):
    """
    GSM8K RL dataset with reasoning format (XML-style).
    
    Formats prompts to encourage reasoning in the following format:
    <reasoning>
    ...
    </reasoning>
    <answer>
    ...
    </answer>
    
    Based on Unsloth's reasoning model training approach.
    
    Note: The path may contain "-reasoning" suffix, which should be removed
    before loading from HuggingFace.
    """
    # Remove "-reasoning" suffix if present to get the actual dataset path
    actual_path = path.replace("-reasoning", "").replace("_reasoning", "")
    dataset = load_dataset(path=actual_path, name="main", split=split)
    
    # System prompt instructing the model to use reasoning format
    SYSTEM_PROMPT = """Respond in the following format:
<reasoning>
[Your step-by-step reasoning process here]
</reasoning>
<answer>
[Your final numerical answer here]
</answer>"""

    def process(sample):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": sample["question"]}
        ]
        return {"messages": messages}

    dataset = dataset.map(process).remove_columns(["question"])

    # Filter out sequences longer than max_length if tokenizer and max_length are provided
    if max_length is not None:
        def filter_length(sample):
            # Tokenize all messages to check total length
            # We need to account for the system prompt + user message
            total_content = ""
            for msg in sample["messages"]:
                total_content += msg["content"] + "\n"
            tokens = tokenizer.encode(total_content)
            return len(tokens) <= max_length

        dataset = dataset.filter(filter_length)

    return dataset
