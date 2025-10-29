from typing import TYPE_CHECKING

from datasets import load_dataset

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerFast


def get_terminal_bench_rl_dataset(
    path: str,
    split: str,
    tokenizer: "PreTrainedTokenizerFast",
    max_length: int | None = None,
):
    """Load terminal-bench dataset for RL training.

    The dataset should be in parquet format with the following columns:
    - prompt: The formatted prompt for the task
    - task_name: Name of the task
    - instruction: Raw instruction text
    - extra_info: JSON string containing task metadata
    """
    # Load from parquet file
    dataset = load_dataset("parquet", data_files={split: path}, split=split)

    # The dataset already has the right format from the converter:
    # - prompt: contains the formatted conversation
    # - task_name, instruction, extra_info: metadata fields

    # For RL training, we need to extract messages from the prompt or extra_info
    def process(sample):
        # The prompt is already formatted, but we need to extract the instruction
        # to create a messages structure for the workflow
        instruction = sample.get("instruction", "")
        task_name = sample.get("task_name", "")
        dockerfile_contents = sample.get("dockerfile_contents", "")

        # Return data in the format expected by the workflow
        return {
            "instruction": instruction,
            "task_name": task_name,
            "dockerfile_contents": dockerfile_contents,
            "extra_info": sample.get("extra_info", ""),
            "data_source": sample.get("data_source", "terminal_bench"),
        }

    dataset = dataset.map(process)

    # Filter out sequences longer than max_length if specified
    if max_length is not None:

        def filter_length(samples):
            # Tokenize instructions in batches for efficiency
            instructions = samples["instruction"]
            tokens_list = tokenizer(instructions, add_special_tokens=False)["input_ids"]
            return [len(tokens) <= max_length for tokens in tokens_list]

        dataset = dataset.filter(filter_length, batched=True)

    return dataset
