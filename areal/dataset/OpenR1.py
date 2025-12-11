import json
from typing import Optional

from datasets import Dataset
from datasets.distributed import split_dataset_by_node


def get_openr1_sft_dataset(
    path: str,
    split: str,
    tokenizer,
    rank: int,
    world_size: int,
    max_length: Optional[int] = None,
):
    """
    Load OpenR1 dataset for supervised fine-tuning.
    
    Args:
        path: Path to the OpenR1-raw-qwen.jsonl file
        split: Dataset split (not used for custom jsonl file)
        tokenizer: Tokenizer to encode the text
        rank: Current process rank for distributed training
        world_size: Total number of processes
        max_length: Maximum sequence length (optional)
    """
    # Load the jsonl file
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    
    def process(sample):
        # Extract the prompt and clean it to get only the math problem
        prompt = sample["prompt"]
        
        # Extract only the math problem content from ChatML format
        # Remove ChatML tags and extract the actual problem text
        if "<|im_start|>user\n" in prompt:
            # Extract content between <|im_start|>user\n and \nPlease reason step by step
            user_content = prompt.split("<|im_start|>user\n")[1]
            if "\nPlease reason step by step" in user_content:
                problem_text = user_content.split("\nPlease reason step by step")[0].strip()
            elif "\n<|im_end|>" in user_content:
                problem_text = user_content.split("\n<|im_end|>")[0].strip()
            else:
                problem_text = user_content.strip()
        else:
            # Fallback: just remove the instruction text
            instruction_to_remove = "Please reason step by step, and put your final answer within \\boxed{}."
            if instruction_to_remove in prompt:
                problem_text = prompt.replace(instruction_to_remove, "").strip()
            else:
                problem_text = prompt.strip()
        
        # Get the solution (assuming we take the first solution)
        solution = sample["solutions"][0] if sample["solutions"] else ""
        
        # Create the full sequence: problem_text + solution + eos_token
        full_text = problem_text + solution + tokenizer.eos_token
        seq_token = tokenizer.encode(full_text)
        
        # Create prompt tokens for loss masking
        prompt_token = tokenizer.encode(problem_text)
        
        # Create loss mask: 0 for prompt tokens, 1 for solution tokens
        loss_mask = [0] * len(prompt_token) + [1] * (len(seq_token) - len(prompt_token))
        
        return {
            "input_ids": seq_token, 
            "loss_mask": loss_mask,
            "query_id": sample["query_id"],
            "task": sample["task"],
            "answer": solution  # Add answer for reward function
        }
    
    # Create dataset from the loaded data
    dataset = Dataset.from_list(data)
    
    # Apply processing
    dataset = dataset.map(process).remove_columns(["prompt", "solutions"])
    
    # Filter out sequences longer than max_length if specified
    if max_length is not None:
        dataset = dataset.filter(lambda x: len(x["input_ids"]) <= max_length)
    
    # Split dataset across nodes for distributed training
    dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)
    return dataset


def get_openr1_rl_dataset(
    path: str,
    split: str,
    tokenizer,
    rank: int,
    world_size: int,
    max_length: Optional[int] = None,
):
    """
    Load OpenR1 dataset for reinforcement learning.
    
    Args:
        path: Path to the OpenR1-raw-qwen.jsonl file
        split: Dataset split (not used for custom jsonl file)
        tokenizer: Tokenizer to encode the text
        rank: Current process rank for distributed training
        world_size: Total number of processes
        max_length: Maximum sequence length (optional)
    """
    # Load the jsonl file
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    
    def process(sample):
        # Extract the prompt and clean it to get only the math problem
        prompt = sample["prompt"]
        
        # Extract only the math problem content from ChatML format
        # Remove ChatML tags and extract the actual problem text
        if "<|im_start|>user\n" in prompt:
            # Extract content between <|im_start|>user\n and \nPlease reason step by step
            user_content = prompt.split("<|im_start|>user\n")[1]
            if "\nPlease reason step by step" in user_content:
                problem_text = user_content.split("\nPlease reason step by step")[0].strip()
            elif "\n<|im_end|>" in user_content:
                problem_text = user_content.split("\n<|im_end|>")[0].strip()
            else:
                problem_text = user_content.strip()
        else:
            # Fallback: just remove the instruction text
            instruction_to_remove = "Please reason step by step, and put your final answer within \\boxed{}."
            if instruction_to_remove in prompt:
                problem_text = prompt.replace(instruction_to_remove, "").strip()
            else:
                problem_text = prompt.strip()
        
        # Get the solution (for reward function)
        solution = sample["solutions"][0] if sample["solutions"] else ""
        
        # Create messages format for RL training
        messages = [
            {
                "role": "user",
                "content": problem_text + "\nPlease reason step by step, and put your final answer within \\boxed{}."
            }
        ]
        
        return {
            "messages": messages,
            "query_id": sample["query_id"],
            "task": sample["task"],
            "answer": solution  # Add answer for reward function
        }
    
    # Create dataset from the loaded data
    dataset = Dataset.from_list(data)
    
    # Apply processing
    dataset = dataset.map(process).remove_columns(["prompt", "solutions"])
    
    # Filter out sequences longer than max_length if specified
    if max_length is not None:
        def filter_length(sample):
            # Tokenize the user content to check length
            content = sample["messages"][0]["content"]
            tokens = tokenizer.encode(content)
            return len(tokens) <= max_length
        
        dataset = dataset.filter(filter_length)
    
    # Split dataset across nodes for distributed training
    dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)
    return dataset