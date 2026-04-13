"""
Data preprocessing script for GSM8K-Code task.

This script downloads the GSM8K dataset and converts it to the format
required by verl-tool for MT-GRPO training.

Usage:
    python gsm8k_code.py --local_dir ~/data/gsm8k_code
"""

import argparse
import logging
import os
import tempfile
import re

import pandas as pd
from datasets import load_dataset

from verl.utils.hdfs_io import copy, makedirs

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# System prompt for code execution task (from Multi-Turn-RL-Agent)
CODE_SYSTEM_PROMPT = """\
Given a math problem, use step-by-step reasoning and code execution to solve the problem. 

For each step:
1. Think through your reasoning inside <reasoning> tags
2. You must write Python scripts inside <code> tags to work out calculations, but it's not required.
   - Functions and variables do not persist across <code> calls and should be redefined each time
   - Scripts should be written in Python 3.10+ syntax, and should run in under 10 seconds
   - Any desired outputs should be printed using print() statements
   - You may import numpy, scipy, and sympy libraries for your calculations
3. You will see the output from print() statements in your code in <output> tags
4. Continue until you can give the final answer inside <answer> tags
"""


def extract_answer_from_solution(solution: str) -> str:
    """Extract the numerical answer from GSM8K solution string."""
    # GSM8K answers are in format "#### <number>"
    match = re.search(r'####\s*([\d,\.]+)', solution)
    if match:
        answer = match.group(1).replace(',', '')
        return answer
    return solution


def process_gsm8k_row(row, split_name: str, row_index: int) -> pd.Series:
    """
    Process a single row of GSM8K data.
    
    Args:
        row: Dictionary containing 'question' and 'answer' fields
        split_name: Name of the current split (train/test)
        row_index: Index of the row
        
    Returns:
        pd.Series: Processed row data
    """
    question = row.get("question", "")
    solution = row.get("answer", "")
    
    # Extract numerical answer
    ground_truth = extract_answer_from_solution(solution)
    
    # Build prompt structure
    user_content = f"Solve the following problem:\n\n{question}"
    prompt = [
        {"role": "system", "content": CODE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]
    
    # Build reward model data
    reward_model_data = {
        "ground_truth": ground_truth,
        "style": "rule",
        "full_solution": solution
    }
    
    # Build tools kwargs for python code execution
    tools_kwargs = {
        "python_code": {
            "create_kwargs": {
                "question": question,
                "ground_truth": ground_truth
            }
        }
    }
    
    # Build extra_info structure
    extra_info = {
        "index": row_index,
        "need_tools_kwargs": True,
        "question": question,
        "split": split_name,
        "tools_kwargs": tools_kwargs,
    }
    
    return pd.Series({
        "data_source": f"gsm8k_{split_name}",
        "prompt": prompt,
        "ability": "math_code",
        "reward_model": reward_model_data,
        "extra_info": extra_info,
        "metadata": {
            "original_question": question,
            "original_solution": solution
        }
    })


def main():
    local_save_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_save_dir, exist_ok=True)
    
    processed_files = []
    
    # Load GSM8K dataset
    logger.info(f"Loading GSM8K dataset from {args.dataset_name}")
    
    try:
        dataset = load_dataset(args.dataset_name, args.dataset_config)
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return
    
    # Process each split
    split_mapping = {
        "train": "train",
        "test": "test"
    }
    
    for hf_split, output_split in split_mapping.items():
        if hf_split not in dataset:
            logger.warning(f"Split {hf_split} not found in dataset")
            continue
        
        logger.info(f"Processing {hf_split} split...")
        
        # Convert to pandas DataFrame
        df_raw = dataset[hf_split].to_pandas()
        logger.info(f"Loaded {len(df_raw)} rows from {hf_split} split")
        
        # Apply processing
        def apply_process_row(row, split_name=output_split):
            return process_gsm8k_row(row, split_name=split_name, row_index=row.name)
        
        df_processed = df_raw.apply(apply_process_row, axis=1)
        
        # Save processed DataFrame
        output_file_path = os.path.join(local_save_dir, f"{output_split}.parquet")
        df_processed.to_parquet(output_file_path, index=False)
        logger.info(f"Saved {len(df_processed)} processed rows to {output_file_path}")
        processed_files.append(output_file_path)
    
    if not processed_files:
        logger.warning("No data was processed or saved")
        return
    
    logger.info(f"Successfully processed {len(processed_files)} files to {local_save_dir}")
    
    # Copy to HDFS if specified
    if args.hdfs_dir:
        try:
            makedirs(args.hdfs_dir)
            copy(src=local_save_dir, dst=args.hdfs_dir)
            logger.info(f"Successfully copied files to HDFS: {args.hdfs_dir}")
        except Exception as e:
            logger.error(f"Error copying files to HDFS: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download GSM8K dataset and preprocess for MT-GRPO training."
    )
    parser.add_argument(
        "--dataset_name",
        default="openai/gsm8k",
        help="HuggingFace dataset name."
    )
    parser.add_argument(
        "--dataset_config",
        default="main",
        help="Dataset configuration name."
    )
    parser.add_argument(
        "--local_dir",
        default="~/data/gsm8k_code",
        help="Local directory to save the processed Parquet files."
    )
    parser.add_argument(
        "--hdfs_dir",
        default=None,
        help="Optional HDFS directory to copy the Parquet files to."
    )
    
    args = parser.parse_args()
    main()

