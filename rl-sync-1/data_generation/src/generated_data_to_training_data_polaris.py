#!/usr/bin/env python

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Prepare Polaris training data with Qwen templates, quality filtering, and JSONL exports.
This script wraps the original notebook-style workflow into a single CLI entrypoint.
"""

import argparse
import concurrent.futures
import json
import os
import random
from typing import Dict, List, Tuple

import datasets
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

# The rewards package is expected to be importable (as in the original workflow).
# You may need to ensure the repo that provides these modules is on PYTHONPATH.
from rewards import rllm_reward_fn_math  # type: ignore


def load_inputs(dataset_json: str, polaris_parquet: str) -> Tuple[List[List[str]], pd.DataFrame]:
    with open(dataset_json, "r") as f:
        data = json.load(f)

    polaris_df = pd.read_parquet(polaris_parquet)
    print(f"Loaded Polaris parquet with {len(polaris_df)} rows")
    return data, polaris_df


def build_tokenizer(qwen_model_name: str) -> AutoTokenizer:
    qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_model_name)
    return qwen_tokenizer


def process_item(
    data_item: List[str],
    df_row: pd.Series,
    instruction: str,
    qwen_tokenizer: AutoTokenizer,
) -> Dict[str, object]:
    question = df_row.prompt[0]["content"]
    response_content_with_reasoning = data_item[0]

    messages = [
        {"role": "user", "content": f"{question} {instruction}"},
        {"role": "assistant", "content": f"{response_content_with_reasoning}"},
    ]

    response_reasoning = response_content_with_reasoning.split("</think>")[0].strip().removeprefix("<think>")
    response_content = response_content_with_reasoning.split("</think>")[-1].strip()

    qwen_text = qwen_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    assert "<think>" in qwen_text, "Qwen text should contain <think> tag"

    return {
        "question": question,
        "response_reasoning": response_reasoning,
        "response_content": response_content,
        "qwen_text": qwen_text,
        "num_qwen_tokens": len(qwen_tokenizer(qwen_text)["input_ids"]),
        "raw_messages": messages,
    }


def filter_correct(data: List[List[str]], polaris_df: pd.DataFrame) -> Tuple[List[List[str]], pd.DataFrame]:
    corrects = []
    for data_item, df_row in zip(data, polaris_df.itertuples()):
        llm_solution = data_item[0]
        ground_truth = df_row.reward_model["ground_truth"]
        corrects.append(rllm_reward_fn_math("polaris", llm_solution, ground_truth))

    filtered_data = [data[i] for i, c in enumerate(corrects) if c]
    filtered_df = polaris_df.iloc[[i for i, c in enumerate(corrects) if c]]
    print(f"Filtered to {len(filtered_data)} correct rows (from {len(data)}).")
    return filtered_data, filtered_df


def repeat_and_shuffle(texts: List[Dict[str, object]], repeats: int, seed: int) -> List[Dict[str, object]]:
    random.seed(seed)
    combined: List[Dict[str, object]] = []
    for _ in range(repeats):
        copy = texts.copy()
        random.shuffle(copy)
        combined.extend(copy)
    return combined


def save_jsonl(texts: List[Dict[str, object]], output_file: str) -> None:
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        for item in texts:
            f.write(json.dumps({"messages": item["raw_messages"]}) + "\n")
    print(f"Saved {len(texts)} rows to {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert generated Polaris data into training splits.")
    parser.add_argument(
        "--dataset-json",
        type=str,
        required=True,
        help="Input JSON file containing generated answers.",
    )
    parser.add_argument(
        "--polaris-parquet",
        type=str,
        required=True,
        help="Parquet file for the Polaris dataset with ground truth.",
    )
    parser.add_argument("--output-dir", type=str, default=".", help="Base directory to place outputs.")
    parser.add_argument("--instruction", type=str, default="Let's think step by step and output the final answer within \\boxed{}.", help="Instruction appended to the user question.")
    parser.add_argument("--qwen-model", type=str, default="Qwen/Qwen3-8B", help="Tokenizer for Qwen text.")
    parser.add_argument("--sample-size", type=int, default=1000, help="How many examples to sample for the 1k split.")
    parser.add_argument("--repeats", type=int, default=1, help="How many shuffles to concatenate for the 8x variant.")
    parser.add_argument("--workers", type=int, default=32, help="Worker threads for processing.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for shuffling.")
    args = parser.parse_args()

    dataset_json = os.path.expanduser(args.dataset_json)
    polaris_parquet = os.path.expanduser(args.polaris_parquet)
    output_dir = os.path.abspath(args.output_dir)

    data, polaris_df = load_inputs(dataset_json, polaris_parquet)
    qwen_tokenizer = build_tokenizer(args.qwen_model)

    # Filter correctness using the reward fn.
    data, polaris_df = filter_correct(data, polaris_df)
    assert len(data) == len(polaris_df), "Lengths diverged after filtering."

    # Process into structured records.
    def wrapper(pair):
        return process_item(pair[0], pair[1], args.instruction, qwen_tokenizer)

    inputs = list(zip(data, polaris_df.itertuples()))
    texts: List[Dict[str, object]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        for processed in tqdm(executor.map(wrapper, inputs), total=len(inputs)):
            texts.append(processed)

    print(f"Processed {len(texts)} examples.")

    dataset_name_to_save = os.path.splitext(os.path.basename(dataset_json))[0].replace("/", "_").replace("-", "_")
    base = os.path.join(output_dir, dataset_name_to_save)

    # Full JSONL and 8x JSONL.
    save_jsonl(texts, os.path.join(base + "_json", "train.jsonl"))
    if args.repeats > 1:
        texts_8x = repeat_and_shuffle(texts, args.repeats, args.seed)
        save_jsonl(texts_8x, os.path.join(base + f"_{args.repeats}x_json", "train.jsonl"))

    # 1k sampled parquet + JSONL.
    random.seed(args.seed)
    sample_size = min(args.sample_size, len(texts))
    if sample_size < args.sample_size:
        print(f"Warning: requested {args.sample_size} but only {sample_size} available; using all.")
    sampled_indices = random.sample(range(len(texts)), sample_size)
    sampled_texts = [texts[i] for i in sampled_indices]
    sampled_dataset = datasets.Dataset.from_list(sampled_texts)
    sampled_dir = os.path.join(output_dir, f"{dataset_name_to_save}_1k")
    os.makedirs(sampled_dir, exist_ok=True)
    sampled_dataset.to_parquet(os.path.join(sampled_dir, "train.parquet"))
    save_jsonl(sampled_texts, os.path.join(output_dir, f"{dataset_name_to_save}_1k_json", "train.jsonl"))

    print("Done.")


if __name__ == "__main__":
    main()
