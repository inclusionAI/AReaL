"""Preprocess GeoEdit datasets to parquet format for verl-agent training.

Expects input JSONL files with fields:
  - task_id: str
  - question: str (or task_prompt)
  - answer: str (or task_answer)
  - image_path: str (absolute or relative path to image)
  - task_type: str (optional: "exact", "contains", "numeric", "option")

Output parquet contains:
  - data_source: str
  - prompt: List[Dict] (chat messages)
  - reward_model: Dict
  - extra_info: Dict
  - env_kwargs: Dict (task_id, task_prompt, task_answer, task_image_path, task_type)
"""

import argparse
import json
import os

import pandas as pd


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def process_dataset(data, split, data_source="geo_edit", image_root=None):
    """Convert raw data to verl-agent parquet format."""
    records = []

    for idx, item in enumerate(data):
        task_id = item.get("task_id", f"{split}_{idx}")
        question = item.get("question", item.get("task_prompt", ""))
        answer = item.get("answer", item.get("task_answer", ""))
        image_path = item.get("image_path", item.get("task_image_path", ""))
        task_type = item.get("task_type", "exact")

        # Resolve image path
        if image_path and image_root and not os.path.isabs(image_path):
            image_path = os.path.join(image_root, image_path)

        # Prompt: minimal - the full system prompt and tools are injected by
        # GeoEditEnvironmentManager.build_text_obs() at reset time.
        # We only need the user question here as the initial prompt.
        prompt = [
            {"role": "user", "content": question},
        ]

        record = {
            "data_source": data_source,
            "prompt": prompt,
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {
                "split": split,
                "index": idx,
                "answer": answer,
                "question": question,
                "task_type": task_type,
            },
            "env_kwargs": {
                "task_id": task_id,
                "task_prompt": question,
                "task_answer": answer,
                "task_image_path": image_path,
                "task_type": task_type,
            },
        }
        records.append(record)

    return records


def main():
    parser = argparse.ArgumentParser(description="Preprocess GeoEdit data to parquet")
    parser.add_argument("--train_file", type=str, required=True, help="Path to train JSONL file")
    parser.add_argument("--test_file", type=str, default=None, help="Path to test JSONL file")
    parser.add_argument("--output_dir", type=str, default="~/data/geo_edit", help="Output directory")
    parser.add_argument("--image_root", type=str, default=None, help="Root directory for resolving relative image paths")
    parser.add_argument("--data_source", type=str, default="geo_edit", help="Data source name")
    args = parser.parse_args()

    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Process train
    train_data = load_jsonl(args.train_file)
    train_records = process_dataset(train_data, "train", args.data_source, args.image_root)
    train_df = pd.DataFrame(train_records)
    train_path = os.path.join(output_dir, "train.parquet")
    train_df.to_parquet(train_path)
    print(f"Train: {len(train_df)} samples -> {train_path}")

    # Process test
    if args.test_file:
        test_data = load_jsonl(args.test_file)
        test_records = process_dataset(test_data, "test", args.data_source, args.image_root)
        test_df = pd.DataFrame(test_records)
        test_path = os.path.join(output_dir, "test.parquet")
        test_df.to_parquet(test_path)
        print(f"Test: {len(test_df)} samples -> {test_path}")


if __name__ == "__main__":
    main()
