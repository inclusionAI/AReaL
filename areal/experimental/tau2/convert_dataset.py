"""
Convert tau2 dataset to huggingface dataset.

Usage:
    python convert_dataset.py --data_dir /path/to/tau2/data --output_dir /path/to/output/dataset --split train
"""

import os

from datasets import Dataset


def convert_dataset(data_dir: str, output_dir: str, split: str = "train"):
    from tau2.registry import registry

    domains = os.listdir(os.path.join(data_dir, "tau2", "domains"))

    all_task_ids = []
    for domain in domains:
        splits_loader_fn = registry.get_task_splits_loader(domain)
        if splits_loader_fn is None:
            print(f"No task splits loader found for domain {domain}, skip")
            continue
        splits = splits_loader_fn()
        if split not in splits:
            print(
                f"Split {split} not found in {splits}, available splits: {list(splits.keys())} for domain {domain}, skip"
            )
            continue
        task_ids = splits[split]
        all_task_ids.extend(
            [{"task_id": task_id, "domain": domain} for task_id in task_ids]
        )
    dataset = Dataset.from_list(all_task_ids)
    dataset.save_to_disk(output_dir)


if __name__ == "__main__":
    import argparse

    from loguru import logger

    logger.remove()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--split", type=str, default="train", choices=["train", "test", "val"]
    )
    args = parser.parse_args()
    os.environ["TAU2_DATA_DIR"] = args.data_dir
    convert_dataset(args.data_dir, args.output_dir, args.split)
