"""
Preprocess grounding train dataset to parquet format with IoU-based sampling.

Train JSON example:
{"image": "...",
 "conversations": [
    {"from": "human", "value": "<image>..."},
    {"from": "gpt",   "value": "...<box>[x1,y1,x2,y2]</box>..."}
 ],
 "iou_main": 0.82,
 "241b_model_iou": 0.75,
 ...}
"""
from PIL import Image
import argparse
import os
import re
import random
import datasets

BASE_IMG_PATH = os.getenv("BASE_IMG_PATH")


def extract_answer(answer_raw: str) -> list[int]:
    pattern = r"<box>\s*\[(.*?)\]\s*</box>"
    match = re.search(pattern, answer_raw)
    if not match:
        raise ValueError(f"Answer format is incorrect: {answer_raw}")
    return list(map(int, match.group(1).strip().split(",")))


def make_map_fn(split: str, fmt: str):
    """
    fmt: "default" or "qwen"
    """

    def process_fn(example, idx):
        conversation = example.pop("conversations")

        orig_h = example.get("height")
        orig_w = example.get("width")

        img_rel = example["image"]
        img_path = os.path.join(BASE_IMG_PATH, img_rel)

        if os.path.exists(img_path):
            with Image.open(img_path) as im:
                orig_w, orig_h = im.size
        else:
            print(f"[Warning] Image not found: {img_path}")

        if fmt == "qwen" and "sent" in example:
            prompt = "<image>\nLocate {sent}, output its bbox coordinates using JSON format"
            question_raw = prompt.format(sent=example["sent"])
        else:
            question_raw = conversation[0]["value"]

        answer_raw = conversation[1]["value"]
        solution = extract_answer(answer_raw)

        data = {
            "data_source": "grounding",
            "prompt": [
                {
                    "role": "user",
                    "content": question_raw,
                }
            ],
            "images": [img_path],
            "ability": "grounding",
            "reward_model": {
                "style": "rule",
                "ground_truth": solution,
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "id": example.get("id"),
                "height": orig_h,
                "width": orig_w,
                "question": question_raw,
            },
        }
        return data

    return process_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess grounding train JSON file to parquet with IoU-based sampling."
    )
    parser.add_argument(
        "--input_json",
        required=True,
        help="Input train JSON file path.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output parquet file directory.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=64,
        help="Number of processes for datasets.map.",
    )
    parser.add_argument(
        "--format",
        choices=["default", "qwen"],
        help="Output prompt format.",
    )

    args = parser.parse_args()

    input_json = os.path.expanduser(args.input_json)
    output_dir = os.path.expanduser(args.output_dir)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    num_workers = int(args.num_workers)
    fmt = args.format

    ds_all = datasets.load_dataset("json", data_files=input_json)["train"]

    ds_train = ds_all.map(
        function=make_map_fn("train", fmt),
        with_indices=True,
        num_proc=num_workers,
    )
    print(f"[grounding preprocess] train samples (final): {ds_train.num_rows}")

    ds_train.to_parquet(os.path.join(output_dir, "train_grounding.parquet"))
    print(f"[grounding preprocess] wrote {os.path.join(output_dir, 'train_grounding.parquet')}")
