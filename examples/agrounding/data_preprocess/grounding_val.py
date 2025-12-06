"""
Preprocess grounding test datasets to parquet format.

Test JSON example:
{"image": "...", "sent": "...", "bbox": [x1,y1,x2,y2], "height": H, "width": W}
"""
from PIL import Image
import argparse
import os
import re
import datasets

BASE_IMG_PATH = os.getenv("BASE_IMG_PATH")
QWEN_MAX_AREA = 641 * 641


def bbox_px_to_norm_1000(bbox, w: int, h: int):
    x1, y1, x2, y2 = bbox
    w_f = float(w)
    h_f = float(h)
    nx1 = float(x1) * 1000.0 / w_f
    ny1 = float(y1) * 1000.0 / h_f
    nx2 = float(x2) * 1000.0 / w_f
    ny2 = float(y2) * 1000.0 / h_f
    return [nx1, ny1, nx2, ny2]


def make_map_fn_test(fmt: str):
    def process_fn(example, idx):
        img_rel = example["image"]
        img_path = os.path.join(BASE_IMG_PATH, img_rel)

        sent = example["sent"]
        bbox = example["bbox"]

        if not os.path.exists(img_path):
            print(f"[Warning] Image not found: {img_path}")
            W, H = example.get("width", 0), example.get("height", 0)
        else:
            with Image.open(img_path) as im:
                W, H = im.size

        if fmt == "qwen":
            content = (
                "<image>\n"
                f"Locate {sent}, output its bbox coordinates using JSON format"
            )
        else:
            content = (
                "<image>\n"
                "Please provide the bounding box coordinate of the region "
                f"this sentence describes: <ref>{sent}</ref>"
            )

        data = {
            "data_source": "grounding",
            "prompt": [{"role": "user", "content": content}],
            "images": [img_path],
            "ability": "grounding",
            "reward_model": {
                "style": "rule",
                "ground_truth": bbox_px_to_norm_1000(bbox, W, H),
            },
            "extra_info": {
                "split": "test",
                "index": idx,
                "height": H,
                "width": W,
                "question": sent,
            },
        }

        return data

    return process_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess grounding test JSON files to parquet."
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing input JSON files.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to write output parquet files.",
    )
    parser.add_argument(
        "--test_files",
        nargs="+",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--format",
        choices=["default", "qwen"],
    )

    args = parser.parse_args()

    input_dir = os.path.expanduser(args.input_dir)
    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    num_workers = int(args.num_workers)
    fmt = args.format

    test_files = []
    for f in args.test_files:
        if os.path.isabs(f):
            test_files.append(f)
        else:
            test_files.append(os.path.join(input_dir, f))

    mixed_slices = []
    for file_path in test_files:
        if not os.path.exists(file_path):
            print(f"[Warning] Test file not found: {file_path}")
            continue

        ds_t = datasets.load_dataset("json", data_files=file_path)["train"]
        ds_t = ds_t.map(
            function=make_map_fn_test(fmt),
            with_indices=True,
            num_proc=num_workers,
        )

        n = len(ds_t)
        k = max(1, int(n * 0.1))
        mixed_slices.append(ds_t.shuffle(seed=42).select(range(k)))

    mixed = datasets.concatenate_datasets(mixed_slices).shuffle(seed=42)
    out_test = os.path.join(output_dir, f"val_grounding.parquet")
    mixed.to_parquet(out_test)
    print(f"[grounding preprocess] Val samples: {len(mixed)}, wrote {out_test}")
