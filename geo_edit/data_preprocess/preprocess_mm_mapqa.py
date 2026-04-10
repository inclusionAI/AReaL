#!/usr/bin/env python3
"""Flatten mm_mapqa multi-turn QA into single-turn parquet with images saved to disk.

Saves each unique source image once to {output_dir}/images/{row_idx}.png,
then stores the path in the parquet `image` column (string). This avoids
duplicating ~70KB image bytes across ~13 QA pairs per image, keeping the
parquet lightweight (~50MB vs ~34GB).
"""
from __future__ import annotations

import argparse
import glob
import os

import pandas as pd
from tqdm import tqdm


def flatten_mm_mapqa(input_dir: str, output_dir: str) -> None:
    parquet_files = sorted(glob.glob(os.path.join(input_dir, "train-*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet shards found in {input_dir}")

    print(f"Found {len(parquet_files)} shard(s) in {input_dir}")

    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    rows: list[dict] = []
    global_row_idx = 0
    skipped_no_images = 0
    skipped_no_bytes = 0
    total_source_rows = 0

    for pf in tqdm(parquet_files, desc="Reading shards"):
        df = pd.read_parquet(pf)
        total_source_rows += len(df)
        for _, row in df.iterrows():
            images = row["images"]
            data = row["data"]

            if len(images) == 0:
                skipped_no_images += 1
                global_row_idx += 1
                continue
            img_bytes = images[0].get("bytes")
            if not img_bytes:
                skipped_no_bytes += 1
                global_row_idx += 1
                continue

            image_path = os.path.join(images_dir, f"{global_row_idx}.png")
            if not os.path.exists(image_path):
                with open(image_path, "wb") as img_f:
                    img_f.write(img_bytes)

            # Data array format: image ref, then alternating user-text / assistant-text pairs
            qa_idx = 0
            i = 0
            while i < len(data):
                entry = data[i]
                if entry["modality"] == "image":
                    i += 1
                    continue
                if entry["role"] == "user" and entry["modality"] == "text":
                    question = entry["data"]
                    if i + 1 < len(data) and data[i + 1]["role"] == "assistant" and data[i + 1]["modality"] == "text":
                        answer = data[i + 1]["data"]
                        rows.append({
                            "id": f"{global_row_idx}_q{qa_idx}",
                            "image": image_path,
                            "question": question,
                            "answer": answer,
                        })
                        qa_idx += 1
                        i += 2
                        continue
                i += 1

            global_row_idx += 1

    print(f"Source rows: {total_source_rows}")
    print(f"Skipped (no images): {skipped_no_images}")
    print(f"Skipped (no bytes): {skipped_no_bytes}")
    print(f"Images saved: {total_source_rows - skipped_no_images - skipped_no_bytes}")
    print(f"Total flattened QA pairs: {len(rows)}")

    output_path = os.path.join(output_dir, "mm_mapqa_flat.parquet")
    df_out = pd.DataFrame(rows)
    df_out.to_parquet(output_path, index=False)
    print(f"Saved to {output_path}")
    print(f"Columns: {df_out.columns.tolist()}")
    print(f"Shape: {df_out.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/storage/openpsi/data/mm_mapqa/data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/storage/openpsi/data/mm_mapqa/processed",
    )
    args = parser.parse_args()
    flatten_mm_mapqa(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
