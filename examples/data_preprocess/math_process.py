#!/usr/bin/env python3
import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional


def load_jsonl(file_path: Path) -> List[Dict]:
    """Load JSONL file with validation"""
    try:
        with file_path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]
    except FileNotFoundError:
        print(f"ERROR: JSONL file not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        print(f"ERROR: JSON parsing failed in {file_path}: {str(e)}")
        raise


def load_id2info(file_path: Path) -> Dict:
    """Load ID mapping file with structure validation"""
    try:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if not all(isinstance(v, dict) and "solutions" in v for v in data.values()):
                raise ValueError("Invalid id2info structure")
            return data
    except FileNotFoundError:
        print(f"ERROR: ID mapping file not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        print(f"ERROR: JSON parsing failed in {file_path}: {str(e)}")
        raise


def process_data(prompts_data: List[Dict], id2info: Dict, output_path: Path) -> None:
    """Process and save transformed data"""
    processed = []
    missing_ids = 0
    for item in prompts_data:
        query_id = item.get("query_id")
        if not query_id or query_id not in id2info:
            missing_ids += 1
            continue
        processed.append(
            {
                "prompt": item.get("prompt", ""),
                "task": "math",
                "query_id": query_id,
                "solutions": id2info[query_id].get("solutions", []),
            }
        )
    if missing_ids:
        print(f"WARNING: Found {missing_ids} items with missing/invalid query_id")
    if not processed:
        print("ERROR: No valid data to process")
        sys.exit(1)
    random.shuffle(processed)

    try:
        with output_path.open("w", encoding="utf-8") as f:
            for item in processed:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"SUCCESS: Wrote {len(processed)} items to {output_path}")
    except IOError as e:
        print(f"ERROR: Failed to write output: {str(e)}")
        raise


def main():
    """
    Command line entry point
    
    Example usage:
    python math_process.py \
      --prompts_path ./input/prompts.jsonl \
      --id2info_path ./input/id2info.json \
      --output_path ./output/processed.jsonl
    """
    parser = argparse.ArgumentParser(
        description="Math dataset processing tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--prompts_path",
        type=Path,
        required=True,
        help="Path to prompts.jsonl input file",
    )
    parser.add_argument(
        "--id2info_path",
        type=Path,
        required=True,
        help="Path to id2info.json input file",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default="output.jsonl",
        help="Path for processed output file",
    )

    args = parser.parse_args()
    print("Starting data processing...")
    try:
        prompts_data = load_jsonl(args.prompts_path)
        id2info = load_id2info(args.id2info_path)
        process_data(prompts_data, id2info, args.output_path)
    except Exception as e:
        print(f"FATAL: Processing failed - {str(e)}")
        sys.exit(1)
    print("Operation completed successfully")


if __name__ == "__main__":
    main()
