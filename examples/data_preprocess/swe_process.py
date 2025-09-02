#!/usr/bin/env python3
import argparse
import json
import sys
import traceback
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


def load_jsonl(file_path: str):
    """Load JSONL file with validation"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]
    except FileNotFoundError:
        print(f"ERROR: JSONL file not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        print(f"ERROR: JSON parsing failed in {file_path}: {str(e)}")
        raise


def process_data(id2info: List, output_path: Path) -> None:
    """Process and save transformed data"""
    processed = []
    for i in range(len(id2info)):
        item = id2info[i]
        # line = line.replace(": NaN", ": null") if "NaN" in line else line

        instance_id = item.get("instance_id")
        artifact_info_str = item.get("pipeline_artifact_info")
        if not instance_id or not artifact_info_str:
            print(
                f"Warning: Skipping line {i} due to missing 'instance_id' or 'pipeline_artifact_info'."
            )
            continue

        artifact_info = json.loads(artifact_info_str)
        function = artifact_info.get("function")
        base_commit = item.get("base_commit")
        instance_id = item.get("instance_id")
        full_patch_str = item.get("full_patch_str")
        test_file_list = item.get("test_file")

        if (
            not function
            or not base_commit
            or not instance_id
            or not full_patch_str
            or not test_file_list
        ):
            print(f"[Fail] invalid data : {item}")
            continue

        processed.append(
            {
                "task": "swe",
                "query_id": item.get("instance_id"),
                "prompt": item.get("prompt", ""),
                "commitId": base_commit,
                "patch": full_patch_str,
                "testcases": json.loads(test_file_list),
                "runtime": function,
            }
        )

    try:
        with output_path.open("w", encoding="utf-8") as f:
            for item in processed:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"SUCCESS: Wrote {len(processed)} items to {output_path}")
    except IOError as e:
        print(f"ERROR: Failed to write output: {str(e)}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Math dataset processing tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
        id2info = load_jsonl(args.id2info_path)
        process_data(id2info, args.output_path)
    except Exception as e:
        print(f"FATAL: Processing failed - {str(e)}")
        traceback.print_exc()
        sys.exit(1)
    print("Operation completed successfully")


if __name__ == "__main__":
    main()
