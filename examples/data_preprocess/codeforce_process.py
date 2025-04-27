# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

"""
Dataset Toolkit - Process and validate code/math datasets with flexible input support
"""
import argparse
import json
import logging
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# Configure console logging
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.DEBUG)
logger = logging.getLogger(__name__)


def load_jsonl(file_path: str) -> List[Dict]:
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


def save_file(output_path: str, processed_data: list):
    with open(output_path, "w") as f:
        for item in processed_data:
            f.write(json.dumps(item) + "\n")


def process_math_data(file_path: str) -> List[Dict]:
    """Process math dataset from JSON/JSONL file"""
    if not file_path:
        return []

    raw_data = load_jsonl(file_path)
    processed = []

    for index, item in enumerate(raw_data):
        processed.append(
            {
                "task": "math",
                "query_id": str(item.get("query_id", f"math-{index}")),
                "prompt": item["context"],
                "solutions": [item["groundtruth"]],
            }
        )

    return processed


def process_code_data(file_path: str) -> List[Dict]:
    """Process code dataset from JSONL file"""
    if not file_path:
        return []

    raw_data = load_jsonl(file_path)
    processed = []

    for item in raw_data:
        # Field extraction and transformation
        input_output = item["code_test_cases"]

        time_limit = item["meta"]["time_limit"]
        seconds = time_limit.get("seconds", 0) + time_limit.get("nanos", 0) / 1e9
        memory = item["meta"]["memory_limit_bytes"] / (1024 * 1024)
        processed.append(
            {
                "task": "code",
                "query_id": str(item["id"]),
                "prompt": item["context"],
                "input_output": json.dumps(
                    {
                        "inputs": [io.get("input") for io in input_output],
                        "outputs": [io.get("output") for io in input_output],
                        "fn_name": item.get("metadata", {}).get("fn_name", ""),
                        "remote": False,
                    }
                ),
                "solutions": [item["groundtruth"]],
                "language": "PYTHON",
                "timeout": seconds,
                "memory": memory,
            }
        )

    return processed


def main():
    parser = argparse.ArgumentParser(
        description="Dataset Toolkit: Process and validate STEM datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--code", help="Path to code dataset (JSONL)")
    parser.add_argument("--math", help="Path to math dataset (JSONL)")
    parser.add_argument("--output", help="Output file path (JSONL)")

    args = parser.parse_args()

    if not args.output:
        logger.error("Output file required in process mode")
        return
    processed_data = []
    stats = defaultdict(int)

    if args.code:
        code_data = process_code_data(args.code)
        logger.info(f"Loaded {len(code_data)} code items")
        processed_data.extend(code_data)
        stats["code"] = len(code_data)
    if args.math:
        math_data = process_math_data(args.math)
        logger.info(f"Loaded {len(math_data)} math items")
        processed_data.extend(math_data)
        stats["math"] = len(math_data)

    random.shuffle(processed_data)
    save_file(args.output, processed_data)
    logger.info("\nProcessing Complete:")
    logger.info(f"Total items: {len(processed_data)}")
    logger.info(f"Code items: {stats['code']}")
    logger.info(f"Math items: {stats['math']}")


if __name__ == "__main__":
    main()
