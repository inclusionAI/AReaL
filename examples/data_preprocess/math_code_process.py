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


def process_code_data(file_path: str) -> List[Dict]:
    """Process code dataset from JSONL file"""
    if not file_path:
        return []

    raw_data = load_jsonl(file_path)
    processed = []

    for item in raw_data:
        # Field extraction and transformation
        input_output = json.loads(item["input_output"])
        processed.append(
            {
                "task": "code",
                "query_id": str(item["id"]),
                "prompt": item["question"],
                "solutions": item.get("solutions", []),
                "input_output": json.dumps(
                    {
                        "inputs": input_output.get("inputs", []),
                        "outputs": input_output.get("outputs", []),
                        "fn_name": item.get("metadata", {}).get("fn_name", ""),
                        "remote": False,
                    }
                ),
                "language": item.get("language", "PYTHON"),
            }
        )

        case_size = sys.getsizeof(processed[-1]["input_output"])
        assert (
            case_size < 500 * 1024
        ), f"'input_output' exceeds 500KB ({case_size} bytes). Use remote testcase instead."

    return processed


def process_math_data(file_path: str) -> List[Dict]:
    """Process math dataset from JSON/JSONL file"""
    if not file_path:
        return []

    raw_data = load_jsonl(file_path)
    processed = []

    for item in raw_data:
        processed.append(
            {
                "task": "math",
                "query_id": str(item["query_id"]),
                "prompt": item["prompt"],
                "solutions": item.get("solutions", []),
            }
        )

    return processed


def validate_raw_code(item: Dict) -> Tuple[bool, List[str]]:
    """Validate raw code item structure"""
    errors = []
    required = {
        "task": str,
        "query_id": str,
        "prompt": str,
        "input_output": str,
    }

    code_input_output_required = {
        "inputs": list,
        "outputs": list,
        #'fn_name': str
    }

    for field, typ in required.items():
        if field not in item:
            errors.append(f"Missing required field: {field}")
        elif not isinstance(item[field], typ):
            errors.append(f"Invalid type for {field}: expected {typ.__name__}")

    input_output = json.loads(item["input_output"])

    for io_field, io_typ in code_input_output_required.items():
        if io_field not in input_output:
            errors.append(f"Missing required field: {io_field} in input_output")
        elif not isinstance(input_output[io_field], io_typ):
            errors.append(
                f"Invalid type for {io_field}: expected {io_typ.__name__} in input_output"
            )

    return (not errors, errors)


def validate_raw_math(item: Dict) -> Tuple[bool, List[str]]:
    """Validate raw math item structure"""
    errors = []
    required = {"query_id": str, "prompt": str, "solutions": list}

    for field, typ in required.items():
        if field not in item:
            errors.append(f"Missing required field: {field}")
        elif not isinstance(item[field], typ):
            type_names = (
                [t.__name__ for t in typ] if isinstance(typ, tuple) else typ.__name__
            )
            errors.append(f"Invalid type for {field}: expected {type_names}")

    return (not errors, errors)


def validate_raw_item(item: Dict) -> Tuple[bool, List[str]]:
    """Validate raw code item structure"""
    if item.get("task", "") == "math":
        return validate_raw_math(item)
    else:
        return validate_raw_code(item)


def check_item_valid(raw_data: list):
    if not raw_data:
        return defaultdict(int)
    logger.info(f"Validating code-math dataset...")
    total_stats = defaultdict(int)
    for index, item in enumerate(raw_data):
        valid, errors = validate_raw_item(item)
        if valid:
            total_stats["valid"] += 1
        else:
            total_stats["invalid"] += 1
            logger.warning(f"item {index}: {errors}")
    return total_stats


def filter_shuffle(shuffle: bool, processed_data):
    # Final validation and write output
    valid_output = []
    invalid_count = 0
    for item in processed_data:
        valid, errors = validate_raw_item(item)
        if valid:
            valid_output.append(item)
        else:
            invalid_count += 1
            logger.error(
                f'{item["task"]} item {item.get("query_id")} is invalid: {errors}'
            )
    if shuffle:
        random.shuffle(valid_output)
    return valid_output, invalid_count


def save_file(output_path: str, processed_data: list):
    with open(output_path, "w") as f:
        for item in processed_data:
            f.write(json.dumps(item) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Dataset Toolkit: Process and validate STEM datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--code", help="Path to code dataset (JSONL)")
    parser.add_argument("--math", help="Path to math dataset (JSONL)")
    parser.add_argument("--output", help="Output file path (JSONL)")
    parser.add_argument(
        "--mode",
        choices=["check", "process"],
        default="check",
        help="Operation mode: check raw data or process datasets",
    )
    parser.add_argument(
        "--shuffle", action="store_true", help="Shuffle output data in process mode"
    )

    args = parser.parse_args()
    if args.mode == "check":
        # Validate raw data structure
        code_data = load_jsonl(args.code) if args.code else []
        math_data = load_jsonl(args.math) if args.math else []
        total_stats = check_item_valid(code_data + math_data)

        # Print validation summary
        logger.info(
            f"\nValidation Summary: {total_stats['valid']} valid, {total_stats['invalid']} invalid"
        )
    elif args.mode == "process":
        # Process and merge datasets
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

        processed_data, invalid_count = filter_shuffle(args.shuffle, processed_data)
        stats["invalid"] = invalid_count
        save_file(args.output, processed_data)
        logger.info("\nProcessing Complete:")
        logger.info(f"Total items: {len(processed_data)}")
        logger.info(f"Code items: {stats['code']}")
        logger.info(f"Math items: {stats['math']}")
        logger.info(f"Invalid items: {stats['invalid']}")


if __name__ == "__main__":
    main()
