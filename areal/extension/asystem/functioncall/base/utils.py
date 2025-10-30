# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").
import json
import uuid
from datetime import datetime

try:
    from areal.base import constants, logging

    logger = logging.getLogger("function call")
except Exception:
    import logging

    constants = None

    logger = logging.getLogger("function call")
    logger.setLevel(logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console.setFormatter(formatter)
    logger.addHandler(console)


def construct_swe_uid(query_id: str):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    try:
        trial_name = f"{constants.experiment_name()}&{constants.trial_name()}"
    except Exception as e:
        trial_name = "test"
    uid = f"{timestamp}-[{trial_name}]-{query_id}-{str(uuid.uuid4())[:4]}"
    return uid


def construct_uid(query_id: str, start_idx: int, end_idx: int):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    try:
        trial_name = f"{constants.experiment_name()}&{constants.trial_name()}"
    except Exception as e:
        trial_name = "test"
    uid = f"{timestamp}-[{trial_name}]-{query_id}-[{start_idx}-{end_idx}]-{str(uuid.uuid4())[:4]}-0"
    return uid


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
