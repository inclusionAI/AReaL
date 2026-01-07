# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").
import json
from datetime import datetime
import uuid

from areal.utils import logging

logger = logging.getLogger("FunctionCall")




def construct_uid(query_id: str, start_idx: int, end_idx: int):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    trial_name = str(uuid.uuid4())
    uid = f"[{timestamp}-{trial_name}]-{query_id}-[{start_idx}-{end_idx}]"
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
