# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").
from datetime import datetime

try:
    from realhf.base import logging, constants

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


def construct_uid(query_id: str, start_idx: int, end_idx: int):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    try:
        trial_name = f"{constants.experiment_name()}-{constants.trial_name()}"
    except Exception as e:
        trial_name = "test"
    uid = f"{timestamp}-{trial_name}-{query_id}-case-{start_idx}-{end_idx}"
    return uid
