# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import dataclasses
import os
import pathlib
import pickle
from typing import Dict, List, Optional, Tuple

from realhf.base import constants, logging

logger = logging.getLogger("recover")

RECOVER_INFO_PATH = None


@dataclasses.dataclass
class StepInfo:
    epoch: int = 0
    epoch_step: int = 0
    global_step: int = 0


@dataclasses.dataclass
class RecoverInfo:
    # Recover start is the counter of the next RLHF interation
    # w.r.t. the counter of the saved checkpoint
    recover_start: StepInfo
    # Last step info is the counter of the saved checkpoint.
    # It exactly lags beind recover_start by 1 iteration.
    last_step_info: StepInfo

    save_ctl_info: Dict
    ckpt_ctl_info: Dict
    eval_ctl_info: Dict

    data_loading_dp_idx: int

    hash_vals_to_ignore: List[int] = dataclasses.field(default_factory=list)


def dump_recover_info(args, recover_info: RecoverInfo):
    global RECOVER_INFO_PATH
    if RECOVER_INFO_PATH is None:
        RECOVER_INFO_PATH = os.path.join(
            constants.get_save_path(args),
            "recover_info.pkl",
        )
        os.makedirs(os.path.dirname(RECOVER_INFO_PATH), exist_ok=True)
    with open(RECOVER_INFO_PATH, "wb") as f:
        pickle.dump(recover_info, f)


def load_recover_info(args) -> Tuple[int, Optional[RecoverInfo]]:
    if os.environ.get("REAL_RECOVER_RUN", "0") != "1":
        return False, None
    global RECOVER_INFO_PATH
    if RECOVER_INFO_PATH is None:
        RECOVER_INFO_PATH = os.path.join(
            constants.get_save_path(args),
            "recover_info.pkl",
        )
        os.makedirs(os.path.dirname(RECOVER_INFO_PATH), exist_ok=True)
    try:
        with open(RECOVER_INFO_PATH, "rb") as f:
            return True, pickle.load(f)
    except FileNotFoundError:
        logger.warning(
            f"Resume info not found at {RECOVER_INFO_PATH}. "
            f"This should not be a resumed experiment!"
        )
        return False, None


class InValidRecoverCkpt(Exception):
    pass


def discover_ckpt(args) -> Tuple[str, List[str], RecoverInfo]:
    expr_name, trial_name = args.experiment_name, args.trial_name
    recover_info_file = pathlib.Path(constants.get_save_path(args)) / "recover_info.pkl"
    if os.path.exists(str(recover_info_file)):
        with open(recover_info_file, "rb") as f:
            info: RecoverInfo = pickle.load(f)
        if info.last_step_info.epoch < 0:
            msg = (
                f"Recover checkpoint is not valid. "
                f"Expected last_step_info.epoch >= 0, "
                f"but found {info.last_step_info.epoch}"
            )
            raise InValidRecoverCkpt(msg)
        model_save_dir = pathlib.Path(constants.get_save_path(args))
        model_ckpt_dirs = []
        for role in os.listdir(model_save_dir):
            if "dataset_indices" in role:
                continue
            if not os.path.isdir(model_save_dir / role):
                continue
            ckpt_dir = (
                model_save_dir
                / role
                / f"epoch{info.last_step_info.epoch + 1}epochstep{info.last_step_info.epoch_step + 1}globalstep{info.last_step_info.global_step + 1}"
            )
            if not ckpt_dir.exists():
                raise InValidRecoverCkpt(
                    f"Guessed checkpoint path does not exist: {ckpt_dir}."
                )
            model_ckpt_dirs.append(str(ckpt_dir))
        return str(recover_info_file), model_ckpt_dirs, info
    raise InValidRecoverCkpt(f"Recover checkpoint not found at: {recover_info_file}")
