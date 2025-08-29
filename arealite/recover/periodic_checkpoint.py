import dataclasses
import os
import pickle
import getpass
from typing import Optional, Tuple

from transformers import PreTrainedTokenizerFast

from arealite.api.cli_args import RecoverConfig
from arealite.api.engine_api import TrainEngine
from arealite.api.io_struct import SaveLoadMeta

from realhf.base import logging

logger = logging.getLogger("recover")

@dataclasses.dataclass
class RecoverInfo:
    epoch: int = 0
    epoch_step: int = 0
    global_step: int = 0
    dataloader_state: dict = dataclasses.field(default_factory=dict)
    step_ctl_state: dict = dataclasses.field(default_factory=dict)
    hf_path: str = ""
    checkpoint_path: str = ""


class Recover:
    def __init__(self, config: RecoverConfig):
        self.config = config

    def get_save_checkpoint_root(
        self,
        name: str,
    ):
        path = os.path.join(
            f"{self.config.fileroot}/recover/{getpass.getuser()}/{self.config.experiment_name}/{self.config.trial_name}/models",
            name,
        )
        os.makedirs(path, exist_ok=True)
        return path

    def get_save_checkpoint_path(
        self,
        epoch: int,
        step: int,
        globalstep: int,
        name: str,
    ):
        path = os.path.join(
            self.get_save_checkpoint_root(name), 
            f"epoch{epoch}epochstep{step}globalstep{globalstep}",
        )
        os.makedirs(path, exist_ok=True)
        return path

    def get_save_huggingface_checkpoint_path(
        self,
        epoch: int,
        step: int,
        globalstep: int,
        name: str,
    ):
        path = os.path.join(
            self.get_save_checkpoint_root(name),
            f"epoch{epoch}epochstep{step}globalstep{globalstep}",
        )
        os.makedirs(path, exist_ok=True)
        return path

    def get_save_meta_root(
        self,
        name: str,
    ):
        path = os.path.join(
            f"{self.config.fileroot}/recover/{getpass.getuser()}/{self.config.experiment_name}/{self.config.trial_name}/metas",
            name,
        )
        os.makedirs(path, exist_ok=True)
        return path

    def get_save_meta_path(
        self,
        epoch: int,
        step: int,
        globalstep: int,
        name: str,
    ):
        path = os.path.join(
            self.get_save_meta_root(name),
            f"epoch{epoch}epochstep{step}globalstep{globalstep}",
        )
        os.makedirs(path, exist_ok=True)
        return path

    def save(
        self,
        engine: TrainEngine,
        epoch: int,
        step: int,
        global_step: int,
        dataloader_state: dict,
        name: str = "periodic_checkpoint",
        tokenizer: PreTrainedTokenizerFast | None = None,
        base_model_path: str | None = None,
        disable_save_hf: bool = False,
    ):
        # save hf model
        if not disable_save_hf:
            path = self.get_save_huggingface_checkpoint_path(
                epoch, step, global_step, f"{name}/huggingface"
            )
            weight_format = "huggingface"
            with_optim = False
            meta = SaveLoadMeta(
                path=path,
                weight_format=weight_format,
                global_step=global_step,
                with_optim=with_optim,
                tokenizer=tokenizer,
                base_model_path=base_model_path,
            )
            engine.save(meta)
            logger.info(f"[Recover] Saved hf model to {path} success.")

        # save checkpoint
        path = self.get_save_checkpoint_path(
            epoch, step, global_step, name
        )
        weight_format = "mcore"
        with_optim = True
        meta = SaveLoadMeta(
            path=path,
            weight_format=weight_format,
            global_step=global_step,
            with_optim=with_optim,
            tokenizer=tokenizer,
            base_model_path=base_model_path,
        )
        engine.save(meta)
        logger.info(f"[Recover] Saved checkpoint to {path} success.")

        # save meta info
        self.save_meta_info(epoch, step, global_step, dataloader_state, name)

    def save_meta_info(self, epoch: int, step: int, global_step: int, dataloader_state: dict, name: str):
        path = self.get_save_meta_path(epoch, step, global_step, name)
        hf_path = self.get_save_checkpoint_path(epoch, step, global_step, f"{name}/huggingface")
        checkpoint_path = self.get_save_checkpoint_path(epoch, step, global_step, name)
        recover_info = RecoverInfo(
            epoch=epoch,
            epoch_step=step,
            global_step=global_step,
            dataloader_state=dataloader_state,
            hf_path=hf_path,
            checkpoint_path=checkpoint_path
        )
        with open(os.path.join(path, "recover_info.pkl"), "wb") as f:
            pickle.dump(recover_info, f)
        logger.info(f"[Recover] Saved recover meta info to {path} success.")

    @staticmethod
    def load(path: str) -> Tuple[bool, Optional[RecoverInfo]]:
        try:
            with open(path, "rb") as f:
                recover_info = pickle.load(f)
            return True, recover_info
        except FileNotFoundError:
            logger.warning(f"Recover info not found at {path}")
            return False, None
        except Exception as e:
            logger.error(f"Failed to load recover info from {path}: {str(e)}")
            return False, None
