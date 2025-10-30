import dataclasses
import getpass
import os
import pickle
import shutil

from transformers import PreTrainedTokenizerFast

from realhf.base import logging

from areal.api.cli_args import RecoverConfig
from areal.api.controller_api import TrainController
from areal.api.io_struct import SaveLoadMeta

logger = logging.getLogger("recover")


@dataclasses.dataclass
class RecoverInfo:
    epoch: int = 0
    epoch_step: int = 0
    global_step: int = 0
    dataloader_state: dict = dataclasses.field(default_factory=dict)
    step_ctl_state: dict = dataclasses.field(default_factory=dict)
    rollout_buffer_state: dict = dataclasses.field(default_factory=dict)
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
        ctl: TrainController,
        epoch: int,
        step: int,
        global_step: int,
        dataloader_state: dict,
        rollout_buffer_state: dict = {},
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

            # save的时候如果失败了，保存路径还会存在，所以再次save的时候要清空一下
            if os.path.exists(path):
                logger.info(f"begin remove {path}")
                shutil.rmtree(path)
                logger.info(f"{path} remove content success.")
            os.makedirs(path, exist_ok=True)
            logger.info(f"{path} recreate success.")

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
            ctl.save(meta)
            logger.info(f"Saved hf model to {path} success.")

        # save checkpoint
        path = self.get_save_checkpoint_path(epoch, step, global_step, name)

        # save的时候如果失败了，保存路径还会存在，所以再次save的时候要清空一下
        if os.path.exists(path):
            logger.info(f"begin remove {path}")
            shutil.rmtree(path)
            logger.info(f"{path} remove content success.")
        os.makedirs(path, exist_ok=True)
        logger.info(f"{path} recreate success.")

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
        ctl.save(meta)
        logger.info(f"Saved checkpoint to {path} success.")

        # save meta info
        self.save_meta_info(
            epoch, step, global_step, dataloader_state, rollout_buffer_state, name
        )

    def save_meta_info(
        self,
        epoch: int,
        step: int,
        global_step: int,
        dataloader_state: dict,
        rollout_buffer_state: dict,
        name: str,
    ):
        path = self.get_save_meta_path(epoch, step, global_step, name)
        hf_path = self.get_save_checkpoint_path(
            epoch, step, global_step, f"{name}/huggingface"
        )
        checkpoint_path = self.get_save_checkpoint_path(epoch, step, global_step, name)
        recover_info = RecoverInfo(
            epoch=epoch,
            epoch_step=step,
            global_step=global_step,
            dataloader_state=dataloader_state,
            rollout_buffer_state=rollout_buffer_state,
            hf_path=hf_path,
            checkpoint_path=checkpoint_path,
        )
        with open(os.path.join(path, "recover_info.pkl"), "wb") as f:
            pickle.dump(recover_info, f)
        logger.info(f"Saved recover meta info to {path} success.")

    @staticmethod
    def load(path: str) -> tuple[bool, RecoverInfo | None]:
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
