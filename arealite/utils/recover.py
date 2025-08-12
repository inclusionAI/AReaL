import dataclasses
import os
import pathlib
import pickle
import getpass
from typing import Dict, List, Optional, Tuple

from transformers import PreTrainedTokenizerFast

from arealite.api.cli_args import SaverConfig
from arealite.api.engine_api import TrainEngine
from arealite.api.io_struct import FinetuneSpec, SaveLoadMeta

from realhf.base import timeutil
from realhf.base import constants, logging

logger = logging.getLogger("recover")

RECOVER_INFO_PATH = None

@dataclasses.dataclass
class RecoverInfo:
    epoch: int = 0
    epoch_step: int = 0
    global_step: int = 0
    dataloader_state: dict = dataclasses.field(default_factory=dict)
    epoch_ctl_state: dict = dataclasses.field(default_factory=dict)
    step_ctl_state: dict = dataclasses.field(default_factory=dict)
    time_ctl_state: dict = dataclasses.field(default_factory=dict)
    hf_path: str = ""
    checkpoint_path: str = ""


class Recover:
    def __init__(self, config: SaverConfig, ft_spec: FinetuneSpec):
        self.config = config
        self.ft_spec = ft_spec
        self.freq_ctl = timeutil.EpochStepTimeFreqCtl(
            freq_epoch=config.freq_epochs,
            freq_step=config.freq_steps,
            freq_sec=config.freq_secs,
        )

    @staticmethod
    def get_save_checkpoint_root(
        config: SaverConfig,
        name: str = "default",
    ):
        path = os.path.join(
            f"{config.fileroot}/recover/models/{getpass.getuser()}/{config.experiment_name}/{config.trial_name}",
            name,
        )
        os.makedirs(path, exist_ok=True)
        return path

    @staticmethod
    def get_save_checkpoint_path(
        config: SaverConfig,
        epoch: int,
        step: int,
        globalstep: int,
        name: str = "default",
    ):
        path = os.path.join(
            Recover.get_save_checkpoint_root(config, name),
            f"epoch{epoch}epochstep{step}globalstep{globalstep}",
        )
        os.makedirs(path, exist_ok=True)
        return path

    @staticmethod
    def get_save_meta_root(
        config: SaverConfig,
        name: str = "default",
    ):
        path = os.path.join(
            f"{config.fileroot}/recover/metas/{getpass.getuser()}/{config.experiment_name}/{config.trial_name}",
            name,
        )
        os.makedirs(path, exist_ok=True)
        return path

    @staticmethod
    def get_save_meta_path(
        config: SaverConfig,
        epoch: int,
        step: int,
        globalstep: int,
        name: str = "default",
    ):
        path = os.path.join(
            Recover.get_save_meta_root(config, name),
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
        name: str = "default",
        tokenizer: PreTrainedTokenizerFast | None = None,
        base_model_path: str | None = None,
        disable_save_hf: bool = False,
    ):
        # save hf model
        if not disable_save_hf:
            path = Recover.get_save_checkpoint_path(
                self.config, epoch, step, global_step, f"{name}/huggingface"
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
        path = Recover.get_save_checkpoint_path(
            self.config, epoch, step, global_step, name
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
        self.save_meta_info(epoch, step, global_step, dataloader_state)

    def save_meta_info(self, epoch: int, step: int, global_step: int, dataloader_state: dict, name: str = "default"):
        path = self.get_save_meta_path(self.config, epoch, step, global_step, name)
        hf_path = self.get_save_checkpoint_path(self.config, epoch, step, global_step, name)
        checkpoint_path = self.get_save_checkpoint_path(self.config, epoch, step, global_step, name) #TODO,diff path
        recover_info = RecoverInfo(
            epoch=epoch,
            epoch_step=step,
            global_step=global_step,
            dataloader_state=dataloader_state,
            epoch_ctl_state=self.freq_ctl.epoch_ctl.state_dict(),
            step_ctl_state=self.freq_ctl.step_ctl.state_dict(),
            time_ctl_state=self.freq_ctl.time_ctl.state_dict(),
            hf_path=hf_path,
            checkpoint_path=checkpoint_path
        )
        with open(os.path.join(path, "recover_info.pkl"), "wb") as f:
            pickle.dump(recover_info, f)
        logger.info(f"[Recover] Saved recover meta info to {path} success.")

    def load_ctl_states(self, recover_info: RecoverInfo):
        if hasattr(recover_info, 'epoch_ctl_state'):
            self.freq_ctl.epoch_ctl.load_state_dict(recover_info.epoch_ctl_state)
        if hasattr(recover_info, 'step_ctl_state'):
            self.freq_ctl.step_ctl.load_state_dict(recover_info.step_ctl_state)
        if hasattr(recover_info, 'time_ctl_state'):
            self.freq_ctl.time_ctl.load_state_dict(recover_info.time_ctl_state)

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
