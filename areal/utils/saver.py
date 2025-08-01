import getpass
import os

from transformers import AutoProcessor, PreTrainedTokenizerFast

from areal.api.cli_args import SaverConfig
from areal.api.engine_api import TrainEngine
from areal.api.io_struct import FinetuneSpec, SaveLoadMeta
from realhf.base import timeutil


class Saver:

    def __init__(self, config: SaverConfig, ft_spec: FinetuneSpec, for_recover: bool):
        self.config = config
        self.ft_sepc = ft_spec
        self.for_recover = for_recover
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
            f"{config.fileroot}/checkpoints/{getpass.getuser()}/{config.experiment_name}/{config.trial_name}",
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
            Saver.get_save_checkpoint_root(config, name),
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
        name: str = "default",
        tokenizer: PreTrainedTokenizerFast | None = None,
        processor: AutoProcessor | None = None,
        base_model_path: str | None = None,
    ):
        if not self.freq_ctl.check(
            epochs=int(step == self.ft_sepc.steps_per_epoch - 1), steps=1
        ):
            return
        path = Saver.get_save_checkpoint_path(
            self.config, epoch, step, global_step, name
        )
        weight_format = "hf"
        with_optim = False
        if self.for_recover:
            weight_format = "dcp"
            with_optim = True

        meta = SaveLoadMeta(
            path=path,
            weight_format=weight_format,
            with_optim=with_optim,
            tokenizer=tokenizer,
            processor=processor,
            base_model_path=base_model_path,
        )
        engine.save(meta)
