import dataclasses
import os
import pickle
import getpass
import shutil
from typing import Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

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
    hf_path: str = ""
    checkpoint_path: str = ""


class Recover:
    def __init__(self, config: RecoverConfig):
        self.config = config

    def get_save_checkpoint_path(
        self,
        name: str,
    ):
        path = os.path.join(
            f"{self.config.fileroot}/recover/{getpass.getuser()}/{self.config.experiment_name}/{self.config.trial_name}/models",
            name,
        )

        os.makedirs(path, exist_ok=True)
        return path


    def get_save_huggingface_checkpoint_path(
        self,
        name: str,
    ):
        path = os.path.join(
            f"{self.config.fileroot}/recover/{getpass.getuser()}/{self.config.experiment_name}/{self.config.trial_name}/models",
            name,
        )
        os.makedirs(path, exist_ok=True)
        return path

    def get_save_meta_path(
        self,
        name: str,
    ):
        path = os.path.join(
            f"{self.config.fileroot}/recover/{getpass.getuser()}/{self.config.experiment_name}/{self.config.trial_name}/metas",
            name,
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
        name: str = "latest_checkpoint",
        tokenizer: PreTrainedTokenizerFast | None = None,
        base_model_path: str | None = None,
        disable_save_hf: bool = True,
    ):
        # Determine checkpoint name based on global_step parity
        checkpoint_name = "latest_checkpoint_odd" if global_step % 2 else "latest_checkpoint_even"
        symlink_name = name
        
        # Clear target directory
        target_dir = self.get_save_checkpoint_path(checkpoint_name)
        if os.path.exists(target_dir):
            import shutil
            shutil.rmtree(target_dir)
        os.makedirs(target_dir, exist_ok=True)

        # save hf model
        if not disable_save_hf:
            hf_path = self.get_save_huggingface_checkpoint_path(f"{checkpoint_name}/huggingface")
            weight_format = "huggingface"
            with_optim = False
            meta = SaveLoadMeta(
                path=hf_path,
                weight_format=weight_format,
                global_step=global_step,
                with_optim=with_optim,
                tokenizer=tokenizer,
                base_model_path=base_model_path,
            )
            engine.save(meta)
            logger.info(f"[Recover] Saved hf model to {hf_path} success.")

        # save checkpoint
        weight_format = "mcore"
        with_optim = True
        meta = SaveLoadMeta(
            path=target_dir,
            weight_format=weight_format,
            global_step=global_step,
            with_optim=with_optim,
            tokenizer=tokenizer,
            base_model_path=base_model_path,
        )
        engine.save(meta)
        logger.info(f"[Recover] Saved checkpoint to {target_dir} success.")

        # save meta info
        self.save_meta_info(epoch, step, global_step, dataloader_state, name)

        # Create/update symlink
        symlink_path = self.get_save_checkpoint_path(symlink_name)
        if os.path.exists(symlink_path):
            if os.path.islink(symlink_path) or os.path.isfile(symlink_path):
                os.remove(symlink_path)
            else:
                shutil.rmtree(symlink_path)
        os.symlink(target_dir, symlink_path)
        logger.info(f"[Recover] global_step: {global_step} Created symlink {symlink_name} -> {checkpoint_name}")

        # Async cleanup of the other checkpoint dir
        other_checkpoint = "latest_checkpoint_even" if global_step % 2 else "latest_checkpoint_odd"
        other_dir = self.get_save_checkpoint_path(other_checkpoint)
        
        def cleanup_dir(dir_path):
            try:
                if os.path.exists(dir_path):
                    shutil.rmtree(dir_path)
                    logger.info(f"[Recover] Async cleanup completed for {dir_path}")
            except Exception as e:
                logger.error(f"[Recover] Async cleanup failed for {dir_path}: {str(e)}")

        # Start async cleanup in thread pool
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(cleanup_dir, other_dir)
            logger.info(f"[Recover] Started async cleanup for {other_dir}")

    def save_meta_info(self, epoch: int, step: int, global_step: int, dataloader_state: dict, name: str = "latest_checkpoint"):
        # Determine meta name based on global_step parity
        meta_name = "latest_checkpoint_odd" if global_step % 2 else "latest_checkpoint_even"
        symlink_name = name
        
        # Save meta to odd/even directory
        path = self.get_save_meta_path(meta_name)
        
        # meta info, use symlink_name
        hf_path = self.get_save_checkpoint_path(f"{symlink_name}/huggingface")
        checkpoint_path = self.get_save_checkpoint_path(symlink_name)
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

        # Create/update meta symlink
        symlink_path = self.get_save_meta_path(symlink_name)
        if os.path.exists(symlink_path):
            if os.path.islink(symlink_path) or os.path.isfile(symlink_path):
                os.remove(symlink_path)
            else:
                shutil.rmtree(symlink_path)
        os.symlink(path, symlink_path)
        logger.info(f"[Recover] global_step: {global_step} Created meta symlink {symlink_name} -> {meta_name}")

        # Sync cleanup of the other meta dir
        other_meta = "latest_checkpoint_even" if global_step % 2 else "latest_checkpoint_odd"
        other_dir = self.get_save_meta_path(other_meta)
        try:
            if os.path.exists(other_dir):
                shutil.rmtree(other_dir)
                logger.info(f"[Recover] Sync cleanup completed for {other_dir}")
        except Exception as e:
            logger.error(f"[Recover] Sync cleanup failed for {other_dir}: {str(e)}")

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
