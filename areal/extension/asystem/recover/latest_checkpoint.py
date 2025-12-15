import dataclasses
import getpass
import os
import pickle
import shutil
from concurrent.futures import ThreadPoolExecutor

from transformers import PreTrainedTokenizerFast

from areal.api.cli_args import RecoverConfig
from areal.api.io_struct import SaveLoadMeta
from areal.extension.asystem.controller import TrainController
from areal.utils import logging
from areal.utils.errors import FrameworkError

logger = logging.getLogger("recover")


@dataclasses.dataclass
class RecoverInfo:
    epoch: int = 0
    epoch_step: int = 0
    global_step: int = 0
    dataloader_state: dict = dataclasses.field(default_factory=dict)
    rollout_buffer_state: dict = dataclasses.field(default_factory=dict)
    hf_path: str = ""
    checkpoint_path: str = ""
    ref_model_path: str = ""


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
        ctl: TrainController,
        epoch: int,
        step: int,
        global_step: int,
        dataloader_state: dict,
        rollout_buffer_state: dict = {},
        name: str = "latest_checkpoint",
        tokenizer: PreTrainedTokenizerFast | None = None,
        base_model_path: str | None = None,
        disable_save_hf: bool = True,
        ref_model_path: str | None = None,
    ):
        # Determine checkpoint name based on global_step parity
        checkpoint_name = (
            "latest_checkpoint_odd" if global_step % 2 else "latest_checkpoint_even"
        )
        symlink_name = name

        # Clear target directory
        target_dir = self.get_save_checkpoint_path(checkpoint_name)
        if os.path.exists(target_dir):
            logger.info(f"begin remove {target_dir}")
            shutil.rmtree(target_dir)
            logger.info(f"{target_dir} remove content success.")
        os.makedirs(target_dir, exist_ok=True)
        logger.info(f"{target_dir} recreate success.")

        # save hf model
        if not disable_save_hf:
            hf_path = self.get_save_huggingface_checkpoint_path(
                f"{checkpoint_name}/huggingface"
            )

            # save的时候如果失败了，保存路径还会存在，所以再次save的时候要清空一下
            if os.path.exists(hf_path):
                logger.info(f"begin remove {hf_path}")
                shutil.rmtree(hf_path)
                logger.info(f"{hf_path} remove content success.")
            os.makedirs(hf_path, exist_ok=True)
            logger.info(f"{hf_path} recreate success.")

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
            ctl.save(meta)
            logger.info(f"Saved hf model to {hf_path} success.")

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
        ctl.save(meta)
        logger.info(f"Saved checkpoint to {target_dir} success.")

        # save meta info
        self.save_meta_info(
            epoch,
            step,
            global_step,
            dataloader_state,
            rollout_buffer_state,
            name,
            ref_model_path=ref_model_path,
        )

        # Create/update symlink
        symlink_path = self.get_save_checkpoint_path(symlink_name)
        if os.path.exists(symlink_path):
            if os.path.islink(symlink_path) or os.path.isfile(symlink_path):
                os.remove(symlink_path)
            else:
                shutil.rmtree(symlink_path)
        os.symlink(target_dir, symlink_path)
        logger.info(
            f"global_step: {global_step} Created symlink {symlink_name} -> {checkpoint_name}"
        )

        # Async cleanup of the other checkpoint dir
        other_checkpoint = (
            "latest_checkpoint_even" if global_step % 2 else "latest_checkpoint_odd"
        )
        other_dir = self.get_save_checkpoint_path(other_checkpoint)

        def cleanup_dir(dir_path):
            try:
                if os.path.exists(dir_path):
                    shutil.rmtree(dir_path)
                    logger.info(f"Async cleanup completed for {dir_path}")
            except Exception as e:
                logger.error(f"Async cleanup failed for {dir_path}: {str(e)}")
                raise FrameworkError("FrameworkError", "RecoverError", e)

        # Start async cleanup in thread pool
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(cleanup_dir, other_dir)
            logger.info(f"Started async cleanup for {other_dir}")

    def save_meta_info(
        self,
        epoch: int,
        step: int,
        global_step: int,
        dataloader_state: dict,
        rollout_buffer_state: dict,
        name: str = "latest_checkpoint",
        ref_model_path: str | None = None,
    ):
        # Determine meta name based on global_step parity
        meta_name = (
            "latest_checkpoint_odd" if global_step % 2 else "latest_checkpoint_even"
        )
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
            rollout_buffer_state=rollout_buffer_state,
            hf_path=hf_path,
            checkpoint_path=checkpoint_path,
            ref_model_path=ref_model_path or "",
        )
        with open(os.path.join(path, "recover_info.pkl"), "wb") as f:
            pickle.dump(recover_info, f)
        logger.info(f"Saved recover meta info to {path} success.")

        # Create/update meta symlink
        symlink_path = self.get_save_meta_path(symlink_name)
        if os.path.exists(symlink_path):
            if os.path.islink(symlink_path) or os.path.isfile(symlink_path):
                os.remove(symlink_path)
            else:
                shutil.rmtree(symlink_path)
        os.symlink(path, symlink_path)
        logger.info(
            f"global_step: {global_step} Created meta symlink {symlink_name} -> {meta_name}"
        )

        # Sync cleanup of the other meta dir
        other_meta = (
            "latest_checkpoint_even" if global_step % 2 else "latest_checkpoint_odd"
        )
        other_dir = self.get_save_meta_path(other_meta)
        try:
            if os.path.exists(other_dir):
                shutil.rmtree(other_dir)
                logger.info(f"Sync cleanup completed for {other_dir}")
        except Exception as e:
            logger.error(f"Sync cleanup failed for {other_dir}: {str(e)}")
            raise FrameworkError("FrameworkError", "RecoverError", e)

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
